"""
智能客服 v4.0 —— 工具模块
升级：从关键词匹配 → 向量语义检索（课6 RAG）

v3.0: search_knowledge_base() 用 if "退款" in line 关键词匹配
v4.0: retrieve_relevant() 用 Chroma 向量语义检索
v8.1: 向量库改为"按内容指纹复用"，修掉每次启动都重复写入的 bug
v9.0: 切块改为"先按【章节】切、再在章节内细分"，修掉检索串题的问题
v9.1: 【常见问题】改为按"一问一答"切块，问题不再被合并稀释
      （v9.0 实测"怎么开发票"检索不到，就是因为发票那条问答
        和支付方式、数据导出等 4 条不相关的问答挤在同一个块里）
v10.0: 检索升级为"BM25 字面匹配 + 向量语义匹配（RRF 融合）+ cross-encoder 重排"，
       实现在 retriever.py；量化评测见 evaluate_retrieval.py

【v9.0 修的是什么】
旧代码用 RecursiveCharacterTextSplitter 直接对整篇文档按 700 字硬切，
结果【退换政策】这种只有 6 行的小节，和【产品与服务】【产品与定价】这些
完全不相关的内容被塞进了同一个块。实测问"退款政策是什么"：

    第1名 → 荣誉资质 + 产品与服务 + 产品与定价   （完全不相关）
    第2名 → 常见问题                              （部分相关）
    第3名 → 才轮到真正的【退换政策】              （真正相关，却排最后）

现在知识库只有 6 个块、k=3 等于捞出半个库，所以还能歪打正着。
但知识库一旦变大，真正的政策块就会被挤出 top-3，
客服就会在"没有拿到政策原文"的情况下编答案（幻觉）。

改成先按【章节标题】切开（共 9 个章节），章节内部再按 400 字细分，
保证每个块主题单一，检索排名就正过来了。
"""
import hashlib
import os
import re

# 必须在导入 HuggingFace 之前设置
# 1. 镜像站：防止被墙
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 2. 离线模式：模型已下载到本地，禁止联网检查（解决 .no_exist 文件反复重试的问题）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from huggingface_hub import snapshot_download

# ========== 路径与参数 ==========
# 用"本文件所在目录"来算路径，这样不管从哪个目录启动都不会找不到文件
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_FILE = os.path.join(BASE_DIR, "knowledge.txt")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")

# 章节内部的细分参数（整篇文档不再按这个尺寸硬切）
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]

# 切分策略的版本号：策略一改就把这里也改掉，
# 指纹会跟着变 → 向量库自动重建，不会拿旧切法的库来用
SPLIT_VERSION = "v9.1-section-qa"

# 问答格式的分隔点：knowledge.txt 的【常见问题】是 "Q: ...\nA: ..." 一段一对。
# 一问一答本身就是完整的语义单元，如果再按字数合并，
# 就会把"发票""密码""试用期"这些不相关的问题混进同一个向量里互相稀释。
QA_SPLIT_RE = re.compile(r"\n(?=Q[:：])")

# 本地中文向量模型（离线可用，不需要 API Key）
MODEL_REPO = "shibing624/text2vec-base-chinese"

# 旧版本用的默认集合名（里面堆积了重复数据），初始化成功后顺手清掉
LEGACY_COLLECTIONS = ["langchain"]

# ========== 全局单例：只加载一次 ==========
_vectorstore = None
_retriever = None      # 混合检索器（BM25 + 向量 + 重排），首次检索时构建


def _count(store) -> int:
    """当前集合里有多少个块"""
    return store._collection.count()


def _knowledge_fingerprint(text: str) -> str:
    """算知识库的"内容指纹"

    把【文件内容 + 向量模型 + 切分策略/参数】一起做 SHA256。
    其中任何一项变了，指纹就变 → 集合名变 → 自动用全新的集合重建，
    所以永远不会出现"新旧内容混在一起、越积越多"的情况。
    """
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(f"|{MODEL_REPO}|{SPLIT_VERSION}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode("utf-8"))
    return h.hexdigest()


def _load_embeddings() -> HuggingFaceEmbeddings:
    """加载本地中文向量模型

    用 local_files_only=True 从本地缓存里找，避免联网下载缺的配置文件
    （配合上面的离线模式，解决 .no_exist 文件反复重试的问题）
    """
    model_path = snapshot_download(repo_id=MODEL_REPO, local_files_only=True)
    return HuggingFaceEmbeddings(model_name=model_path)


def _split_by_section(text: str) -> list:
    """把知识库按【章节标题】切成 [(标题, 正文), ...]

    识别规则：整行就是 【xxx】 的行算章节标题（knowledge.txt 里共 9 个）。
    第一个标题之前的内容（文档大标题）并入第一节，避免产生"只有一个标题"的无效块。
    """
    sections = []
    title = None
    buf = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((title, body))

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("【") and stripped.endswith("】"):
            flush()          # 先收好上一节
            title = stripped  # 开新的一节
            buf = [line]
        else:
            buf.append(line)
    flush()

    # 把文档大标题并进第一节
    if sections and sections[0][0] is None:
        head = sections.pop(0)[1]
        if sections:
            first_title, first_body = sections[0]
            sections[0] = (first_title, f"{head}\n{first_body}")
    return sections


def _split_body(body: str, splitter) -> list:
    """切一节正文

    - 问答格式（【常见问题】）：按"一问一答"切，一条问答一个块
    - 其他章节：按字数细分（章节内部已经是同一主题，切碎一点没关系）
    """
    if QA_SPLIT_RE.search(body):
        pairs = [p.strip() for p in QA_SPLIT_RE.split(body) if p.strip()]
        pieces = []
        for pair in pairs:
            # 万一某一条问答特别长，再用字数兜底细分
            pieces.extend(splitter.split_text(pair) if len(pair) > CHUNK_SIZE else [pair])
        return pieces
    return splitter.split_text(body)


def _split_knowledge() -> list:
    """第1、2步：加载文档 → 先按章节切、章节内再细分（课6 L1 + L2）"""
    # 第1步：加载文档（课6 L1）
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    text = loader.load()[0].page_content

    # 第2步：切分文档（课6 L2）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    from langchain_core.documents import Document

    docs = []
    sections = _split_by_section(text)
    for title, body in sections:
        pieces = _split_body(body, splitter)
        for i, piece in enumerate(pieces):
            # 每块都带上章节标题：既是上下文，也让 embedding 更容易命中主题
            content = f"{title}\n{piece}" if title else piece
            docs.append(Document(
                page_content=content,
                metadata={"section": title or "", "source": "knowledge.txt", "part": i},
            ))

    print(f"  文档切成 {len(sections)} 个章节，共 {len(docs)} 个块")
    return docs


def _drop_legacy_collections(embeddings):
    """清理旧版本留下的重复集合（v8.0 及以前用的是默认名 langchain）

    清不掉也不影响使用——新代码只读新的指纹集合，旧集合只是白占点磁盘。
    """
    for name in LEGACY_COLLECTIONS:
        try:
            old = Chroma(
                collection_name=name,
                embedding_function=embeddings,
                persist_directory=PERSIST_DIR,
            )
            if _count(old) > 0:
                old.delete_collection()
                print(f"  已清理旧集合 '{name}'（历史重复数据）")
        except Exception:
            pass  # 集合不存在 / 删除失败都忽略


def _drop_stale_kb_collections(store, keep: str):
    """清理知识库改版后遗留的旧 kb_* 集合，只保留当前这个"""
    try:
        names = [
            c if isinstance(c, str) else getattr(c, "name", "")
            for c in store._client.list_collections()
        ]
    except Exception:
        return
    for name in names:
        if name.startswith("kb_") and name != keep:
            try:
                store._client.delete_collection(name)
                print(f"  已清理过期集合 '{name}'")
            except Exception:
                pass


def _init_vectorstore(force_rebuild: bool = False):
    """初始化向量库（首次调用时执行；内容没变则直接复用磁盘上的）

    Args:
        force_rebuild: True 时忽略已有集合，强制重新切块 + 向量化
    """
    global _vectorstore
    if _vectorstore is not None and not force_rebuild:
        return

    # 读知识库原文（用来算指纹）
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    fingerprint = _knowledge_fingerprint(text)
    collection_name = f"kb_{fingerprint[:16]}"

    embeddings = _load_embeddings()

    # 第3步：向量化（课6 L3）
    # 注意这里用 Chroma(...) 而不是 from_documents：
    # 集合不存在时它只是建一个空集合，不会像 from_documents 那样直接写数据
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    count = _count(store)
    if count > 0 and not force_rebuild:
        print(f"  [RAG] 复用已有向量库（集合 {collection_name}，{count} 个块，跳过向量化）")
        _vectorstore = store
        return

    print("  [初始化 RAG：加载文档 → 按章节切块 → 向量化]")

    if force_rebuild and count > 0:
        # 强制重建：先把当前集合清空，否则又会追加成重复数据
        try:
            store.delete_collection()
        except Exception:
            pass
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )

    chunks = _split_knowledge()
    store.add_documents(chunks)  # Chroma 会自动落盘
    print(f"  向量库初始化完成（{len(chunks)} 个块，集合 {collection_name}）")

    # 顺手清掉历史遗留的集合（旧的默认集合 + 旧版切法留下的 kb_* 集合）
    _drop_legacy_collections(embeddings)
    _drop_stale_kb_collections(store, collection_name)

    _vectorstore = store


def retrieve_relevant(query: str, k: int = 3) -> str:
    """
    检索与 query 最相关的文档段落

    v4.0 ：纯向量语义检索（课6 L4）
    v10.0：升级为"BM25 字面匹配 + 向量语义匹配（RRF 融合）+ cross-encoder 重排"。
           起因是评测脚本实测出纯向量的两个固有弱点：
           专有名词（问"客服电话是多少"）和否定表述（答案"仅支持一个设备登录"
           被问成"能多人同时登录吗"）都检索不到。实现在 retriever.py 里。
    """
    global _retriever
    _init_vectorstore()
    if _retriever is None:
        from retriever import HybridRetriever

        _retriever = HybridRetriever(_vectorstore)
    docs = _retriever.search(query, k=k)
    return "\n---\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    # 手动自检用法：
    #   python tools.py            检查/构建向量库，并跑几条测试检索看排名
    #   python tools.py --rebuild  强制重建（改了 knowledge.txt 但指纹没生效时用）
    import sys

    _init_vectorstore(force_rebuild="--rebuild" in sys.argv)
    print(f"\n向量库块数：{_count(_vectorstore)}")

    # 用几条典型问题验证检索排名（分数是距离，越小越相关）
    for q in ["退款政策是什么", "怎么开发票", "OA 专业版多少钱", "数据安全吗",
              "忘记密码怎么办", "有试用期吗"]:
        print(f"\n[检索] {q}")
        for doc, score in _vectorstore.similarity_search_with_score(q, k=3):
            section = doc.metadata.get("section", "?")
            print(f"  [{score:.3f}] {section} | {doc.page_content[:30]}...")
