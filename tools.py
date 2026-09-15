"""
智能客服 v4.0 —— 工具模块
升级：从关键词匹配 → 向量语义检索（课6 RAG）

v3.0: search_knowledge_base() 用 if "退款" in line 关键词匹配
v4.0: retrieve_relevant() 用 Chroma 向量语义检索
v8.1: 向量库改为"按内容指纹复用"，修掉每次启动都重复写入的 bug

【v8.1 修的是什么】
旧代码是 Chroma.from_documents(..., persist_directory="./chroma_db")，
它每次启动都把同一批知识块【追加】进同一个集合里。跑 10 次，知识库里
就有 10 份重复内容（实测 chroma.sqlite3 里同一段话出现了约 10 次），
导致 similarity_search(k=3) 经常返回 3 个几乎一样的块，白白占用上下文。

现在的做法：
  集合名 = "kb_" + 知识库内容指纹
  → 内容没变，就是同一个集合，直接复用（不再重新向量化，启动更快）
  → 内容/切块参数/向量模型变了，指纹就变，自动用新集合重建
  → 永远不会重复累积
"""
import hashlib
import os

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

# 切块参数提成常量：它一变，指纹就变，会自动触发重建
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]

# 本地中文向量模型（离线可用，不需要 API Key）
MODEL_REPO = "shibing624/text2vec-base-chinese"

# 旧版本用的默认集合名（里面已经堆积了重复数据），初始化成功后顺手清掉
LEGACY_COLLECTIONS = ["langchain"]

# ========== 全局单例：只加载一次 ==========
_vectorstore = None


def _count(store) -> int:
    """当前集合里有多少个块"""
    return store._collection.count()


def _knowledge_fingerprint(text: str) -> str:
    """算知识库的"内容指纹"

    把【文件内容 + 向量模型 + 切块参数】一起做 SHA256。
    其中任何一项变了，指纹就变 → 集合名变 → 自动用全新的集合重建，
    所以永远不会出现"新旧内容混在一起、越积越多"的情况。
    """
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    h.update(f"|{MODEL_REPO}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode("utf-8"))
    return h.hexdigest()


def _load_embeddings() -> HuggingFaceEmbeddings:
    """加载本地中文向量模型

    用 local_files_only=True 从本地缓存里找，避免联网下载缺的配置文件
    （配合上面的离线模式，解决 .no_exist 文件反复重试的问题）
    """
    model_path = snapshot_download(repo_id=MODEL_REPO, local_files_only=True)
    return HuggingFaceEmbeddings(model_name=model_path)


def _split_knowledge() -> list:
    """第1、2步：加载文档 → 切块（课6 L1 + L2）"""
    loader = TextLoader(KNOWLEDGE_FILE, encoding="utf-8")
    docs = loader.load()

    # chunk_size 700：让每个块内容更完整，避免 FAQ 问答被切碎
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
    print(f"  文档已切分为 {len(chunks)} 个块")
    return chunks


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

    print("  [初始化 RAG：加载文档 → 切块 → 向量化]")

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

    # 顺手清掉旧版本堆积的重复集合
    _drop_legacy_collections(embeddings)

    _vectorstore = store


def retrieve_relevant(query: str, k: int = 3) -> str:
    """
    语义检索：搜索与 query 最相关的文档段落（课6 L4）
    替代 v3.0 的 search_knowledge_base() 关键词匹配
    """
    _init_vectorstore()
    chunks = _vectorstore.similarity_search(query, k=k)
    return "\n---\n".join([chunk.page_content for chunk in chunks])


if __name__ == "__main__":
    # 手动自检用法：
    #   python tools.py            检查/构建向量库，并跑一条测试检索
    #   python tools.py --rebuild  强制重建（改了 knowledge.txt 但指纹没生效时用）
    import sys

    _init_vectorstore(force_rebuild="--rebuild" in sys.argv)
    print(f"\n向量库块数：{_count(_vectorstore)}")
    print("\n[测试检索] 退款政策是什么？")
    print(retrieve_relevant("退款政策是什么", k=3))
