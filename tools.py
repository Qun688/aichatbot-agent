"""
智能客服 v4.0 —— 工具模块
升级：从关键词匹配 → 向量语义检索（课6 RAG）

v3.0: search_knowledge_base() 用 if "退款" in line 关键词匹配
v4.0: retrieve_relevant() 用 Chroma 向量语义检索
"""
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

# ========== 全局单例：只加载一次 ==========
_vectorstore = None


def _init_vectorstore():
    """初始化向量库（首次调用时执行）"""
    global _vectorstore
    if _vectorstore is not None:
        return

    print("  [初始化 RAG：加载文档 → 切块 → 向量化]")

    # 第1步：加载文档（课6 L1）
    loader = TextLoader("knowledge.txt", encoding="utf-8")
    docs = loader.load()

    # 第2步：切分文档（课6 L2）
    # chunk_size 调大到 700：让每个块内容更完整，避免 FAQ 问答被切碎
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"  文档已切分为 {len(chunks)} 个块")

    # 第3步：向量化 + 存入 Chroma（课6 L3）
    # 使用本地中文模型，离线可用，无需 API Key
    # 用 local_files_only=True 从本地缓存找路径，避免联网下载缺的配置文件
    model_path = snapshot_download(
        repo_id="shibing624/text2vec-base-chinese",
        local_files_only=True,
    )
    embeddings = HuggingFaceEmbeddings(model_name=model_path)

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("  向量库初始化完成")


def retrieve_relevant(query: str, k: int = 3) -> str:
    """
    语义检索：搜索与 query 最相关的文档段落（课6 L4）
    替代 v3.0 的 search_knowledge_base() 关键词匹配
    """
    _init_vectorstore()
    chunks = _vectorstore.similarity_search(query, k=k)
    return "\n---\n".join([chunk.page_content for chunk in chunks])
