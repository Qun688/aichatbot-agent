"""
混合检索 + 重排（v10.0）
========================
`evaluate_retrieval.py` 实测发现纯向量检索有两个固有弱点：

  1. 专有名词/数字：问"客服电话是多少"，向量语义相似度捞回来的是
     同章节的邻居块，电话号那一块反而排不进来。
  2. 否定表述：答案是"仅支持一个设备登录"，用户问"能多人同时登录吗"，
     语义方向相反，向量相似度不升反降。

两个标准解法（本模块实现）：

  1. 混合检索 —— BM25 负责"字面命中"（专有名词、数字、型号），
     向量负责"语义命中"，两路结果用 RRF（倒数排名融合）合并。
     RRF 的好处是不用调和两种分数的量纲，也不需要调权重。
  2. 重排 —— 先粗排取 top-N 候选，再用 cross-encoder 精排取 top-k，
     解决"命中但排名靠后"的问题。

设计上刻意不依赖 langchain 的 EnsembleRetriever：
一是少一层依赖，二是 RRF 自己实现只要十几行，面试时能讲清每一步。
"""
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# 重排模型（中文效果主流选择，约 1.1GB，首次使用会自动下载）
RERANK_MODEL = "BAAI/bge-reranker-base"

# 粗排候选数：先取这么多，重排后只留 k 个给大模型
DEFAULT_CANDIDATES = 10

# RRF 公式里的平滑常数，取自原论文的默认值 60
RRF_K = 60

# 全局单例：BM25 索引和重排模型都只构建/加载一次
_reranker = None
_reranker_tried = False


def tokenize(text: str) -> list:
    """中文分词 —— BM25 是按"词"匹配的，不切词的话中文整句会被当成一个词

    jieba.lcut 是精确模式分词，例如
      "客服电话是多少" → ['客服', '电话', '是', '多少']
    """
    import jieba

    return [tok for tok in jieba.lcut(text) if tok.strip()]


def load_reranker():
    """加载 cross-encoder 重排模型；失败就返回 None（自动降级，不影响主流程）"""
    global _reranker, _reranker_tried
    if _reranker_tried:
        return _reranker
    _reranker_tried = True

    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder

        # 用 snapshot_download(local_files_only=True) 先拿到本地缓存目录，再加载。
        #
        # 为什么不直接写 CrossEncoder("BAAI/bge-reranker-base")：
        #   huggingface_hub 的"离线开关/目标地址"是【导入那一刻】读环境变量决定的，
        #   而本项目里 langchain_community 会先于 tools.py 把 huggingface_hub 导进来，
        #   导致 tools.py 设的 HF_HUB_OFFLINE / HF_ENDPOINT 根本没生效。
        #   结果它跑去 huggingface.co 发 HEAD 请求，国内网络重试 5 次、每次几十秒，
        #   实测能把整个评测卡住好几分钟。
        #   显式传 local_files_only=True 是唯一不受导入顺序影响的写法，
        #   也和 tools.py 加载向量模型的方式保持一致。
        local_path = snapshot_download(RERANK_MODEL, local_files_only=True)
        _reranker = CrossEncoder(local_path)
        print(f"  [重排] 模型已加载：{RERANK_MODEL}")
    except Exception as exc:            # noqa: BLE001 —— 任何失败都降级，不能拖垮问答
        print(f"  [重排] 模型不可用，本次降级为纯混合检索：{exc}")
        print("         如需启用：联网执行一次 CrossEncoder 构造，把模型拉到本地缓存即可")
        _reranker = None
    return _reranker


def _docs_from_store(store) -> list:
    """把向量库里的块读出来，供 BM25 使用

    直接从库里取而不是重新切一遍文件，
    保证 BM25 和向量检索面对的是完全相同的一批块。
    """
    data = store.get(include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(data["documents"], data["metadatas"])
    ]


def rrf_fuse(ranked_lists: list, top_n: int = None) -> list:
    """RRF（Reciprocal Rank Fusion）倒数排名融合

    每路结果按名次给分：score += 1 / (RRF_K + rank)
    只看名次、不看原始分数，所以 BM25 的分数和向量的距离可以直接相加。
    """
    scores = {}
    doc_of = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, 1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_of.setdefault(key, doc)

    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    docs = [doc_of[key] for key in ordered]
    return docs[:top_n] if top_n else docs


class HybridRetriever:
    """BM25 + 向量 融合检索，可选 cross-encoder 重排"""

    def __init__(self, store, use_rerank: bool = True, candidates: int = DEFAULT_CANDIDATES):
        self.store = store
        self.candidates = candidates

        docs = _docs_from_store(store)
        self.bm25 = BM25Retriever.from_documents(docs, preprocess_func=tokenize)
        self.bm25.k = candidates

        self.reranker = load_reranker() if use_rerank else None

    def search(self, query: str, k: int = 3) -> list:
        # 第 1 路：BM25 字面匹配
        bm25_hits = self.bm25.invoke(query)
        # 第 2 路：向量语义匹配
        vector_hits = self.store.similarity_search(query, k=self.candidates)

        # 融合
        fused = rrf_fuse([vector_hits, bm25_hits])

        # 第 3 步：重排（有则用，没有就跳过）
        if self.reranker is not None:
            candidates = fused[: self.candidates]
            pairs = [(query, doc.page_content) for doc in candidates]
            scores = self.reranker.predict(pairs)
            fused = [
                doc for _, doc in sorted(
                    zip(scores, candidates), key=lambda pair: pair[0], reverse=True
                )
            ]

        return fused[:k]
