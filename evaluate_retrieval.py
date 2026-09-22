"""
检索评测脚本 —— 量化 RAG 检索质量
====================================
干什么：
  用同一套测试问题，对比两种切块策略的检索效果
    A. 基线（baseline）：整篇文档按 700 字硬切      —— v4.0~v8.0 的老做法
    B. 现状（current） ：先按【章节】切、【常见问题】按一问一答切 —— v9.1
  输出 Hit@1 / Hit@3 / MRR 三个指标，以及每道题的命中排名。
  A→B 的差值，就是可以写进简历的"检索命中率提升 XX%"。

为什么不用 RAGAS：
  RAGAS 的生成类指标要调 LLM 打分，慢、要花钱、结果还不稳定。
  而"检索有没有捞到正确内容"这件事用关键词判定就足够客观，
  而且免费、可复现、几秒跑完。先把这一步做扎实，
  再上 RAGAS 的 faithfulness / answer relevancy 才是有价值的补充。

用法：
  python evaluate_retrieval.py
"""
import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tools

# 评测用的向量库目录（跟生产用的 chroma_db 分开，随时可删）
EVAL_DIR = os.path.join(tools.BASE_DIR, "chroma_db_eval")

# 基线策略参数（复刻 v4.0~v8.0 的切块方式）
BASELINE_CHUNK_SIZE = 700
BASELINE_CHUNK_OVERLAP = 100
BASELINE_SEPARATORS = ["\n\n", "\n", "。", "，", " ", ""]

# 检索条数：跟生产环境 retrieve_relevant() 的默认值保持一致
TOP_K = 3

# ============================================================
# 测试集
#   q      : 用户会怎么问
#   expect : 正确答案里必然出现的关键词（只要检索到的任意一个块包含它，就算命中）
#   关键词与切块方式无关，所以两套策略用的是完全公平的同一把尺子
# ============================================================
TEST_SET = [
    # 【公司信息】
    {"q": "你们公司的客服电话是多少？", "expect": ["400-888-6666"]},
    {"q": "云帆科技是什么时候成立的？", "expect": ["2019年3月"]},
    {"q": "公司现在有多少员工？", "expect": ["320 人"]},
    {"q": "你们公司的具体地址在哪？", "expect": ["文三路"]},
    # 【产品与定价】
    {"q": "OA 专业版多少钱一个月？", "expect": ["15元/人/月"]},
    {"q": "CRM 高级版的价格和包含什么功能？", "expect": ["40元/人/月", "AI 客户洞察"]},
    {"q": "数据智能平台怎么收费？", "expect": ["50元/人/月"]},
    # 【技术保障】
    {"q": "你们的数据安全是怎么保障的？", "expect": ["AES-256"]},
    {"q": "服务可用性承诺是多少？", "expect": ["99.9%"]},
    {"q": "数据备份策略是什么样的？", "expect": ["每日自动备份", "30 天"]},
    # 【售后服务】
    {"q": "提交工单之后多久能响应？", "expect": ["4 小时内响应", "24 小时内响应"]},
    # 【客户案例】
    {"q": "有没有具体的客户案例？", "expect": ["审批效率提升 60%", "某制造企业"]},
    # 【合作伙伴与生态】
    {"q": "你们和哪些云厂商有合作？", "expect": ["阿里云", "腾讯云"]},
    # 【常见问题】
    {"q": "怎么开发票？", "expect": ["增值税普通发票", "发票管理"]},
    {"q": "忘记密码了怎么办？", "expect": ["重置密码"]},
    {"q": "有免费试用吗？", "expect": ["14 天免费试用"]},
    {"q": "支持私有化部署吗？", "expect": ["单独购买部署服务"]},
    {"q": "怎么接入钉钉？", "expect": ["集成设置", "单点登录"]},
    {"q": "数据可以导出来吗？", "expect": ["Excel"]},
    {"q": "同一个账号能不能多人同时登录？", "expect": ["同时仅支持一个设备登录"]},
    {"q": "支持哪些支付方式？", "expect": ["支付宝", "微信支付"]},
    # 【退换政策】
    {"q": "退款政策是什么？", "expect": ["7 天内支持无理由退款"]},
]


def build_baseline_docs() -> list:
    """基线策略：整篇文档按 700 字硬切（复刻老版本的切块方式）"""
    text = TextLoader(tools.KNOWLEDGE_FILE, encoding="utf-8").load()[0].page_content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASELINE_CHUNK_SIZE,
        chunk_overlap=BASELINE_CHUNK_OVERLAP,
        separators=BASELINE_SEPARATORS,
    )
    return splitter.create_documents([text])


def evaluate(store, label: str) -> dict:
    """在给定向量库上跑一遍测试集，算 Hit@1 / Hit@k / MRR"""
    details = []
    for case in TEST_SET:
        hits = store.similarity_search(case["q"], k=TOP_K)
        rank = None
        for i, doc in enumerate(hits, 1):
            if any(kw in doc.page_content for kw in case["expect"]):
                rank = i
                break
        details.append({"q": case["q"], "rank": rank, "keyword": case["expect"][0]})

    n = len(details)
    hit1 = sum(1 for d in details if d["rank"] == 1) / n
    hitk = sum(1 for d in details if d["rank"] is not None) / n
    mrr = sum(1.0 / d["rank"] for d in details if d["rank"]) / n
    return {"label": label, "hit1": hit1, "hitk": hitk, "mrr": mrr, "details": details}


def print_result(res: dict, total_chunks: int):
    print(f"\n【{res['label']}】共 {total_chunks} 个块")
    print(f"  Hit@1 : {res['hit1'] * 100:5.1f}%   （第一名就命中）")
    print(f"  Hit@{TOP_K} : {res['hitk'] * 100:5.1f}%   （前 {TOP_K} 名内命中）")
    print(f"  MRR   : {res['mrr']:.3f}   （命中越靠前越高，满分 1.000）")

    missed = [d for d in res["details"] if d["rank"] is None]
    late = [d for d in res["details"] if d["rank"] and d["rank"] > 1]
    if late:
        print(f"  命中但排名靠后（{len(late)} 条）：")
        for d in late:
            print(f"    - 第{d['rank']}名  {d['q']}")
    if missed:
        print(f"  ❌ 完全没检索到（{len(missed)} 条）：")
        for d in missed:
            print(f"    - {d['q']}   （期望关键词：{d['keyword']}）")


def main():
    # 每次重建，保证结果可复现
    if os.path.exists(EVAL_DIR):
        shutil.rmtree(EVAL_DIR, ignore_errors=True)

    print("=" * 62)
    print("  RAG 检索评测：基线切块 vs 章节+问答切块")
    print("=" * 62)

    embeddings = tools._load_embeddings()

    baseline_docs = build_baseline_docs()
    current_docs = tools._split_knowledge()

    baseline_store = Chroma.from_documents(
        documents=baseline_docs, embedding=embeddings,
        collection_name="baseline", persist_directory=EVAL_DIR,
    )
    current_store = Chroma.from_documents(
        documents=current_docs, embedding=embeddings,
        collection_name="current", persist_directory=EVAL_DIR,
    )

    base_res = evaluate(baseline_store, "基线：整篇 700 字硬切")
    curr_res = evaluate(current_store, "现状：章节 + 一问一答切块")

    print_result(base_res, len(baseline_docs))
    print_result(curr_res, len(current_docs))

    print("\n" + "=" * 62)
    print("  对比结论")
    print("=" * 62)
    print(f"  Hit@1 : {base_res['hit1'] * 100:.1f}%  →  {curr_res['hit1'] * 100:.1f}%"
          f"   （{'+' if curr_res['hit1'] >= base_res['hit1'] else ''}"
          f"{(curr_res['hit1'] - base_res['hit1']) * 100:.1f} 个百分点）")
    print(f"  Hit@{TOP_K} : {base_res['hitk'] * 100:.1f}%  →  {curr_res['hitk'] * 100:.1f}%"
          f"   （{'+' if curr_res['hitk'] >= base_res['hitk'] else ''}"
          f"{(curr_res['hitk'] - base_res['hitk']) * 100:.1f} 个百分点）")
    print(f"  MRR   : {base_res['mrr']:.3f}  →  {curr_res['mrr']:.3f}")
    print(f"\n  测试集共 {len(TEST_SET)} 道题，检索条数 k={TOP_K}")


if __name__ == "__main__":
    main()
