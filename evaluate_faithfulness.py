"""
幻觉评测脚本 —— 测答案是否忠于检索到的原文
==============================================
`evaluate_retrieval.py` 只能证明"资料捞对了"，但捞对了不代表答案没编。
对客服来说最危险的就是编价格、编电话、编退款政策 —— 这个脚本专门测这个。

它跑的是**完整 Agent 链路**（和真实用户走同一条路：审核 → 工具调用 → 质检），
对每道题做三层检测：

  1. 是否真的调用了知识库工具？
     没调工具就直接作答 = 凭模型记忆回答 = 幻觉高风险（这也是之前实测发现的真实问题）
  2. 数字溯源（纯代码判定，免费且客观）
     把答案里的电话/价格/百分比/天数等"可验证事实片段"抽出来，
     逐个检查是否能在**本次检索到的原文**里找到。找不到的就是编的。
  3. LLM 裁判（每题 1 次调用）
     判断答案是否忠实于资料、是否回答了用户问题 —— 语义层面的幻觉靠这层兜住。

用法：
  python evaluate_faithfulness.py            # 全部 22 题（约 6~8 分钟，几毛钱）
  python evaluate_faithfulness.py --limit 5  # 只跑前 5 题（几分钱，先验证方法）
"""
import os
import re
import sqlite3
import sys

from langchain_core.messages import ToolMessage

import agent_v6
from evaluate_retrieval import TEST_SET
from helper_functions import get_llm_response

# 用来识别"质检护栏介入"（答案被兜底话术替换）。
# 这个指标是必要的：兜底话术本身永远是"忠实"的，
# 如果护栏误杀全部回答，忠实率反而会显示 100% —— 那是评测的盲区。
GUARD_MARK = agent_v6.QUALITY_FALLBACK[:12]

# 测试用的身份：线程跑完会清理掉，不会污染你正常的对话列表
TEST_USER = "faith_eval"
THREAD_PREFIX = "faith_eval_"

# ============================================================
# 第一层：数字/事实片段溯源
#   这些模式覆盖了客服场景里"最不能编错"的信息
# ============================================================
FACT_PATTERNS = [
    (r"400-\d{3}-\d{4}", "客服电话"),
    # 金额后面不能吃掉 Markdown 的 * 和括号，否则 "15 元/人/月**" 这种
    # 脏片段跟原文比对必然失败（这是个真实踩过的坑）
    (r"\d+(?:\.\d+)?\s*元(?:/[^\s，。、；：（）()*#\n]*)?", "金额"),
    (r"\d+(?:\.\d+)?\s*%", "百分比"),
    (r"\d+\s*天", "天数"),
    (r"AES-\d+", "加密规格"),
    (r"\d{4}\s*年\s*\d{1,2}\s*月", "日期"),
]

# 系统提示里给定的信息（比如人工客服电话）也属于"可信来源"：
# 模型引用它不算幻觉。只按检索原文判定会产生大量假阳性。
TRUSTED_SYSTEM_INFO = agent_v6.SYSTEM_PROMPT

JUDGE_PROMPT = """你是严格的事实核查员。请**只依据下面的资料**判断【客服回复】。

【用户问题】
{question}

【知识库检索结果】
{context}

【系统给定的客服信息】（同样属于可信来源）
{system_info}

【客服回复】
{answer}

请判断两件事，并严格按下面三行格式输出，不要输出别的内容：
忠实性：忠实 或 不忠实
相关性：相关 或 不相关
理由：一句话（若不忠实，指出哪句话在资料里找不到）"""


def _norm(text: str) -> str:
    """去掉所有空白再做比对，避免"15元/人/月"和"15 元/人/月"这类空格差异误判"""
    return re.sub(r"\s+", "", text or "")


def extract_facts(text: str) -> list:
    """从文本里抽出可验证的事实片段，返回 [(片段, 类型), ...]"""
    found = []
    for pattern, kind in FACT_PATTERNS:
        for match in re.findall(pattern, text or ""):
            found.append((match.strip(), kind))
    return found


def check_facts(answer: str, context: str) -> tuple:
    """逐个检查答案里的事实片段能否在"检索原文 + 系统提示"里找到

    返回 (已证实的片段列表, 找不到的片段列表)
    """
    context_norm = _norm(context) + _norm(TRUSTED_SYSTEM_INFO)
    supported, unsupported = [], []
    for fragment, kind in extract_facts(answer):
        (supported if _norm(fragment) in context_norm else unsupported).append(f"{fragment}（{kind}）")
    return supported, unsupported


def _parse_line(text: str, key: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith(key):
            return line[len(key):].strip()
    return ""


def judge(question: str, answer: str, context: str) -> dict:
    """让 LLM 当裁判，判断忠实性与相关性"""
    raw = get_llm_response(
        JUDGE_PROMPT.format(
            question=question, context=context,
            system_info=TRUSTED_SYSTEM_INFO, answer=answer,
        ),
        temperature=0,
    )
    faith_text = _parse_line(raw, "忠实性：")
    rel_text = _parse_line(raw, "相关性：")
    return {
        # "不忠实".startswith("忠实") 是 False，正好能把两种情况区分开
        "faithful": faith_text.startswith("忠实") if faith_text else None,
        "relevant": rel_text.startswith("相关") if rel_text else None,
        "reason": _parse_line(raw, "理由：") or raw.strip()[:60],
    }


def ask_agent(question: str, index: int) -> dict:
    """跑一次完整 Agent 链路，并把它实际用到的资料捞出来"""
    thread_id = f"{THREAD_PREFIX}{index}"
    answer = agent_v6.process_message(question, thread_id=thread_id, user_id=TEST_USER)

    # 工具返回的内容会以 ToolMessage 的形式留在会话状态里 —— 这就是本次用到的原文
    messages = agent_v6.get_thread_messages(thread_id)
    contexts = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content
            contexts.append(content if isinstance(content, str) else str(content))

    return {"answer": answer, "context": "\n".join(contexts), "thread_id": thread_id}


def cleanup(thread_ids: list):
    """清掉评测产生的线程和测试用户记忆，别污染正常使用"""
    base = os.path.dirname(os.path.abspath(__file__))

    conn = sqlite3.connect(os.path.join(base, "checkpoints.db"))
    for tid in thread_ids:
        for table in ("checkpoints", "writes"):
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id=?", (tid,))
            except sqlite3.Error:
                pass
    conn.commit()
    conn.close()

    import json

    meta_path = os.path.join(base, "thread_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        for tid in thread_ids:
            meta.pop(tid, None)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    conn = sqlite3.connect(os.path.join(base, "memory_store.db"))
    try:
        conn.execute("DELETE FROM store WHERE prefix LIKE ?", (f"%{TEST_USER}%",))
        conn.commit()
    except sqlite3.Error:
        pass
    conn.close()


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    cases = TEST_SET[:limit] if limit else TEST_SET

    print("=" * 70)
    print(f"  幻觉评测：{len(cases)} 道题（完整 Agent 链路 + 双层检测）")
    print("=" * 70)

    rows = []
    thread_ids = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['q']}")
        result = ask_agent(case["q"], i)
        thread_ids.append(result["thread_id"])

        contexts = result["context"]
        tool_called = bool(contexts.strip())

        supported, unsupported = check_facts(result["answer"], contexts)
        verdict = judge(case["q"], result["answer"], contexts) if tool_called else {
            "faithful": False, "relevant": None, "reason": "未调用知识库工具，无法核对"
        }

        guarded = GUARD_MARK in result["answer"]
        rows.append({
            "q": case["q"], "answer": result["answer"], "tool_called": tool_called,
            "guarded": guarded,
            "facts_total": len(supported) + len(unsupported),
            "unsupported": unsupported, **verdict,
        })

        print(f"  工具调用: {'是' if tool_called else '❌ 否'}"
              f" | 事实片段: {len(supported)} 证实 / {len(unsupported)} 存疑"
              f" | 忠实: {verdict['faithful']} | 相关: {verdict['relevant']}"
              f"{' | ⚠️ 护栏介入' if guarded else ''}")

    # ---------------- 汇总 ----------------
    n = len(rows)
    with_tool = sum(1 for r in rows if r["tool_called"])
    no_tool = n - with_tool
    # 只统计"真的检索了、并且答案里有可验证事实"的题：
    # 没检索的题压根没有原文可比，混进分母会低估真实忠实度 ——
    # 它们属于"流程不可靠"，单独由工具调用率反映。
    checked = [r for r in rows if r["tool_called"] and r["facts_total"] > 0]
    fact_ok = [r for r in checked if not r["unsupported"]]
    faithful = [r for r in rows if r["faithful"]]
    relevant = [r for r in rows if r["relevant"]]

    print("\n" + "=" * 70)
    print("  汇总")
    print("=" * 70)
    print(f"  工具调用率      : {with_tool}/{n} = {with_tool / n * 100:.1f}%"
          f"   （{no_tool} 题没调库 = 凭模型记忆作答，幻觉高风险）")
    if checked:
        print(f"  数字溯源通过率  : {len(fact_ok)}/{len(checked)} = {len(fact_ok) / len(checked) * 100:.1f}%"
              f"   （分母只算「检索过且含可验证事实」的题）")
    print(f"  LLM 判定忠实率  : {len(faithful)}/{n} = {len(faithful) / n * 100:.1f}%")
    print(f"  LLM 判定相关性  : {len(relevant)}/{n} = {len(relevant) / n * 100:.1f}%")
    guarded = [r for r in rows if r["guarded"]]
    print(f"  质检护栏介入    : {len(guarded)}/{n} = {len(guarded) / n * 100:.1f}%"
          f"   （介入 = 答案被兜底话术替换，需人工确认是否误杀）")

    bad = [r for r in rows if r["unsupported"] or r["faithful"] is False]
    if bad:
        print(f"\n  需要人工复核的 {len(bad)} 题：")
        for r in bad:
            print(f"    - {r['q']}")
            if r["unsupported"]:
                print(f"        原文找不到的事实: {'、'.join(r['unsupported'])}")
            print(f"        裁判理由: {r['reason']}")

    cleanup(thread_ids)
    print("\n  （已清理评测用的临时线程与记忆）")


if __name__ == "__main__":
    main()
