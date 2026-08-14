"""
长期记忆模块（课7知识）
========================
用 SqliteStore 跨会话记住用户信息

和 Checkpoint（checkpoints.db）的区别：
  Checkpoint    -> 存对话历史，按 thread_id 隔离（一次对话的存档）
  Memory Store  -> 存用户信息，按 user_id 隔离（跨所有对话的永久记忆）

记忆的存储结构：
  namespace = ("aichatbot", user_id)   # 第一级是应用名，第二级是用户ID
  key       = 随机UUID                 # 每条记忆的唯一标识
  value     = {"content": "用户喜欢简洁的回复"}  # 提炼后的事实

工作流程：
  1. 对话前：get_user_memory(user_id) 查出记忆，注入 system prompt
  2. 对话后：extract_memory() 用 LLM 提取新信息，save_user_memory() 存起来
"""
import sqlite3
import uuid

from langgraph.store.sqlite import SqliteStore
from helper_functions import llm_precise

# ========== 全局单例：只初始化一次 ==========
# isolation_level=None 是自动提交模式，SqliteStore 需要它自己管理事务
_conn = sqlite3.connect("memory_store.db", check_same_thread=False, isolation_level=None)
store = SqliteStore(_conn)
store.setup()  # 创建数据库表

APP_NAMESPACE = "aichatbot"  # 应用名，作为 namespace 的第一级


def get_user_memory(user_id: str) -> str:
    """查某个用户的所有长期记忆，拼接成文本（没有则返回空字符串）"""
    # search 的 namespace_prefix 用前缀匹配，查 ("aichatbot", user_id) 下所有记忆
    items = store.search((APP_NAMESPACE, user_id), query=None, limit=50)
    if not items:
        return ""

    lines = []
    for item in items:
        content = item.value.get("content", "")
        if content:
            lines.append(f"- {content}")
    return "\n".join(lines)


def save_user_memory(user_id: str, content: str):
    """存一条用户长期记忆"""
    key = str(uuid.uuid4())  # 每条记忆用随机ID
    store.put((APP_NAMESPACE, user_id), key, {"content": content})


def extract_memory(user_input: str, reply: str, existing_memory: str = "") -> str | None:
    """用 LLM 从一轮对话中提取值得长期记住的用户信息

    Args:
        user_input: 用户输入
        reply: 客服回复
        existing_memory: 该用户已有的记忆（用于去重，避免重复提取）

    返回：提炼出的一句事实（如"用户喜欢简洁的回复"），没有新信息则返回 None
    """
    prompt = f"""分析下面这段客服对话，判断用户是否透露了值得长期记住的【新】信息。

值得记住的信息包括：
- 用户的姓名、身份、职业
- 用户的偏好（如"喜欢简洁的回复"、"希望用正式语气"）
- 用户提到的重要背景（如"我是学生"、"我们公司有100人"）

如果没有任何值得记住的【新】信息，只回复"无"。
如果有新信息，用一句简洁的事实陈述（提炼成客观事实，不要复述原话）。

用户说：{user_input}
客服回复：{reply}"""

    # 如果已有记忆，告诉 LLM 不要重复记录
    if existing_memory:
        prompt += f"""

【该用户已有的记忆】（不要重复记录这些）
{existing_memory}"""

    prompt += """

值得记住的新信息（没有就回复"无"）："""

    result = llm_precise.invoke(prompt).content.strip()
    if not result or result == "无":
        return None
    return result
