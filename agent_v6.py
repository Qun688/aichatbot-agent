"""
智能客服 Agent v6.0 —— LangGraph 状态图版（课4知识）
升级点：用 StateGraph 管理整个对话流程，将 v5.0 的隐式控制流
显式化为图节点和条件边

v6.0 完整特性：
  - StateGraph 节点+条件边 管理流程
  - MemorySaver 持久化（进程内，后续可升级 SqliteSaver）
  - thread_id 多线程对话（每个线程独立上下文）
  - .stream() 流式输出（实时显示节点执行进度）

图结构：
  START → safety_check ─┬─ 不通过 → reject → END
                        └─ 通过 → agent_node ─┬─ 需要工具 → tool_node → agent_node（循环）
                                               └─ 不需要 → quality_check → END
"""
from typing import TypedDict, Annotated
from datetime import datetime
import json
import os
import sqlite3

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

from tools import retrieve_relevant
from helper_functions import llm_precise, llm_creative
from memory import get_user_memory, save_user_memory, extract_memory

# ============================================================
# 复用 v4.0 的安全审核、质检（从 agent.py 导入）
# ============================================================
from agent import check_input_safety, check_output

# ============================================================
# 工具定义（原来在 agent_v5.py 中）
# ============================================================

@tool
def search_knowledge(question: str) -> str:
    """搜索云帆科技公司知识库。当用户询问公司信息、产品功能、价格方案、
    使用方法、常见问题、技术规格等知识类问题时，必须使用此工具。

    Args:
        question: 用户的具体问题，用完整自然语言描述。
    """
    return retrieve_relevant(question)


@tool
def check_order(order_id: str) -> str:
    """根据订单号查询订单的物流状态和发货进度。
    当用户提供订单号并询问订单状态、发货情况、物流信息时使用此工具。

    Args:
        order_id: 订单号，格式为 ORD- 开头加数字编号，例如 ORD-2024001。
    """
    mock_orders = {
        "ORD-2024001": "已发货 | 顺丰 SF1234567890 | 预计7月3日送达",
        "ORD-2024002": "处理中 | 预计7月2日发货",
        "ORD-2024003": "已签收 | 6月28日签收",
    }
    result = mock_orders.get(order_id, "未找到该订单，请确认订单号是否正确。")
    return f"订单 {order_id} 状态：{result}"


@tool
def check_refund_policy() -> str:
    """查询云帆科技公司的退款政策和退款流程。
    当用户询问退款条件、退款金额、退款流程、退款时效时使用此工具。
    """
    return retrieve_relevant("退款 退款政策 退款流程 退款条件")


TOOLS = [search_knowledge, check_order, check_refund_policy]

SYSTEM_PROMPT = """你是云帆科技的智能客服助手。你的职责是：
1. 理解用户问题，判断需要使用哪个工具来获取信息
2. 调用合适的工具获取必要信息
3. 基于工具返回的结果，用自然、友好的语气回答用户
4. 如果手头信息不足以回答，如实告知并建议联系人工客服 400-888-6666
重要：绝对不要编造工具返回结果中没有的信息。"""

# llm_precise(t=0) 绑定工具
llm_with_tools = llm_precise.bind_tools(TOOLS)

# ============================================================
# 第1部分：AgentState 定义
# ============================================================

class AgentState(TypedDict):
    """贯穿所有节点的共享状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    safety_passed: bool
    user_id: str


# ============================================================
# 第2部分：节点函数（纯逻辑，不含 print——显示由 streaming 层处理）
# ============================================================

def safety_check(state: AgentState) -> dict:
    """节点1：检查用户消息是否包含 prompt injection"""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not user_msgs:
        return {"safety_passed": False}
    user_input = user_msgs[-1].content
    is_safe, _ = check_input_safety(user_input)
    return {"safety_passed": is_safe}


def reject_node(state: AgentState) -> dict:
    """节点2：安全审核不通过时返回拒绝消息"""
    return {"messages": [AIMessage(content="抱歉，您的输入包含不安全内容，无法处理。")]}


def agent_node(state: AgentState) -> dict:
    """节点3：LLM 分析消息历史，决定是否调用工具"""
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        # 查用户的长期记忆，注入 system prompt（课7 Memory Store）
        user_id = state.get("user_id", "anonymous")
        memory_text = get_user_memory(user_id)
        enhanced_prompt = SYSTEM_PROMPT
        if memory_text:
            enhanced_prompt += f"\n\n【关于该用户的长期记忆】\n{memory_text}\n（如与当前对话相关，自然地运用这些信息）"
        messages = [SystemMessage(content=enhanced_prompt)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# 节点4：使用 LangGraph 内置 ToolNode，自动处理工具执行
tool_node = ToolNode(tools=TOOLS)


def quality_check(state: AgentState) -> dict:
    """节点5：质检最后一条 AI 回复"""
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
    if not user_msgs or not ai_msgs:
        return {}

    user_input = user_msgs[-1].content
    reply = ai_msgs[-1].content
    check_output(reply, user_input)  # 质检结果只做日志，不修改消息
    return {}


# ============================================================
# 第3部分：条件路由函数
# ============================================================

def route_after_safety(state: AgentState) -> str:
    """审核后路由：通过→agent_node，不通过→reject_node"""
    if state["safety_passed"]:
        return "agent_node"
    return "reject_node"


# ============================================================
# 第4部分：构建并编译图（注入 SqliteSaver 持久化）
# ============================================================

# 用 sqlite3 连接构造 SqliteSaver，checkpoints.db 存对话历史，重启不丢
_conn = sqlite3.connect("checkpoints.db", check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


def build_graph():
    """构建 LangGraph 状态图"""
    builder = StateGraph(AgentState)

    builder.add_node("safety_check", safety_check)
    builder.add_node("reject_node", reject_node)
    builder.add_node("agent_node", agent_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("quality_check", quality_check)

    builder.add_edge(START, "safety_check")
    builder.add_edge("reject_node", END)

    builder.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {"agent_node": "agent_node", "reject_node": "reject_node"}
    )

    builder.add_conditional_edges(
        "agent_node",
        tools_condition,
        {"tools": "tool_node", "__end__": "quality_check"}
    )

    builder.add_edge("tool_node", "agent_node")
    builder.add_edge("quality_check", END)

    return builder.compile(checkpointer=_checkpointer)


agent_graph = build_graph()


# ============================================================
# 第5部分：线程元数据管理（JSON 文件持久化，重启不丢）
# ============================================================

META_FILE = "thread_meta.json"


def _load_thread_meta() -> dict:
    """从 JSON 文件加载线程元数据"""
    if os.path.exists(META_FILE):
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_thread_meta():
    """把线程元数据保存到 JSON 文件"""
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(_thread_meta, f, ensure_ascii=False, indent=2)


# 存储线程的元信息（标题/创建时间，跟随 SqliteSaver 一起持久化）
_thread_meta: dict[str, dict] = _load_thread_meta()


def _get_thread_title(thread_id: str) -> str:
    """获取线程标题，无则返回 '新对话'"""
    meta = _thread_meta.get(thread_id, {})
    return meta.get("title", "新对话")


def _ensure_thread_meta(thread_id: str, first_message: str = ""):
    """确保线程存在元数据，首次对话时自动用首条消息生成标题"""
    if thread_id not in _thread_meta:
        title = first_message[:20] + ("..." if len(first_message) > 20 else "")
        _thread_meta[thread_id] = {
            "title": title if title else "新对话",
            "created_at": datetime.now().strftime("%m-%d %H:%M"),
        }
        _save_thread_meta()


def list_threads() -> list[dict]:
    """列出所有线程（按创建时间倒序）"""
    result = []
    for tid, meta in _thread_meta.items():
        # 检查该线程是否有历史消息
        config = {"configurable": {"thread_id": tid}}
        snapshot = agent_graph.get_state(config)
        msg_count = len(snapshot.values.get("messages", [])) if snapshot and snapshot.values else 0
        result.append({
            "id": tid,
            "title": meta.get("title", "新对话"),
            "created_at": meta.get("created_at", "未知"),
            "messages": msg_count,
        })
    # 按创建时间倒序
    result.sort(key=lambda x: x["created_at"], reverse=True)
    return result


def get_thread_state(thread_id: str) -> dict:
    """查看某线程的当前状态"""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent_graph.get_state(config)
    return snapshot.values if snapshot else {}


def get_thread_messages(thread_id: str) -> list:
    """获取某线程的消息历史"""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = agent_graph.get_state(config)
    if snapshot and snapshot.values:
        return snapshot.values.get("messages", [])
    return []


# ============================================================
# 第6部分：流式处理 - 节点显示映射
# ============================================================

def _format_node_event(node_name: str, node_output: dict) -> str:
    """将节点输出格式化为可读的进度信息"""
    if node_name == "safety_check":
        status = "安全" if node_output.get("safety_passed") else "拦截"
        return f"[审核: {status}]"

    elif node_name == "reject_node":
        return "[已拦截]"

    elif node_name == "agent_node":
        msgs = node_output.get("messages", [])
        if msgs:
            msg = msgs[0]
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                names = [tc["name"] for tc in msg.tool_calls]
                return f"[Agent 调用工具: {', '.join(names)}]"
            else:
                return "[Agent 生成回复]"
        return "[Agent 处理中]"

    elif node_name == "tool_node":
        return "[工具执行完毕]"

    elif node_name == "quality_check":
        return "[质检完成]"

    return f"[{node_name}]"


# ============================================================
# 第7部分：对外入口
# ============================================================

def process_message(user_input: str, thread_id: str = "default", user_id: str = "anonymous") -> str:
    """用 LangGraph 图执行一轮对话（支持流式输出 + 持久化 + 长期记忆）

    Args:
        user_input: 用户输入
        thread_id: 线程 ID，同一线程保持对话上下文
        user_id: 用户 ID，用于跨会话的长期记忆
    """
    _ensure_thread_meta(thread_id, user_input)

    config = {"configurable": {"thread_id": thread_id}}
    input_state = {"messages": [HumanMessage(content=user_input)], "user_id": user_id}

    print(f"  [{_get_thread_title(thread_id)}] ", end="")

    # stream_mode="updates"：每个节点执行完就 yield 一次
    for event in agent_graph.stream(
        input_state, config,
        stream_mode="updates",
        recursion_limit=10
    ):
        for node_name, node_output in event.items():
            print(_format_node_event(node_name, node_output), end=" → ")

    print("完成")

    # 从持久化状态中提取最后一条 AI 回复
    final_state = agent_graph.get_state(config)
    if final_state and final_state.values:
        ai_msgs = [m for m in final_state.values.get("messages", [])
                   if isinstance(m, AIMessage)]
        if ai_msgs:
            reply = ai_msgs[-1].content

            # 对话结束后，用 LLM 提取值得记住的用户信息，存入长期记忆
            memory = extract_memory(user_input, reply)
            if memory:
                save_user_memory(user_id, memory)
                print(f"  [记忆: 已记住「{memory}」]")

            return reply

    return "抱歉，系统处理出现异常，请稍后重试或拨打 400-888-6666。"
