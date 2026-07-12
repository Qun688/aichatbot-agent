"""
智能客服 Agent v5.0 —— Tool Calling 升级版（课3知识）
升级点：用 bind_tools() + @tool 替代 v4.0 的 if/elif 硬路由
LLM 自主决定调用哪个工具，而非被动分类后走预设分支

流水线：①审核 → ②LLM思考+工具调用 → ③执行工具 → ④生成答案 → ⑤质检
"""
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from tools import retrieve_relevant
from helper_functions import llm_precise, llm_creative

# ============================================================
# 复用 v4.0 的安全审核和质检（从 agent.py 导入，零修改）
# ============================================================
from agent import check_input_safety, check_output

# ============================================================
# 第1部分：定义工具（课3核心——@tool 装饰器）
# LLM 通过 docstring 理解工具的用途和参数，自主决定何时调用
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


# ============================================================
# 第2部分：工具列表 + 绑定到 LLM
# ============================================================

TOOLS = [search_knowledge, check_order, check_refund_policy]
# 构建工具名→工具对象的映射，供后续查找执行
_tool_map = {t.name: t for t in TOOLS}

# llm_precise(t=0) 绑定工具：LLM 严格按工具描述决定调哪个
llm_with_tools = llm_precise.bind_tools(TOOLS)

# llm_creative(t=0.3) 不带工具：用于最终生成回复（语气更自然）
llm_final = llm_creative

# ============================================================
# 第3部分：系统角色提示词
# ============================================================

SYSTEM_PROMPT = """你是云帆科技的智能客服助手。你的职责是：
1. 理解用户问题，判断需要使用哪个工具来获取信息
2. 调用合适的工具获取必要信息
3. 基于工具返回的结果，用自然、友好的语气回答用户
4. 如果手头信息不足以回答，如实告知并建议联系人工客服 400-888-6666
重要：绝对不要编造工具返回结果中没有的信息。"""

# ============================================================
# 第4部分：Agent 核心循环
# ============================================================

def run_agent(user_input: str) -> str:
    """Agent 核心：LLM 自主决定是否调工具，执行后生成最终答案"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input)
    ]

    # 第一轮：LLM 分析问题，决定是否需要工具
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # 如果 LLM 决定调用工具
    if response.tool_calls:
        print(f"  [Agent 决定调用 {len(response.tool_calls)} 个工具]", end="")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            print(f" → {tool_name}({tool_args})")

            # 执行工具
            tool_result = _tool_map[tool_name].invoke(tool_args)

            # 工具结果作为 ToolMessage 追加
            messages.append(ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            ))

        print(f"  [工具执行完毕，生成回复中...]")
        # 第二轮：将工具结果发给 LLM，生成最终答案
        final_response = llm_final.invoke(messages)
        return final_response.content
    else:
        print(f"  [Agent 无需工具，直接回复]")
        return response.content


# ============================================================
# 第5部分：对外入口（保持与 v4.0 相同的接口）
# ============================================================

def process_message(user_input: str, chat_history: list) -> str:
    """端到端流水线：①审核 → ②③Agent思考+工具调用 → ④质检 → 返回"""
    print("  [流水线启动]")

    # ① 安全审核
    is_safe, verdict = check_input_safety(user_input)
    print(f"  [① 审核: {'安全' if is_safe else '拦截'}]")
    if not is_safe:
        return "抱歉，您的输入包含不安全内容，无法处理。"

    # ②③ Agent 自主决策 + 执行工具
    reply = run_agent(user_input)
    print(f"  [②③ Agent处理完成]")

    # ④ 输出质检
    quality = check_output(reply, user_input)
    print(f"  [④ 质检: {quality[:30]}...]")

    return reply
