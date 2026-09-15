"""
智能客服 v8.0 —— Streamlit 网页界面
====================================
作用：把 FastAPI 接口包装成可视化聊天界面（前端）

架构链路（前后端分离）：
  浏览器 → Streamlit(app.py) → HTTP请求 → FastAPI(api.py) → Agent(agent_v6.py)

运行（需要两个终端，先启动后端再启动前端）：
  终端1：uvicorn api:app --reload          # 后端 API（8000 端口）
  终端2：streamlit run app.py              # 前端界面（8501 端口）
"""
import uuid

import streamlit as st
import requests

# ============================================================
# 第0步：页面基础配置
# ============================================================
# set_page_config 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="云帆科技智能客服",   # 浏览器标签页标题
    page_icon="🤖",                  # 标签页图标
    layout="wide",                   # 宽屏布局（侧边栏 + 主区域更宽敞）
)

API_BASE = "http://localhost:8000"   # FastAPI 服务地址（api.py 默认端口）


# ============================================================
# 第1步：初始化会话状态
# Streamlit 每次交互（点按钮、发消息）都会从头重跑整个脚本，
# 所以必须用 st.session_state 把"需要记住的数据"存起来，否则每次重跑就丢了。
# ============================================================
if "thread_id" not in st.session_state:
    # 页面首次打开时，自动开一个空白对话（用随机ID，保证每个对话独立上下文）
    st.session_state.thread_id = uuid.uuid4().hex[:8]
if "messages" not in st.session_state:
    # 当前对话的消息列表，每条格式：{"role": "user"或"assistant", "content": "文本"}
    st.session_state.messages = []


# ============================================================
# 第2步：封装"调用 FastAPI"的三个函数
# ============================================================
def api_get_threads() -> list:
    """GET /threads —— 获取所有对话线程列表。API 没启动时返回空列表"""
    try:
        resp = requests.get(f"{API_BASE}/threads", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.ConnectionError:
        pass  # 连不上后端（没启动），静默返回空列表，界面会提示
    return []


def api_get_history(thread_id: str) -> list:
    """GET /history/{thread_id} —— 获取某个对话线程的历史消息"""
    try:
        resp = requests.get(f"{API_BASE}/history/{thread_id}", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("messages", [])
    except requests.exceptions.ConnectionError:
        pass
    return []


def api_chat(message: str, thread_id: str):
    """POST /chat —— 发送消息，返回客服回复文本。失败返回 None"""
    try:
        resp = requests.post(
            f"{API_BASE}/chat",
            json={"message": message, "thread_id": thread_id, "user_id": "web_user"},
            timeout=120,  # Agent 要调 LLM + 工具（首次还要加载 RAG 模型），给足超时时间
        )
        if resp.status_code == 200:
            return resp.json().get("reply", "")
    except requests.exceptions.ConnectionError:
        pass
    return None


# ============================================================
# 第3步：侧边栏 —— 「新建对话」按钮 + 历史对话列表
# ============================================================
with st.sidebar:
    st.title("💬 对话列表")

    # 3.1 「新建对话」按钮
    if st.button("➕ 新建对话", use_container_width=True):
        # 生成全新线程ID + 清空消息，等于开了一个空白对话
        st.session_state.thread_id = uuid.uuid4().hex[:8]
        st.session_state.messages = []
        st.rerun()  # 立即重跑脚本，让界面刷新成空对话

    st.divider()  # 一条分隔线

    # 3.2 加载线程列表
    threads = api_get_threads()
    if not threads:
        st.caption("暂无历史对话")  # 灰色小字提示（后端没启动 或 还没发过消息）

    # 3.3 逐个渲染线程按钮，点击切换到这个对话
    for t in threads:
        label = f"{t['title']}  ·  {t['messages']}条"
        # key 用线程ID，保证每个按钮唯一
        if st.button(label, key=t["id"], use_container_width=True):
            # 切换线程：记住ID，并加载它的历史消息
            st.session_state.thread_id = t["id"]
            st.session_state.messages = api_get_history(t["id"])
            st.rerun()


# ============================================================
# 第4步：主区域 —— 聊天窗口
# ============================================================
st.title("云帆科技 · 智能客服")
st.caption("基于 LangGraph 的知识问答 + 订单查询 + 退款咨询")

# 4.1 渲染当前对话的所有历史消息（每条一个聊天气泡）
for msg in st.session_state.messages:
    # st.chat_message 会根据 role 自动画成"用户"或"客服"的气泡
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4.2 底部输入框（st.chat_input 会自动在页面底部生成一个输入栏）
if user_input := st.chat_input("请输入你的问题..."):
    # (1) 先记录用户说的话
    st.session_state.messages.append({"role": "user", "content": user_input})

    # (2) 调用后端 Agent，等待期间显示"正在思考"转圈提示
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            reply = api_chat(user_input, st.session_state.thread_id)

    # (3) 记录客服回复（失败时记录一条错误提示）
    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ 无法连接后端服务，请先在终端运行：`uvicorn api:app --reload`",
        })

    # (4) 重跑脚本，让消息列表和侧边栏线程列表都刷新
    st.rerun()
