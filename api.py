"""
智能客服 v7.0 —— FastAPI 入口
把 agent_v6.py 的 process_message() 包装成 REST API

运行:
  cd aichatbot
  pip install fastapi uvicorn
  uvicorn api:app --reload

然后浏览器打开 http://localhost:8000/docs
"""
import sys, os

# 把当前目录加入 Python 搜索路径，确保能 import agent_v6
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---- 导入你已有的 Agent 核心逻辑 ----
from agent_v6 import (
    process_message,
    get_thread_messages,
    list_threads,
)


# ============================================================
# 第1步：创建 FastAPI 应用
# ============================================================
app = FastAPI(
    title="云帆科技智能客服 API",
    description="基于 LangGraph 的智能客服系统 | 知识问答 + 订单查询 + 退款咨询",
    version="7.0",
)


# ============================================================
# 第2步：定义数据结构（请求长什么样，返回长什么样）
# ============================================================

class ChatRequest(BaseModel):
    """用户发来的聊天请求"""
    message: str                          # 用户说的话（必填）
    thread_id: str = "default"            # 对话线程ID，同一个ID共享上下文
    user_id: str = "anonymous"            # 用户标识，后续做长期记忆用


class ChatResponse(BaseModel):
    """返回给用户的聊天响应"""
    reply: str                            # 客服回复内容
    thread_id: str                        # 当前线程ID
    status: str = "ok"                    # ok / blocked / error


class ThreadInfo(BaseModel):
    """线程摘要信息"""
    id: str
    title: str
    messages: int
    created_at: str


# ============================================================
# 第3步：定义 API 接口
# ============================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    POST /chat  —— 最核心的聊天接口

    前端调用示例:
      fetch("http://localhost:8000/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
              "message": "退货政策是什么",
              "thread_id": "user_001"
          })
      })
    """
    try:
        # 调用你的 Agent（就是之前 main_v6.py 里调的那个函数）
        reply = process_message(
            user_input=request.message,
            thread_id=request.thread_id,
        )
        return ChatResponse(reply=reply, thread_id=request.thread_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent 处理失败: {str(e)}")


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """
    GET /history/{thread_id}  —— 查看某个线程的对话历史

    例: GET http://localhost:8000/history/user_001
    """
    messages = get_thread_messages(thread_id)
    return {
        "thread_id": thread_id,
        "messages": [
            {
                "role": "user" if m.__class__.__name__ == "HumanMessage" else "assistant",
                "content": m.content,
            }
            for m in messages
        ],
    }


@app.get("/threads", response_model=list[ThreadInfo])
async def get_all_threads():
    """
    GET /threads  —— 列出所有对话线程
    """
    return [ThreadInfo(**t) for t in list_threads()]


@app.get("/health")
async def health():
    """
    GET /health  —— 健康检查，判断服务是否存活
    """
    return {"status": "ok"}
