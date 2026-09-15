# 云帆科技智能客服 —— AI Agent 实战项目

一个基于 **LangGraph + RAG + 长期记忆** 的智能客服系统，可以像真人客服一样回答问题、查询订单、解答退款政策，并跨对话记住用户信息。

> 这是学习 AI Agent 开发的完整实战项目，从零一步步迭代而来（`v1` 简单问答 → `v8` 完整前后端系统）。

---

## 一、功能特性

| 功能 | 说明 |
|------|------|
| 💬 知识问答 | 用 RAG（向量语义检索）回答公司产品、价格、售后等问题 |
| 📦 订单查询 | 根据订单号查询物流状态 |
| ↩️ 退款咨询 | 查询退款政策、退款流程 |
| 🧵 多轮对话 | 每个对话线程独立上下文，互不干扰，重启不丢 |
| 🧠 长期记忆 | 跨对话记住用户偏好（如"喜欢简洁回复"） |
| 🛡️ 安全审核 | 拦截 prompt injection 恶意输入 |
| 🖥️ 三种使用方式 | 命令行 / REST API / Web 界面 |

---

## 二、技术架构

```
┌─────────────┐     HTTP      ┌──────────────┐    函数调用    ┌────────────────┐
│  Web 界面    │ ────────────▶ │  FastAPI     │ ────────────▶ │  LangGraph Agent │
│  (Streamlit) │   requests   │  (api.py)    │  process_msg  │   (agent_v6.py) │
└─────────────┘               └──────────────┘               └───────┬────────┘
                                                                      │ 调用工具
                                                          ┌───────────┴───────────┐
                                                          │  RAG检索  订单查询  退款  │
                                                          │ (tools.py) (mock)  (mock)│
                                                          └─────────────────────────┘
                                                                      │
                                        ┌─────────────────────────────┼─────────────┐
                                        ▼                             ▼             ▼
                                  Chroma向量库                  SqliteSaver      SqliteStore
                                  (知识检索)                   (对话历史)       (长期记忆)
```

**核心设计理念**：Agent = LLM + 工具 + 循环。LangGraph 负责编排流程（审核 → 调工具 → 回复 → 质检），前端只负责展示，通过 REST API 和 Agent 解耦。

---

## 三、项目结构

| 文件 | 作用 | 对应知识点 |
|------|------|-----------|
| `agent_v6.py` | Agent 核心：LangGraph 状态图 + 工具调用 | 课4 LangGraph |
| `tools.py` | RAG 向量检索（Chroma + 中文向量模型） | 课6 RAG |
| `memory.py` | 长期记忆（SqliteStore 跨会话记用户） | 课7 长期记忆 |
| `api.py` | FastAPI 接口（REST API） | 工程化 |
| `app.py` | Streamlit 网页界面（前端） | 工程化 |
| `helper_functions.py` | LLM 客户端（DeepSeek/OpenAI 兼容） | 课5 |
| `knowledge.txt` | 知识库（公司 FAQ 文档） | 数据 |
| `checkpoints.db` | 对话历史持久化（自动生成） | 运行时数据 |
| `memory_store.db` | 长期记忆库（自动生成） | 运行时数据 |

---

## 四、快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

项目使用 DeepSeek 大模型（也兼容 OpenAI），需要设置环境变量：

```bash
# Windows (CMD)
set DEEPSEEK_API_KEY=你的key

# Windows (PowerShell)
$env:DEEPSEEK_API_KEY="你的key"
```

> 也可以改用 OpenAI：设置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 即可。

### 3. 启动（三选一）

#### 方式一：命令行（最简单）

```bash
python main_v6.py
```

#### 方式二：REST API

```bash
uvicorn api:app --reload
# 打开 http://localhost:8000/docs 查看接口文档
```

#### 方式三：Web 界面（推荐）

需要开两个终端：

```bash
# 终端1：先启动后端 API
uvicorn api:app --reload

# 终端2：再启动前端界面
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`，就能和客服对话了。

---

## 五、API 接口一览

| 方法 | 路径 | 作用 |
|------|------|------|
| POST | `/chat` | 发送消息，返回客服回复 |
| GET | `/history/{thread_id}` | 查看某对话的历史消息 |
| GET | `/threads` | 列出所有对话线程 |
| GET | `/health` | 健康检查 |

调用示例：

```python
import requests
resp = requests.post("http://localhost:8000/chat", json={
    "message": "退款政策是什么",
    "thread_id": "user_001",
})
print(resp.json()["reply"])
```

---

## 六、技术栈

- **LangGraph** —— Agent 编排框架（状态图）
- **LangChain** —— 工具封装、消息处理
- **ChromaDB + text2vec** —— 向量检索（RAG）
- **SqliteSaver / SqliteStore** —— 对话持久化 / 长期记忆
- **FastAPI + uvicorn** —— 后端接口
- **Streamlit** —— 前端界面
- **DeepSeek API** —— 大语言模型

---

## 七、版本演进

| 版本 | 升级内容 |
|------|---------|
| v1~v3 | 简单 LLM 问答 → 关键词匹配知识库 |
| v4 | 安全审核 + RAG 向量检索 |
| v5 | 工具调用（订单查询、退款政策） |
| v6 | LangGraph 状态图 + SqliteSaver 持久化 + 多线程对话 |
| v7 | FastAPI 接口 + 长期记忆（Memory Store） |
| v8 | Streamlit 网页界面 + 知识库扩充 |
