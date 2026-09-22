# 云帆科技智能客服 —— AI Agent 实战项目

一个基于 **LangGraph + RAG + 长期记忆** 的智能客服系统，可以像真人客服一样回答问题、查询订单、解答退款政策，并跨对话记住用户信息。

> 这是学习 AI Agent 开发的完整实战项目，从零一步步迭代而来（`v1` 简单问答 → `v8` 完整前后端系统）。

---

## ✨ 核心亮点

- **检索质量提升 45.5 个百分点**：自建 22 题评测集，`Hit@1` 从 54.5% 提升到 **100%**，`MRR` 从 0.667 提升到 **1.000**
- **提升可归因**：切块策略优化（按章节切 + FAQ 按一问一答切）贡献 +18.2pp；混合检索（BM25 + 向量 RRF 融合）与 cross-encoder 重排贡献 +27.3pp
- **可复现的量化评测**：`evaluate_retrieval.py` 一条命令跑出全部指标，不依赖 LLM 打分 —— 免费、3 秒、结果稳定
- **完整的 Agent 骨架**：LangGraph 状态图编排（安全审核 → 工具调用 → 输出质检）+ 对话持久化 + 跨会话长期记忆 + FastAPI / Streamlit 前后端

## 📊 评测结果

同一套 22 道测试题、同一向量模型、检索条数 `k=3`，对比三代检索方案：

| 指标 | A 基线<br>整篇 700 字硬切 | B 切块优化<br>章节 + 一问一答 | C 混合检索 + 重排<br>（当前版本） |
|------|------|------|------|
| **Hit@1** | 54.5% | 72.7% | **100.0%** |
| **Hit@3** | 81.8% | 86.4% | **100.0%** |
| **MRR** | 0.667 | 0.788 | **1.000** |
| 完全检索不到 | 4 道 | 3 道 | **0 道** |

复现方式：

```bash
python evaluate_retrieval.py
```

> 说明：100% 是「22 道自建题目 + 单领域知识库 + 25 个知识块」条件下的结果，
> 它说明的是**同一套题下的相对提升**，不代表任意知识库上的绝对水平。

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
| `agent_v6.py` | **Agent 核心**：LangGraph 状态图 + 工具调用 + 输出质检护栏 | 课4 LangGraph |
| `tools.py` | **RAG 索引层**：章节感知切块 + 按内容指纹复用向量库 | 课6 RAG |
| `retriever.py` | **混合检索层**：BM25 + 向量 RRF 融合 + cross-encoder 重排 | RAG 进阶 |
| `evaluate_retrieval.py` | **检索评测**：22 题、三代方案对比（Hit@k / MRR） | 评测 |
| `evaluate_faithfulness.py` | **幻觉评测**：工具调用率 + 数字溯源 + LLM 忠实度裁判 | 评测 |
| `memory.py` | 长期记忆（SqliteStore 跨会话记用户） | 课7 长期记忆 |
| `api.py` | FastAPI 接口（REST API） | 工程化 |
| `app.py` | Streamlit 网页界面（前端） | 工程化 |
| `helper_functions.py` | LLM 客户端（DeepSeek/OpenAI 兼容） | 课5 |
| `main_v6.py` | 命令行入口（多线程对话管理） | 工程化 |
| `knowledge.txt` | 知识库（公司 FAQ，共 9 个章节 / 25 个块） | 数据 |
| `agent.py` · `main.py` | 历史版本（v4 流水线 / v3 CLI）；注意 `agent.py` 的审核与质检函数**仍被当前版本复用** | 演进记录 |
| `checkpoints.db` · `memory_store.db` | 对话历史 / 长期记忆（自动生成，已 gitignore） | 运行时数据 |

---

## 四、快速开始

### 1. 准备环境

项目自带虚拟环境 `.venv`，**请务必用它**：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

> ⚠️ 不要用裸 `python`。系统 PATH 上的 Python 和项目 `.venv` 是**两套独立环境**，
> 全局环境可能缺少依赖（表现为 `ModuleNotFoundError`）。

### 2. 配置 API Key

项目使用 DeepSeek 大模型（也兼容 OpenAI），需要设置环境变量：

```powershell
# Windows (PowerShell) —— 当前窗口临时生效
$env:DEEPSEEK_API_KEY="你的key"

# 永久生效（写入用户环境变量，需重开终端）
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "你的key", "User")
```

> 也可以改用 OpenAI：设置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL` 即可。

### 3. 启动（三选一）

#### 方式一：命令行（最简单）

```powershell
.\.venv\Scripts\python.exe main_v6.py
```

#### 方式二：REST API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
# 打开 http://localhost:8000/docs 查看接口文档
```

#### 方式三：Web 界面（推荐）

需要开两个终端：

```powershell
# 终端1：先启动后端 API
.\.venv\Scripts\python.exe -m uvicorn api:app --reload

# 终端2：再启动前端界面
.\.venv\Scripts\python.exe -m streamlit run app.py
```

浏览器打开 `http://localhost:8501` 就能和客服对话。

> **在 VS Code 里更省事**：直接按 `Ctrl+Shift+B` 一键启动。
> `.vscode/tasks.json` 里写死了 `.venv` 的完整路径，不会用错解释器。

### 4. 跑评测（可选，用来验证检索与幻觉指标）

```powershell
# 检索评测：22 题、三代方案对比（纯本地计算，免费，约 2 分钟）
.\.venv\Scripts\python.exe evaluate_retrieval.py

# 幻觉评测：跑完整 Agent 链路（需要 LLM 调用，约 8 分钟）
.\.venv\Scripts\python.exe evaluate_faithfulness.py
```

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
- **ChromaDB + text2vec** —— 向量检索（RAG 基础层）
- **BM25（rank_bm25 + jieba 分词）+ BAAI/bge-reranker-base** —— 混合检索与重排
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
| v8.1 | 向量库按内容指纹复用（修掉重复累积）；网页端按用户隔离长期记忆 |
| v9 | 章节感知切块（FAQ 按一问一答切），检索 Hit@1 54.5% → 72.7% |
| v10 | 混合检索（BM25 + 向量 RRF 融合）+ cross-encoder 重排，Hit@1 → 100%；新增检索/幻觉评测脚本；质检节点接上输出护栏 |
