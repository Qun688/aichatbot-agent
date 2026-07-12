"""
智能客服项目 —— LLM 调用工具
v3.0: 新增 LangChain 模型封装（课5知识）
兼容 DeepSeek / OpenAI 等 API
"""
import os
from openai import OpenAI

# ---- 配置 API ----
API_KEY = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com"
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return _client


def get_llm_response(prompt: str, temperature: float = 0.3) -> str:
    """调用 LLM，返回纯文本（兼容旧接口）"""
    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[API 调用失败] {e}"


# ========== 课5新增：LangChain 模型封装 ==========
# 用 LangChain 的 ChatOpenAI 连 DeepSeek，支持 | 管道语法
from langchain_openai import ChatOpenAI

# temperature=0 用于分类/审核等确定性任务
llm_precise = ChatOpenAI(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0,
)

# temperature=0.3 用于生成回复
llm_creative = ChatOpenAI(
    model=MODEL,
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0.3,
)
