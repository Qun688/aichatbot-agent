"""
智能客服 Agent v3.0
用课5 LangChain 语法重写

升级对比：
  v1.0: 整个知识库塞进一个 prompt → LLM 直接回答
  v2.0: 手写 if/elif + f-string 链式调用
  v3.0: ChatPromptTemplate + StrOutputParser + | 管道语法

运行方式：python main.py
"""
from agent import process_message

# 手动管理对话历史（替代已废弃的 langchain.memory）
chat_history = []  # [{input, output}, ...]


def main():
    print("=" * 50)
    print("云帆科技智能客服 v3.0（输入 quit 退出）")
    print("=" * 50)
    print("升级：LangChain 管道语法 | ChatPromptTemplate | Memory")
    print("流水线：①审核 → ②分类 → ③④链式处理 → ⑤质检")
    print("=" * 50)

    while True:
        user_input = input("\n您：").strip()

        if user_input == "":
            continue
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("\n客服：感谢您的咨询，再见！")
            break

        # 取最近3轮对话作为上下文
        recent = chat_history[-3:] if chat_history else []

        # Agent 流水线
        reply = process_message(user_input, recent)

        # 保存本轮对话
        chat_history.append({"input": user_input, "output": reply})

        print(f"\n客服：{reply}")


if __name__ == "__main__":
    main()
