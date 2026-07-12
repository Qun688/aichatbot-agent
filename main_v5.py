"""
智能客服 Agent v5.0 —— CLI 入口
用 Tool Calling 替代 v4.0 的 if/elif 硬路由

运行方式：python main_v5.py
"""
from agent_v5 import process_message

chat_history = []  # [{input, output}, ...]


def main():
    print("=" * 50)
    print("云帆科技智能客服 v5.0（输入 quit 退出）")
    print("=" * 50)
    print("升级：Tool Calling | bind_tools() | LLM 自主选择工具")
    print("流水线：①审核 → ②Agent思考+工具调用 → ③生成答案 → ④质检")
    print("=" * 50)

    while True:
        user_input = input("\n您：").strip()

        if user_input == "":
            continue
        if user_input.lower() in ["quit", "exit", "退出"]:
            print("\n客服：感谢您的咨询，再见！")
            break

        recent = chat_history[-3:] if chat_history else []
        reply = process_message(user_input, recent)
        chat_history.append({"input": user_input, "output": reply})

        print(f"\n客服：{reply}")


if __name__ == "__main__":
    main()
