"""
智能客服 Agent v6.0 —— CLI 入口
用 LangGraph 状态图管理对话流程

特性：
  - 流式输出：实时显示每个节点的执行进度
  - 持久化：MemorySaver 保存对话上下文（进程内）
  - 多线程：/new /switch 管理多个独立对话

命令：
  /help     - 显示帮助
  /list     - 列出所有对话线程
  /new      - 创建新线程
  /switch N - 切换到线程 N
  /history  - 查看当前线程的对话历史

运行方式：python main_v6.py
"""
from agent_v6 import process_message, list_threads, get_thread_messages, agent_graph

# 当前活跃线程
current_thread = "default"


def print_banner():
    """启动横幅"""
    print("=" * 55)
    print("  云帆科技智能客服 v6.0 —— LangGraph 状态图版")
    print("=" * 55)
    print("  流式输出 | MemorySaver 持久化 | 多线程对话")
    print("  输入 /help 查看命令列表")
    print("=" * 55)

    # 打印图结构
    try:
        print("\n[Agent 状态图]")
        print(agent_graph.get_graph().draw_ascii())
    except Exception:
        print("  (图结构可视化需要安装 grandalf: pip install grandalf)")
    print()


def show_current_thread():
    """显示当前线程信息和最近对话"""
    messages = get_thread_messages(current_thread)
    print(f"[当前线程: {current_thread}]", end="")
    if messages:
        # 统计轮数（HumanMessage 的数量）
        rounds = len([m for m in messages if m.__class__.__name__ == "HumanMessage"])
        print(f" | {rounds} 轮对话", end="")

        # 显示最近 3 轮摘要
        recent = messages[-6:]  # 最近 3 轮 = 6 条消息（3问3答）
        print("\n[最近对话]")
        for m in recent:
            role = "您" if m.__class__.__name__ == "HumanMessage" else "客服"
            content = m.content[:60] + ("..." if len(m.content) > 60 else "")
            print(f"  {role}: {content}")
    else:
        print(" | 新对话")
    print()


def handle_command(cmd: str) -> bool:
    """处理命令，返回 True 表示需要退出"""
    global current_thread

    parts = cmd.strip().split()
    action = parts[0].lower()

    if action == "/help":
        print("\n命令列表：")
        print("  /help      - 显示此帮助")
        print("  /list      - 列出所有对话线程")
        print("  /new [标题] - 创建新线程（可选标题）")
        print("  /switch N  - 切换到指定线程")
        print("  /history   - 查看当前线程完整对话历史")
        print("  /quit      - 退出")
        print()

    elif action == "/list":
        threads = list_threads()
        if not threads:
            print("\n暂无对话线程。输入任意内容开始对话。\n")
        else:
            print(f"\n{'ID':<20} {'标题':<25} {'消息':<6} {'创建时间'}")
            print("-" * 65)
            for t in threads:
                marker = " ← 当前" if t["id"] == current_thread else ""
                print(f"{t['id']:<20} {t['title']:<25} {t['messages']:<6} {t['created_at']}{marker}")
            print()

    elif action == "/new":
        from datetime import datetime
        title = parts[1] if len(parts) > 1 else ""
        # 用时间戳生成唯一 ID
        thread_id = datetime.now().strftime("%m%d%H%M")
        if title:
            from agent_v6 import _thread_meta, _ensure_thread_meta
            _ensure_thread_meta(thread_id, title)
            _thread_meta[thread_id]["title"] = title
        current_thread = thread_id
        print(f"\n已创建并切换到线程: {thread_id}\n")

    elif action == "/switch":
        if len(parts) < 2:
            print("\n用法: /switch <线程ID>\n")
            return False
        target = parts[1]
        # 检查线程是否存在
        from agent_v6 import get_thread_state
        state = get_thread_state(target)
        if state is not None and target in [t["id"] for t in list_threads()]:
            current_thread = target
            print(f"\n已切换到线程: {target}")
            show_current_thread()
        else:
            print(f"\n线程 '{target}' 不存在。使用 /list 查看所有线程。\n")

    elif action == "/history":
        messages = get_thread_messages(current_thread)
        if not messages:
            print("\n当前线程无对话历史。\n")
        else:
            print(f"\n=== {current_thread} 完整对话历史 ===\n")
            for m in messages:
                role = "您" if m.__class__.__name__ == "HumanMessage" else "客服"
                print(f"[{role}]: {m.content}\n")
            print("=== 历史结束 ===\n")

    else:
        print(f"\n未知命令: {action}。输入 /help 查看可用命令。\n")

    return False


def main():
    print_banner()
    show_current_thread()

    while True:
        user_input = input("您：").strip()

        if user_input == "":
            continue

        # 处理命令
        if user_input.startswith("/"):
            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("\n客服：感谢您的咨询，再见！")
                break
            handle_command(user_input)
            continue

        # 正常对话（流式输出由 process_message 内部处理）
        reply = process_message(user_input, thread_id=current_thread)
        print(f"\n客服：{reply}\n")


if __name__ == "__main__":
    main()
