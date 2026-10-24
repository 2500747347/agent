"""个人AI助理 Agent —— Gradio 对话界面。

启动： python ui/app.py  （默认 http://127.0.0.1:7860）
功能：多轮对话、工具调用过程实时展示、重置会话。
"""
from __future__ import annotations

import os
import sys

# 保证无论从哪个目录启动，都能找到项目根下的 agent / tools 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr  # noqa: E402

from agent.factory import build_agent  # noqa: E402

agent = build_agent()

_HELP_EXAMPLES = [
    "今天是几号？",
    "帮我计算 128 * 37 等于多少",
    "记一下明天上午10点开周会",
    "芜湖今天天气怎么样？",
    "什么是 RAG？",
    "记住我正在准备面试",
    "我之前说过什么",
]


def format_trace(trace: list[dict]) -> str:
    """把一次对话的工具调用轨迹渲染为 Markdown（供右侧面板展示）。"""
    lines: list[str] = []
    for step in trace:
        if step["type"] == "tool_call":
            args = step.get("arguments") or {}
            args_str = ", ".join(f"{k}={v}" for k, v in args.items()) or "（无参数）"
            lines.append(f"**步骤 {step['step']}** 调用 `{step['name']}({args_str})`")
            lines.append(f"> 返回：{step['result']}")
        elif step["type"] == "answer":
            lines.append(f"**最终回复**（步骤 {step['step']}）：{step['content']}")
        elif step["type"] == "limit":
            lines.append(f"⚠️ {step['content']}")
    return "\n\n".join(lines) if lines else "（本轮未调用工具）"


def respond(message: str, history: list | None, trace_md: str) -> tuple:
    """单轮对话回调：调用 Agent，返回更新后的聊天记录与工具轨迹。"""
    history = history or []
    history.append({"role": "user", "content": message})
    result = agent.run(message)
    history.append({"role": "assistant", "content": result["answer"]})
    trace_md = format_trace(result["trace"])
    return history, trace_md


def reset_session() -> tuple:
    agent.memory.clear()
    return [], "（会话已重置）"


def main() -> None:
    with gr.Blocks(title="个人AI助理 Agent") as demo:
        gr.Markdown(
            "# 🤖 个人AI助理 Agent\n"
            "手写 ReAct 内核 + Function Calling + 记忆。试试下面的示例，或直接提问。"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=480, label="对话")
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="例如：记一下明天上午10点开周会",
                        show_label=False,
                        scale=4,
                    )
                    send = gr.Button("发送", variant="primary", scale=1)
                    reset = gr.Button("重置会话", scale=1)
            with gr.Column(scale=2):
                gr.Markdown("### 工具调用轨迹")
                trace_box = gr.Markdown("（本轮未调用工具）")

        gr.Examples(examples=_HELP_EXAMPLES, inputs=msg, label="试试这些")

        msg.submit(respond, [msg, chatbot, trace_box], [chatbot, trace_box]).then(
            lambda: "", None, msg
        )
        send.click(respond, [msg, chatbot, trace_box], [chatbot, trace_box]).then(
            lambda: "", None, msg
        )
        reset.click(reset_session, None, [chatbot, trace_box])

    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
