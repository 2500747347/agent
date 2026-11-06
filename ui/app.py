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

from agent.factory import build_agent, load_config  # noqa: E402

agent = build_agent()

_HELP_EXAMPLES = [
    "今天是几号？",
    "帮我计算 128 * 37 等于多少",
    "记一下明天上午10点开周会",
    "芜湖今天天气怎么样？",
    "什么是 RAG？",
    "记住我周末常去健身",
    "我之前说过什么",
]


def _runtime_info() -> tuple[str, str]:
    """读取当前运行配置，用于顶部状态展示。"""
    try:
        llm_cfg = load_config().get("llm", {})
    except Exception:
        llm_cfg = {}
    provider = llm_cfg.get("provider", "mock")
    if provider == "mock":
        model_label = "Mock 模式（离线确定性）"
    else:
        model_label = llm_cfg.get("model", "OpenAI 兼容模型")
    try:
        tool_count = len(agent.registry.tools)
    except Exception:
        tool_count = 14
    return provider, model_label, tool_count


_PROVIDER, _MODEL_LABEL, _TOOL_COUNT = _runtime_info()
_PROVIDER_TEXT = "在线模型" if _PROVIDER != "mock" else "离线演练"
_PROVIDER_CLASS = "badge-online" if _PROVIDER != "mock" else "badge-offline"

HERO_HTML = f"""
<div class="hero">
  <div class="hero-main">
    <div class="hero-eyebrow">PERSONAL AI ASSISTANT / REACT CORE</div>
    <h1>个人 AI 助理 Agent</h1>
    <p class="hero-sub">从零手写的 ReAct 推理内核 —— 工具调度、双记忆、RAG 检索、全程可观测</p>
    <div class="hero-chips">
      <span class="chip">{_TOOL_COUNT} 个工具</span>
      <span class="chip">SQLite 持久化</span>
      <span class="chip">58 项单测</span>
      <span class="chip">45 条评测</span>
    </div>
  </div>
  <div class="hero-status">
    <div class="status-line"><span class="dot {_PROVIDER_CLASS}"></span>{_PROVIDER_TEXT}</div>
    <div class="status-model">{_MODEL_LABEL}</div>
    <div class="status-note">会话内自动记忆 · 工具调用可见</div>
  </div>
</div>
"""

TRACE_HEAD = """
<div class="panel-title">
  <div>
    <h3>工具调用轨迹</h3>
    <p>每一轮的工具名、参数与返回值都会实时显示在这里</p>
  </div>
  <span class="panel-tag">TRACE</span>
</div>
"""

FOOTER_HTML = """
<div class="page-footer">
  <span>ReAct 内核 · Function Calling · 双记忆 · 自研 RAG · 可观测性</span>
  <span class="footer-dot">·</span>
  <span>本地运行，数据不出机器</span>
</div>
"""

CSS = """
:root {
  --brand: #0f766e;
  --brand-2: #14b8a6;
  --ink: #0f172a;
  --muted: #64748b;
  --line: #e2e8ef;
  --card: #ffffff;
  --page: #eef3f5;
}

body, .gradio-container {
  background: linear-gradient(180deg, #f7fafb 0%, #eaf1f3 100%) !important;
  font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif !important;
}

.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  padding: 22px 18px 30px !important;
  color: var(--ink);
}

/* ---------- Hero ---------- */
.hero {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  flex-wrap: wrap;
  background: linear-gradient(135deg, #0b1220 0%, #123f3b 62%, #0f766e 100%);
  border-radius: 22px;
  padding: 28px 32px;
  color: #f8fafc;
  box-shadow: 0 18px 40px rgba(11, 18, 32, 0.22);
  margin-bottom: 18px;
}
.hero-eyebrow {
  font-size: 11px;
  letter-spacing: 0.22em;
  color: #7dd3c7;
  font-weight: 600;
  margin-bottom: 10px;
}
.hero h1 {
  margin: 0 0 8px 0 !important;
  font-size: 30px !important;
  font-weight: 700 !important;
  color: #ffffff !important;
  line-height: 1.2 !important;
}
.hero-sub {
  margin: 0;
  font-size: 14px;
  color: #cbd5e1;
  line-height: 1.6;
}
.hero-chips {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 16px;
}
.chip {
  font-size: 12px;
  color: #d1fae5;
  background: rgba(20, 184, 166, 0.16);
  border: 1px solid rgba(94, 234, 212, 0.32);
  border-radius: 999px;
  padding: 4px 12px;
}
.hero-status {
  min-width: 210px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.14);
  border-radius: 16px;
  padding: 14px 18px;
  backdrop-filter: blur(6px);
}
.status-line {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 600;
  color: #ecfeff;
}
.dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  display: inline-block;
}
.badge-online { background: #34d399; box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.18); }
.badge-offline { background: #fbbf24; box-shadow: 0 0 0 4px rgba(251, 191, 36, 0.16); }
.status-model {
  font-size: 12px;
  color: #a7f3d0;
  margin-top: 6px;
  word-break: break-all;
}
.status-note {
  font-size: 11px;
  color: #94a3b8;
  margin-top: 8px;
}

/* ---------- 面板卡片 ---------- */
#chat-panel, #trace-panel {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px 16px 12px;
  box-shadow: 0 6px 22px rgba(15, 23, 42, 0.05);
}
#chat-panel { min-height: 570px; }
.panel-title {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}
.panel-title h3 {
  margin: 0 !important;
  font-size: 15px !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
}
.panel-title p {
  margin: 4px 0 0 !important;
  font-size: 12px !important;
  color: var(--muted) !important;
  line-height: 1.5 !important;
}
.panel-tag {
  font-size: 10px;
  letter-spacing: 0.16em;
  color: var(--brand);
  border: 1px solid #99f6e4;
  background: #f0fdfa;
  border-radius: 999px;
  padding: 3px 9px;
  white-space: nowrap;
}

/* ---------- 聊天区 ---------- */
/* 兜底：Gradio 6 会跟随系统深色模式，这里统一拉回浅色 */
.gradio-container {
  color-scheme: light !important;
  --body-text-color: #0f172a !important;
  --body-text-color-subdued: #64748b !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #f8fafc !important;
  --block-background-fill: #ffffff !important;
  --border-color-primary: #e2e8f0 !important;
}
#chat, #chat > div, #chat .chatbot, #chat [class*="message"], #chat [class*="chat"],
#chat [class*="bubble"] {
  background: #fbfdfe !important;
  color: #0f172a !important;
}
#chat .placeholder, #chat [class*="placeholder"], #chat ::placeholder {
  color: #64748b !important;
  opacity: 1 !important;
}
/* 隐藏消息旁的空头像占位方块与按钮底板 */
#chat [class*="avatar"] { display: none !important; }
#chat [class*="icon-button"], #chat [class*="message-buttons"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  color: #94a3b8 !important;
}
/* 消息气泡：用户淡青、助手浅灰白，靠底色区分 */
#chat .message, #chat [class*="message"] {
  font-size: 13.5px !important;
  line-height: 1.7 !important;
}
#chat [class*="user"] {
  background: #e8f7f4 !important;
  border: 1px solid #cdeee8 !important;
}
#chat [class*="bot"] {
  background: #ffffff !important;
  border: 1px solid #e6edf3 !important;
}
#chat .bubble-wrap, #chat .message {
  border-radius: 14px !important;
}
#chat {
  border: none !important;
  background: #fbfdfe !important;
  border-radius: 14px !important;
}

/* ---------- 输入区 ---------- */
#composer {
  margin-top: 12px;
  gap: 10px !important;
}
#msg-box textarea {
  border-radius: 14px !important;
  border: 1px solid #d7e1e7 !important;
  background: #fbfdfe !important;
  font-size: 14px !important;
  padding: 12px 14px !important;
  box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.04);
}
#msg-box textarea:focus {
  border-color: var(--brand-2) !important;
  box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.15) !important;
}
#send-btn {
  background: linear-gradient(135deg, #0f766e, #14b8a6) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 14px !important;
  font-weight: 600 !important;
  min-width: 96px !important;
  box-shadow: 0 8px 18px rgba(13, 148, 136, 0.28) !important;
}
#send-btn:hover { filter: brightness(1.06); }
#reset-btn {
  background: #ffffff !important;
  color: #334155 !important;
  border: 1px solid #d7e1e7 !important;
  border-radius: 14px !important;
  min-width: 96px !important;
}
#reset-btn:hover { border-color: #94a3b8 !important; }

/* ---------- 轨迹面板 ---------- */
#trace-content {
  background: #f8fafc;
  border: 1px dashed #dbe4ec;
  border-radius: 14px;
  padding: 12px 14px;
  min-height: 455px;
  max-height: 500px;
  overflow-y: auto;
  font-size: 13px;
  line-height: 1.65;
  color: #334155;
}
#trace-content p, #trace-content span, #trace-content div, #trace-content li,
#trace-content em, #trace-content strong {
  color: #334155 !important;
}
#trace-content p { margin: 0 0 6px 0; }
#trace-content code {
  background: #e6f7f4;
  color: #0f766e;
  border-radius: 6px;
  padding: 1px 6px;
  font-size: 12.5px;
}
#trace-content blockquote {
  border-left: 3px solid #99f6e4;
  margin: 6px 0;
  padding: 2px 0 2px 10px;
  color: #475569;
}

/* ---------- 示例区 ---------- */
#examples {
  margin-top: 18px;
}
#examples [class*="label"], #examples label, #examples span {
  color: #64748b !important;
}
#examples .gallery, #examples .wrap {
  gap: 8px !important;
}

/* ---------- 页脚 ---------- */
.page-footer {
  margin-top: 22px;
  text-align: center;
  font-size: 12px;
  color: #94a3b8;
  display: flex;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}
.footer-dot { color: #cbd5e1; }

@media (max-width: 760px) {
  .hero { padding: 22px 20px; border-radius: 18px; }
  .hero h1 { font-size: 24px !important; }
  .gradio-container { padding: 14px 12px 24px !important; }
}
"""


def format_trace(trace: list[dict]) -> str:
    """把一次对话的工具调用轨迹渲染为 Markdown（供右侧面板展示）。"""
    lines: list[str] = []
    for step in trace:
        if step["type"] == "tool_call":
            args = step.get("arguments") or {}
            args_str = ", ".join(f"{k}={v}" for k, v in args.items()) or "（无参数）"
            lines.append(f"**步骤 {step['step']}** · 调用 `{step['name']}({args_str})`")
            lines.append(f"> 返回：{step['result']}")
        elif step["type"] == "answer":
            lines.append(f"**最终回复**（步骤 {step['step']}）：{step['content']}")
        elif step["type"] == "limit":
            lines.append(f"⚠️ {step['content']}")
    return "\n\n".join(lines) if lines else "本轮未调用工具 · 直接回复"


def respond(message: str, history: list | None, trace_md: str) -> tuple:
    """单轮对话回调：调用 Agent，返回更新后的聊天记录与工具轨迹。"""
    history = history or []
    message = (message or "").strip()
    if not message:
        return history, trace_md
    history.append({"role": "user", "content": message})
    result = agent.run(message)
    history.append({"role": "assistant", "content": result["answer"]})
    trace_md = format_trace(result["trace"])
    return history, trace_md


def reset_session() -> tuple:
    agent.memory.clear()
    return [], "会话已重置 · 等待新的提问"


def main() -> None:
    with gr.Blocks(title="个人AI助理 Agent") as demo:
        gr.HTML(HERO_HTML)

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, elem_id="chat-panel"):
                chatbot = gr.Chatbot(
                    height=420,
                    show_label=False,
                    elem_id="chat",
                    placeholder="开始对话——问时间、算个数、记待办，或追问上一轮",
                )
                with gr.Row(elem_id="composer"):
                    msg = gr.Textbox(
                        placeholder="输入你的问题，回车发送…",
                        show_label=False,
                        scale=4,
                        elem_id="msg-box",
                    )
                    send = gr.Button("发送", variant="primary", scale=1, elem_id="send-btn")
                    reset = gr.Button("重置会话", scale=1, elem_id="reset-btn")

            with gr.Column(scale=2, elem_id="trace-panel"):
                gr.HTML(TRACE_HEAD)
                trace_box = gr.Markdown(
                    "本轮未调用工具 · 直接回复",
                    elem_id="trace-content",
                )

        gr.Examples(
            examples=_HELP_EXAMPLES,
            inputs=msg,
            label="试试这些问题",
            elem_id="examples",
        )

        gr.HTML(FOOTER_HTML)

        msg.submit(respond, [msg, chatbot, trace_box], [chatbot, trace_box]).then(
            lambda: "", None, msg
        )
        send.click(respond, [msg, chatbot, trace_box], [chatbot, trace_box]).then(
            lambda: "", None, msg
        )
        reset.click(reset_session, None, [chatbot, trace_box])

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="emerald", neutral_hue="slate"),
    )


if __name__ == "__main__":
    main()
