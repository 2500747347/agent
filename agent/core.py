"""Agent 主循环：ReAct + Function Calling。

处理流程（详见 docs/01-需求与架构.md 4.1）：
    用户输入
      → LLM(会话历史 + 工具schema)
      → 有 tool_calls?  → 执行工具 → 结果回填 → 循环（最多 max_iterations 轮）
      → 无 tool_calls   → 输出最终回复

上下文预算（P1）：组装时估算总 token，超预算按可牺牲性依次收缩——
先砍检索注入片段，再收缩窗口旧轮次（被裁轮次照常落库，不失忆）。
"""
from __future__ import annotations

import json
from typing import Any

from agent.history_memory import estimate_tokens
from agent.llm import LLMResponse  # noqa: F401  (类型标注)
from agent.memory import ConversationMemory
from agent.tools import ToolRegistry

# 上下文总预算（token，启发式估算）：system + 检索 + 窗口 + 当前输入
_DEFAULT_MAX_CONTEXT_TOKENS = 8000
# 检索注入片段预算
_DEFAULT_RETRIEVE_BUDGET = 1500
# 单条工具结果最大字符数（超长截断，防止单轮撑爆上下文）
_DEFAULT_TOOL_RESULT_MAX_CHARS = 4000


def _ctx_tokens(msgs: list[dict]) -> int:
    """一组消息的 token 估算。"""
    return sum(estimate_tokens(m.get("content") or "") for m in msgs)


class Agent:
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        memory: ConversationMemory | None = None,
        max_iterations: int = 5,
        system_prompt: str | None = None,
        max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS,
        retrieve_budget: int = _DEFAULT_RETRIEVE_BUDGET,
        tool_result_max_chars: int = _DEFAULT_TOOL_RESULT_MAX_CHARS,
    ):
        self._llm = llm
        self._registry = registry
        self._memory = memory or ConversationMemory()
        self._max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._max_context_tokens = max_context_tokens
        self._retrieve_budget = retrieve_budget
        self._tool_result_max_chars = tool_result_max_chars

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def run(self, user_input: str) -> dict[str, Any]:
        """执行一轮对话。

        返回结构：
        {
            "answer": str,        # 最终回复
            "trace": [            # 过程轨迹（供界面/评测使用）
                {"type": "tool_call", "step": n, "name": ..., "arguments": ..., "result": ...},
                {"type": "answer", "step": n, "content": ...},
                ...
            ]
        }
        """
        # ---- 上下文组装（P1：总 token 预算控制） ----
        # 1) 不可牺牲部分：System + 当前输入
        base: list[dict] = []
        if self._system_prompt:
            base.append({"role": "system", "content": self._system_prompt})
        user_message = {"role": "user", "content": user_input}
        base_tokens = _ctx_tokens(base) + estimate_tokens(user_input)

        # 2) 检索注入（可牺牲）：独立 system 消息，不污染 user 输入；
        #    Mock 模式只匹配 user/tool 消息，天然忽略该注入，评测行为不变
        retrieved = self._memory.retrieve_context(
            user_input, token_budget=self._retrieve_budget
        )
        injected: list[dict] = []
        if retrieved:
            injected.append(
                {
                    "role": "system",
                    "content": "【对话历史中的相关信息，供参考】\n"
                    + "\n".join(f"- {c}" for c in retrieved),
                }
            )

        # 3) 窗口历史 + 当前输入
        messages = base + injected + self._memory.messages + [user_message]

        # 4) 总预算收缩：超预算先牺牲检索片段，再收缩窗口旧轮次（被裁轮次照常落库）
        if _ctx_tokens(messages) > self._max_context_tokens:
            messages = base + self._memory.messages + [user_message]
        if _ctx_tokens(messages) > self._max_context_tokens:
            self._memory.ensure_token_budget(self._max_context_tokens - base_tokens)
            messages = base + self._memory.messages + [user_message]

        trace: list[dict] = []

        for step in range(self._max_iterations):
            resp = self._llm.chat(messages, tools=self._registry.schemas())

            # 情况A：模型不再调用工具 → 产出最终回复
            if not resp.has_tool_calls:
                answer = resp.content or "（模型未返回内容）"
                trace.append({"type": "answer", "step": step, "content": answer})
                self._memory.add_user(user_input)
                self._memory.add_assistant(answer)
                return {"answer": answer, "trace": trace}

            # 情况B：模型发起工具调用 → 记录并逐个执行
            raw_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ]
            messages.append({"role": "assistant", "content": None, "tool_calls": raw_calls})

            for tc in resp.tool_calls:
                tool = self._registry.get(tc.name)
                if tool is None:
                    result = f"[错误] 未知工具：{tc.name}"
                else:
                    result = tool.run(**tc.arguments)
                    if len(result) > self._tool_result_max_chars:
                        result = result[: self._tool_result_max_chars] + "…[已截断，结果过长]"
                messages.append(
                    {"role": "tool", "content": result, "tool_call_id": tc.id, "name": tc.name}
                )
                trace.append(
                    {
                        "type": "tool_call",
                        "step": step,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result": result,
                    }
                )

        # 达到轮次上限：强制兜底，防止死循环
        fallback = "（已达到最大推理轮次，请简化问题或稍后再试。）"
        trace.append({"type": "limit", "step": self._max_iterations, "content": fallback})
        self._memory.add_user(user_input)
        self._memory.add_assistant(fallback)
        return {"answer": fallback, "trace": trace}
