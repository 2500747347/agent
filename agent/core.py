"""Agent 主循环：ReAct + Function Calling。

处理流程（详见 docs/01-需求与架构.md 4.1）：
    用户输入
      → LLM(会话历史 + 工具schema)
      → 有 tool_calls?  → 执行工具 → 结果回填 → 循环（最多 max_iterations 轮）
      → 无 tool_calls   → 输出最终回复
"""
from __future__ import annotations

import json
from typing import Any

from agent.llm import LLMResponse  # noqa: F401  (类型标注)
from agent.memory import ConversationMemory
from agent.tools import ToolRegistry


class Agent:
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        memory: ConversationMemory | None = None,
        max_iterations: int = 5,
    ):
        self._llm = llm
        self._registry = registry
        self._memory = memory or ConversationMemory()
        self._max_iterations = max_iterations

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
        messages = self._memory.messages + [{"role": "user", "content": user_input}]
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
