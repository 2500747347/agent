"""LLM 统一封装。

对外提供统一的 LLMResponse 结构（content + tool_calls），
屏蔽不同后端（真实 OpenAI 兼容接口 / Mock）的差异。

- OpenAILLM : 通过 openai SDK 调用任意 OpenAI 兼容接口（Codex/GPT/DeepSeek/通义…）
- MockLLM   : 规则式模拟，离线可用，用于开发期与评测基线
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 统一响应结构
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """模型发起的一次工具调用。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """模型一次输出的统一表示：要么有文本，要么有工具调用（可两者都有）。"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------------------
# 真实模型（OpenAI 兼容接口）
# ---------------------------------------------------------------------------


class OpenAILLM:
    """通过 openai SDK 调用 OpenAI 兼容接口（chat.completions）。"""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.3):
        from openai import OpenAI  # 延迟导入，避免 Mock 模式下无谓加载

        self._client = OpenAI(base_url=base_url or None, api_key=api_key)
        self._model = model
        self._temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            **kwargs,
        )
        msg = resp.choices[0].message
        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return LLMResponse(content=msg.content, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# Mock 模式（离线规则式）
# ---------------------------------------------------------------------------


class MockLLM:
    """规则式模拟 LLM：按工具声明的 keywords 命中用户输入 → 发起工具调用。

    用于：
    1) 开发期无 Key 时跑通 Agent 全链路；
    2) 评测时的确定性基线（真实模型可对比）。
    """

    def __init__(self, registry: "ToolRegistry", name: str = "mock"):
        self._registry = registry
        self._name = name

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        # 规则1：最后一条消息是 tool 结果 → 简单总结后结束
        if messages and messages[-1].get("role") == "tool":
            content = str(messages[-1].get("content", ""))
            return LLMResponse(content=f"[Mock] 工具已执行，结果为：{content}")

        # 规则2：取最近一条用户消息，按 keywords 匹配工具
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = str(m.get("content", ""))
                break

        for tool in self._registry.all:
            for kw in tool.keywords:
                if kw and kw in user_text:
                    # Mock 模式参数补给：固定参数 > 从输入提取 > 空参数
                    if tool.mock_args is not None:
                        arguments = dict(tool.mock_args)
                    elif tool.mock_args_fn is not None:
                        arguments = tool.mock_args_fn(user_text) or {}
                    else:
                        arguments = {}
                    return LLMResponse(
                        content=None,
                        tool_calls=[ToolCall(id="mock-call-1", name=tool.name, arguments=arguments)],
                    )

        # 规则3：无匹配 → 直接回复
        return LLMResponse(content=f"[Mock] 收到：{user_text or '（空）'}")
