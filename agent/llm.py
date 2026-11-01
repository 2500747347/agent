"""LLM 统一封装。

对外提供统一的 LLMResponse 结构（content + tool_calls），
屏蔽不同后端（真实 OpenAI 兼容接口 / Mock）的差异。

- OpenAILLM : 通过 openai SDK 调用任意 OpenAI 兼容接口（Codex/GPT/DeepSeek/通义…）
- MockLLM   : 规则式模拟，离线可用，用于开发期与评测基线

错误处理（P1）：
- 瞬时错误（网络/限流/5xx）→ 指数退避自动重试
- 上下文超窗（context_length_exceeded）→ 抛 ContextLengthExceeded，由 Agent 层收缩上下文重试
- 工具参数非法 JSON → 不再静默吞成 {}，抛 LLMError 回填给模型自纠
- 重试耗尽/致命错误 → 抛本地异常，Agent 层兜底为文本，绝不崩溃
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 本地异常体系（跨层可识别，不依赖具体 SDK）
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """LLM 调用致命错误（认证失败、非法参数、模型输出损坏等），重试无意义。"""


class LLMRetryableError(Exception):
    """瞬时性错误（网络/限流/5xx），重试指定次数后仍失败抛出。"""


class ContextLengthExceeded(Exception):
    """输入超过模型上下文窗口，需收缩上下文后重试。"""


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

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        _client: Any = None,
    ):
        if _client is not None:
            self._client = _client  # 测试注入
        else:
            from openai import OpenAI  # 延迟导入，避免 Mock 模式下无谓加载

            self._client = OpenAI(base_url=base_url or None, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    def _classify(self, exc: Exception) -> str:
        """把 SDK 异常分类：retryable（可重试）/ context（超窗）/ fatal（致命）。"""
        try:
            import openai
        except ImportError:  # 无 SDK 时无法分类，按致命处理
            return "fatal"
        if isinstance(
            exc,
            (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.RateLimitError,
                openai.InternalServerError,
            ),
        ):
            return "retryable"
        if isinstance(exc, openai.BadRequestError):
            text = str(exc)
            if "context_length" in text or "maximum context" in text:
                return "context"
            return "fatal"
        return "fatal"

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kwargs: dict[str, Any] = {}
        if tools:
            kwargs["tools"] = tools

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
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
                    except json.JSONDecodeError as jexc:
                        # 模型输出损坏：不静默吞成 {}，明确报错让上层回填给模型自纠
                        raise LLMError(
                            f"工具 {tc.function.name!r} 的参数不是合法 JSON："
                            f"{tc.function.arguments!r}"
                        ) from jexc
                    tool_calls.append(
                        ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                    )
                return LLMResponse(content=msg.content, tool_calls=tool_calls)
            except (LLMError, ContextLengthExceeded):
                raise
            except Exception as exc:  # noqa: BLE001 —— SDK 异常统一分类
                kind = self._classify(exc)
                if kind == "context":
                    raise ContextLengthExceeded(
                        f"输入超过模型上下文窗口：{exc}"
                    ) from exc
                if kind == "retryable" and attempt < self._max_retries:
                    delay = self._retry_base_delay * (2**attempt)
                    time.sleep(delay)
                    last_exc = exc
                    continue
                if kind == "retryable":
                    raise LLMRetryableError(
                        f"重试 {self._max_retries} 次仍失败：{exc}"
                    ) from exc
                raise LLMError(f"模型调用失败：{exc}") from exc

        # 理论不可达（retryable 分支已 raise）
        raise LLMRetryableError(f"重试 {self._max_retries} 次仍失败：{last_exc}")


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
        # 策略：最长关键词优先（更具体的表达优先），平局按注册顺序
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = str(m.get("content", ""))
                break

        best: tuple[int, Tool] | None = None  # (关键词长度, 工具)
        for tool in self._registry.all:
            for kw in tool.keywords:
                if kw and kw in user_text:
                    if best is None or len(kw) > best[0]:
                        best = (len(kw), tool)

        if best is not None:
            _, tool = best
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
