"""LLM 错误处理层测试（P1）：重试 / 超窗识别 / 非法输出 / Agent 层兜底。"""
from __future__ import annotations

from types import SimpleNamespace

from agent.core import Agent
from agent.llm import (
    ContextLengthExceeded,
    LLMError,
    LLMResponse,
    LLMRetryableError,
    OpenAILLM,
)
from agent.tools import ToolRegistry
from tools import register_all


def _openai_error(cls, msg: str = "boom", status: int = 500):
    import openai

    resp = SimpleNamespace(
        status_code=status, headers={}, text=msg, request=SimpleNamespace()
    )
    if issubclass(cls, openai.APIStatusError):
        return cls(msg, response=resp, body={"error": {"message": msg}})
    return cls(msg)


def _chat_resp(content=None, tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _make_client(create):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _make_llm(create, **kwargs):
    return OpenAILLM(
        base_url="",
        api_key="test-key",
        model="test-model",
        max_retries=kwargs.pop("max_retries", 3),
        retry_base_delay=kwargs.pop("retry_base_delay", 0.01),
        _client=_make_client(create),
        **kwargs,
    )


# -- OpenAILLM 重试 ----------------------------------------------------------


def test_retry_then_success():
    """瞬时错误（429 限流）自动重试：前 2 次失败，第 3 次成功。"""
    import openai

    n = {"count": 0}

    def create(**kwargs):
        n["count"] += 1
        if n["count"] < 3:
            raise _openai_error(openai.RateLimitError, "rate limited", 429)
        return _chat_resp(content="重试成功")

    llm = _make_llm(create, max_retries=3)
    out = llm.chat([{"role": "user", "content": "hi"}])
    assert out.content == "重试成功"
    assert n["count"] == 3


def test_retry_exhausted_raises_retryable():
    """重试耗尽 → LLMRetryableError（可识别、可兜底）。"""
    import openai

    def create(**kwargs):
        raise _openai_error(openai.RateLimitError, "always limited", 429)

    llm = _make_llm(create, max_retries=2)
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("应当抛出 LLMRetryableError")
    except LLMRetryableError:
        pass


def test_context_length_raises_special():
    """超窗（BadRequestError 含 context_length）→ ContextLengthExceeded。"""
    import openai

    def create(**kwargs):
        raise _openai_error(
            openai.BadRequestError,
            "this model's maximum context length is 8192 tokens",
            400,
        )

    llm = _make_llm(create, max_retries=2)
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("应当抛出 ContextLengthExceeded")
    except ContextLengthExceeded:
        pass


def test_invalid_json_raises_llm_error():
    """工具参数非法 JSON → LLMError（不再静默吞成 {}）。"""
    bad_tc = SimpleNamespace(
        id="c1", function=SimpleNamespace(name="calculate", arguments="{not json")
    )
    llm = _make_llm(lambda **kw: _chat_resp(content=None, tool_calls=[bad_tc]))
    try:
        llm.chat([{"role": "user", "content": "hi"}])
        raise AssertionError("应当抛出 LLMError")
    except LLMError as exc:
        assert "不是合法 JSON" in str(exc)


# -- Agent 层兜底 ------------------------------------------------------------


def _make_agent(llm) -> Agent:
    registry = ToolRegistry()
    register_all(registry)
    return Agent(llm=llm, registry=registry, max_iterations=3)


def test_agent_fatal_error_fallback():
    """致命错误（认证失败）→ 兜底文本，不崩溃。"""

    class BoomLLM:
        def chat(self, messages, tools=None):
            raise LLMError("认证失败")

    result = _make_agent(BoomLLM()).run("你好")
    assert "模型调用失败" in result["answer"]
    assert "认证失败" in result["answer"]


def test_agent_retryable_fallback():
    """重试耗尽 → 兜底文本提示稍后再试。"""

    class RetryLLM:
        def chat(self, messages, tools=None):
            raise LLMRetryableError("重试 3 次仍失败：timeout")

    result = _make_agent(RetryLLM()).run("你好")
    assert "模型服务暂时不可用" in result["answer"]


def test_agent_ctx_shrink_retry_succeeds():
    """超窗 → 收缩上下文后原轮重试成功。"""
    n = {"count": 0}

    class CtxLLM:
        def chat(self, messages, tools=None):
            n["count"] += 1
            if n["count"] == 1:
                raise ContextLengthExceeded("输入太长")
            return LLMResponse(content="收缩后成功", tool_calls=[])

    result = _make_agent(CtxLLM()).run("你好")
    assert result["answer"] == "收缩后成功"
    assert n["count"] == 2, "应收缩后重试一次"


def test_agent_ctx_shrink_twice_fails():
    """收缩一次仍超窗 → 兜底返回明确提示，不崩溃。"""

    class CtxLLM2:
        def chat(self, messages, tools=None):
            raise ContextLengthExceeded("仍超限")

    result = _make_agent(CtxLLM2()).run("你好")
    assert "上下文超限" in result["answer"]
