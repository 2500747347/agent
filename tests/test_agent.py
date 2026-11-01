"""Agent 核心链路测试（P2 验收）。

覆盖：无工具对话、单工具调用、工具结果回填、多轮记忆。
"""
from __future__ import annotations

import os
import sys

# 保证从项目根导入 agent/tools 包（pytest 默认路径不含项目根）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.core import Agent  # noqa: E402
from agent.llm import MockLLM  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402
from tools import register_all  # noqa: E402


def make_agent() -> Agent:
    registry = ToolRegistry()
    register_all(registry)
    return Agent(llm=MockLLM(registry), registry=registry)


def test_no_tool_direct_reply():
    agent = make_agent()
    result = agent.run("你好呀")
    assert result["answer"]
    assert not any(t["type"] == "tool_call" for t in result["trace"])


def test_time_tool_triggered():
    agent = make_agent()
    result = agent.run("今天是几号")
    calls = [t for t in result["trace"] if t["type"] == "tool_call"]
    assert calls, "应当发起工具调用"
    assert calls[0]["name"] == "get_current_time"
    assert "年" in result["answer"]


def test_calculator_tool_result():
    agent = make_agent()
    result = agent.run("计算 128 * 37 等于多少")
    calls = [t for t in result["trace"] if t["type"] == "tool_call"]
    assert calls and calls[0]["name"] == "calculate"
    assert "4736" in result["answer"], f"计算结果应回填到回复，实际：{result['answer']}"


def test_multi_turn_memory_kept():
    agent = make_agent()
    agent.run("今天是几号")
    agent.run("再算一下 100 / 4")
    # 两轮对话后，历史应保留多轮消息
    roles = [m["role"] for m in agent.memory.messages]
    assert roles.count("user") == 2
    assert roles.count("assistant") == 2


def test_unknown_tool_error_survives():
    """LLM 调用不存在的工具时，错误应作为文本回填，不崩溃。"""
    from agent.llm import ToolCall, LLMResponse

    registry = ToolRegistry()
    register_all(registry)

    class BadLLM:
        def chat(self, messages, tools=None):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="x", name="not_exist_tool", arguments={})],
            )

    agent = Agent(llm=BadLLM(), registry=registry, max_iterations=2)
    result = agent.run("随便问问")
    assert any("未知工具" in t.get("result", "") for t in result["trace"] if t["type"] == "tool_call")


# -- 循环保险丝（Loop 终止机制补全）-----------------------------------------


def test_repetition_detection_stops_loop():
    """连续相同调用达 repetition_limit → 判定死循环提前终止，不跑满轮数。"""
    from agent.llm import ToolCall, LLMResponse

    registry = ToolRegistry()
    register_all(registry)

    class RepeatLLM:
        def chat(self, messages, tools=None):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="x", name="get_current_time", arguments={})],
            )

    agent = Agent(llm=RepeatLLM(), registry=registry, max_iterations=10, repetition_limit=3)
    result = agent.run("一直问时间")
    assert "循环终止" in result["answer"]
    calls = [t for t in result["trace"] if t["type"] == "tool_call"]
    # 第 3 次重复即触发终止，而不是傻跑 10 轮
    assert len(calls) == 3
    assert calls[-1]["result"].startswith("[循环终止]")


def test_repetition_not_false_positive():
    """参数变化不算重复：不同参数依次调用 → 正常完成。"""
    from agent.llm import ToolCall, LLMResponse

    registry = ToolRegistry()
    register_all(registry)

    class AltLLM:
        def __init__(self) -> None:
            self._n = 0

        def chat(self, messages, tools=None):
            self._n += 1
            if self._n <= 3:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(id="x", name="calculate", arguments={"expression": f"{self._n}+1"})
                    ],
                )
            return LLMResponse(content="算完了", tool_calls=[])

    agent = Agent(llm=AltLLM(), registry=registry, max_iterations=5, repetition_limit=3)
    result = agent.run("连续算几道题")
    assert "循环终止" not in result["answer"]
    assert "算完了" in result["answer"]


def test_tool_timeout_interrupts_wait():
    """慢工具超过 tool_timeout → 回填 [工具超时]，不阻塞主循环。"""
    import time

    from agent.llm import ToolCall, LLMResponse
    from agent.tools import Tool

    registry = ToolRegistry()
    register_all(registry)

    def slow_fn() -> str:
        time.sleep(0.5)
        return "慢工具跑完了"

    registry.register(Tool(name="slow_tool", description="慢速演示工具", fn=slow_fn))

    class SlowLLM:
        """第一轮调慢工具（超时），第二轮看到提示后直接回答。"""

        def __init__(self) -> None:
            self._n = 0

        def chat(self, messages, tools=None):
            self._n += 1
            if self._n == 1:
                return LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="x", name="slow_tool", arguments={})],
                )
            return LLMResponse(content="工具太慢，我换个方式回答", tool_calls=[])

    start = time.time()
    agent = Agent(llm=SlowLLM(), registry=registry, max_iterations=3, tool_timeout=0.1)
    result = agent.run("调用慢工具")
    elapsed = time.time() - start
    assert elapsed < 0.5, f"超时应快速返回，实际耗时 {elapsed:.2f}s"
    tool_calls = [t for t in result["trace"] if t["type"] == "tool_call"]
    assert tool_calls and tool_calls[0]["result"].startswith("[工具超时]")
    assert "工具太慢" in result["answer"]


def test_timeout_off_by_default():
    """默认 tool_timeout=0 → 不启用超时线程，正常执行。"""
    from agent.llm import ToolCall, LLMResponse

    registry = ToolRegistry()
    register_all(registry)

    class NormalLLM:
        def chat(self, messages, tools=None):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="x", name="calculate", arguments={"expression": "2+2"})],
            )

    agent = Agent(llm=NormalLLM(), registry=registry, max_iterations=2)
    result = agent.run("算 2+2")
    assert "工具超时" not in result["answer"]
