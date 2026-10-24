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
