"""可观测性测试：stats 结构 / 终止原因 / 计数 / 结构化日志写入。"""
from __future__ import annotations

import json
import os

from agent.core import Agent
from agent.llm import LLMError, LLMResponse, LLMRetryableError, ToolCall
from agent.tools import ToolRegistry
from tools import register_all


def _make_agent(llm=None, log_dir=None, **kwargs) -> Agent:
    registry = ToolRegistry()
    register_all(registry)
    return Agent(llm=llm, registry=registry, log_dir=log_dir, **kwargs)


def _make_mock_agent(log_dir=None) -> Agent:
    from agent.llm import MockLLM

    registry = ToolRegistry()
    register_all(registry)
    return Agent(llm=MockLLM(registry), registry=registry, log_dir=log_dir)


def test_stats_structure_normal_answer():
    """正常回答 → stats 字段齐全，terminated_by=answer。"""
    result = _make_mock_agent().run("你好呀")
    stats = result["stats"]
    assert stats["terminated_by"] == "answer"
    assert stats["llm_calls"] >= 1
    assert stats["output_tokens"] >= 1
    assert stats["duration_ms"] >= 0
    assert stats["steps"] >= 1


def test_stats_tool_call_counts():
    """工具调用链路 → tool_calls 计数、无错误。"""
    result = _make_mock_agent().run("计算 128 * 37 等于多少")
    stats = result["stats"]
    assert stats["tool_calls"] == 1
    assert stats["tool_errors"] == 0
    assert "4736" in result["answer"]


def test_stats_terminated_by_loop_detected():
    """连续重复调用 → terminated_by=loop_detected。"""

    class RepeatLLM:
        def chat(self, messages, tools=None):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="x", name="get_current_time", arguments={})],
            )

    agent = _make_agent(RepeatLLM(), max_iterations=10, repetition_limit=3)
    stats = agent.run("一直问时间")["stats"]
    assert stats["terminated_by"] == "loop_detected"


def test_stats_terminated_by_llm_error():
    """LLM 致命错误 → terminated_by=llm_error。"""

    class BoomLLM:
        def chat(self, messages, tools=None):
            raise LLMError("认证失败")

    stats = _make_agent(BoomLLM(), max_iterations=3).run("你好")["stats"]
    assert stats["terminated_by"] == "llm_error"


def test_stats_terminated_by_limit():
    """模型永远只调工具不回答 → 跑满轮数 → terminated_by=limit。"""

    class ForeverLLM:
        def chat(self, messages, tools=None):
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="x", name="get_current_time", arguments={})],
            )

    agent = _make_agent(ForeverLLM(), max_iterations=2, repetition_limit=0)
    stats = agent.run("一直问时间")["stats"]
    assert stats["terminated_by"] == "limit"
    assert stats["steps"] == 2


def test_stats_tool_error_counted():
    """工具执行失败 → tool_errors 计数。"""

    class BadToolLLM:
        def chat(self, messages, tools=None):
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="x", name="calculate", arguments={"expression": "1/0"})
                ],
            )

    stats = _make_agent(BadToolLLM(), max_iterations=1).run("算一下")["stats"]
    assert stats["tool_calls"] == 1
    assert stats["tool_errors"] == 1


def test_jsonl_log_written(tmp_path):
    """run 后结构化日志写入：含分步事件与 summary，行行合法 JSON。"""
    log_dir = str(tmp_path)
    result = _make_mock_agent(log_dir=log_dir).run("计算 100 / 4 等于多少")
    path = os.path.join(log_dir, "agent.jsonl")
    assert os.path.exists(path)

    with open(path, encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    events = [ev for ev in lines if ev["event"] != "summary"]
    summary = [ev for ev in lines if ev["event"] == "summary"]
    assert len(summary) == 1
    assert summary[0]["terminated_by"] == result["stats"]["terminated_by"]
    # 工具调用事件与统计一致
    tool_events = [ev for ev in events if ev["event"] == "tool_call"]
    assert len(tool_events) == result["stats"]["tool_calls"] == 1
    assert tool_events[0]["ok"] is True
