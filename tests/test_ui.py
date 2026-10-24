"""P4 界面逻辑测试：respond 回调与轨迹渲染（不启动服务器）。"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.app import format_trace, respond  # noqa: E402
from tools.data_store import get_connection  # noqa: E402


@pytest.fixture(autouse=True)
def clean_store():
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM long_term_memory")
        conn.execute("DELETE FROM todos")
    conn.close()
    yield


def test_respond_calculator_trace():
    history, trace = respond("帮我计算 128 * 37 等于多少", [], "")
    assert history[-1]["role"] == "assistant"
    assert "4736" in history[-1]["content"]
    assert "calculate" in trace


def test_respond_weather():
    history, _ = respond("芜湖今天天气怎么样", [], "")
    assert "芜湖" in history[-1]["content"]


def test_respond_cross_session_memory():
    respond("记住我正在准备面试", [], "")
    history, _ = respond("我之前说过什么", [], "")
    assert "面试" in history[-1]["content"]


def test_format_trace_no_tool():
    md = format_trace([{"type": "answer", "step": 0, "content": "你好"}])
    assert "你好" in md
