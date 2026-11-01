"""P0 会话历史持久层 + 向量检索测试。

覆盖：历史落库、关键词/语义检索召回、token 预算截断、
滑动窗口裁剪后旧消息落库并可被检索注入。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.history_memory import HistoryMemory, TfVectorizer  # noqa: E402
from agent.memory import ConversationMemory  # noqa: E402
from tools.data_store import get_connection  # noqa: E402


@pytest.fixture(autouse=True)
def clean_store():
    """每个测试前清理全部数据表，保证可重复。"""
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM conversation_history")
        conn.execute("DELETE FROM long_term_memory")
        conn.execute("DELETE FROM todos")
        conn.execute("DELETE FROM notes")
    conn.close()
    yield


def test_add_and_count():
    hm = HistoryMemory(session_id="t1")
    assert hm.count() == 0
    hm.add_messages([{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"}])
    assert hm.count() == 2


def test_add_skips_empty_content():
    hm = HistoryMemory(session_id="t2")
    hm.add_messages(
        [
            {"role": "user", "content": "有内容"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
            {"role": "tool", "content": ""},
        ]
    )
    assert hm.count() == 1


def test_retrieve_keyword_recall():
    hm = HistoryMemory(session_id="t3")
    hm.add_messages(
        [
            {"role": "user", "content": "我的项目目标是找一份 Agent 开发工作"},
            {"role": "assistant", "content": "好的，已记住"},
            {"role": "user", "content": "今天天气怎么样"},
            {"role": "assistant", "content": "芜湖今天晴"},
        ]
    )
    hits = hm.retrieve("我的项目目标")
    assert len(hits) >= 1
    assert "Agent 开发工作" in hits[0]


def test_retrieve_empty_query_returns_recent():
    hm = HistoryMemory(session_id="t4", top_k=2)
    for i in range(5):
        hm.add_messages([{"role": "user", "content": f"消息{i}"}])
    hits = hm.retrieve("")
    assert len(hits) == 2
    assert hits[0] == "消息4"  # 最近的在最前


def test_retrieve_token_budget():
    hm = HistoryMemory(session_id="t5", top_k=10, token_budget=12)
    hm.add_messages([{"role": "user", "content": "关键词A " + "很长" * 20}])
    hm.add_messages([{"role": "user", "content": "关键词A 短"}])
    hits = hm.retrieve("关键词A")
    # 预算 12：超长条被跳过，只应取回短条
    assert len(hits) == 1
    assert all(len(h) <= 12 for h in hits)


def test_session_isolation():
    h1 = HistoryMemory(session_id="sess-a")
    h2 = HistoryMemory(session_id="sess-b")
    h1.add_messages([{"role": "user", "content": "只有A有这条"}])
    assert h2.count() == 0


def test_trim_offloads_to_history():
    """滑动窗口裁掉的旧消息落库，且可被检索注入召回。"""
    mem = ConversationMemory(max_turns=2)  # 只保留最近 2 轮
    mem.add_user("第一轮：我的求职目标是找 Agent 开发的工作")
    mem.add_assistant("好的，已记录你的求职目标")
    mem.add_user("第二轮：随便聊聊")
    mem.add_assistant("嗯")
    mem.add_user("第三轮：今天的安排")
    mem.add_assistant("好的")
    # 第一轮已被整轮裁掉，不在窗口内
    assert len(mem.messages) == 4
    assert "求职目标" not in str(mem.messages)
    # 但已整轮落库（user + assistant），按相关性可检索召回
    hits = mem.retrieve_context("我第一轮说的求职目标是什么")
    assert any("Agent 开发" in h for h in hits)
    assert any("已记录你的求职目标" in h for h in hits)


def test_retrieve_context_empty_without_history():
    mem = ConversationMemory(max_turns=2, history=None)
    assert mem.retrieve_context("随便问问") == []
