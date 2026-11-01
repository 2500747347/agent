"""P0 会话历史持久层 + 向量检索测试。

覆盖：历史落库、关键词/语义检索召回、token 预算截断、
滑动窗口裁剪后旧消息落库并可被检索注入。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.history_memory import (  # noqa: E402
    EmbeddingVectorizer,
    HistoryMemory,
    TfVectorizer,
    estimate_tokens,
    try_build_embedding_vectorizer,
)
from agent.memory import ConversationMemory  # noqa: E402
from agent.core import Agent  # noqa: E402
from agent.llm import LLMResponse, MockLLM  # noqa: E402
from agent.tools import ToolRegistry, tool  # noqa: E402
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


# -- P1：token 预算 ----------------------------------------------------------


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") == 4          # 中文 1 字 ≈ 1 token
    assert estimate_tokens("hello world") == 2       # ASCII 4 字符 ≈ 1 token
    assert estimate_tokens("你好abc") == 2           # 混合


def test_ensure_token_budget_trims_and_offloads():
    mem = ConversationMemory(max_turns=10)
    for i in range(10):
        mem.add_user(f"第{i + 1}轮" + "测" * 100)     # 每轮 user ≈ 104 token
        mem.add_assistant("答" * 50)                  # 每轮 assistant ≈ 50 token
    assert mem.retrieve_context  # 窗口已接近满
    window_tokens = sum(estimate_tokens(m.get("content") or "") for m in mem.messages)
    assert window_tokens > 800                        # 确认超预算场景成立

    remaining = mem.ensure_token_budget(800)
    assert remaining <= 800
    # 被裁轮次已落库且可检索召回（不失忆）
    hits = mem.retrieve_context("第1轮")
    assert any("第1轮" in h for h in hits)


class _CaptureLLM:
    """捕获每次发给 LLM 的 messages。"""

    def __init__(self):
        self.seen = []

    def chat(self, messages, tools=None):
        self.seen.append(list(messages))
        return LLMResponse(content="（测试回复）", tool_calls=None)


def test_agent_total_budget_shrink():
    """总预算超限时：先砍检索片段，再收缩窗口，最终上下文 ≤ 预算。"""
    cap = _CaptureLLM()
    agent = Agent(
        llm=cap,
        registry=ToolRegistry(),
        memory=ConversationMemory(max_turns=10),
        max_context_tokens=600,
        retrieve_budget=100,
        max_iterations=1,
    )
    for i in range(6):
        agent.run(f"第{i + 1}轮" + "长" * 120)         # 每轮 ≈ 150 token
    last = cap.seen[-1]
    total = sum(estimate_tokens(m.get("content") or "") for m in last)
    assert total <= 600, f"上下文超预算: {total}"


def test_tool_result_truncation():
    """超长工具结果被截断，防止单轮撑爆上下文。"""

    @tool("big_result", "返回超长结果", keywords=["超长"])
    def big_result() -> str:
        return "长" * 1000

    registry = ToolRegistry()
    registry.register(big_result)
    agent = Agent(
        llm=MockLLM(registry),
        registry=registry,
        memory=ConversationMemory(max_turns=2),
        tool_result_max_chars=100,
        max_iterations=1,
    )
    r = agent.run("给我超长结果")
    calls = [t for t in r["trace"] if t["type"] == "tool_call"]
    assert calls, "Mock 模式应触发 big_result"
    result = calls[0]["result"]
    assert len(result) <= 100 + len("…[已截断，结果过长]")
    assert "已截断" in result


# -- P2：语义 Embedding -------------------------------------------------------

_EMB_AVAILABLE = try_build_embedding_vectorizer() is not None


@pytest.mark.skipif(not _EMB_AVAILABLE, reason="本地 embedding 模型未就绪")
def test_embedding_semantic_recall():
    """语义召回：字面零重合但语义相关（TF 判 0 分，Embedding 可命中）。"""
    hm = HistoryMemory(session_id="emb1", vectorizer=EmbeddingVectorizer())
    hm.add_messages([{"role": "user", "content": "我想找一份工作"}])
    hm.add_messages([{"role": "user", "content": "今天天气不错"}])
    hits = hm.retrieve("求职意向")  # 与"工作"零字面重合
    assert hits, "Embedding 应能召回语义相关历史"
    assert any("工作" in h for h in hits), f"未召回目标历史: {hits}"


@pytest.mark.skipif(not _EMB_AVAILABLE, reason="本地 embedding 模型未就绪")
def test_embedding_end_to_end_injection():
    """端到端：窗口裁掉的旧轮次，经 Embedding 检索注入后续对话。

    裁剪发生在每轮 run 结束（add_user 时），故第 4 轮 query 才会
    命中第 3 轮结束时才落库的第 1 轮历史。
    """
    cap = _CaptureLLM()
    mem = ConversationMemory(
        max_turns=2, history=HistoryMemory(vectorizer=EmbeddingVectorizer())
    )
    agent = Agent(llm=cap, registry=ToolRegistry(), memory=mem, max_iterations=1)
    agent.run("第一轮：我想找一份 Agent 开发的工作")
    agent.run("第二轮：随便聊聊天气")
    agent.run("第三轮：今天吃了吗")
    agent.run("第四轮：我之前的求职意向是什么")  # 语义相关，应召回第 1 轮
    last = cap.seen[-1]
    injected = [m for m in last if "【对话历史" in m["content"]]
    assert injected, "未注入任何检索片段"
    assert any("Agent 开发" in m["content"] for m in injected), "语义相关旧历史未被注入"
