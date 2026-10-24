"""工具：长期记忆（记住 / 回忆）。

用户说“记住XXX” → 写入长期记忆；问“我之前说过/上次…” → 语义检索召回。
跨会话生效（SQLite 持久化）。
"""
from __future__ import annotations

import re

from agent.long_term_memory import LongTermMemory
from agent.tools import tool

_store = LongTermMemory()


def _extract_content(text: str) -> dict:
    cleaned = re.sub(r"记住|帮我|请|记下|一下|这条", "", text)
    cleaned = re.sub(r"[?？!！。，,]+$", "", cleaned).strip()
    return {"content": cleaned} if cleaned else {}


def _extract_query(text: str) -> dict:
    cleaned = re.sub(r"回忆|回想|上次|之前|说过|说了|我|的|什么|内容|一下", "", text)
    cleaned = re.sub(r"[?？!！。，,\s]+$", "", cleaned).strip()
    return {"query": cleaned} if cleaned else {}


@tool(
    "remember",
    "记住一条关于用户的长期事实或偏好（如职业、目标、喜好），跨会话生效",
    keywords=["记住", "记住了"],
    mock_args_fn=_extract_content,
)
def remember(content: str) -> str:
    """content: 要记住的事实或偏好"""
    _store.remember(content)
    return f"已记住：{content}"


@tool(
    "recall_memory",
    "回忆之前记住的用户信息（如“我上次说了什么”），按主题语义检索",
    keywords=["上次", "之前说过", "回忆", "我说过", "说了什么"],
    mock_args_fn=_extract_query,
)
def recall_memory(query: str = "") -> str:
    """query: 回忆的主题，如 面试 / 健身 / 目标（可为空）"""
    results = _store.recall(query)
    if not results:
        return "我还没有记住相关的信息。你可以说“记住XXX”来保存。"
    return "我记得：\n" + "\n".join(f"- {r['content']}" for r in results)
