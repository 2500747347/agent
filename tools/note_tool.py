"""工具：笔记（新增 / 检索）。"""
from __future__ import annotations

import re

from agent.tools import tool
from tools.data_store import get_connection, init_db

init_db()


def _extract_note(text: str) -> dict:
    cleaned = re.sub(r"帮我|记个|记一下|笔记|备忘|写进|一条", "", text)
    cleaned = re.sub(r"^[：:\s]+|[?？!！。，,:：\s]+$", "", cleaned).strip()
    if not cleaned:
        return {}
    return {"content": cleaned}


def _extract_keyword(text: str) -> dict:
    # 优先提取“笔记里/笔记中的X”模式
    m = re.search(r"笔记[里中有的]*([\u4e00-\u9fa5a-z0-9]{2,})", text)
    if m:
        return {"keyword": m.group(1)}
    cleaned = re.sub(r"找|查|笔记|里|有|什么|关于|一下|的", "", text)
    cleaned = re.sub(r"[?？!！。，,:\s]+$", "", cleaned).strip()
    if not cleaned:
        return {}
    return {"keyword": cleaned}


@tool(
    "add_note",
    "新增一条笔记，content 为笔记内容",
    keywords=["记笔记", "写笔记", "备忘", "记一条", "笔记"],
    mock_args_fn=_extract_note,
)
def add_note(content: str) -> str:
    """content: 笔记内容"""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute("INSERT INTO notes (content) VALUES (?)", (content,))
        return f"已保存笔记 #{cur.lastrowid}：{content}"
    finally:
        conn.close()


@tool(
    "search_notes",
    "按关键词检索已保存的笔记，返回匹配的笔记内容",
    keywords=["找笔记", "查笔记", "笔记里有", "翻翻笔记", "查一下笔记", "找一下笔记", "的笔记"],
    mock_args_fn=_extract_keyword,
)
def search_notes(keyword: str) -> str:
    """keyword: 检索关键词"""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, content FROM notes WHERE content LIKE ? ORDER BY id DESC LIMIT 5",
            (f"%{keyword}%",),
        ).fetchall()
        if not rows:
            return f"没有找到包含“{keyword}”的笔记。"
        return "\n".join(f"#{r['id']} {r['content']}" for r in rows)
    finally:
        conn.close()
