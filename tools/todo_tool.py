"""工具：待办事项管理（新增 / 查询 / 完成）。

数据持久化到 SQLite（data/assistant.db），重启不丢失。
"""
from __future__ import annotations

import re

from agent.tools import tool
from tools.data_store import get_connection, init_db

init_db()  # 确保表存在（幂等）


def _extract_add_todo(text: str) -> dict:
    """Mock 模式参数提取：去掉指令词，剩余内容作为待办内容。"""
    cleaned = re.sub(r"帮我|记一下|请|添加|新增|待办|事项|一条|一个", "", text)
    cleaned = re.sub(r"[?？!！。，,]+$", "", cleaned).strip()
    if not cleaned:
        return {}
    return {"content": cleaned}


def _extract_todo_id(text: str) -> dict:
    m = re.search(r"\d+", text)
    return {"todo_id": int(m.group())} if m else {}


@tool(
    "add_todo",
    "新增一条待办事项。content 必填；due 可填截止时间描述（如 明天上午10点）。",
    keywords=["记一下", "提醒我", "别忘了", "添加待办", "新增待办"],
    mock_args_fn=_extract_add_todo,
    level="write",
)
def add_todo(content: str, due: str = "") -> str:
    """content: 待办内容；due: 截止时间描述（可选）"""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO todos (content, due) VALUES (?, ?)", (content, due)
            )
        return f"已新增待办 #{cur.lastrowid}：{content}" + (f"（{due}）" if due else "")
    finally:
        conn.close()


@tool(
    "list_todos",
    "列出当前所有未完成的待办事项",
    keywords=["待办", "待办列表", "有什么待办", "待办有哪些", "我的待办"],
)
def list_todos() -> str:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, content, due FROM todos WHERE status='open' ORDER BY id"
        ).fetchall()
        if not rows:
            return "当前没有未完成的待办事项。"
        return "待办列表：\n" + "\n".join(
            f"{r['id']}. {r['content']}" + (f"（{r['due']}）" if r["due"] else "")
            for r in rows
        )
    finally:
        conn.close()


@tool(
    "complete_todo",
    "把指定编号的待办事项标记为已完成",
    keywords=["完成", "搞定", "做完", "办完"],
    mock_args_fn=_extract_todo_id,
    level="write",
)
def complete_todo(todo_id: int) -> str:
    """todo_id: 待办编号（list_todos 返回的 id）"""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "UPDATE todos SET status='done' WHERE id=? AND status='open'", (todo_id,)
            )
        if cur.rowcount == 0:
            return f"未找到编号为 {todo_id} 的未完成待办。"
        return f"待办 #{todo_id} 已标记为完成。"
    finally:
        conn.close()


@tool(
    "delete_todo",
    "删除指定编号的待办事项（不可恢复，属于危险操作，执行前需人工确认）",
    keywords=["删除待办", "删掉待办", "删除", "删掉"],
    mock_args_fn=_extract_todo_id,
    level="danger",
)
def delete_todo(todo_id: int) -> str:
    """todo_id: 待办编号（list_todos 返回的 id）"""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                "DELETE FROM todos WHERE id=? AND status='open'", (todo_id,)
            )
        if cur.rowcount == 0:
            return f"未找到编号为 {todo_id} 的未完成待办。"
        return f"待办 #{todo_id} 已删除。"
    finally:
        conn.close()
