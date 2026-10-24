"""工具：当前日期时间。"""
from __future__ import annotations

from datetime import datetime

from agent.tools import tool

_WEEKDAYS_CN = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]


@tool(
    "get_current_time",
    "获取当前日期、星期与时间，如 2026年10月24日 星期六 14:30",
    keywords=["时间", "日期", "几号", "星期", "几点", "今天几号", "今天星期"],
)
def get_current_time() -> str:
    now = datetime.now()
    weekday = _WEEKDAYS_CN[now.weekday()]
    return now.strftime(f"%Y年%m月%d日 {weekday} %H:%M")
