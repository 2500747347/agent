"""工具：掷骰子（演示 mock_args_fn 参数提取 + Pydantic 参数护栏）。"""
from __future__ import annotations

import random
import re

from pydantic import BaseModel, Field

from agent.tools import tool


def _extract_faces(text: str) -> dict:
    """从输入中提取骰子面数，如“20面的骰子”→ {"faces": 20}"""
    m = re.search(r"(\d+)\s*面", text)
    if m:
        return {"faces": int(m.group(1))}
    return {}


class DiceParams(BaseModel):
    """Pydantic 护栏：骰子面数必须是 2~100 的整数。"""

    faces: int = Field(default=6, ge=2, le=100, description="骰子面数")


@tool(
    "roll_dice",
    "掷一个骰子，可指定面数，默认六面",
    keywords=["骰子", "掷骰", "roll"],
    mock_args_fn=_extract_faces,    # ← 加上这一行，Mock 才会提取
    params_model=DiceParams,        # ← Pydantic 护栏：脏参数执行前被拦截
)
def roll_dice(faces: int = 6) -> str:
    """faces: 骰子面数，默认 6"""
    return f"骰子点数：{random.randint(1, faces)}"
