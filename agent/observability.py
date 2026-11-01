"""可观测性：结构化日志 + 运行统计。

每次 Agent.run 产出两类信息（写入 logs/agent.jsonl，JSON Lines 格式，
每行一个 JSON 对象，可直接用 jq / pandas 消费）：
- 分步事件：每步 LLM 调用 / 工具调用的明细（含耗时、token 估算、结果长度）
- summary 事件：一次对话的汇总统计（终止原因、调用次数、错误数、token、总耗时）

设计：
- JsonlLogger 每次 run 打开-写入-关闭，无句柄泄漏，长驻服务安全
- 日志路径默认 logs/，可用 Agent(log_dir=...) 覆盖或传 None 关闭
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class AgentStats:
    """一次 Agent.run 的汇总统计。"""

    steps: int = 0
    terminated_by: str = "answer"  # answer | limit | loop_detected | llm_error
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlLogger:
    """把事件字典追加写入 JSON Lines 文件（一次打开，写完关闭）。"""

    def __init__(self, log_dir: str = "logs", filename: str = "agent.jsonl"):
        self._path = os.path.join(log_dir, filename)
        os.makedirs(log_dir, exist_ok=True)

    @property
    def path(self) -> str:
        return self._path

    def log_many(self, events: list[dict[str, Any]]) -> None:
        with open(self._path, "a", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
