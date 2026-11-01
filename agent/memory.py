"""会话记忆（滑动窗口 + 历史持久化）。

职责：按 OpenAI 消息格式维护多轮对话历史，
超长时自动裁掉最旧的一整轮（一轮 = 一个 user 消息及其后续关联消息），
保证发送给 LLM 的上下文可控。

被裁掉的旧消息不再丢弃：转发给 HistoryMemory 落库（SQLite），
后续可通过向量检索（retrieve_context）按相关性取回，实现"不失忆"。
"""
from __future__ import annotations

from agent.history_memory import HistoryMemory, estimate_tokens


class ConversationMemory:
    def __init__(self, max_turns: int = 10, history: HistoryMemory | None = None):
        self._messages: list[dict] = []
        self._user_count = 0
        self._max_turns = max_turns
        # 历史持久层：默认启用（惰性创建默认实例）
        self._history: HistoryMemory | None = history if history is not None else HistoryMemory()

    # -- 写入 --------------------------------------------------------------

    def append(self, message: dict) -> None:
        if message.get("role") == "user":
            self._user_count += 1
        self._messages.append(message)
        self._trim()

    def add_user(self, content: str) -> None:
        self.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.append({"role": "assistant", "content": content})

    def add_tool_call_message(self, tool_calls: list) -> None:
        """把模型发起的工具调用作为 assistant 消息写入历史。"""
        self.append({"role": "assistant", "content": None, "tool_calls": tool_calls})

    def add_tool_result(self, name: str, content: str, tool_call_id: str) -> None:
        self.append({"role": "tool", "content": content, "tool_call_id": tool_call_id, "name": name})

    # -- 读取 --------------------------------------------------------------

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
        self._user_count = 0

    def retrieve_context(
        self,
        query: str,
        top_k: int | None = None,
        token_budget: int | None = None,
    ) -> list[str]:
        """按相关性检索已落库的历史片段（供 Agent 注入上下文）。

        query 为空时返回最近 top_k 条兜底；结果带 token 预算上限。
        """
        if self._history is None:
            return []
        return self._history.retrieve(query, top_k=top_k, token_budget=token_budget)

    def ensure_token_budget(self, budget: int) -> int:
        """窗口总 token 超预算时，从最旧轮继续整轮裁剪（同样落库）。

        返回收缩后窗口的 token 估算值。至少保留 1 轮。
        """
        while self._user_count > 1:
            total = sum(estimate_tokens(m.get("content") or "") for m in self._messages)
            if total <= budget:
                return total
            self._trim_one()
        return sum(estimate_tokens(m.get("content") or "") for m in self._messages)

    # -- 裁剪 --------------------------------------------------------------

    def _trim(self) -> None:
        """超过 max_turns 时，从最旧轮开始整轮裁剪（落库）。"""
        while self._user_count > self._max_turns:
            self._trim_one()

    def _trim_one(self) -> None:
        """无条件裁掉最旧一整轮（user 及其同轮后续消息），落库到历史持久层。

        避免留下"没有 user 的孤立 assistant"污染上下文。
        """
        idx = None
        for i, m in enumerate(self._messages):
            if m.get("role") == "user":
                idx = i
                break
        if idx is None:
            return
        end = len(self._messages)
        for i in range(idx + 1, len(self._messages)):
            if self._messages[i].get("role") == "user":
                end = i
                break
        # 旧消息不丢弃：落库到历史持久层（content 非空的才存）
        removed = self._messages[idx:end]
        if self._history is not None:
            self._history.add_messages(removed)
        del self._messages[idx:end]
        self._user_count -= 1
