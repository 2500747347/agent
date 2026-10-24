"""会话记忆（滑动窗口）。

职责：按 OpenAI 消息格式维护多轮对话历史，
超长时自动裁掉最旧的一整轮（一轮 = 一个 user 消息及其后续关联消息），
保证发送给 LLM 的上下文可控。
"""
from __future__ import annotations


class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self._messages: list[dict] = []
        self._user_count = 0
        self._max_turns = max_turns

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

    # -- 裁剪 --------------------------------------------------------------

    def _trim(self) -> None:
        while self._user_count > self._max_turns:
            # 从头找到第一个 user 消息的位置，删掉它及之前的全部消息
            idx = None
            for i, m in enumerate(self._messages):
                if m.get("role") == "user":
                    idx = i
                    break
            if idx is None:
                break
            del self._messages[: idx + 1]
            self._user_count -= 1
