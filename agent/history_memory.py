"""会话历史持久层 + 向量检索注入（P0：零依赖 TF 版）。

职责：解决"滑动窗口截断即失忆"与"长对话上下文爆炸"两个问题。
- 被滑动窗口裁掉的旧消息不再丢弃，全部落库（SQLite）；
- 每轮对话开始时，用当前用户输入做 query，向量检索取回相关历史片段注入上下文；
- 向量器接口化：默认 TfVectorizer（词袋 TF + 余弦，零依赖），
  P2 可平滑替换为 EmbeddingVectorizer（本地/API 语义向量）而不动本模块逻辑。

token 预算（P0 简化）：以字符数粗估（中文约 1 字 ≈ 1 token），
P1 再接 tiktoken 精确计数。
"""
from __future__ import annotations

from typing import Any, Protocol

from tools.data_store import get_connection, init_db

from rag.vectorizer import cosine, tokenize

# 落库单条内容上限：超大工具结果（如 RAG 检索）截断再存，防止单条无限膨胀
MAX_CONTENT = 2000
# 检索时参与向量化的内容截断：只取头部，避免超长文本噪声
RETRIEVE_TRUNCATE = 300
# 单次检索最多扫过的历史条数（防库无限增长拖慢检索）
MAX_SCAN = 5000


class Vectorizer(Protocol):
    """向量器接口：P2 换 Embedding 时实现同一协议即可。"""

    def vectorize(self, text: str) -> Any: ...
    def similarity(self, a: Any, b: Any) -> float: ...


class TfVectorizer:
    """词袋 TF 向量器（零依赖，字面匹配，P0 默认实现）。"""

    def vectorize(self, text: str) -> dict[str, int]:
        tf: dict[str, int] = {}
        for tok in tokenize(text):
            tf[tok] = tf.get(tok, 0) + 1
        return tf

    def similarity(self, a: dict[str, int], b: dict[str, int]) -> float:
        return cosine(a, b)


def estimate_tokens(text: str) -> int:
    """估算 token 数（零依赖启发式）。

    - CJK 字符与全角符号（\\u3000-\\u9fff）：约 1 字符 ≈ 1 token；
    - ASCII（英文/数字/半角符号）：约 4 字符 ≈ 1 token。
    对中文对话足够精确；如需 tiktoken 级精度，替换本函数即可。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u3000" <= ch <= "\u9fff")
    ascii_count = max(0, len(text) - cjk)
    return max(1, cjk + max(1, ascii_count) // 4)


class HistoryMemory:
    """会话历史持久层：落库 + 按相关性检索。"""

    def __init__(
        self,
        session_id: str = "default",
        top_k: int = 5,
        token_budget: int = 1500,
        vectorizer: Vectorizer | None = None,
    ) -> None:
        init_db()
        self._session_id = session_id
        self._top_k = top_k
        self._token_budget = token_budget
        self._vectorizer = vectorizer or TfVectorizer()

    # -- 写入 --------------------------------------------------------------

    def add_messages(self, messages: list[dict]) -> None:
        """把一批消息落库（只存非空 content，超长截断）。"""
        rows = [
            (self._session_id, str(m.get("role", "")), str(m.get("content", ""))[:MAX_CONTENT])
            for m in messages
            if m.get("content")
        ]
        if not rows:
            return
        conn = get_connection()
        try:
            with conn:
                conn.executemany(
                    "INSERT INTO conversation_history (session_id, role, content) VALUES (?, ?, ?)",
                    rows,
                )
        finally:
            conn.close()

    # -- 读取 --------------------------------------------------------------

    def count(self) -> int:
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM conversation_history WHERE session_id = ?",
                (self._session_id,),
            ).fetchone()
            return int(row["n"]) if row else 0
        finally:
            conn.close()

    def retrieve(
        self,
        query: str = "",
        top_k: int | None = None,
        token_budget: int | None = None,
    ) -> list[str]:
        """检索相关历史片段。

        - query 为空：返回最近 top_k 条（兜底）；
        - 否则按余弦相似度打分降序，逐条累计到 token_budget 为止。
        """
        top_k = top_k or self._top_k
        token_budget = token_budget or self._token_budget

        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT content FROM conversation_history WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (self._session_id, MAX_SCAN),
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return []

        if not query.strip():
            return [r["content"] for r in rows[:top_k]]

        q_vec = self._vectorizer.vectorize(query)
        if not q_vec:
            return [r["content"] for r in rows[:top_k]]

        scored: list[tuple[float, str]] = []
        for r in rows:
            doc_vec = self._vectorizer.vectorize(r["content"][:RETRIEVE_TRUNCATE])
            if not doc_vec:
                continue
            score = self._vectorizer.similarity(q_vec, doc_vec)
            if score > 0:
                scored.append((score, r["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)

        picked: list[str] = []
        used = 0
        for _, content in scored:
            cost = estimate_tokens(content)
            if used + cost > token_budget:
                continue
            picked.append(content)
            used += cost
            if len(picked) >= top_k:
                break
        return picked
