"""会话历史持久层 + 向量检索注入（P0 零依赖 TF 版 / P2 语义 Embedding 版）。

职责：解决"滑动窗口截断即失忆"与"长对话上下文爆炸"两个问题。
- 被滑动窗口裁掉的旧消息不再丢弃，全部落库（SQLite）；
- 每轮对话开始时，用当前用户输入做 query，向量检索取回相关历史片段注入上下文；
- 向量器接口化：默认 TfVectorizer（词袋 TF + 余弦，零依赖）；
  P2 可注入 EmbeddingVectorizer（本地 bge-small-zh，语义检索）而不动本模块流程。

token 预算（P1）：estimate_tokens 启发式估算（中文≈1字/token、英文≈4字符/token）。
"""
from __future__ import annotations

import os
from typing import Any, Protocol

from tools.data_store import get_connection, init_db

from rag.vectorizer import cosine, tokenize

# 本地语义 Embedding 模型路径（P2：用户下载到项目 models/ 目录）
_EMBEDDING_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
_DEFAULT_EMBEDDING_PATH = os.path.join(_EMBEDDING_DIR, "bge-small-zh-v1.5")
# BGE 官方建议：检索 query 加指令前缀，可显著提升召回
_BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 落库单条内容上限：超大工具结果（如 RAG 检索）截断再存，防止单条无限膨胀
MAX_CONTENT = 2000
# 检索时参与向量化的内容截断：只取头部，避免超长文本噪声
RETRIEVE_TRUNCATE = 300
# 单次检索最多扫过的历史条数（防库无限增长拖慢检索）
MAX_SCAN = 5000


class Vectorizer(Protocol):
    """向量器接口：P2 换 Embedding 时实现同一协议即可。

    is_query=True 时表示该文本是检索 query（供语义模型加指令前缀）。
    """

    def vectorize(self, text: str, is_query: bool = False) -> Any: ...
    def similarity(self, a: Any, b: Any) -> float: ...


class TfVectorizer:
    """词袋 TF 向量器（零依赖，字面匹配，默认实现）。"""

    def vectorize(self, text: str, is_query: bool = False) -> dict[str, int]:
        tf: dict[str, int] = {}
        for tok in tokenize(text):
            tf[tok] = tf.get(tok, 0) + 1
        return tf

    def similarity(self, a: dict[str, int], b: dict[str, int]) -> float:
        return cosine(a, b)


class EmbeddingVectorizer:
    """语义向量器（P2）：sentence-transformers 加载本地模型，稠密向量 + 余弦。

    依赖 models/bge-small-zh-v1.5（约 95MB）与 sentence-transformers；
    模型缺失或库未安装时由调用方回退 TfVectorizer。
    """

    def __init__(self, model_path: str = _DEFAULT_EMBEDDING_PATH, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_path, device=device)

    def vectorize(self, text: str, is_query: bool = False) -> list[float]:
        if is_query:
            text = _BGE_QUERY_PREFIX + text
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def similarity(self, a: list[float], b: list[float]) -> float:
        # 已归一化向量的点积 = 余弦相似度
        return float(sum(x * y for x, y in zip(a, b)))


def try_build_embedding_vectorizer() -> EmbeddingVectorizer | None:
    """尝试构建语义向量器；模型/依赖不可用时返回 None（调用方回退 TF）。"""
    if not os.path.isdir(_DEFAULT_EMBEDDING_PATH):
        return None
    try:
        return EmbeddingVectorizer()
    except Exception:
        return None


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

        q_vec = self._vectorizer.vectorize(query, is_query=True)
        if not q_vec:
            return [r["content"] for r in rows[:top_k]]

        scored: list[tuple[float, str]] = []
        for r in rows:
            doc_vec = self._vectorizer.vectorize(r["content"][:RETRIEVE_TRUNCATE], is_query=False)
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
