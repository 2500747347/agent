"""长期记忆：跨会话存储用户偏好/事实，支持语义检索。

检索实现（无外部依赖，可讲原理）：
1. 中文按二元组（bigram）切分，英文/数字按单词切分 → 词频(TF)向量；
2. 余弦相似度打分，Top-K 召回。
语料为个人记忆（规模小），该方案足够；规模增大后可替换为 Embedding + FAISS。
"""
from __future__ import annotations

import re

import numpy as np

from tools.data_store import get_connection, init_db

_STOP = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它",
    "吗", "呢", "吧", "啊", "和", "与", "及", "这", "那", "有", "就", "也", "个",
}


def _tokenize(text: str) -> list[str]:
    """中文 bigram + 英文/数字单词 混合分词。"""
    text = text.lower()
    tokens: list[str] = []
    cn_chars = re.findall(r"[\u4e00-\u9fa5]", text)
    tokens.extend(cn_chars[i] + cn_chars[i + 1] for i in range(len(cn_chars) - 1))
    tokens.extend(re.findall(r"[a-z0-9]+", text))
    return [t for t in tokens if t not in _STOP and len(t) > 1]


def _tf_vector(tokens: list[str]) -> dict[str, int]:
    vec: dict[str, int] = {}
    for t in tokens:
        vec[t] = vec.get(t, 0) + 1
    return vec


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LongTermMemory:
    def __init__(self) -> None:
        init_db()

    def remember(self, content: str) -> int:
        """写入一条长期记忆，返回 id。"""
        conn = get_connection()
        try:
            with conn:
                cur = conn.execute(
                    "INSERT INTO long_term_memory (content) VALUES (?)", (content,)
                )
            return cur.lastrowid
        finally:
            conn.close()

    def recall(self, query: str = "", top_k: int = 3) -> list[dict]:
        """检索长期记忆。query 为空时返回最近 top_k 条。"""
        conn = get_connection()
        try:
            rows = conn.execute(
                "SELECT id, content FROM long_term_memory ORDER BY id DESC LIMIT 200"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return []

        if not query.strip():
            return [{"id": r["id"], "content": r["content"], "score": 0.0} for r in rows[:top_k]]

        q_vec = _tf_vector(_tokenize(query))
        if not q_vec:
            return [{"id": r["id"], "content": r["content"], "score": 0.0} for r in rows[:top_k]]

        scored: list[tuple[float, int, str]] = []
        for r in rows:
            doc_vec = _tf_vector(_tokenize(r["content"]))
            if not doc_vec:
                continue
            score = _cosine(q_vec, doc_vec)
            if score > 0:
                scored.append((score, r["id"], r["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": i, "content": c, "score": round(s, 3)} for s, i, c in scored[:top_k]
        ]
