"""RAG 存储与检索模块：SQLite 存 chunk + 稀疏向量，余弦检索 top-k。

数据量小（几十 MB 文本 -> 数千块），全量余弦完全够快；
量级上去后可无缝换 FAISS（第四站埋过这个升级口）。
"""
from __future__ import annotations

import json
import os

from rag.vectorizer import TfidfVectorizer, cosine

from tools.data_store import get_connection

# 知识库索引路径（可配置）
_DEFAULT_SOURCE_DIR = r"D:\study\gaopu\VR"


def _ensure_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS docs_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            tokens TEXT NOT NULL
        )
        """
    )


def build_index(source_dir: str = _DEFAULT_SOURCE_DIR,
                max_chunks: int | None = None) -> dict:
    """完整索引流程：扫描 -> 切块 -> 向量化 -> 入库。返回统计信息。"""
    from rag.ingest import scan_and_extract
    from rag.chunker import split_into_chunks

    records = scan_and_extract(source_dir)
    chunk_records: list[dict] = []  # [{source, index, text}]
    for rec in records:
        chunks = split_into_chunks(rec["rel_path"], rec["text"])
        for i, c in enumerate(chunks):
            chunk_records.append(
                {"source": rec["rel_path"], "index": i, "text": c}
            )
    if max_chunks:
        chunk_records = chunk_records[:max_chunks]

    # 向量化：先 fit IDF，再向量化每一块
    texts = [c["text"] for c in chunk_records]
    vec = TfidfVectorizer().fit(texts)

    conn = get_connection()
    try:
        with conn:
            _ensure_table(conn)
            conn.execute("DELETE FROM docs_chunks")
            for c in chunk_records:
                tokens = vec.vectorize(c["text"])
                conn.execute(
                    "INSERT INTO docs_chunks (source, chunk_index, text, tokens)"
                    " VALUES (?, ?, ?, ?)",
                    (c["source"], c["index"], c["text"],
                     json.dumps(tokens, ensure_ascii=False)),
                )
    finally:
        conn.close()

    return {
        "files": len(records),
        "chunks": len(chunk_records),
        "source_dir": source_dir,
    }


def search(query: str, top_k: int = 3, min_score: float = 0.02) -> list[dict]:
    """检索：向量化 query -> 与所有块算余弦 -> top-k 带出处返回。"""
    conn = get_connection()
    try:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT id, source, chunk_index, text, tokens FROM docs_chunks"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return []

    texts = [r[3] for r in rows]
    vec = TfidfVectorizer().fit(texts)  # 用索引库的 IDF 口径
    qvec = vec.vectorize(query)
    scored: list[tuple[float, dict]] = []
    for r in rows:
        toks = json.loads(r[4])
        s = cosine(qvec, toks)
        if s >= min_score:
            scored.append((s, {
                "id": r[0], "source": r[1], "chunk_index": r[2],
                "text": r[3], "score": round(s, 4),
            }))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]
