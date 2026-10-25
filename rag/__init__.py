"""RAG 检索模块：摄入 -> 切块 -> TF-IDF 向量化 -> 余弦检索。

对外主要接口：
- build_index()  重建知识库索引（摄入 VR 项目资料）
- search(query)  检索相关片段（带来源出处）
"""
from rag.ingest import clean_query
from rag.retriever import build_index, search

__all__ = ["build_index", "search", "clean_query"]
