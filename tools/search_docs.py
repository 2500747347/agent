"""工具：检索本地 VR 质检项目知识库（RAG 检索）。

通过 tools/__init__.py 的 register_all 自动注册，内核零改动。
真实模型：语义理解自己决定是否调用；
Mock 模式：命中 keywords（架构/文档/方案/质检…）即触发，参数由 _extract_query 提取。
"""
from __future__ import annotations

import re

from agent.tools import tool
from rag import clean_query, search


def _extract_query(text: str) -> dict:
    """Mock 参数提取：优先取引号内的内容，否则清洗整句。"""
    m = re.search(r"[\"'“”‘’](.+?)[\"'“”‘’]", text)
    if m:
        return {"query": clean_query(m.group(1))}
    return {"query": clean_query(text)}


@tool(
    "search_docs",
    "从本地VR质检项目知识库检索资料（技术方案、架构文档、算法说明、源代码），返回相关片段及来源文件",
    keywords=[
        # 项目专属内容词（与 search_knowledge 的"什么是/查一下"类概念词错开，
        # 避免 Mock 最长关键词匹配时撞车）
        "架构", "文档", "资料", "方案", "规格", "质检",
        "算法", "标定", "缺陷", "相机", "项目", "高普",
        "视觉检测", "检测算法", "技术方案",
    ],
    mock_args_fn=_extract_query,
)
def search_docs(query: str, top_k: int = 3) -> str:
    """query: 要检索的问题描述；top_k: 返回片段数量"""
    results = search(query, top_k=top_k)
    if not results:
        return "（知识库中未检索到相关内容）"
    lines: list[str] = []
    for item in results:
        text = item["text"].strip()
        if len(text) > 300:
            text = text[:300] + "…"
        lines.append(
            f"[来源：{item['source']} 第{item['chunk_index']}块，"
            f"相关度 {item['score']}]"
        )
        lines.append(text)
    return "\n\n".join(lines)
