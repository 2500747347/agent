"""工具：领域知识检索。

Mock 版内置小型知识库（离线可演示）；真实版只需把 _KB 替换为
搜索引擎 API / 向量库检索结果即可，工具接口不变。
"""
from __future__ import annotations

import re

from agent.tools import tool

_KB = [
    {"title": "Python 3.13", "content": "Python 3.13（2024年10月发布）：新增交互式解释器增强、实验性 JIT 编译器、类型参数默认值、f-string 改进等特性。"},
    {"title": "Git 基础", "content": "Git 是分布式版本控制系统（2005年创建）。核心命令：init / add / commit / status / log / branch / merge / push / pull。"},
    {"title": "AI Agent", "content": "AI Agent 是能自主感知、规划、调用工具并完成任务闭环的智能体系统，核心组件：大模型（LLM）、工具调用（Function Calling）、记忆（Memory）、规划（Planning）。"},
    {"title": "RAG", "content": "RAG（检索增强生成）在生成前先从知识库检索相关资料注入大模型上下文，降低幻觉、提升事实性。流程：文档切分→向量化→检索→拼接→生成。"},
    {"title": "Function Calling", "content": "Function Calling 让大模型以结构化 JSON 输出工具调用意图（工具名+参数），由应用层执行真实函数并把结果回填，形成工具使用闭环。"},
    {"title": "ReAct", "content": "ReAct 是推理+行动交替的 Agent 范式：Thought（思考）→ Action（调用工具）→ Observation（观察结果）循环，直到得出最终答案。"},
    {"title": "Transformer", "content": "Transformer 是 2017 年提出的注意力架构，由 Encoder-Decoder 组成，是大模型（GPT/LLaMA）的基础架构，核心是自注意力（Self-Attention）。"},
    {"title": "向量检索", "content": "向量检索把文本映射为高维向量，用余弦相似度等度量找最相似文本。流程：Embedding 模型 → 索引（FAISS/ChromaDB）→ Top-K 召回。"},
]


def _extract_kw(text: str) -> dict:
    cleaned = re.sub(r"查一下|什么是|介绍一下|知识库|原理|讲讲|搜索|帮我", "", text)
    cleaned = re.sub(r"[?？!！。，,\s]+$", "", cleaned).strip()
    return {"keyword": cleaned} if cleaned else {}


@tool(
    "search_knowledge",
    "检索内置领域知识库（Python/Git/Agent/RAG/ReAct/Transformer 等主题），返回相关知识",
    keywords=["查一下", "什么是", "介绍一下", "知识库", "讲讲", "原理"],
    mock_args_fn=_extract_kw,
)
def search_knowledge(keyword: str) -> str:
    """keyword: 主题关键词，如 python / git / agent / rag / react / transformer"""
    kw = keyword.lower()
    hits = [
        doc for doc in _KB
        if kw in doc["title"].lower() or kw in doc["content"].lower()
    ]
    if not hits:
        return f"知识库中没有找到与“{keyword}”相关的内容。可尝试：python / git / agent / rag / react / transformer。"
    return "\n".join(f"【{doc['title']}】{doc['content']}" for doc in hits[:3])
