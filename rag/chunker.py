"""RAG 切块模块：把长文本切成"语义完整、大小适中"的块。

切块是 RAG 的关键工程细节：
- 块太小 -> 每块信息量不足，检索召回碎片化
- 块太大 -> 语义混杂，且喂给 LLM 消耗 token 多
- 纯按固定字符切 -> 会把一句话/一段逻辑劈成两半

本实现：按段落（空行分隔）累积，接近目标大小即成块；
相邻块重叠 overlap 字符，避免跨块语义被切断。
"""
from __future__ import annotations

_DEFAULT_SIZE = 600   # 目标块大小（字符）
_DEFAULT_OVERLAP = 120  # 相邻块重叠（字符）


def chunk_text(text: str, chunk_size: int = _DEFAULT_SIZE,
               overlap: int = _DEFAULT_OVERLAP) -> list[str]:
    """把普通文档文本切成块：按段落累积 + 重叠窗口。"""
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in re_split_paragraphs(text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        # 单段超过上限：先强切，再继续累积
        while len(para) > chunk_size:
            if buf:
                chunks.append(buf)
                buf = buf[-overlap:]  # 把上一块尾部拼到下一块，保证连贯
            chunks.append(para[:chunk_size])
            para = para[chunk_size:]
        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= chunk_size + overlap:
            buf = buf + "\n" + para
        else:
            chunks.append(buf)
            buf = buf[-overlap:] + "\n" + para
    if buf:
        chunks.append(buf)
    return chunks


def chunk_code(text: str, lines_per_chunk: int = 60,
               overlap_lines: int = 8) -> list[str]:
    """代码按行切块：60 行一块，重叠 8 行，避免函数/类定义被劈开。"""
    if not text.strip():
        return []
    lines = text.splitlines()
    if len(lines) <= lines_per_chunk:
        return [text]
    chunks: list[str] = []
    i = 0
    while i < len(lines):
        end = min(i + lines_per_chunk, len(lines))
        chunks.append("\n".join(lines[i:end]))
        if end >= len(lines):
            break
        i = end - overlap_lines
    return chunks


def split_into_chunks(rel_path: str, text: str) -> list[str]:
    """按文件类型分发：代码按行切，文档按段落切。"""
    if rel_path.lower().endswith(".py"):
        return chunk_code(text)
    return chunk_text(text)


def re_split_paragraphs(text: str) -> list[str]:
    """按空行分隔段落（兼容 \r\n / \n / \r）。"""
    import re

    return re.split(r"\n\s*\n", text)
