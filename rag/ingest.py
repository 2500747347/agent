"""RAG 摄入模块：扫描知识源目录 -> 白名单过滤 -> 解析文本。

为什么需要白名单：真实项目的目录往往是"大杂烩"（缓存、模型权重、图片
混杂在一起）。直接全量索引会把浏览器缓存等噪音喂给向量检索，所以必须
显式声明"哪些目录/哪些扩展名才是有知识价值的文本"。
"""
from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET

# ---- 白名单规则 ----
_EXCLUDE_DIRS = {
    ".local", ".git", ".svn", "outputs_dl", "__pycache__", ".venv",
    "node_modules", "debug_crops", "dist", "build",
}
# 文本类源码（直接按文本读）
_TEXT_EXTS = {".py", ".md", ".txt", ".csv"}
# 文档类（需要专门解析）
_DOC_EXTS = {".docx", ".pdf"}
# 单个文件大小上限（超过视为二进制/权重文件，跳过）
_MAX_FILE_BYTES = 20 * 1024 * 1024

_DOCX_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _is_allowed(rel_path: str, ext: str, size: int) -> bool:
    """白名单判断：目录黑名单 + 扩展名白名单 + 大小上限。"""
    if size > _MAX_FILE_BYTES:
        return False
    parts = set(p.lower() for p in rel_path.split(os.sep))
    if parts & _EXCLUDE_DIRS:
        return False
    return ext.lower() in (_TEXT_EXTS | _DOC_EXTS)


def extract_docx(path: str) -> str:
    """自研 docx 文本提取：docx 本质是 zip，正文在 word/document.xml。

    只取 <w:t>（真正文字）标签内容，按 <w:p>（段落）换行。
    零依赖，且能讲清楚"Office 文件内部是什么"。
    """
    with zipfile.ZipFile(path) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for para in root.iter(f"{_DOCX_NS}p"):
        texts = [t.text or "" for t in para.iter(f"{_DOCX_NS}t")]
        if texts:
            paragraphs.append("".join(texts))
    return "\n".join(paragraphs)


def extract_pdf(path: str) -> str:
    """PDF 文本提取：使用轻量库 pypdf（唯一新增依赖）。"""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(pages)


def extract_text_file(path: str) -> str:
    """文本类文件：优先 utf-8，失败回退 gbk（Windows 老文档常见）。"""
    for enc in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # 双编码都失败：忽略坏字节，尽量取内容
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(path: str) -> str:
    """按扩展名分发解析。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return extract_docx(path)
    if ext == ".pdf":
        return extract_pdf(path)
    return extract_text_file(path)


def scan_and_extract(source_dir: str) -> list[dict]:
    """扫描整个知识源目录，返回 [{path, rel_path, text}]。

    path 用于回答时标注出处；rel_path 用于白名单判断。
    """
    records: list[dict] = []
    for root, dirs, files in os.walk(source_dir):
        # 就地修剪黑名单目录（os.walk 需要改 dirs 才会生效）
        dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
        for fn in files:
            full = os.path.join(root, fn)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            ext = os.path.splitext(fn)[1]
            rel = os.path.relpath(full, source_dir)
            if not _is_allowed(rel, ext, size):
                continue
            try:
                text = extract_text(full)
            except Exception:
                continue  # 单文件解析失败不阻断整体摄入
            if not text.strip():
                continue
            records.append({"path": full, "rel_path": rel, "text": text})
    return records


def clean_query(raw: str) -> str:
    """清洗口语化查询：去掉请求/疑问语气词，保留实质内容。"""
    raw = re.sub(r"[？?。！!，,\s]+", " ", raw)
    raw = re.sub(
        r"^(帮我|请|麻烦你|帮我看看|查一下|检索|搜索|查询|找一下|找找|看看|里面|文档里|资料里)"
        r"(关于|一下|下|找|查|看|有哪些|是什么|的内容|的内容是什么)?",
        "",
        raw,
    )
    return raw.strip()
