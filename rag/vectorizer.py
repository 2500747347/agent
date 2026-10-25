"""RAG 向量化模块：自研 TF-IDF 向量（零外部依赖，可平滑替换为 Embedding）。

与第四站长记忆的 bigram+TF 一脉相承，这里升级为文档级 TF-IDF：
- TF（词频）：词在本文档块中出现的次数
- IDF（逆文档频率）：log((N+1)/(df+1))+1，词越罕见越有区分力
  （"的/是"这类高频词 IDF 极低，检索时几乎不贡献分数——这就是
   为什么不用纯 TF：会把所有块都带出来，毫无区分度）

接口设计为可替换：只要实现 vectorize(text)->dict 的类，
就能换成 sentence-transformers / OpenAI Embedding 而不动检索逻辑。
"""
from __future__ import annotations

import math
import re

_EN_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{1,}")


def tokenize(text: str) -> list[str]:
    """分词：中文按相邻两字（bigram），英文按单词。

    中文单字无信息量，三字太稀疏，两字是"语义单元"的最短稳定形式
    （与长记忆系统一致）。
    """
    tokens: list[str] = []
    cn = re.findall(r"[\u4e00-\u9fff]+", text.lower())
    for seg in cn:
        tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
        if len(seg) == 1:
            tokens.append(seg)
    tokens.extend(_EN_WORD.findall(text.lower()))
    return tokens


class TfidfVectorizer:
    """自研 TF-IDF 向量器。"""

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        """根据全部文档块统计 IDF。"""
        n = len(texts)
        df: dict[str, int] = {}
        for t in texts:
            for tok in set(tokenize(t)):
                df[tok] = df.get(tok, 0) + 1
        self._idf = {
            tok: math.log((n + 1) / (freq + 1)) + 1.0
            for tok, freq in df.items()
        }
        return self

    def vectorize(self, text: str) -> dict[str, float]:
        """文本 -> 稀疏向量 {token: tf * idf}。"""
        tf: dict[str, float] = {}
        for tok in tokenize(text):
            tf[tok] = tf.get(tok, 0.0) + 1.0
        vec: dict[str, float] = {}
        for tok, freq in tf.items():
            w = self._idf.get(tok, 1.0)  # 未见过的词给基础权重
            vec[tok] = freq * w
        return vec


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """稀疏向量余弦相似度。"""
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a  # 遍历小的那个
    dot = 0.0
    for tok, va in a.items():
        vb = b.get(tok)
        if vb is not None:
            dot += va * vb
    if dot == 0.0:
        return 0.0
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
