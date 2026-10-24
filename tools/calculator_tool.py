"""工具：数学计算（安全求值，避免 eval 注入）。"""
from __future__ import annotations

import ast
import operator
import re

from agent.tools import tool

_BIN_OPS: dict[type, object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST):
    """仅允许常量、四则/幂/取模/取负等算术运算。"""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError("表达式包含不支持的操作")


def _extract_expr(text: str) -> dict:
    """从自然语言输入中提取数学表达式（仅 Mock 模式使用）。"""
    cleaned = re.sub(r"计算|等于|多少|请问|帮我|一下|结果", "", text)
    cleaned = re.sub(r"[?？]+$", "", cleaned).strip()
    expr = re.sub(r"[^\d+\-*/().%\s]", "", cleaned).strip()
    expr = re.sub(r"\s+", " ", expr)
    if not re.search(r"\d", expr):
        return {}
    return {"expression": expr}


@tool(
    "calculate",
    "计算数学表达式并返回结果，支持 + - * / ** // % 与括号",
    keywords=["计算", "等于", "多少", "乘以", "除以", "平方", "帮我算", "算一下"],
    mock_args_fn=_extract_expr,
)
def calculate(expression: str) -> str:
    """expression: 数学表达式，如 '128 * 37'"""
    tree = ast.parse(expression, mode="eval")
    result = _safe_eval(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)
