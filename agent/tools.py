"""工具注册机制。

设计目标：新增工具 = 写一个普通函数 + 加一个 @tool 装饰器，
内核与注册表零改动。装饰器自动完成：
- 函数 docstring 首行 → 工具描述
- 类型注解 → 参数 schema（OpenAI 兼容格式）
- keywords（可选）→ Mock 模式的触发词（真实模型忽略）
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable

# Python 类型 → JSON Schema 类型
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _type_name(t: type) -> str:
    try:
        return t.__name__
    except AttributeError:  # 如 typing.List 等
        return str(t)


def _infer_schema(fn: Callable) -> dict:
    """从函数签名与类型注解推断 OpenAI 兼容的 parameters schema。"""
    sig = inspect.signature(fn)
    hints = getattr(fn, "__annotations__", {})
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        ann = hints.get(name, str)
        ptype = _TYPE_MAP.get(ann, "string")
        desc = f"参数 {name}（{_type_name(ann)}）"
        # 可选参数（带默认值）不加入 required
        if param.default is inspect.Parameter.empty:
            required.append(name)
        properties[name] = {"type": ptype, "description": desc}
    return {"type": "object", "properties": properties, "required": required}


class Tool:
    """一个可被 Agent 调用的工具。

    mock_args / mock_args_fn 仅 Mock 模式使用（真实模型忽略）：
    - mock_args     : 固定的参数字典
    - mock_args_fn  : callable，接收用户输入文本 → 参数字典（用于从输入中提取参数）

    params_model（可选，Pydantic 护栏）：参数校验模型（Pydantic BaseModel 子类）。
    配置后，run() 执行前先校验/强转参数，脏参数（类型错、缺必填、超范围）
    在进入真实工具函数前被拦截，返回错误文本回填给 LLM 自行纠错。
    """

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable,
        keywords: list[str] | None = None,
        mock_args: dict | None = None,
        mock_args_fn: Callable[[str], dict] | None = None,
        params_model: type | None = None,
    ):
        self.name = name
        self.description = description
        self.fn = fn
        self.keywords = keywords or []
        self.mock_args = mock_args
        self.mock_args_fn = mock_args_fn
        self.params_model = params_model
        self.schema = _infer_schema(fn)

    def run(self, **kwargs: Any) -> str:
        """执行工具，任何异常都转成文本结果（回填给 LLM 自行纠错）。

        有 params_model 时先校验：校验失败直接返回错误文本，不执行工具。
        """
        if self.params_model is not None:
            try:
                validated = self.params_model(**kwargs)
                kwargs = validated.model_dump()
            except Exception as exc:  # noqa: BLE001 —— 护栏拒绝脏参数，不向上抛
                return f"[参数校验失败] {exc}"
        try:
            result = self.fn(**kwargs)
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 —— 工具错误必须兜底为文本
            return f"[工具执行错误] {type(exc).__name__}: {exc}"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


class ToolRegistry:
    """工具注册表：登记、查找、批量导出 schema。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具名重复：{tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.all]

    def __len__(self) -> int:
        return len(self._tools)


def tool(
    name: str,
    description: str,
    keywords: list[str] | None = None,
    mock_args: dict | None = None,
    mock_args_fn: Callable[[str], dict] | None = None,
    params_model: type | None = None,
) -> Callable[[Callable], Tool]:
    """装饰器：把普通函数声明为 Agent 工具。

    用法：
        @tool("get_time", "获取当前日期时间", keywords=["时间", "日期", "几号", "星期"])
        def get_time() -> str:
            return ...

    可选（Mock 模式参数补给）：
        mock_args    : 固定的参数字典
        mock_args_fn : 从用户输入文本提取参数字典的函数

    可选（Pydantic 护栏）：
        params_model : Pydantic BaseModel 子类，run() 执行前校验/强转参数
    """

    def deco(fn: Callable) -> Tool:
        return Tool(
            name=name,
            description=description,
            fn=fn,
            keywords=keywords or [],
            mock_args=mock_args,
            mock_args_fn=mock_args_fn,
            params_model=params_model,
        )

    return deco
