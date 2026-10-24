"""工具包：自动装配所有工具模块。

约定：每个工具模块在顶层通过 @tool 装饰器创建 Tool 对象，
register_all 自动扫描并注册，新增工具无需修改任何装配代码。
"""
from __future__ import annotations

import importlib
import pkgutil

from agent.tools import Tool, ToolRegistry


def register_all(registry: ToolRegistry) -> ToolRegistry:
    for mod_info in pkgutil.iter_modules(__path__):
        module = importlib.import_module(f"{__name__}.{mod_info.name}")
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if isinstance(obj, Tool):
                registry.register(obj)
    return registry
