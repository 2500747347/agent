"""工具：天气查询。

Mock 版内置示例数据（离线可演示）；真实版可替换为高德/和风天气 API 调用。
"""
from __future__ import annotations

import re

from agent.tools import tool

_MOCK_WEATHER: dict[str, tuple] = {
    "芜湖": ("多云转晴", 12, 21, "东北风2级"),
    "合肥": ("晴", 10, 22, "东风2级"),
    "南京": ("小雨", 14, 19, "东南风3级"),
    "上海": ("阴", 16, 20, "东风3级"),
    "北京": ("晴", 5, 15, "西北风4级"),
    "深圳": ("多云", 20, 27, "南风3级"),
}
_DEFAULT = ("晴", 10, 20, "微风")


def _extract_city(text: str) -> dict:
    for city in _MOCK_WEATHER:
        if city in text:
            return {"city": city}
    cleaned = re.sub(r"天气|气温|怎么样|如何|多少|查一下|帮我|预报|今天", "", text)
    cleaned = re.sub(r"[?？!！。，,\s]+$", "", cleaned).strip()
    return {"city": cleaned[:8]} if cleaned else {}


@tool(
    "get_weather",
    "查询指定城市今日天气（含天气状况、气温、风力）",
    keywords=["天气", "气温", "下雨", "多少度", "温度"],
    mock_args_fn=_extract_city,
)
def get_weather(city: str) -> str:
    """city: 城市名，如 芜湖 / 合肥 / 南京"""
    city = city.strip()
    desc, t_low, t_high, wind = _MOCK_WEATHER.get(city, _DEFAULT)
    return f"{city}今日天气：{desc}，气温 {t_low}~{t_high}℃，{wind}。"
