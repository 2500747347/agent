"""从配置构建 Agent 实例。

配置来源：
1) config/config.yaml   —— 模板/默认值（入库）
2) config/secret.yaml   —— 真实 Key（不入库，优先覆盖）

providers:
- mock   : 规则式 MockLLM，离线可用（开发期/评测基线）
- openai : OpenAI 兼容接口（Codex/GPT/DeepSeek/通义…）
"""
from __future__ import annotations

import os

import yaml

from agent.core import Agent
from agent.llm import MockLLM, OpenAILLM
from agent.tools import ToolRegistry

_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))


def load_config() -> dict:
    config_path = os.path.join(_CONFIG_DIR, "config.yaml")
    secret_path = os.path.join(_CONFIG_DIR, "secret.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # secret.yaml 存在则合并覆盖（仅覆盖非空字段）
    if os.path.exists(secret_path):
        with open(secret_path, "r", encoding="utf-8") as f:
            secret = yaml.safe_load(f) or {}
        llm_cfg = cfg.setdefault("llm", {})
        llm_cfg.update(
            {k: v for k, v in (secret.get("llm") or {}).items() if v not in (None, "")}
        )
    return cfg


def build_registry() -> ToolRegistry:
    from tools import register_all  # 延迟导入，避免循环依赖

    registry = ToolRegistry()
    register_all(registry)
    return registry


def build_agent() -> Agent:
    cfg = load_config()
    registry = build_registry()
    llm_cfg = cfg.get("llm", {})

    if llm_cfg.get("provider", "mock") == "openai":
        llm = OpenAILLM(
            base_url=llm_cfg.get("base_url", ""),
            api_key=llm_cfg.get("api_key", ""),
            model=llm_cfg.get("model", "gpt-5-codex"),
            temperature=float(llm_cfg.get("temperature", 0.3)),
        )
    else:
        llm = MockLLM(registry)

    return Agent(
        llm=llm,
        registry=registry,
        max_iterations=int(llm_cfg.get("max_iterations", 5)),
    )
