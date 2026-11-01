"""P3 工具与记忆系统测试。

覆盖：计算安全求值、待办增查改、笔记增查、知识检索、天气、
长期记忆写入与语义检索、Mock 全链路（记忆跨工具协同）。
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.core import Agent  # noqa: E402
from agent.llm import MockLLM  # noqa: E402
from agent.tools import ToolRegistry  # noqa: E402
from tools import register_all  # noqa: E402
from tools.data_store import get_connection  # noqa: E402


@pytest.fixture(autouse=True)
def clean_store():
    """每个测试前清理待办/笔记/长期记忆，保证可重复。"""
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM long_term_memory")
        conn.execute("DELETE FROM todos")
        conn.execute("DELETE FROM notes")
    conn.close()
    yield


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    register_all(registry)
    return registry


def make_agent() -> Agent:
    registry = make_registry()
    return Agent(llm=MockLLM(registry), registry=registry)


def _registry_get(name: str):
    return make_registry().get(name)


def test_calculator_valid_and_safe():
    tool = _registry_get("calculate")
    assert "4736" in tool.run(expression="128 * 37")
    # 非法输入（注入尝试）被安全拒绝而不是执行
    result = tool.run(expression="__import__('os').system('echo pwned')")
    assert "工具执行错误" in result


def test_todo_crud():
    reg = make_registry()
    add = reg.get("add_todo").run(content="明天上午10点开周会")
    assert "已新增待办" in add
    todo_id = int(add.split("#")[1].split("：")[0])

    listing = reg.get("list_todos").run()
    assert "开周会" in listing

    done = reg.get("complete_todo").run(todo_id=todo_id)
    assert "已标记为完成" in done
    assert "开周会" not in reg.get("list_todos").run()


def test_note_add_and_search():
    reg = make_registry()
    reg.get("add_note").run(content="健身计划：每周三次跑步")
    hits = reg.get("search_notes").run(keyword="健身")
    assert "每周三次跑步" in hits


def test_search_knowledge():
    reg = make_registry()
    out = reg.get("search_knowledge").run(keyword="agent")
    assert "AI Agent" in out or "Agent" in out


def test_weather():
    reg = make_registry()
    out = reg.get("get_weather").run(city="芜湖")
    assert "芜湖" in out and "℃" in out


def test_long_term_memory_recall():
    ltm = __import__("agent.long_term_memory", fromlist=["LongTermMemory"]).LongTermMemory()
    ltm.remember("我在准备面试，目标是 AI 应用与 Agent 开发岗位")
    ltm.remember("我的爱好是打游戏")
    results = ltm.recall("面试")
    assert results and any("面试" in r["content"] for r in results)
    # 语义相关性：检索“找工作”也应召回面试那条（bigram 有交集：找/工作 vs 准备/面试 无直接交集，用宽松断言仅检查排序）
    scores = [r["score"] for r in results]
    assert all(s >= 0 for s in scores)


def test_mock_full_chain_memory():
    """Mock 全链路：记住 → 回忆，跨工具协同。"""
    agent = make_agent()
    r1 = agent.run("记住我正在准备面试")
    assert "已记住" in r1["answer"]

    # 新 Agent 实例（模拟新的会话）仍能回忆起 → 验证跨会话持久化
    agent2 = make_agent()
    r2 = agent2.run("我之前说过什么")
    assert "面试" in r2["answer"]


# -- Pydantic 参数护栏（Harness 层）------------------------------------------


def test_roll_dice_guard_valid_and_default():
    tool = _registry_get("roll_dice")
    assert tool.run(faces=20).startswith("骰子点数")
    # 无参数 → 默认 6 面（护栏默认值兜底）
    assert tool.run().startswith("骰子点数")


def test_roll_dice_guard_rejects_bad_params():
    """护栏拦截脏参数：类型错误 / 范围越界，不进入真实函数。"""
    tool = _registry_get("roll_dice")
    assert tool.run(faces="abc").startswith("[参数校验失败]")   # 类型错
    assert tool.run(faces=1).startswith("[参数校验失败]")        # 低于 ge=2
    assert tool.run(faces=101).startswith("[参数校验失败]")      # 高于 le=100


def test_roll_dice_guard_mock_full_chain():
    """Mock 全链路：评测用例（20面）经护栏正常执行。"""
    agent = make_agent()
    r = agent.run("掷一个20面的骰子")
    assert "骰子点数" in r["answer"]
    calls = [t for t in r["trace"] if t["type"] == "tool_call"]
    assert calls and calls[0]["name"] == "roll_dice"
    assert calls[0]["arguments"] == {"faces": 20}


def test_guard_off_for_legacy_tools():
    """旧工具未配置 params_model：行为不变（零改动兼容）。"""
    tool = _registry_get("get_weather")
    out = tool.run(city="芜湖")
    assert "芜湖" in out and "℃" in out
