"""评测脚本：执行测试集，输出指标与逐条结果，生成报告。

用法：
    python eval/run_eval.py              # 评测前自动清空数据表（保证顺序用例可复现）
    python eval/run_eval.py --no-clean   # 不清空（在已有数据上追加评测）

指标口径：
- 工具调用准确率 tool_acc : 实际调用的工具名 == 期望工具（闲聊类期望不调用工具）
- 参数准确率     param_acc: 工具参数与期望一致（未声明期望参数时视为通过）
- 回答命中率     answer_acc: 最终回复包含期望关键词
- 综合通过率     overall  : 以上三者同时通过

报告输出：eval/output/report.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.factory import build_agent  # noqa: E402
from tools.data_store import get_connection  # noqa: E402

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_EVAL_DIR, "output")


def clean_store() -> None:
    """清空数据表并重置自增 id，保证顺序用例（todo id、记忆顺序）可复现。"""
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM long_term_memory")
        conn.execute("DELETE FROM todos")
        conn.execute("DELETE FROM notes")
        conn.execute("DELETE FROM sqlite_sequence")  # 重置自增序列
    conn.close()


def run_case(agent, case: dict) -> dict:
    result = agent.run(case["input"])
    calls = [t for t in result["trace"] if t["type"] == "tool_call"]
    tool_names = [t["name"] for t in calls]
    answer = result["answer"]

    expected_tool = case.get("expected_tool", "")
    ok_tool: bool
    ok_params = True

    if expected_tool:
        ok_tool = expected_tool in tool_names
        exp_args = case.get("expected_arguments") or {}
        matched = [t for t in calls if t["name"] == expected_tool]
        if exp_args:
            ok_params = bool(matched) and all(
                matched[0]["arguments"].get(k) == v for k, v in exp_args.items()
            )
    else:
        ok_tool = len(calls) == 0  # 闲聊类：期望不调用任何工具

    ok_answer = all(kw in answer for kw in case.get("answer_contains", []))
    return {
        "tools": tool_names,
        "ok_tool": ok_tool,
        "ok_params": ok_params,
        "ok_answer": ok_answer,
        "answer": answer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="个人AI助理Agent 评测")
    parser.add_argument("--no-clean", action="store_true", help="评测前不清空数据表")
    args = parser.parse_args()

    if not args.no_clean:
        clean_store()

    with open(os.path.join(_EVAL_DIR, "test_suite.json"), "r", encoding="utf-8") as f:
        cases = json.load(f)

    agent = build_agent()
    results: list[dict] = []
    for case in cases:
        r = run_case(agent, case)
        passed = r["ok_tool"] and r["ok_params"] and r["ok_answer"]
        results.append(
            {
                "id": case["id"],
                "input": case["input"],
                "expected_tool": case.get("expected_tool", ""),
                "tools": r["tools"],
                "ok_tool": r["ok_tool"],
                "ok_params": r["ok_params"],
                "ok_answer": r["ok_answer"],
                "passed": passed,
                "answer": r["answer"],
            }
        )

    n = len(results)
    tool_acc = sum(1 for r in results if r["ok_tool"]) / n
    param_acc = sum(1 for r in results if r["ok_params"]) / n
    answer_acc = sum(1 for r in results if r["ok_answer"]) / n
    overall = sum(1 for r in results if r["passed"]) / n

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    lines = [
        "# 评测报告（Mock 基线）",
        "",
        "## 总指标",
        "",
        f"- 用例总数：{n}",
        f"- 工具调用准确率：{tool_acc:.1%}",
        f"- 参数准确率：{param_acc:.1%}",
        f"- 回答命中率：{answer_acc:.1%}",
        f"- **综合通过率：{overall:.1%}**",
        "",
        "## 逐条结果",
        "",
        "| 用例 | 输入 | 期望工具 | 实际工具 | 工具✓ | 回答（截断） | 结果 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        tools_str = ",".join(r["tools"]) or "（无）"
        answer_short = r["answer"][:42].replace("|", "/").replace("\n", " ")
        status = "✅" if r["passed"] else "❌"
        lines.append(
            f"| {r['id']} | {r['input']} | {r['expected_tool'] or '（无）'} "
            f"| {tools_str} | {'✓' if r['ok_tool'] else '✗'} | {answer_short} | {status} |"
        )

    report_path = os.path.join(_OUTPUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 摘要同时写入 stdout（部分环境回显不稳定，报告文件为权威来源）
    print(f"用例数={n} 工具调用={tool_acc:.1%} 参数={param_acc:.1%} 回答={answer_acc:.1%} 综合={overall:.1%}")
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    main()
