"""Agent 主循环：ReAct + Function Calling。

处理流程（详见 docs/01-需求与架构.md 4.1）：
    用户输入
      → LLM(会话历史 + 工具schema)
      → 有 tool_calls?  → 执行工具 → 结果回填 → 循环（最多 max_iterations 轮）
      → 无 tool_calls   → 输出最终回复

上下文预算（P1）：组装时估算总 token，超预算按可牺牲性依次收缩——
先砍检索注入片段，再收缩窗口旧轮次（被裁轮次照常落库，不失忆）。

循环保险丝（Loop 终止机制补全）：
- 重复调用检测：连续 N 次相同 (工具, 参数) → 判定死循环，提前终止
- 单轮工具超时：工具执行超过上限 → 中断等待并回填提示（默认关闭）
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from agent.history_memory import estimate_tokens
from agent.llm import (
    ContextLengthExceeded,
    LLMError,
    LLMResponse,  # noqa: F401  (类型标注)
    LLMRetryableError,
)
from agent.memory import ConversationMemory
from agent.observability import AgentStats, JsonlLogger
from agent.tools import ToolRegistry

# 上下文总预算（token，启发式估算）：system + 检索 + 窗口 + 当前输入
_DEFAULT_MAX_CONTEXT_TOKENS = 8000
# 检索注入片段预算
_DEFAULT_RETRIEVE_BUDGET = 1500
# 单条工具结果最大字符数（超长截断，防止单轮撑爆上下文）
_DEFAULT_TOOL_RESULT_MAX_CHARS = 4000
# 重复调用检测：同一 (工具, 参数) 连续出现此次数 → 判定死循环
_DEFAULT_REPETITION_LIMIT = 3


def _ctx_tokens(msgs: list[dict]) -> int:
    """一组消息的 token 估算。"""
    return sum(estimate_tokens(m.get("content") or "") for m in msgs)


class Agent:
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        memory: ConversationMemory | None = None,
        max_iterations: int = 5,
        system_prompt: str | None = None,
        max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS,
        retrieve_budget: int = _DEFAULT_RETRIEVE_BUDGET,
        tool_result_max_chars: int = _DEFAULT_TOOL_RESULT_MAX_CHARS,
        human_confirm: Callable[[str, dict], bool] | None = None,
        repetition_limit: int = _DEFAULT_REPETITION_LIMIT,
        tool_timeout: float = 0,
        log_dir: str = "logs",
    ):
        self._llm = llm
        self._registry = registry
        self._memory = memory or ConversationMemory()
        self._max_iterations = max_iterations
        self._system_prompt = system_prompt
        self._max_context_tokens = max_context_tokens
        self._retrieve_budget = retrieve_budget
        self._tool_result_max_chars = tool_result_max_chars
        # 人工确认器：danger 工具执行前的闸门（工具名, 参数）-> 是否放行。
        # 未配置时危险操作默认拒绝（fail-safe）。
        self._human_confirm = human_confirm
        # 循环保险丝：连续相同 (工具, 参数) 达此数 → 判定死循环提前终止（0 关闭）
        self._repetition_limit = repetition_limit
        # 单轮工具超时（秒）：超过则中断等待并回填提示；0 表示关闭（默认）
        self._tool_timeout = tool_timeout
        # 可观测性：结构化日志目录（传 None 关闭日志）
        self._log_dir = log_dir

    def _run_tool(self, tool: Any, **kwargs: Any) -> str:
        """执行工具；启用 tool_timeout 时包一层超时，超时不阻塞主循环。

        注意：shutdown(wait=False)——超时后不等待慢工具线程，避免把超时变成摆设。
        """
        if self._tool_timeout <= 0:
            return tool.run(**kwargs)
        ex = ThreadPoolExecutor(max_workers=1)
        future = ex.submit(tool.run, **kwargs)
        try:
            return future.result(timeout=self._tool_timeout)
        except TimeoutError:
            return f"[工具超时] 工具执行超过 {self._tool_timeout} 秒，已中断等待。"
        finally:
            ex.shutdown(wait=False)

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    def run(self, user_input: str) -> dict[str, Any]:
        """执行一轮对话。

        返回结构：
        {
            "answer": str,        # 最终回复
            "trace": [            # 过程轨迹（供界面/评测使用）
                {"type": "tool_call", "step": n, "name": ..., "arguments": ..., "result": ...},
                {"type": "answer", "step": n, "content": ...},
                ...
            ]
        }
        """
        # ---- 上下文组装（P1：总 token 预算控制） ----
        # 1) 不可牺牲部分：System + 当前输入
        base: list[dict] = []
        if self._system_prompt:
            base.append({"role": "system", "content": self._system_prompt})
        user_message = {"role": "user", "content": user_input}
        base_tokens = _ctx_tokens(base) + estimate_tokens(user_input)

        # 2) 检索注入（可牺牲）：独立 system 消息，不污染 user 输入；
        #    Mock 模式只匹配 user/tool 消息，天然忽略该注入，评测行为不变
        retrieved = self._memory.retrieve_context(
            user_input, token_budget=self._retrieve_budget
        )
        injected: list[dict] = []
        if retrieved:
            injected.append(
                {
                    "role": "system",
                    "content": "【对话历史中的相关信息，供参考】\n"
                    + "\n".join(f"- {c}" for c in retrieved),
                }
            )

        # 3) 窗口历史 + 当前输入
        messages = base + injected + self._memory.messages + [user_message]

        # 4) 总预算收缩：超预算先牺牲检索片段，再收缩窗口旧轮次（被裁轮次照常落库）
        if _ctx_tokens(messages) > self._max_context_tokens:
            messages = base + self._memory.messages + [user_message]
        if _ctx_tokens(messages) > self._max_context_tokens:
            self._memory.ensure_token_budget(self._max_context_tokens - base_tokens)
            messages = base + self._memory.messages + [user_message]

        trace: list[dict] = []

        # ---- 可观测性：计时 / 计数 / 事件收集 ----
        t0 = time.perf_counter()
        llm_calls = 0
        input_tokens_total = 0
        context_tokens_last = _ctx_tokens(messages)
        tool_calls = 0
        tool_errors = 0
        events: list[dict[str, Any]] = []

        # 重复调用检测状态：连续相同 (工具, 参数) 计数
        last_call_key: str | None = None
        repeat_count = 0
        loop_detected: str | None = None
        # 上下文超窗收缩标志：同一轮对话内只收缩一次
        ctx_shrunk = False

        def _finish(
            answer: str, step: int, terminated_by: str, steps: int | None = None
        ) -> dict[str, Any]:
            """统一收口：组装 stats、写结构化日志、返回结果。"""
            stats = AgentStats(
                steps=steps if steps is not None else step + 1,
                terminated_by=terminated_by,
                llm_calls=llm_calls,
                tool_calls=tool_calls,
                tool_errors=tool_errors,
                input_tokens=input_tokens_total,
                output_tokens=estimate_tokens(answer),
                context_tokens=context_tokens_last,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            if self._log_dir:
                summary = {"event": "summary", "session": "default", **stats.to_dict()}
                JsonlLogger(self._log_dir).log_many([*events, summary])
            return {"answer": answer, "trace": trace, "stats": stats.to_dict()}

        def llm_fail(msg: str, step: int) -> dict[str, Any]:
            """LLM 调用失败统一兜底：记录 trace、写入记忆、返回可读文本（不崩溃）。"""
            trace.append({"type": "limit", "step": step, "content": msg})
            self._memory.add_user(user_input)
            self._memory.add_assistant(msg)
            return _finish(msg, step, "llm_error")

        for step in range(self._max_iterations):
            llm_calls += 1
            input_tokens_total += _ctx_tokens(messages)
            context_tokens_last = _ctx_tokens(messages)
            events.append(
                {
                    "event": "llm_call",
                    "session": "default",
                    "step": step,
                    "input_tokens": _ctx_tokens(messages),
                }
            )
            try:
                resp = self._llm.chat(messages, tools=self._registry.schemas())
            except ContextLengthExceeded:
                # 上下文超窗（模型侧拒绝，重试同样失败）：收缩上下文后原轮重试一次
                if not ctx_shrunk:
                    ctx_shrunk = True
                    self._memory.ensure_token_budget(self._max_context_tokens // 2)
                    messages = base + self._memory.messages + [user_message]
                    continue
                return llm_fail(
                    "（模型上下文超限：已收缩对话历史重试仍失败，请简化问题或开启新会话。）",
                    step,
                )
            except LLMRetryableError as exc:
                return llm_fail(
                    f"（模型服务暂时不可用，自动重试后仍失败，请稍后再试。{exc}）", step
                )
            except LLMError as exc:
                return llm_fail(f"（模型调用失败：{exc}）", step)
            except Exception as exc:  # noqa: BLE001 —— LLM 未知异常兜底
                return llm_fail(f"（模型调用出现未知异常：{exc}）", step)

            # 情况A：模型不再调用工具 → 产出最终回复
            if not resp.has_tool_calls:
                answer = resp.content or "（模型未返回内容）"
                trace.append({"type": "answer", "step": step, "content": answer})
                self._memory.add_user(user_input)
                self._memory.add_assistant(answer)
                return _finish(answer, step, "answer")

            # 情况B：模型发起工具调用 → 记录并逐个执行
            raw_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in resp.tool_calls
            ]
            messages.append({"role": "assistant", "content": None, "tool_calls": raw_calls})

            for tc in resp.tool_calls:
                # 保险丝①：重复调用检测——同一 (工具, 参数) 连续出现达上限 → 死循环
                call_key = (
                    tc.name,
                    json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False),
                )
                if call_key == last_call_key:
                    repeat_count += 1
                else:
                    last_call_key = call_key
                    repeat_count = 1
                if (
                    self._repetition_limit
                    and repeat_count >= self._repetition_limit
                ):
                    result = (
                        f"[循环终止] 检测到重复调用 {tc.name}（参数 {tc.arguments}）"
                        f"连续 {repeat_count} 次，判定陷入循环，已提前终止。"
                    )
                    loop_detected = result
                    events.append(
                        {
                            "event": "loop_detected",
                            "session": "default",
                            "step": step,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
                    messages.append(
                        {"role": "tool", "content": result, "tool_call_id": tc.id, "name": tc.name}
                    )
                    trace.append(
                        {
                            "type": "tool_call",
                            "step": step,
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": result,
                        }
                    )
                    break

                tool_calls += 1
                tool = self._registry.get(tc.name)
                if tool is None:
                    result = f"[错误] 未知工具：{tc.name}"
                elif tool.level == "danger" and self._human_confirm is None:
                    # 安全默认（fail-safe）：未配置人工确认 → 拒绝危险操作
                    result = (
                        f"[需人工确认] 工具 {tc.name} 属于危险操作（{tool.description}），"
                        "当前未配置人工确认，已拒绝执行。"
                    )
                elif tool.level == "danger":
                    # 有人工确认器 → 执行前询问用户
                    if self._human_confirm(tc.name, tc.arguments):
                        result = self._run_tool(tool, **tc.arguments)
                    else:
                        result = f"[已取消] 用户拒绝了 {tc.name} 的操作（参数：{tc.arguments}）。"
                else:
                    result = self._run_tool(tool, **tc.arguments)
                if len(result) > self._tool_result_max_chars:
                    result = result[: self._tool_result_max_chars] + "…[已截断，结果过长]"
                ok = not result.startswith(
                    ("[工具执行错误]", "[参数校验失败]", "[工具超时]")
                )
                if not ok:
                    tool_errors += 1
                events.append(
                    {
                        "event": "tool_call",
                        "session": "default",
                        "step": step,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result_len": len(result),
                        "ok": ok,
                    }
                )
                messages.append(
                    {"role": "tool", "content": result, "tool_call_id": tc.id, "name": tc.name}
                )
                trace.append(
                    {
                        "type": "tool_call",
                        "step": step,
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "result": result,
                    }
                )

            if loop_detected:
                break

        # 达到轮次上限或循环终止：强制兜底，防止死循环
        if loop_detected:
            fallback = (
                f"（检测到重复工具调用，已提前终止，避免死循环。）\n{loop_detected}"
            )
            terminated_by = "loop_detected"
        else:
            fallback = "（已达到最大推理轮次，请简化问题或稍后再试。）"
            terminated_by = "limit"
        trace.append({"type": "limit", "step": self._max_iterations, "content": fallback})
        self._memory.add_user(user_input)
        self._memory.add_assistant(fallback)
        steps = step + 1 if loop_detected else self._max_iterations
        return _finish(fallback, self._max_iterations, terminated_by, steps=steps)
