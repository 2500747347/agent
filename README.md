# 个人AI助理 Agent（Personal AI Assistant Agent）

> 从零手写 ReAct 内核 + Function Calling + 记忆系统的通用 AI 助理，不依赖任何现成 Agent 框架。
> 面向 AI 应用 / Agent 开发岗位的完整项目经历：可运行、可评测、可演示、可讲解。

---

## 特性一览

- **手写 Agent 内核**：ReAct 推理循环 + 原生 Function Calling 工具调度，无 LangChain/LangGraph 依赖
- **热插拔工具机制**：`@tool` 装饰器 + 注册表，新增工具零侵入内核（当前 11 个工具）
- **双记忆系统**：会话记忆（滑动窗口）+ 长期记忆（跨会话持久化，自研 bigram TF + 余弦相似度检索）
- **双运行模式**：真实模型（Codex/GPT/DeepSeek/通义，OpenAI 兼容接口可切换）/ Mock 模式（离线可跑）
- **评测体系**：40 条测试集，工具调用 / 参数 / 回答三指标，综合通过率 **100%**
- **可视化 Demo**：Gradio Web 界面，多轮对话 + 工具调用过程实时展示

## 技术栈

| 层 | 技术 |
|---|---|
| 语言 | Python 3.13 |
| LLM | OpenAI 兼容接口（openai SDK v3.x），内置规则式 MockLLM |
| 存储 | SQLite（标准库，零依赖） |
| 检索 | 自实现 bigram 词频向量 + 余弦相似度（numpy） |
| 界面 | Gradio 6.x |
| 测试/评测 | pytest + 自研评测脚本 |

## 目录结构

```
Agent/
├── agent/                  # Agent 内核
│   ├── core.py             #   ReAct 主循环（意图→工具→回复，轮次上限防死循环）
│   ├── llm.py              #   LLM 统一封装（OpenAILLM + MockLLM）
│   ├── tools.py            #   @tool 装饰器 + ToolRegistry（schema 自动推断）
│   ├── memory.py           #   会话记忆（滑动窗口）
│   ├── long_term_memory.py #   长期记忆（TF 向量 + 余弦检索）
│   └── factory.py          #   配置加载与 Agent 装配
├── tools/                  # 11 个工具（时间/计算/待办/笔记/知识检索/天气/记忆…）
│   └── data_store.py       #   SQLite 公共存储
├── ui/app.py               # Gradio 对话界面（工具轨迹可视化）
├── eval/                   # 评测：test_suite.json + run_eval.py → output/report.md
├── tests/                  # 单元测试（16 项）
├── config/config.yaml      # LLM 配置模板（provider: mock | openai）
└── docs/                   # 需求架构文档 + 求职材料
```

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖（一次性）
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 启动对话界面（默认 http://127.0.0.1:7860）
.\.venv\Scripts\python.exe ui\app.py

# 3. 运行测试与评测
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe eval\run_eval.py
```

**接入真实模型（Codex 等）**：复制 `config/config.yaml` 为 `config/secret.yaml`，填好
`base_url / api_key / model` 并将 `provider` 改为 `openai`（secret.yaml 优先覆盖且不入库）。
不配置则默认 Mock 模式，完全离线可运行。

## 试试这些

```
今天是几号？            → 时间工具
帮我计算 128 * 37       → 计算工具（安全求值，结果 4736）
记一下明天上午10点开周会 → 待办工具（SQLite 持久化）
芜湖今天天气怎么样？      → 天气工具
什么是 RAG？            → 知识检索工具
记住我正在准备面试       → 长期记忆
我之前说过什么           → 记忆召回（跨会话）
```

## Agent 工作原理（30 秒版）

```
用户输入
  → LLM（携带会话历史 + 11 个工具的 JSON Schema）
  → 模型返回结构化 tool_calls（工具名 + 参数）
  → 注册表查找 → 参数校验 → 执行工具 → 结果回填为 observation
  → 循环（最多 5 轮）直到模型不再调用工具 → 输出最终回复
```

细节见 [docs/01-需求与架构.md](docs/01-需求与架构.md)。

## 项目价值（求职向）

一次完整的 Agent 开发闭环：需求分析 → 架构设计 → 内核实现 → 工具与记忆 →
应用落地 → 评测迭代 → 文档沉淀。简历条目与面试要点见 [docs/02-求职材料.md](docs/02-求职材料.md)。

## 开发进度

- [x] P0 环境与项目初始化
- [x] P1 需求与架构设计
- [x] P2 Agent 内核实现（5 项链路测试）
- [x] P3 工具与记忆系统（11 工具，12 项测试）
- [x] P4 对话界面与演示（Gradio + 轨迹可视化）
- [x] P5 评测与迭代（40 条用例，综合通过率 100%）
- [x] P6 文档与求职呈现
