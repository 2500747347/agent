# 个人AI助理 Agent（Personal AI Assistant Agent）

> 从零手写一个具备工具调用、规划与记忆能力的通用 AI 助理 Agent。

## 项目简介

本项目不依赖现成 Agent 框架，从零实现 ReAct 推理循环、Function Calling 工具调度、
会话记忆与长期记忆，最终交付一个可本地运行的多轮对话 Web 应用（Gradio）。

可用于日程/待办管理、笔记、信息查询、计算等日常助理场景。

## 技术栈

- Python 3.13
- LLM：OpenAI 兼容接口（Codex / GPT / DeepSeek / 通义等可切换），内置 Mock 模式
- Agent 内核：手写 ReAct + Function Calling + 工具注册机制
- 记忆：会话记忆（SQLite）+ 长期记忆（用户偏好/事实）
- 界面：Gradio

## 目录结构

```
Agent/
├── agent/          # Agent 内核（核心循环、LLM封装、工具调度、记忆）
├── tools/          # 工具实现（时间/计算/待办/笔记/搜索/天气）
├── config/         # 配置文件（LLM Key 等，不入库）
├── data/           # 运行时数据（待办、笔记、记忆库）
├── ui/             # Gradio 对话界面
├── eval/           # 评测测试集与脚本
├── docs/           # 需求、架构、设计文档
└── tests/          # 单元测试
```

## 快速开始

（待 P4 完成后补充）

## 开发进度

- [ ] P0 环境与项目初始化
- [ ] P1 需求与架构设计
- [ ] P2 Agent 内核实现
- [ ] P3 工具与记忆系统
- [ ] P4 对话界面与演示
- [ ] P5 评测与迭代
- [ ] P6 文档与求职呈现
