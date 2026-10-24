"""个人AI助理 Agent —— 核心包。

模块一览：
- llm.py   : LLM 统一封装（OpenAI 兼容接口 + Mock 模式）
- tools.py : 工具注册机制（@tool 装饰器 + ToolRegistry）
- memory.py: 会话记忆（滑动窗口）
- core.py  : Agent 主循环（ReAct + Function Calling）
"""
