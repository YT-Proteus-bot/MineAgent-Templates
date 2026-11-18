"""开源 · AutoGen 对话驱动 · AutoGen

由 MineAgent 根据画布生成，可以直接改、也可以直接拿去部署。
每个画布节点对应下面一个带 ═══ 标记的代码块。

用法：
    python agent.py --input "你的问题"
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, TypedDict

try:
    from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
except ImportError as _missing:  # pragma: no cover
    raise SystemExit(
        f"缺少 AutoGen 依赖（{_missing}）。先执行：pip install pyautogen ，装好后重新运行本文件即可。"
    ) from _missing
from openai import OpenAI

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))

llm_config = {{
    "config_list": [
        {{
            "model": os.environ.get("MINEAGENT_MODEL") or CFG.get("model", "deepseek-chat"),
            "base_url": os.environ.get("MINEAGENT_BASE_URL") or CFG.get("base_url"),
            "api_key": os.environ.get("MINEAGENT_API_KEY") or "EMPTY",
        }}
    ],
    "temperature": CFG.get("temperature", 0.3),
}}

client = OpenAI(
    base_url=os.environ.get("MINEAGENT_BASE_URL") or CFG.get("base_url"),
    api_key=os.environ.get("MINEAGENT_API_KEY") or "EMPTY",
)


class AgentState(TypedDict, total=False):
    data: Any
    route: str


# ═══════════════════════════════════════════════════════════════
# 节点实现：每个节点对应画布上的一个方块
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# 节点：输入  (input_1)  类型：input
# ═══════════════════════════════════════════════════════════════

def input_1(state: AgentState) -> dict:
    """输入 —— 智能体的入口，接收用户输入。"""
    return {"data": state.get("data", "")}


# ═══════════════════════════════════════════════════════════════
# 节点：GroupChat 主持人  (group_chat_2)  类型：group_chat
# ═══════════════════════════════════════════════════════════════

def group_chat_2(state: AgentState) -> dict:
    """GroupChat 主持人 —— 推理模式：react，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = _get_agent("n2", lambda: create_agent(
        model=llm,
        tools=TOOLS_group_chat_2,
        system_prompt="你是一个能调用工具的智能体。回答前先想清楚需要哪些信息，必要时调用工具，最后给出最终答案。",
    ))
    try:
        answer = _agent_stream(agent, task, "n2", "group_chat_2",
                               14)
    except Exception:  # noqa: BLE001
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": 14},
        )
        messages = result.get("messages") or []
        answer = str(getattr(messages[-1], "content", "")) if messages else ""
    return {"data": answer, "answer": answer}


# ═══════════════════════════════════════════════════════════════
# 节点：AssistantAgent  (agent_3)  类型：agent
# ═══════════════════════════════════════════════════════════════

def agent_3(state: AgentState) -> dict:
    """AssistantAgent —— 推理模式：simple，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = _get_agent("n3", lambda: create_agent(
        model=llm,
        tools=TOOLS_agent_3,
        system_prompt="你是一个能调用工具的智能体。回答前先想清楚需要哪些信息，必要时调用工具，最后给出最终答案。",
    ))
    try:
        answer = _agent_stream(agent, task, "n3", "agent_3",
                               14)
    except Exception:  # noqa: BLE001
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": 14},
        )
        messages = result.get("messages") or []
        answer = str(getattr(messages[-1], "content", "")) if messages else ""
    return {"data": answer, "answer": answer}


# ═══════════════════════════════════════════════════════════════
# 节点：UserProxyAgent  (agent_4)  类型：agent
# ═══════════════════════════════════════════════════════════════

def agent_4(state: AgentState) -> dict:
    """UserProxyAgent —— 推理模式：simple，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = _get_agent("n4", lambda: create_agent(
        model=llm,
        tools=TOOLS_agent_4,
        system_prompt="你是代理用户，负责提需求、挑毛病。",
    ))
    try:
        answer = _agent_stream(agent, task, "n4", "agent_4",
                               14)
    except Exception:  # noqa: BLE001
        result = agent.invoke(
            {"messages": [{"role": "user", "content": task}]},
            config={"recursion_limit": 14},
        )
        messages = result.get("messages") or []
        answer = str(getattr(messages[-1], "content", "")) if messages else ""
    return {"data": answer, "answer": answer}


# ═══════════════════════════════════════════════════════════════
# 节点：收敛结论  (formatter_5)  类型：formatter
# ═══════════════════════════════════════════════════════════════

FORMATTER_5_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是格式化工具：把输入原样输出，最多调整 Markdown 结构；不得增删内容、不得添加解释或判断。"),
    ("human", "{input}"),
])

def formatter_5(state: AgentState) -> dict:
    """收敛结论"""
    model = llm.bind(temperature=0.3)
    text = _stream(model, FORMATTER_5_PROMPT.format_messages(input=state.get("data", "")),
                   "n5", "formatter_5")
    return {"data": text}


# ═══════════════════════════════════════════════════════════════
# 节点：输出  (output_6)  类型：output
# ═══════════════════════════════════════════════════════════════

def output_6(state: AgentState) -> dict:
    """输出 —— 智能体的出口，把结果交给调用方。"""
    source = "auto"
    if source == "data":
        return {"data": state.get("data", "")}
    if source in ("answer", "agent", "auto"):
        answer = state.get("answer")
        if answer not in (None, ""):
            return {"data": answer}
        if source == "answer":
            return {"data": ""}
    for item in reversed(list(state.get("trace") or [])):
        if item.get("node") == source or item.get("kind") == source:
            return {"data": item.get("output", "")}
    return {"data": state.get("data", "")}


# ═══════════════════════════════════════════════════════════════
# 把节点组装成可执行的图
# ═══════════════════════════════════════════════════════════════

TOOLS: list = []

TOOLS_group_chat_2 = []  # GroupChat 主持人 可见的工具
TOOLS_agent_3 = []  # AssistantAgent 可见的工具
TOOLS_agent_4 = []  # UserProxyAgent 可见的工具

# AutoGen：把画布里的 agent 节点变成 AssistantAgent，按对话推进
agent_3_agent = AssistantAgent(
    name="agent_3",
    llm_config=llm_config,
    system_message="自主 Agent，可选 ReAct / Plan / Goal 等推理模式；连到它的 tool 子图会自动注入工具。",
)
agent_4_agent = AssistantAgent(
    name="agent_4",
    llm_config=llm_config,
    system_message="自主 Agent，可选 ReAct / Plan / Goal 等推理模式；连到它的 tool 子图会自动注入工具。",
)

user_proxy = UserProxyAgent(name="user_proxy", code_execution_config=False)
groupchat = GroupChat(agents=[user_proxy, agent_3_agent, agent_4_agent], messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════
def main(user_input: str, thread_id: str = "") -> str:
    """入口：把用户输入交给 input_1，一路跑完整张图。"""
    user_proxy.initiate_chat(manager, message=user_input)
    return groupchat.messages[-1].get("content", "") if groupchat.messages else ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="开源 · AutoGen 对话驱动")
    parser.add_argument("--input", "-i", default="", help="交给智能体的任务")
    args = parser.parse_args()
    text = args.input or input("请输入任务：").strip()
    answer = main(text)
    print("")
    print("=" * 16, "最终结果", "=" * 16)
    print(answer)
