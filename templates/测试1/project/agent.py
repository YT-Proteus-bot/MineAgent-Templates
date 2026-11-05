"""second · AutoGen

由 MineAgent 根据画布生成，可以直接改、也可以直接拿去部署。
每个画布节点对应下面一个带 ═══ 标记的代码块。

用法：
    python agent.py --input "你的问题"
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypedDict

from autogen import AssistantAgent, GroupChat, GroupChatManager, UserProxyAgent
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
# 节点：生成  (agent_2)  类型：agent
# ═══════════════════════════════════════════════════════════════

def agent_2(state: AgentState) -> dict:
    """生成 —— 推理模式：simple，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt="你是一个能调用工具的智能体。回答前先想清楚需要哪些信息，必要时调用工具，最后给出最终答案。",
    )
    try:
        answer = _agent_stream(agent, task, "n2", "agent_2",
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
# 节点：评审  (critic_3)  类型：critic
# ═══════════════════════════════════════════════════════════════

def critic_3(state: AgentState) -> dict:
    """按维度打分并给改进项。"""
    # 这个节点类型还没有专门的原生写法，先用一次模型调用占位
    msg = llm.invoke(f"按维度打分并给改进项。\n\n输入：{state.get('data', '')}")
    return {"data": msg.content}


# ═══════════════════════════════════════════════════════════════
# 节点：还需要改吗  (condition_4)  类型：condition
# ═══════════════════════════════════════════════════════════════

def condition_4(state: AgentState) -> dict:
    """还需要改吗 —— 按判别问题决定走 true 还是 false。"""
    text = str(state.get("data", ""))
    question = "issues > 0"
    verdict = llm.bind(temperature=0).invoke([
        {"role": "system", "content": "你只输出 true 或 false，不要解释。"},
        {"role": "user", "content": f"判别问题：{question}\n\n待判内容：\n{text}\n\n只回答 true 或 false。"},
    ]).content.strip().lower()
    branch = "true" if verdict.startswith("true") else "false"
    rounds = int(state.get("loop_count") or 0) + 1
    max_loops = 0
    if branch == "false" and rounds > max_loops:
        branch = "true"
        _emit({"event": "log", "level": "warn", "node": "n4", "name": "condition_4",
               "message": f"已到循环上限 {max_loops} 次，强制收敛"})
    _emit({"event": "route", "node": "n4", "name": "condition_4",
           "branch": branch, "reason": _brief(verdict, 120)})
    return {"data": text, "route": branch, "loop_count": rounds}


# ═══════════════════════════════════════════════════════════════
# 节点：输出  (output_5)  类型：output
# ═══════════════════════════════════════════════════════════════

def output_5(state: AgentState) -> dict:
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

# AutoGen：把画布里的 agent 节点变成 AssistantAgent，按对话推进
agent_2_agent = AssistantAgent(
    name="agent_2",
    llm_config=llm_config,
    system_message="自主 Agent，可选 ReAct / Plan / Goal 等推理模式；连到它的 tool 子图会自动注入工具。",
)

user_proxy = UserProxyAgent(name="user_proxy", code_execution_config=False)
groupchat = GroupChat(agents=[user_proxy, agent_2_agent], messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════
def main(user_input: str) -> str:
    """入口：把用户输入交给 input_1，一路跑完整张图。"""
    user_proxy.initiate_chat(manager, message=user_input)
    return groupchat.messages[-1].get("content", "") if groupchat.messages else ""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="second")
    parser.add_argument("--input", "-i", default="", help="交给智能体的任务")
    args = parser.parse_args()
    text = args.input or input("请输入任务：").strip()
    answer = main(text)
    print("")
    print("=" * 16, "最终结果", "=" * 16)
    print(answer)
