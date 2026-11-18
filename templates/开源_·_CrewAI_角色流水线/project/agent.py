"""开源 · CrewAI 角色流水线 · CrewAI

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

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from langchain_openai import ChatOpenAI

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))

llm = ChatOpenAI(
    model=os.environ.get("MINEAGENT_MODEL") or CFG.get("model", "deepseek-chat"),
    base_url=os.environ.get("MINEAGENT_BASE_URL") or CFG.get("base_url"),
    api_key=os.environ.get("MINEAGENT_API_KEY") or "EMPTY",
    temperature=CFG.get("temperature", 0.3),
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
# 节点：任务拆解  (planner_2)  类型：planner
# ═══════════════════════════════════════════════════════════════

PLANNER_2_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个任务规划专家。把目标拆成有序、可执行的步骤，每行一步；输入只是寒暄、闲聊或没有可拆解目标时，原样返回输入，不要追问。"),
    ("human", "{input}"),
])

def planner_2(state: AgentState) -> dict:
    """任务拆解"""
    model = llm.bind(temperature=0.3)
    text = _stream(model, PLANNER_2_PROMPT.format_messages(input=state.get("data", "")),
                   "n2", "planner_2")
    return {"data": text}


# ═══════════════════════════════════════════════════════════════
# 节点：研究员（Researcher）  (agent_3)  类型：agent
# ═══════════════════════════════════════════════════════════════

def agent_3(state: AgentState) -> dict:
    """研究员（Researcher） —— 推理模式：simple，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt="你负责收集资料、核实事实。",
    )
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
# 节点：写手（Writer）  (agent_4)  类型：agent
# ═══════════════════════════════════════════════════════════════

def agent_4(state: AgentState) -> dict:
    """写手（Writer） —— 推理模式：simple，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt="你负责把资料写成结构清晰的成稿。",
    )
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
# 节点：审校（Reviewer）  (critic_5)  类型：critic
# ═══════════════════════════════════════════════════════════════

def critic_5(state: AgentState) -> dict:
    """审校（Reviewer） —— 按标准挑问题并判定是否通过。"""
    text = str(state.get("data", ""))
    standard = "准确性、完整性、可用性"
    review = llm.invoke([
        {"role": "system", "content": "你是严格的评审者。先列问题，再判定是否通过。"},
        {"role": "user", "content": f"评审标准：{standard}\n\n待审内容：\n{text}"},
    ]).content.strip()
    passed = "不通过" not in review
    _emit({"event": "log", "node": "n5", "name": "critic_5",
           "message": "评审结论：" + ("通过" if passed else "不通过")})
    return {"data": text, "review": review, "passed": passed}


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

# CrewAI：把画布里的 agent 节点变成 CrewAI 的 Agent + Task
agent_3_agent = Agent(
    role="研究员（Researcher）",
    goal="自主 Agent，可选 ReAct / Plan / Goal 等推理模式；连到它的 tool 子图会自动注入工具。",
    backstory="自主 Agent，可选 ReAct / Plan / Goal 等推理模式；连到它的 tool 子图会自动注入工具。",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)
agent_4_agent = Agent(
    role="写手（Writer）",
    goal="自主 Agent，可选 ReAct / Plan / Goal 等推理模式；连到它的 tool 子图会自动注入工具。",
    backstory="自主 Agent，可选 ReAct / Plan / Goal 等推理模式；连到它的 tool 子图会自动注入工具。",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)

_agents = [agent_3_agent, agent_4_agent]
tasks = [
    Task(description="{input}", expected_output="完成的结果", agent=agent_3_agent),
    Task(description="{input}", expected_output="完成的结果", agent=agent_4_agent),
]
crew = Crew(agents=_agents, tasks=tasks, process=Process.sequential, verbose=False)


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════
def main(user_input: str, thread_id: str = "") -> str:
    """入口：把用户输入交给 input_1，一路跑完整张图。"""
    result = crew.kickoff(inputs={"input": user_input})
    return str(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="开源 · CrewAI 角色流水线")
    parser.add_argument("--input", "-i", default="", help="交给智能体的任务")
    args = parser.parse_args()
    text = args.input or input("请输入任务：").strip()
    answer = main(text)
    print("")
    print("=" * 16, "最终结果", "=" * 16)
    print(answer)
