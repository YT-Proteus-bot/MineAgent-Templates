"""开源 · CrewAI 层级模式 · CrewAI

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
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool
except ImportError as _missing:  # pragma: no cover
    raise SystemExit(
        f"缺少 CrewAI 依赖（{_missing}）。先执行：pip install crewai ，装好后重新运行本文件即可。"
    ) from _missing
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
# 节点：Manager  (supervisor_2)  类型：supervisor
# ═══════════════════════════════════════════════════════════════

def supervisor_2(state: AgentState) -> dict:
    """Manager —— 推理模式：react，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = _get_agent("n2", lambda: create_agent(
        model=llm,
        tools=TOOLS_supervisor_2,
        system_prompt="你是一个能调用工具的智能体。回答前先想清楚需要哪些信息，必要时调用工具，最后给出最终答案。",
    ))
    try:
        answer = _agent_stream(agent, task, "n2", "supervisor_2",
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
# 节点：成员 A  (worker_3)  类型：worker
# ═══════════════════════════════════════════════════════════════

def worker_3(state: AgentState) -> dict:
    """成员 A —— 推理模式：react，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = _get_agent("n3", lambda: create_agent(
        model=llm,
        tools=TOOLS_worker_3,
        system_prompt="你是一个能调用工具的智能体。回答前先想清楚需要哪些信息，必要时调用工具，最后给出最终答案。",
    ))
    try:
        answer = _agent_stream(agent, task, "n3", "worker_3",
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
# 节点：成员 B  (worker_4)  类型：worker
# ═══════════════════════════════════════════════════════════════

def worker_4(state: AgentState) -> dict:
    """成员 B —— 推理模式：react，最多 6 步。"""
    task = state.get("data", "")
    origin = state.get("user_input", "")
    if origin and origin != task:
        task = f"{origin}\n\n## 上一轮观察 / 反馈\n{task}\n\n请结合原始问题作答。"
    agent = _get_agent("n4", lambda: create_agent(
        model=llm,
        tools=TOOLS_worker_4,
        system_prompt="你是一个能调用工具的智能体。回答前先想清楚需要哪些信息，必要时调用工具，最后给出最终答案。",
    ))
    try:
        answer = _agent_stream(agent, task, "n4", "worker_4",
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
# 节点：验收  (critic_5)  类型：critic
# ═══════════════════════════════════════════════════════════════

def critic_5(state: AgentState) -> dict:
    """验收 —— 按标准挑问题并判定是否通过。"""
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

TOOLS_supervisor_2 = []  # Manager 可见的工具
TOOLS_worker_3 = []  # 成员 A 可见的工具
TOOLS_worker_4 = []  # 成员 B 可见的工具

# CrewAI：把画布里的 agent 节点变成 CrewAI 的 Agent + Task
supervisor_2_agent = Agent(
    role="Manager",
    goal="Router 角色：把任务并行派给多个下级 agent，收齐结果再汇总；单个下级失败不拖垮整体。",
    backstory="Router 角色：把任务并行派给多个下级 agent，收齐结果再汇总；单个下级失败不拖垮整体。",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)
worker_3_agent = Agent(
    role="成员 A",
    goal="执行被分派的子任务。",
    backstory="执行被分派的子任务。",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)
worker_4_agent = Agent(
    role="成员 B",
    goal="执行被分派的子任务。",
    backstory="执行被分派的子任务。",
    llm=llm,
    verbose=False,
    allow_delegation=False,
)

_agents = [supervisor_2_agent, worker_3_agent, worker_4_agent]
tasks = [
    Task(description="{input}", expected_output="完成的结果", agent=supervisor_2_agent),
    Task(description="{input}", expected_output="完成的结果", agent=worker_3_agent),
    Task(description="{input}", expected_output="完成的结果", agent=worker_4_agent),
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

    parser = argparse.ArgumentParser(description="开源 · CrewAI 层级模式")
    parser.add_argument("--input", "-i", default="", help="交给智能体的任务")
    args = parser.parse_args()
    text = args.input or input("请输入任务：").strip()
    answer = main(text)
    print("")
    print("=" * 16, "最终结果", "=" * 16)
    print(answer)
