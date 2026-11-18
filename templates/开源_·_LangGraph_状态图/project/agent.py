"""开源 · LangGraph 状态图 · LangChain / LangGraph

由 MineAgent 根据画布生成，可以直接改、也可以直接拿去部署。

代码是主体，画布上的每个节点都对应下面一个带 ═══ 标记的代码块；
在网页里编辑节点代码，改的就是这里的同一段。

用法：
    python agent.py --input "你的问题"
"""
from __future__ import annotations

import contextvars
import json
import os
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


# ── token 用量：按运行上下文累计，一次 run 结束由 take_usage() 取走 ──
# 用 ContextVar 而不是线程局部变量：节点是在线程池里执行的，只有复制上下文
# （_run_with_timeout 里做了）才能把用量带回来。
_USAGE = contextvars.ContextVar("mineagent_usage", default=None)


def reset_usage() -> None:
    _USAGE.set({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0})


def _usage_add(usage) -> None:
    """累计一次模型调用的 token（同时兼容 OpenAI 和 LangChain 的字段名）。"""
    if not isinstance(usage, dict):
        return
    bucket = _USAGE.get()
    if bucket is None:
        reset_usage()
        bucket = _USAGE.get()
    for dst, keys in (("prompt_tokens", ("input_tokens", "prompt_tokens")),
                      ("completion_tokens", ("output_tokens", "completion_tokens")),
                      ("total_tokens", ("total_tokens",))):
        for key in keys:
            value = usage.get(key)
            if value:
                try:
                    bucket[dst] += int(value)
                except (TypeError, ValueError):
                    pass
                break
    bucket["calls"] += 1


def take_usage() -> dict:
    """取走本次运行的用量统计，并清空。"""
    bucket = dict(_USAGE.get() or {"prompt_tokens": 0, "completion_tokens": 0,
                                   "total_tokens": 0, "calls": 0})
    reset_usage()
    return bucket


class _UsageHandler(BaseCallbackHandler):
    """把所有模型调用的 token 用量记进当前运行上下文。

    挂在模型实例上，invoke / stream / create_agent 内部的调用都能覆盖到，
    比在每个节点里手写记账可靠。
    """

    def on_llm_end(self, response, **kwargs):  # noqa: D401
        llm_output = getattr(response, "llm_output", None) or {}
        _usage_add(llm_output.get("token_usage") or llm_output.get("usage"))


_USAGE_HANDLER = _UsageHandler()


# ── 模型：key 从环境变量或 .env 读，不写死在代码里 ──────────────
llm = ChatOpenAI(
    model=os.environ.get("MINEAGENT_MODEL") or CFG.get("model", "deepseek-chat"),
    base_url=os.environ.get("MINEAGENT_BASE_URL") or CFG.get("base_url"),
    api_key=os.environ.get("MINEAGENT_API_KEY") or "EMPTY",
    temperature=CFG.get("temperature", 0.3),
    max_tokens=CFG.get("max_tokens", 2048),
    # 超时必须在 HTTP 层生效：节点级 _track 的超时只是兜底，
    # 真正能中断请求的是这里（默认 120s，可用环境变量覆盖）
    timeout=float(os.environ.get("MINEAGENT_LLM_TIMEOUT") or 120),
    max_retries=int(os.environ.get("MINEAGENT_LLM_MAX_RETRIES") or 2),
    # 流式调用也要带上 token 用量（OpenAI 兼容接口需要显式要 usage）。
    # 个别服务商不认 stream_options，遇到就设 MINEAGENT_STREAM_USAGE=0 关掉。
    stream_usage=str(os.environ.get("MINEAGENT_STREAM_USAGE", "1")).lower() not in ("0", "false", "no"),
    # callbacks 挂在模型上：所有节点（含 create_agent 内部）的调用都会记账
    callbacks=[_USAGE_HANDLER],
)

# ── 运行事件：每个节点执行前后的输入/输出都回传给网页运行面板 ──────
EVENT_PREFIX = "@@EVENT@@"


def _emit(event: dict) -> None:
    try:
        print(EVENT_PREFIX + json.dumps(event, ensure_ascii=False, default=str), flush=True)
    except Exception:  # noqa: BLE001
        pass


def _parse_json(text: str, default: Any = None) -> Any:
    """从模型输出里尽力抠出 JSON（允许带 ```json 包裹或前后废话）。"""
    raw = (text or "").strip()
    if not raw:
        return default if default is not None else {}
    fence = raw.find("```")
    if fence >= 0:
        body = raw[fence + 3:]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
        end = body.find("```")
        raw = body[:end] if end >= 0 else body
        raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0] or [-1])
    end = max(raw.rfind("}"), raw.rfind("]"))
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            pass
    return default if default is not None else {}


def _brief(value: Any, limit: int = 1500) -> Any:
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001
        return str(value)[:limit]


def _stream(model, messages, node_id: str = "", name: str = "", interval: float = 0.05) -> str:
    """逐段调模型：边生成边把 token 推给运行面板，最后拼成完整文本。"""
    parts: list = []
    last = 0.0
    usage_seen = None
    try:
        for chunk in model.stream(messages):
            meta = getattr(chunk, "usage_metadata", None)
            if meta:
                usage_seen = meta
            text = getattr(chunk, "content", "")
            if not isinstance(text, str) or not text:
                continue
            parts.append(text)
            now = time.time()
            if now - last >= interval:
                last = now
                _emit({"event": "token", "node": node_id, "name": name, "text": text})
    except Exception:  # noqa: BLE001
        msg = model.invoke(messages)
        return str(getattr(msg, "content", msg))
    if usage_seen:
        # 流式返回的用量：on_llm_end 拿到的 llm_output 可能是空的，这里补一次
        _usage_add(usage_seen)
    return "".join(parts)


def _agent_stream(agent, task: str, node_id: str = "", name: str = "",
                  recursion_limit: int = 14) -> str:
    """跑 agent 并把生成片段推给运行面板，最后从 values 状态里取最终回答。"""
    last_state: dict = {}
    for mode, payload in agent.stream(
        {"messages": [{"role": "user", "content": task}]},
        config={"recursion_limit": recursion_limit},
        stream_mode=["messages", "values"],
    ):
        if mode == "messages":
            chunk, _meta = payload
            text = getattr(chunk, "content", "")
            if isinstance(text, str) and text:
                _emit({"event": "token", "node": node_id, "name": name, "text": text})
        elif mode == "values" and isinstance(payload, dict):
            last_state = payload
    messages = last_state.get("messages") or []
    return str(getattr(messages[-1], "content", "")) if messages else ""


def _run_with_timeout(fn, state, timeout_s: float):
    """给同步节点加超时。

    超时后立刻把控制权交回调用方：cancel + shutdown(wait=False)，
    不再同步等一个仍卡在网络上的调用。真正的硬中断由模型客户端自己的
    timeout（环境变量 MINEAGENT_LLM_TIMEOUT）兜底。
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    pool = ThreadPoolExecutor(max_workers=1)
    # 复制上下文：节点里累计的 token 用量要能带回调用方
    ctx = contextvars.copy_context()
    future = pool.submit(ctx.run, fn, state)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout:
        future.cancel()
        raise TimeoutError(f"节点执行超时（>{timeout_s:g}s）") from None
    finally:
        pool.shutdown(wait=False)


def _retryable(exc: Exception) -> bool:
    """只有网络 / 限流 / 服务端 5xx 这类明确可重放的错误才值得重试。

    超时不重试：那次调用可能还在服务端跑，重试等于重复计费。
    """
    if isinstance(exc, TimeoutError):
        return False
    name = type(exc).__name__.lower()
    if any(k in name for k in ("connection", "ratelimit")):
        return True
    text = str(exc).lower()
    return any(k in text for k in ("429", "503", "connection reset", "connection aborted"))


def _track(fn, node_id: str, name: str = "", label: str = "", kind: str = "llm",
           opts: dict | None = None):
    """包一层：执行超时 / 失败重试 / 错误分流 / 事件回传 / 轨迹记录。"""
    opts = opts or {}
    timeout_s = float(opts.get("timeout_s") or 0)
    max_retries = int(opts.get("max_retries") or 0)
    backoff = float(opts.get("retry_backoff_s") or 0)
    on_error = str(opts.get("on_error") or "block")
    rate = int(opts.get("rate_per_minute") or 0)

    def wrapped(state):
        started = time.time()
        if rate > 0:
            now = time.time()
            window = [t for t in (state.get("__rate__") or []) if now - t < 60]
            if len(window) >= rate:
                wait = 60 - (now - window[0])
                if wait > 0:
                    _emit({"event": "log", "level": "warn", "node": node_id,
                           "name": name or node_id,
                           "message": f"触发限流，等待 {wait:.1f}s"})
                    time.sleep(min(wait, 30))
                window = []
            window.append(time.time())
            state["__rate__"] = window[-rate:]
        _emit({"event": "node_start", "node": node_id, "name": name or node_id,
               "label": label, "kind": kind})
        attempt = 0
        while True:
            try:
                out = _run_with_timeout(fn, state, timeout_s) if timeout_s > 0 else fn(state)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt < max_retries and _retryable(exc):
                    attempt += 1
                    _emit({"event": "log", "level": "warn", "node": node_id,
                           "name": name or node_id,
                           "message": f"第 {attempt} 次重试：{type(exc).__name__}: {exc}"})
                    time.sleep(backoff * attempt)
                    continue
                _emit({"event": "error", "node": node_id, "name": name or node_id,
                       "message": str(exc)})
                if on_error == "edge":
                    # 不中断流程：把错误写进状态，由 conditional edge 送到错误出口
                    out = {"data": f"[节点失败] {exc}", "__error__": str(exc)}
                    break
                raise
        data = out.get("data", "") if isinstance(out, dict) else out
        trace = list(state.get("trace") or [])
        trace.append({"node": name or node_id, "id": node_id, "kind": kind,
                      "output": _brief(data), "attempts": attempt})
        _emit({"event": "node_end", "node": node_id, "name": name or node_id, "kind": kind,
               "duration_ms": int((time.time() - started) * 1000),
               "attempts": attempt, "output": _brief(data)})
        if isinstance(out, dict):
            out["trace"] = trace[-40:]
            # 成功执行后清掉上一轮的错误标记
            if "__error__" not in out:
                out["__error__"] = ""
        return out

    return wrapped


def _merge_value(old: Any, new: Any) -> Any:
    """并行分支同时写同一个键时，用后到的值覆盖，不抛 InvalidUpdateError。"""
    return new if new is not None else old


class AgentState(TypedDict, total=False):
    """整张图共享的状态：data 在节点之间流动，route 承载分支信号。

    并行分支（一个节点连多条出边）会同时回写，所以这些键都挂了合并函数，
    否则 LangGraph 会报 "Can receive only one value per step"。
    """

    data: Annotated[Any, _merge_value]
    route: Annotated[str, _merge_value]
    history: Annotated[list, _merge_value]
    trace: Annotated[list, _merge_value]
    answer: Annotated[Any, _merge_value]
    loop_count: Annotated[int, _merge_value]
    user_input: Annotated[str, _merge_value]


# ── agent 实例缓存：工具集和 system prompt 都是静态的，没必要每次调用重建 ──
_AGENTS: dict = {}


def _get_agent(key: str, build):
    agent = _AGENTS.get(key)
    if agent is None:
        agent = build()
        _AGENTS[key] = agent
    return agent


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
# 节点：规划  (planner_2)  类型：planner
# ═══════════════════════════════════════════════════════════════

PLANNER_2_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是一个任务规划专家。把目标拆成有序、可执行的步骤，每行一步；输入只是寒暄、闲聊或没有可拆解目标时，原样返回输入，不要追问。"),
    ("human", "{input}"),
])

def planner_2(state: AgentState) -> dict:
    """规划"""
    model = llm.bind(temperature=0.3)
    text = _stream(model, PLANNER_2_PROMPT.format_messages(input=state.get("data", "")),
                   "n2", "planner_2")
    return {"data": text}


# ═══════════════════════════════════════════════════════════════
# 节点：Agent（ReAct）  (agent_3)  类型：agent
# ═══════════════════════════════════════════════════════════════

def agent_3(state: AgentState) -> dict:
    """Agent（ReAct） —— 推理模式：react，最多 6 步。"""
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
# 节点：条件边：继续还是收尾？  (router_4)  类型：router
# ═══════════════════════════════════════════════════════════════

def router_4(state: AgentState) -> dict:
    """条件边：继续还是收尾？ —— 按判别问题决定走 true 还是 false。"""
    text = str(state.get("data", ""))
    question = "是否还需要再调用一次工具补充信息？"
    verdict = llm.bind(temperature=0).invoke([
        {"role": "system", "content": "你只输出 true 或 false，不要解释。"},
        {"role": "user", "content": f"判别问题：{question}\n\n待判内容：\n{text}\n\n只回答 true 或 false。"},
    ]).content.strip().lower()
    branch = "true" if verdict.startswith("true") else "false"
    rounds = int(state.get("loop_count") or 0) + 1
    max_loops = 4
    if rounds > max_loops and branch != "false":
        branch = "false"
        _emit({"event": "log", "level": "warn", "node": "n4", "name": "router_4",
               "message": f"已到循环上限 {max_loops} 次，强制走出口"})
    _emit({"event": "route", "node": "n4", "name": "router_4",
           "branch": branch, "reason": _brief(verdict, 120)})
    return {"data": text, "route": branch, "loop_count": rounds}


# ═══════════════════════════════════════════════════════════════
# 节点：工具  (tool_5)  类型：tool
# ═══════════════════════════════════════════════════════════════

@tool
def tool_5(query: str) -> str:
    """自定义工具：执行节点里的 Python 逻辑。"""
    return f"[tool_5] 尚未实现：{query}"


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

# 全量工具（手写代码里可以直接引用；agent 默认只拿自己下游那几个）
TOOLS = [tool_5]

TOOLS_agent_3 = []  # Agent（ReAct） 可见的工具

builder = StateGraph(AgentState)
builder.add_node("input_1", _track(input_1, "n1", "input_1", "输入", "input", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))
builder.add_node("planner_2", _track(planner_2, "n2", "planner_2", "规划", "planner", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))
builder.add_node("agent_3", _track(agent_3, "n3", "agent_3", "Agent（ReAct）", "agent", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))
builder.add_node("router_4", _track(router_4, "n4", "router_4", "条件边：继续还是收尾？", "router", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))
builder.add_node("output_6", _track(output_6, "n6", "output_6", "输出", "output", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))

builder.add_edge(START, "input_1")
builder.add_edge("input_1", "planner_2")
builder.add_edge("planner_2", "agent_3")
builder.add_edge("agent_3", "router_4")
builder.add_conditional_edges(
    "router_4",
    lambda s: s.get("route") or "true",
    {"true": "agent_3", "false": "output_6"},
)
builder.add_edge("output_6", END)


def build_graph(checkpointer=None):
    """构建（可选带持久化 checkpointer 的）LangGraph 图，服务入口会复用。"""
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════
def main(user_input: str, thread_id: str = "") -> str:
    """入口：把用户输入交给 input_1，一路跑完整张图。"""
    config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
    result = graph.invoke({"data": user_input, "user_input": user_input, "route": ""},
                         config=config)
    data = result.get("data", "")
    _emit({"event": "final", "output": data if isinstance(data, str) else _brief(data)})
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="开源 · LangGraph 状态图")
    parser.add_argument("--input", "-i", default="", help="交给智能体的任务")
    args = parser.parse_args()
    text = args.input or input("请输入任务：").strip()
    try:
        answer = main(text)
    except Exception as exc:  # noqa: BLE001
        _emit({"event": "run_end", "status": "error", "message": str(exc), "output": ""})
        raise
    print("")
    print("=" * 16, "最终结果", "=" * 16)
    print(answer)
    _emit({"event": "run_end", "status": "ok", "output": answer if isinstance(answer, str) else _brief(answer)})
