"""开源 · Langflow 线性链 · 空白画布

由 MineAgent 根据画布生成，可以直接改、也可以直接拿去部署。

代码是主体，画布上的每个节点都对应下面一个带 ═══ 标记的代码块；
在网页里编辑节点代码，改的就是这里的同一段。

用法：
    python agent.py --input "你的问题"
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))

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
    try:
        for chunk in model.stream(messages):
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
    future = pool.submit(fn, state)
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
# 节点：系统提示词  (system_prompt_2)  类型：system_prompt
# ═══════════════════════════════════════════════════════════════

def system_prompt_2(state: AgentState) -> dict:
    """系统提示词 —— 生成并放进 state，供下游复用。"""
    base = "你是一名严谨的分析师。"
    task = str(state.get("data", ""))
    prompt = llm.invoke([
        {"role": "system", "content": "擅长写提示词，输出一段可直接使用的 system prompt。"},
        {"role": "user", "content": f"人设：{base}\n本次任务：{task}\n请据此写出 system prompt。"},
    ]).content.strip()
    return {"data": task, "system_prompt": prompt}


# ═══════════════════════════════════════════════════════════════
# 节点：LLM  (llm_3)  类型：llm
# ═══════════════════════════════════════════════════════════════

LLM_3_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "一次大模型调用，最基础的推理单元。"),
    ("human", "{input}"),
])

def llm_3(state: AgentState) -> dict:
    """LLM"""
    model = llm.bind(temperature=0.3)
    text = _stream(model, LLM_3_PROMPT.format_messages(input=state.get("data", "")),
                   "n3", "llm_3")
    return {"data": text}


# ═══════════════════════════════════════════════════════════════
# 节点：输出整理  (formatter_4)  类型：formatter
# ═══════════════════════════════════════════════════════════════

FORMATTER_4_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是格式化工具：把输入原样输出，最多调整 Markdown 结构；不得增删内容、不得添加解释或判断。"),
    ("human", "{input}"),
])

def formatter_4(state: AgentState) -> dict:
    """输出整理"""
    model = llm.bind(temperature=0.3)
    text = _stream(model, FORMATTER_4_PROMPT.format_messages(input=state.get("data", "")),
                   "n4", "formatter_4")
    return {"data": text}


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

builder = StateGraph(AgentState)
builder.add_node("input_1", _track(input_1, "n1", "input_1", "输入", "input", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))
builder.add_node("system_prompt_2", _track(system_prompt_2, "n2", "system_prompt_2", "系统提示词", "system_prompt", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))
builder.add_node("llm_3", _track(llm_3, "n3", "llm_3", "LLM", "llm", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))
builder.add_node("formatter_4", _track(formatter_4, "n4", "formatter_4", "输出整理", "formatter", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))
builder.add_node("output_5", _track(output_5, "n5", "output_5", "输出", "output", {"timeout_s": 120.0, "max_retries": 0, "retry_backoff_s": 2.0, "rate_per_minute": 0, "on_error": "block"}))

builder.add_edge(START, "input_1")
builder.add_edge("input_1", "system_prompt_2")
builder.add_edge("system_prompt_2", "llm_3")
builder.add_edge("llm_3", "formatter_4")
builder.add_edge("formatter_4", "output_5")
builder.add_edge("output_5", END)


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

    parser = argparse.ArgumentParser(description="开源 · Langflow 线性链")
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
