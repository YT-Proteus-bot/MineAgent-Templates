# 闭源 · DeepSeek Super-agent

- 画布 slug：`闭源_·_DeepSeek_Super-agent`
- 框架：**LangChain / LangGraph**（可运行）
- 节点数：6，连线数：6

## 这个目录是自包含的

整个文件夹可以单独拷走、单独部署，不依赖 MineAgent 平台：
`runtime_sdk.py`、`runner.py` 都已经放在目录里，只用 Python 标准库；
只要有 Python 3.9+，换一台机器也能直接跑。

- 节点代码内联在 `agent.py` 里（不再有 `nodes/` 目录），网页上编辑的就是同一段
- 不要把 `runtime_sdk.py` / `runner.py` 拆出去单独运行
- 项目目录内不要有 `__pycache__`（.gitignore 已忽略，执行器也不会生成）

## 运行

```powershell
# Windows：一键脚本（自动检查依赖、找不到 .venv 就用系统 Python）
.\run.ps1 "你的问题"

# 或者手动执行
python agent.py --input "你的问题"   # 框架原生入口，没装框架会自动回退
python runner.py --input "你的问题"  # 内置执行器，零依赖，任何画布都能跑
```

```bash
# macOS / Linux
./run.sh "你的问题"
```

## 配置模型

1. 网页「用户中心」或画布设置里填 key（推荐，运行时自动注入）；
2. 或复制 `.env.example` 为 `.env` 并填写 `MINEAGENT_API_KEY`。

## 目录

```
agent.json         # 画布图定义（保存时自动覆盖）
agent.py           # 框架原生入口：一张画布 = 一个文件，节点按 ═══ 分块
runner.py          # 内置图执行器（零依赖，任何画布都能跑）
runtime_sdk.py     # 节点可用的全部能力（ctx.*）
server.py          # FastAPI 服务入口：/chat、/chat/stream、/health、/ready、/metrics
settings.py        # 服务配置（全部走环境变量）
telemetry.py       # JSON 日志 + Prometheus 指标
checkpoint.py      # 会话持久化（SQLite / Postgres）
Dockerfile         # 镜像（非 root 运行 + HEALTHCHECK）
.dockerignore      # 保证 .env 不会被打进镜像
docker-compose.yml # 一条命令起服务
integrations/      # 外部数据中台 / 评测中台连接器
tests/             # pytest 冒烟测试（不需要真实模型）
run.ps1            # Windows 一键运行
run.sh             # macOS / Linux 一键运行
skills/            # skill.md 技能文件
config.json        # 模型配置（不含 key）
outputs/           # 运行产物
.env.example       # 复制成 .env 填 key，脱离网页独立运行用
```

## 当成服务跑

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
# 或者
docker compose up --build
```

返回里的 `usage` 是本次运行的 token 统计；`/metrics` 输出 Prometheus 文本，
`/ready` 会明确列出缺哪项配置。

## 节点清单

| 节点 | 类型 |
| --- | --- |
| 多渠道消息（IM / GitHub / 邮件） | `input` |
| Router（常驻协调） | `supervisor` |
| Worker：问题复现 | `worker` |
| Worker：代码审查 | `worker` |
| 汇总与汇报 | `merger` |
| 输出 | `output` |
