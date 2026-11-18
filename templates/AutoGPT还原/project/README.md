# AutoGPT还原

- 画布 slug：`AutoGPT还原`
- 框架：**LangChain / LangGraph**（可运行）
- 节点数：9，连线数：9

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
VERSION            # 版本号 / 内容 hash / 生成时间（/version 直接返回它）
CHANGELOG.md       # 每次内容变化追加一行
.history/          # 最近 5 版 agent.py，回滚就拷这一个文件
runner.py          # 内置图执行器（零依赖，任何画布都能跑）
runtime_sdk.py     # 节点可用的全部能力（ctx.*）
sandbox.py         # shell / exec_python 的隔离实现（MINEAGENT_SANDBOX 控制）
server.py          # FastAPI 服务入口：/chat、/chat/stream、/health、/ready、/metrics
settings.py        # 服务配置（pydantic-settings，启动即校验）
telemetry.py       # JSON 日志 + Prometheus 指标
checkpoint.py      # 会话持久化（SQLite / Postgres）
requirements.txt   # 依赖声明
requirements.lock.txt  # 版本锁（直接依赖全部钉死）
scripts/           # make_lock / offline_bundle / install_offline
wheels/            # 离线包目录（offline_bundle.py 把 wheel 下到这里）
Dockerfile         # 镜像（非 root 运行 + HEALTHCHECK）
.dockerignore      # 保证 .env 不会被打进镜像
docker-compose.yml # 一条命令起服务
integrations/      # 外部数据中台 / 评测中台连接器
evals/             # 评测用例 + 打分脚本（改完 agent 用它证明没变差）
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
pip install -r requirements.lock.txt
uvicorn server:app --host 0.0.0.0 --port 8000
# 或者
docker compose up --build
```

返回里的 `usage` 是本次运行的 token 统计；`/metrics` 输出 Prometheus 文本，
`/ready` 会明确列出缺哪项配置。配错配置时进程**直接起不来**，
想看当前生效的配置（key 会打码）：

```bash
python settings.py
```

## 离线部署（客户机房没有外网）

```bash
# ① 在一台能上网、Python 版本与目标机一致的机器上打离线包
python scripts/offline_bundle.py      # 生成 wheels/

# ② 整个项目目录（含 wheels/）拷到目标机，然后
./scripts/install_offline.sh          # Linux / macOS
.\scripts\install_offline.ps1         # Windows
```

`requirements.lock.txt` 是生成时的版本锁；依赖变动后可以用
`python scripts/make_lock.py` 在装好依赖的环境里重新锁定。

## 改动后怎么证明没变差

```bash
python evals/run.py --check   # CI：只校验用例格式与打分逻辑，不调模型
python evals/run.py           # 真跑：需要 key，结果写到 evals/report.json
```

用例在 `evals/cases.jsonl`，一行一条；`pass_rate` 掉了就是改坏了。
每次请求都会带 `X-Trace-Id`（也可以由调用方传进来），日志和节点事件里都带这个 id，
排查问题时按 trace id 一搜就是完整一条链路。

## 执行沙箱

节点里的 `ctx.shell()` / `ctx.exec_python()` 都走 `sandbox.py`，由 `MINEAGENT_SANDBOX` 控制：

| 取值 | 行为 |
| --- | --- |
| `restricted`（默认） | 限项目目录、清掉密钥环境变量、超时、输出截断 |
| `none` | 不额外限制，只保留超时与输出截断（仅本地开发用） |
| `container` | 在 docker 里跑：无网络、只读根、限内存/CPU/PID；**没装 docker 会直接报错**，不会悄悄退回宿主 |

节点勾了「工具调用幂等」后，同一次运行里相同参数只会真正执行一次，重跑直接回放结果
（记录在同目录 `.store.db` 的 `tool_calls` 表）。

## 可观测

```bash
curl -s localhost:8000/metrics | head        # Prometheus 文本（/metrics/json 是 JSON）
curl -s localhost:8000/runs/<thread_id>      # 会话停在哪、有没有待审批
```

- 每次请求都带 `X-Trace-Id`；日志里的 `trace_id` 可用来串整条链路
- trace 落盘：`MINEAGENT_TRACE_FILE=outputs/traces/{date}.jsonl`（按天切）
- 接 OTel：`MINEAGENT_OTEL_ENDPOINT=http://collector:4318/v1/traces`，
  需要 `pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http`；
  没装会在启动日志里明确告警，不会静默不导出
- 装了 OTel 后每个节点也会是一个子 span（带 `node.id` / `node.kind`），
  在 Jaeger/Tempo 里能直接看到「请求 → 哪几个节点各花多久」
- 看板：`docs/grafana-dashboard.json` 可直接 import 到 Grafana，导入时选一下 Prometheus 数据源
- 采样：`MINEAGENT_TRACE_SAMPLE` 控制 trace 落盘与 OTel span 的采样率（0 不落 / 1 全落）
- 会话治理：`MINEAGENT_MAX_TURNS` 到顶、`MINEAGENT_SESSION_TTL` 过期时会清掉该 thread 的检查点并从头开始，
  `/metrics` 的 `session_reset_total` 记了次数；一次性请求（不传 thread_id）不受影响
- 成本：填 `MINEAGENT_PRICE_INPUT_PER_1K` / `_OUTPUT_PER_1K` 后，响应里的 `usage.cost_usd` 就是本次花费，
  `/metrics` 会累加 `mineagent_cost_usd_total`；设 `MINEAGENT_BUDGET_USD` 后超预算直接 429

## 密钥放哪

默认还是环境变量（跟以前一样）；现场想用受控文件或 Vault 时改 `MINEAGENT_SECRETS`，代码不用动：

| 取值 | 需要配 | 说明 |
| --- | --- | --- |
| `env`（默认） | — | 只读环境变量 |
| `file` | `MINEAGENT_SECRETS_FILE=/etc/mineagent/secrets.json` | JSON 键值，建议 `chmod 600` |
| `vault` | `MINEAGENT_VAULT_ADDR` / `_TOKEN` / `_MOUNT` / `_PATH` | Vault KV v2 |

查找顺序默认「环境变量 → provider」，设 `MINEAGENT_SECRETS_PRIORITY=provider` 可反过来。
provider 连不上只告警一次并退回环境变量，服务照常起；`/ready.secrets` 会显示当前来源与是否配齐。

## 改完之后的版本与回滚

每次保存画布，生成器都会按内容算一个 hash：

- 内容没变 → `VERSION` 不动
- 内容变了 → patch +1，`CHANGELOG.md` 追加一行，上一版 `agent.py` 存进 `.history/`（保留最近 5 版）

```bash
curl -s localhost:8000/version     # {"version":"1.0.3","hash":"…","generated_at":"…"}
curl -s localhost:8000/health      # {"ok":true,"version":"1.0.3"}

# 回滚到上一版：把留档文件拷回去，重启即可
cp .history/agent-1.0.2.py agent.py
```

## 评测飞轮：把线上失败变成用例

`/chat` 出错时会自动把这条请求写进 `evals/candidates.jsonl`（含 input / error / trace_id）。

```bash
python evals/run.py --candidates   # 重跑这些失败输入，看现在还报不报错
python evals/triage.py             # 按「错误类型 + 归一化输入」聚类，看哪类问题最该先修
python evals/triage.py --write     # 生成 draft-cases.jsonl，补断言后挪进 cases.jsonl
```

不再报错的输入，抄进 `cases.jsonl` 再补上 `expect_contains`，就成了一条正式用例——
修过的 bug 从此有了回归测试。

## 灰度发布

配一套 canary 模型，让一小部分流量先走新版本：

```bash
MINEAGENT_CANARY_BASE_URL=https://新模型地址/v1 \
MINEAGENT_CANARY_MODEL=新模型名 \
MINEAGENT_CANARY_PERCENT=10 \
uvicorn server:app --host 0.0.0.0 --port 8000
```

- 分流按 `trace_id` 做稳定哈希，同一个请求不会在两条链路间摇摆；
- 也可以用 header 强制：`X-Canary: 1` 走 canary、`X-Canary: 0` 走主版本（做对比验证很方便）；
- 响应体有 `variant` 字段、响应头有 `X-Variant`，日志与 trace 里也带；
- `/metrics` 里 `mineagent_chat_canary_total` 与 `mineagent_chat_primary_total` 可直接对比成功率与延迟，
  延迟还按变体拆成了 `chat_latency_primary` / `chat_latency_canary`；
- `/version` 会同时给出主版本与灰度版本（`variants.canary.version/label`），
  灰度期间用 `MINEAGENT_CANARY_VERSION=1.1.0` 标一下，验收材料里能看出「当时两边各是哪版」。

## 安全：护栏 / 脱敏 / 审计

三项都由服务设置（`settings.py`）驱动，`/ready.security` 能看到当前值：

| 开关 | 取值 | 行为 |
| --- | --- | --- |
| `MINEAGENT_GUARDRAIL` | `off` / `log` / `block` | 输入过长或像提示词注入时：只记录 / 直接 400 |
| `MINEAGENT_MAX_INPUT_CHARS` | 默认 8000，0 为不限 | 输入长度上限 |
| `MINEAGENT_PII` | `1` / `0` | 输出、日志、trace、评测候选里的手机号/身份证/邮箱/卡号替换成 `[phone]` 这类占位符 |
| `MINEAGENT_AUDIT` | `1` / `0` | 把每次 chat / resume / 被拦截写入 `outputs/audit/<日期>.jsonl` |

审计只记「谁（token 指纹 + IP + UA）在什么时候做了什么、结果如何」，**不记用户正文和原始 token**。

```bash
MINEAGENT_GUARDRAIL=block MINEAGENT_PII=1 MINEAGENT_AUDIT=1 uvicorn server:app
curl -s localhost:8000/ready | jq .security
```

## 备份 / 恢复 / 演练

```bash
python scripts/backup.py              # → backups/<项目>-<时间>.tar.gz（写完回读校验）
python scripts/backup.py --tag pre-upgrade
python scripts/restore.py backups/xxx.tar.gz   # 覆盖前会自动再备份一次现状
python scripts/drill.py               # 灾备演练：临时副本里 备份→写坏→恢复→核对，退出码 0 为通过
```

备份内容：`VERSION`、`CHANGELOG.md`、`.history/`、会话库、`.store.db`、`outputs/`、`config.json`、`agent.json`。
交付前把 `drill.py` 跑一遍，把输出截图放进验收材料；日常运维细节见 `docs/runbook.md`。

## 压测与验收

```bash
python scripts/loadtest.py --url http://127.0.0.1:8000/chat \
    --requests 200 --concurrency 8 --slo-p95-ms 30000
python scripts/loadtest.py --url http://127.0.0.1:8000/chat/stream \
    --stream --requests 50 --slo-p95-ms 8000
```

脚本输出 JSON 摘要，末行是「SLO 判定：达标/不达标」，退出码可直接进流水线。
完整验收清单在 `docs/acceptance.md`（服务能起、冒烟、SLO、可观测、安全、交付物）。

## 人工审批（挂起 / 放行）

画布上的 `human_approval` 节点会真的把整条 run 挂起，等外部放行后再继续：

> 开关：`MINEAGENT_RESUME=1` 才允许 `/resume`（默认 0 返回 501）。画布上有审批节点就把这个打开。

```bash
# ① 发起：可能直接返回结果，也可能返回待审批
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"message":"草稿","thread_id":"t1"}'
# → {"status":"interrupted","interrupt":{"question":"…"},"thread_id":"t1"}

# ② 看看停在哪
curl localhost:8000/runs/t1

# ③ 放行（decision 也可以是 reject）
curl -X POST localhost:8000/resume -H 'Content-Type: application/json' \
     -d '{"thread_id":"t1","decision":"approve"}'
```

挂起状态靠 checkpointer 保存：`MINEAGENT_SESSION_STORE=sqlite`（默认）重启也能续跑，
`none` 只在进程内有效，`postgres` 适合多副本部署。

## 节点清单

| 节点 | 类型 |
| --- | --- |
| 目标 | `input` |
| 任务拆解 | `planner` |
| 任务队列 | `memory` |
| 取队首任务 | `kv_store` |
| 执行任务 | `agent` |
| 自我批评 | `critic` |
| 结果写回记忆 | `memory` |
| 目标是否达成？ | `router` |
| 最终结果 | `output` |
