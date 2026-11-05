# second

- 画布 slug：`second`
- 框架：**AutoGen**（可运行）
- 节点数：5，连线数：5

## 这个目录是自包含的

整个文件夹可以单独拷走、单独部署，不依赖 MineAgent 平台：
`runtime_sdk.py`、`runner.py` 都已经放在目录里，只用 Python 标准库；
只要有 Python 3.9+，换一台机器也能直接跑。

- 不要把 `nodes/` 或 `runtime_sdk.py` 拆出去单独运行
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
agent.json       # 画布图定义（保存时自动覆盖）
agent.py         # 框架原生入口
runner.py        # 内置图执行器（任何画布都能跑）
runtime_sdk.py   # 节点可用的全部能力（ctx.*）
run.ps1          # Windows 一键运行
run.sh           # macOS / Linux 一键运行
nodes/           # 每个节点一个 .py，网页里编辑的就是这里
skills/          # skill.md 技能文件
config.json      # 模型配置（不含 key）
outputs/         # 运行产物
.env.example     # 复制成 .env 填 key，脱离网页独立运行用
```

## 节点清单

| 节点 | 类型 | 文件 |
| --- | --- | --- |
| 输入 | `input` | `nodes/input_1.py` |
| 生成 | `agent` | `nodes/agent_2.py` |
| 评审 | `critic` | `nodes/critic_3.py` |
| 还需要改吗 | `condition` | `nodes/condition_4.py` |
| 输出 | `output` | `nodes/output_5.py` |
