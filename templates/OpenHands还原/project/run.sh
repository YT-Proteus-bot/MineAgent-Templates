#!/usr/bin/env bash
# 一键运行： ./run.sh "你的任务"
#   ./run.sh --install          安装 requirements.txt 里的依赖
#   ./run.sh --framework "..."  走框架原生入口
set -e
cd "$(dirname "$0")"
PY=python3
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

if [ "$1" = "--install" ]; then
  "$PY" -m pip install -r requirements.txt
  exit 0
fi

ENTRY=runner.py
if [ "$1" = "--framework" ]; then ENTRY=agent.py; shift; fi

TEXT="${1:-}"
[ -z "$TEXT" ] && TEXT="${MINEAGENT_INPUT:-}"
echo ">>> 用 $ENTRY 开始运行"
exec "$PY" "$ENTRY" --input "$TEXT"
