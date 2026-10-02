#!/usr/bin/env bash
# alice-memory 环境引导（零依赖：仅需 Python 3.10+ 与 stdlib sqlite3）。
set -euo pipefail
cd "$(dirname "$0")"
cmd="${1:-help}"
case "$cmd" in
  setup)  python3 -c "import yaml" 2>/dev/null || pip install -e ".[dev]";;
  test)   PYTHONPATH=src python3 -m pytest tests/ "$@";;
  shell)  PYTHONPATH=src python3;;
  *) echo "用法: ./bootstrap.sh {setup|test|shell}"; exit 2;;
esac
