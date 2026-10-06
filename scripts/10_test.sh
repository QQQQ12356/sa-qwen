#!/usr/bin/env bash
# 默认运行不下载模型的单元测试；FULL_MODEL_TEST=1 时追加真实 Qwen3.5 Beacon 验证。
set -euo pipefail
source "$(dirname "$0")/common.sh"

PYTHON_BIN=$(resolve_python)
LOG_PATH=${LOG_PATH:-$(new_log_path tests)}

run_logged "$LOG_PATH" "$PYTHON_BIN" -u -m pytest tests -q "$@"

if [[ ${FULL_MODEL_TEST:-0} == 1 ]]; then
  LOG_PATH=${MODEL_LOG_PATH:-$(new_log_path beacon_model_verify)} \
    bash scripts/verify_beacon.sh
fi
