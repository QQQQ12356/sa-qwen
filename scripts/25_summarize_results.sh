#!/usr/bin/env bash
# 用法：bash scripts/25_summarize_results.sh OUTPUT.json RESULT1.jsonl [RESULT2.jsonl ...]
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ $# -lt 2 ]]; then
  echo "用法: $0 OUTPUT.json RESULT1.jsonl [RESULT2.jsonl ...]" >&2
  exit 2
fi

OUTPUT_PATH=$1
shift
PYTHON_BIN=$(resolve_python)
LOG_PATH=${LOG_PATH:-$(new_log_path summarize_results)}
ARGS=()
for result_path in "$@"; do
  ARGS+=(--result_path "$result_path")
done

run_logged "$LOG_PATH" "$PYTHON_BIN" -u -m search_comp.evaluation.statistics \
  "${ARGS[@]}" --output_path "$OUTPUT_PATH"
