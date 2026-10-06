#!/usr/bin/env bash
# 汇总评测指标 JSON（*_metrics.json / tmp.json 格式）为 Excel 对比表。
# 直接修改下面的 RESULT_PATHS 与 OUTPUT_PATH 即可运行：bash scripts/26_export_excel.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

# 输入：评测脚本产出的指标 JSON，可列多行
RESULT_PATHS=(
  "tmp.json"
  "outputs/results/beacon_qwen3_searchr1_v1/ckpt1000_predictions_metrics.json"
)
# 输出：Excel 文件路径
OUTPUT_PATH="outputs/results/eval_summary.xlsx"

PYTHON_BIN=$(resolve_python)
LOG_PATH=${LOG_PATH:-$(new_log_path export_excel)}
ARGS=()
for result_path in "${RESULT_PATHS[@]}"; do
  ARGS+=(--result_path "$result_path")
done

run_logged "$LOG_PATH" "$PYTHON_BIN" -u -m search_comp.evaluation.export_excel \
    "${ARGS[@]}" --output_path "$OUTPUT_PATH"
