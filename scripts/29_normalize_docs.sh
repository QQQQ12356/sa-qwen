#!/usr/bin/env bash
# 把训练轨迹里的检索文档重写为统一格式（Doc N + 标题行 + 去重正文，无 ID/Score）。
# 训练与评测共用同一渲染函数，保证模型看到的 <information> 块逐字一致。
# 输出到新文件（不覆盖输入，可重复执行）；语料无需重建。
#
# 用法：
#   bash scripts/29_normalize_docs.sh                       # Search-R1 轨迹（默认）
#   INPUT_PATH=... OUTPUT_PATH=... bash scripts/29_normalize_docs.sh
#   TOKENIZER=Qwen/Qwen3.5-2B bash scripts/29_normalize_docs.sh   # 附带 token 统计
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)

INPUT_PATH=${INPUT_PATH:-outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-outputs/data/searchr1/qwen3-4b-instruct-sft-normalized.jsonl}
TOKENIZER=${TOKENIZER:-}
LIMIT=${LIMIT:-}

mkdir -p "$(dirname "$OUTPUT_PATH")"

if [ ! -f "$INPUT_PATH" ]; then
  echo "[normalize-docs] 输入文件不存在: $INPUT_PATH" >&2
  exit 1
fi

ARGS=()
if [ -n "$TOKENIZER" ]; then ARGS+=(--tokenizer "$TOKENIZER"); fi
if [ -n "$LIMIT" ]; then ARGS+=(--limit "$LIMIT"); fi

echo "[normalize-docs] $INPUT_PATH -> $OUTPUT_PATH"
"$PYTHON_BIN" -u -m search_comp.data.normalize_docs \
    --input_path "$INPUT_PATH" \
    --output_path "$OUTPUT_PATH" \
    "${ARGS[@]}"
echo "[normalize-docs] 完成。训练时请指向新文件："
echo "    --set train_data_path=$OUTPUT_PATH"
