#!/usr/bin/env bash
# Qwen3 / Qwen3.5 纯文本 SFT 训练（标准 Trainer + tqdm 可视化，无 Beacon、不压缩）。
# 数据：Search-R1 官方 messages 轨迹（与 beacon_qwen35_searchr1.yaml 使用
# 同一份文件，但这里不挂任何压缩参数，可直接对比压缩 vs 不压缩）。
# 训练器：search_comp.trainer.plain_sft_trainer
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)

CONFIG=${1:-configs/train/qwen3_plain_sft.yaml}
if [[ $# -gt 0 ]]; then shift; fi
SEARCHR1_DATA=${SEARCHR1_DATA:-outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl}

mkdir -p outputs/data outputs/models

echo "[plain-sft] 训练数据: $SEARCHR1_DATA"
if [ ! -f "$SEARCHR1_DATA" ]; then
  cat >&2 <<EOF
[plain-sft] 未找到 $SEARCHR1_DATA（Search-R1 SFT 轨迹约 60MB，不在本仓库内）。
[plain-sft] 先下载：
  mkdir -p outputs/data/searchr1
  huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train \\
      --include '*instruct-sft.jsonl' --local-dir outputs/data/searchr1
[plain-sft] 或复用本地已有文件（避免重复下载）：
  mkdir -p outputs/data/searchr1
  ln -s /path/to/qwen3-4b-instruct-sft.jsonl $SEARCHR1_DATA
EOF
  exit 1
fi

echo "[plain-sft] 开始训练（标准 Trainer，可视化到终端）"
LOG_PATH=${LOG_PATH:-$(new_log_path plain_train)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$PYTHON_BIN" -u -m search_comp.trainer.plain_sft_trainer --config "$CONFIG" "$@"
echo "[plain-sft] 训练完成"
