#!/usr/bin/env bash
# 评测原生无训练模型直接回答问题的能力（EM/F1）。
# 不加载语料/检索器/Beacon，只输入 question + 简洁作答提示，贪心生成回答。
# 产出 predictions.jsonl 与 *_metrics.json，字段与其它评测一致，便于汇总对比。
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)

MODEL_PATH=${1:-Qwen/Qwen3-4B-Thinking-2507}
MAX_QUESTIONS=${MAX_QUESTIONS:-10}
RESULT_PATH=${2:-outputs/results/untrained_direct/${MAX_QUESTIONS}qa_predictions.jsonl}
shift $(( $# >= 2 ? 2 : $# ))
DATASET_NAME=${DATASET_NAME:-hotpotqa/hotpot_qa}
DATASET_CONFIG=${DATASET_CONFIG:-distractor}

MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-10}
DO_SAMPLE=${DO_SAMPLE:-0}

mkdir -p outputs/results

SAMPLE_ARGS=()
if [ "$DO_SAMPLE" = "1" ]; then SAMPLE_ARGS=(--do_sample --temperature "${TEMPERATURE:-0.8}"); fi

EXTRA_ARGS=("$@")
if [[ ${RESUME:-0} == 1 ]]; then EXTRA_ARGS+=(--resume); fi
LOG_PATH=${LOG_PATH:-$(new_log_path direct_untrained_eval)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$PYTHON_BIN" -u -m search_comp.evaluation.direct_untrained_eval \
    --model_path "$MODEL_PATH" \
    --output_path "$RESULT_PATH" \
    --split validation \
    --dataset_name "$DATASET_NAME" --dataset_config "$DATASET_CONFIG" \
    --max_questions "$MAX_QUESTIONS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    "${SAMPLE_ARGS[@]}" "${EXTRA_ARGS[@]}"
echo "[direct-eval] 完成 -> $RESULT_PATH"