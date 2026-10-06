#!/usr/bin/env bash
# Qwen3.5 未训练原始模型 纯文本（无 Beacon）交互式 SearchAgent 评估（EM/F1）。
# 与 scripts/21_plain_eval.sh 保持一致：真实交互式 Agent think -> <search> -> <information> -> <answer>，
# 不做任何压缩。唯一区别是模型为未训练的原始模型 Qwen/Qwen3.5-2B。
# 评测入口：search_comp.evaluation.plain_interactive_eval
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)

MODEL_PATH=${1:-Qwen/Qwen3-4B-Thinking-2507}
MAX_QUESTIONS=${MAX_QUESTIONS:-10}
RESULT_PATH=${2:-outputs/results/untrained_plain/${MAX_QUESTIONS}qa_predictions.jsonl}
shift $(( $# >= 2 ? 2 : $# ))
DATASET_NAME=${DATASET_NAME:-hotpotqa/hotpot_qa}
DATASET_CONFIG=${DATASET_CONFIG:-distractor}
CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
CORPUS_PER_SPLIT=${CORPUS_PER_SPLIT:-7405}

MAX_TURNS=${MAX_TURNS:-4}
TOP_K=${TOP_K:-3}
MAX_DOCS_TOKENS=${MAX_DOCS_TOKENS:-1024}
MAX_NEW_TOKENS_PER_TURN=${MAX_NEW_TOKENS_PER_TURN:-768}
DO_SAMPLE=${DO_SAMPLE:-0}

mkdir -p outputs/data outputs/results

# searchr1 模式下训练脚本不再构建检索语料，这里补上兜底。
if [ ! -f "$CORPUS" ]; then
  echo "[plain-eval] 语料库（不存在则构建）: $CORPUS"
  "$PYTHON_BIN" -u -m search_comp.data.build_corpus \
      --output_path "$CORPUS" --splits validation --max_per_split "$CORPUS_PER_SPLIT"
fi

SAMPLE_ARGS=()
if [ "$DO_SAMPLE" = "1" ]; then SAMPLE_ARGS=(--do_sample --temperature 0.8); fi

EXTRA_ARGS=("$@")
if [[ ${RESUME:-0} == 1 ]]; then EXTRA_ARGS+=(--resume); fi
LOG_PATH=${LOG_PATH:-$(new_log_path plain_untrained_eval)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    "$PYTHON_BIN" -u -m search_comp.evaluation.plain_interactive_eval \
    --model_path "$MODEL_PATH" \
    --corpus_path "$CORPUS" \
    --output_path "$RESULT_PATH" \
    --split validation \
    --dataset_name "$DATASET_NAME" --dataset_config "$DATASET_CONFIG" \
    --max_questions "$MAX_QUESTIONS" \
    --max_turns "$MAX_TURNS" --topk "$TOP_K" \
    --max_docs_tokens "$MAX_DOCS_TOKENS" \
    --max_new_tokens_per_turn "$MAX_NEW_TOKENS_PER_TURN" \
    "${SAMPLE_ARGS[@]}" "${EXTRA_ARGS[@]}"
echo "[plain-eval] 完成 -> $RESULT_PATH"