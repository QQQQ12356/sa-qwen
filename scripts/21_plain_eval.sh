#!/usr/bin/env bash
# Qwen3.5 纯文本（无 Beacon）交互式 SearchAgent 评估（EM/F1）。
# 真实交互式 Agent：think -> <search> -> <information> -> <answer>，不做任何压缩。
# 评测入口：search_comp.evaluation.plain_interactive_eval
#
# 默认 jsonl 模式（与 scripts/23_beacon_eval.sh 一致）：对训练 Search-R1 轨迹做在线
# 检索评估，题目取 jsonl（NQ+HotpotQA），语料从 jsonl 的 <information> 抽文档。
# 注意默认即 jsonl，无法用空值关闭；要跑 HotpotQA-validation 老路径，参考
# scripts/28_plain_untrained_interactive_eval.sh 的非 jsonl 写法。
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)
MAX_QUESTIONS=${MAX_QUESTIONS:-1000}
MODEL_PATH=${1:-outputs/models/qwen35-4B_plain_sft_v1/final_merged}
RESULT_PATH=${2:-outputs/results/qwen35-4B_plain_sft_v1/${MAX_QUESTIONS}qa_predictions.jsonl}
shift $(( $# >= 2 ? 2 : $# ))
DATASET_NAME=${DATASET_NAME:-hotpotqa/hotpot_qa}
DATASET_CONFIG=${DATASET_CONFIG:-distractor}
QUESTIONS_JSONL=${QUESTIONS_JSONL:-outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl}
CORPUS_PER_SPLIT=${CORPUS_PER_SPLIT:-7405}

# jsonl 模式（对训练 Search-R1 轨迹做在线检索评估）：
# 题目从 jsonl 取（NQ+HotpotQA），语料也直接从 jsonl 的 <information> 抽文档（含 NQ）。
# 需先按模式定 CORPUS 默认值，避免被下方非 jsonl 的默认值抢先占用。
CORPUS=${CORPUS:-}
if [ -n "$QUESTIONS_JSONL" ]; then
  CORPUS=${CORPUS:-outputs/data/searchr1_corpus.jsonl}
else
  CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
fi
MAX_TURNS=${MAX_TURNS:-4}
TOP_K=${TOP_K:-3}
MAX_DOCS_TOKENS=${MAX_DOCS_TOKENS:-1024}
MAX_NEW_TOKENS_PER_TURN=${MAX_NEW_TOKENS_PER_TURN:-768}
DO_SAMPLE=${DO_SAMPLE:-0}

mkdir -p outputs/data outputs/results

# searchr1 模式下训练脚本不再构建检索语料，这里补上兜底。
if [ ! -f "$CORPUS" ]; then
  echo "[plain-eval] 语料库（不存在则构建）: $CORPUS"
  if [ -n "$QUESTIONS_JSONL" ]; then
    "$PYTHON_BIN" -u -m search_comp.data.build_corpus_searchr1 \
        --jsonl_path "$QUESTIONS_JSONL" --output_path "$CORPUS"
  else
    "$PYTHON_BIN" -u -m search_comp.data.build_corpus \
        --output_path "$CORPUS" --splits validation --max_per_split "$CORPUS_PER_SPLIT"
  fi
fi

SAMPLE_ARGS=()
if [ "$DO_SAMPLE" = "1" ]; then SAMPLE_ARGS=(--do_sample --temperature 0.8); fi

EXTRA_ARGS=("$@")
if [[ ${RESUME:-0} == 1 ]]; then EXTRA_ARGS+=(--resume); fi
if [ -n "$QUESTIONS_JSONL" ]; then EXTRA_ARGS+=(--questions_jsonl "$QUESTIONS_JSONL"); fi
LOG_PATH=${LOG_PATH:-$(new_log_path plain_eval)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
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
