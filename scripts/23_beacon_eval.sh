#!/usr/bin/env bash
# Qwen3 / Qwen3.5 Beacon 交互式 SearchAgent 评估（EM/F1）。
# 真实构建：语料库（不存在则构建）+ BM25 索引（BM25Retriever 内建），
# 真实交互式 Agent（think -> <search> -> <information> -> <answer>，Beacon 压缩生效）。
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)

RATIO=${RATIO:-32}
MAX_QUESTIONS=${MAX_QUESTIONS:-1000}
MODEL_PATH=${1:-outputs/models/beacon_qwen3_searchr1/final}
RESULT_PATH=${2:-outputs/results/beacon_qwen3_searchr1/predictions-${MAX_QUESTIONS}qa.jsonl}
shift $(( $# >= 2 ? 2 : $# ))
DATASET_NAME=${DATASET_NAME:-hotpotqa/hotpot_qa}
DATASET_CONFIG=${DATASET_CONFIG:-distractor}
SPLIT=${SPLIT:-validation}
QUESTIONS_JSONL=${QUESTIONS_JSONL:-outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl}
CORPUS_PER_SPLIT=${CORPUS_PER_SPLIT:-7405}
# 生成预算与 Search-R1 训练轨迹分布对齐（见 search_comp/evaluation/budget.py）：
# 训练轨迹最多 3 次 search + 1 次 answer，assistant 段 p99≈580 token。
MAX_TURNS=${MAX_TURNS:-4}
TOP_K=${TOP_K:-3}
MAX_NEW_TOKENS_PER_TURN=${MAX_NEW_TOKENS_PER_TURN:-768}

# jsonl 模式（对训练 Search-R1 轨迹做在线检索评估）：
# 题目从 jsonl 取（NQ+HotpotQA），语料也直接从 jsonl 的 <information> 抽文档（含 NQ）。
# 需先按模式定 CORPUS 默认值，避免被下方非 jsonl 的默认值抢先占用。
CORPUS=${CORPUS:-}
if [ -n "$QUESTIONS_JSONL" ]; then
  CORPUS=${CORPUS:-outputs/data/searchr1_corpus.jsonl}
else
  CORPUS=${CORPUS:-outputs/data/hotpotqa_corpus.jsonl}
fi

mkdir -p outputs/data outputs/results

echo "[beacon-eval] 语料库（不存在则构建）: $CORPUS"
if [ ! -f "$CORPUS" ]; then
  if [ -n "$QUESTIONS_JSONL" ]; then
    "$PYTHON_BIN" -u -m search_comp.data.build_corpus_searchr1 \
        --jsonl_path "$QUESTIONS_JSONL" --output_path "$CORPUS"
  else
    "$PYTHON_BIN" -u -m search_comp.data.build_corpus \
        --output_path "$CORPUS" --splits validation --max_per_split "$CORPUS_PER_SPLIT"
  fi
fi
echo "[beacon-eval] BM25 索引 + 交互式 Agent 评估（Beacon 压缩生效）"

EXTRA_ARGS=("$@")
if [[ ${RESUME:-0} == 1 ]]; then EXTRA_ARGS+=(--resume); fi
if [ -n "$QUESTIONS_JSONL" ]; then EXTRA_ARGS+=(--questions_jsonl "$QUESTIONS_JSONL"); fi
LOG_PATH=${LOG_PATH:-$(new_log_path beacon_eval)}
run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
  "$PYTHON_BIN" -u -m search_comp.evaluation.beacon_interactive_eval \
  --model_path "$MODEL_PATH" --corpus_path "$CORPUS" --output_path "$RESULT_PATH" \
  --split "$SPLIT" --dataset_name "$DATASET_NAME" --dataset_config "$DATASET_CONFIG" \
  --max_questions "$MAX_QUESTIONS" --max_turns "$MAX_TURNS" --topk "$TOP_K" \
  --max_docs_tokens "${MAX_DOCS_TOKENS:-1024}" \
  --max_new_tokens_per_turn "$MAX_NEW_TOKENS_PER_TURN" "${EXTRA_ARGS[@]}"
echo "[beacon-eval] 完成 -> $RESULT_PATH"
