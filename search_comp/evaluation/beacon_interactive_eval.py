"""Qwen3.5 Beacon 交互式 SearchAgent 评估（EM/F1）。

对训练后的 Beacon 模型跑 Search-R1 风格交互式搜索：模型用 ``beacon_generate``
分段生成 ``think → <search> → <information> → <answer>``，检索文档块在生成时被
压缩为 beacon K/V。输出 JSONL 并计算 EM/F1。

用法::

    python -m search_comp.evaluation.beacon_interactive_eval \
        --model_path outputs/models/beacon_qwen3_sft_v1/final \
        --corpus_path outputs/data/hotpotqa_corpus.jsonl \
        --output_path outputs/results/beacon_qwen3_v1/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch
from tqdm import tqdm

from ..data.trajectory import truncate_docs_by_tokens
from ..data.retrieval import BM25Retriever, format_docs_as_reference
from ..data.searchr1_dataset import align_to_eval_prompt
from ..data.trajectory import (
    INFO_PREFIX,
    INFO_SUFFIX,
    build_search_chat_prompt,
    extract_search_query,
    split_document_regions,
)
from ..models.beacon_config import BeaconConfig
from .budget import MAX_NEW_TOKENS_PER_TURN, MAX_TURNS
from .em_f1 import answer_or_placeholder, extract_answer
from .statistics import ProgressCheckpointer, summarize_results
from ..utils.runtime import append_jsonl


def load_questions_from_searchr1_jsonl(
    jsonl_path: str, max_questions: int | None = None
) -> List[Dict[str, Any]]:
    """从 Search-R1 训练 jsonl 读取题目 + 真值，供在线交互评估使用。

    题目取 ``align_to_eval_prompt`` 后首个 user 消息的裸问题；真值取末条 assistant
    消息 ``<answer>`` 内容。jsonl 无原生 id，用 ``sr-N``（N 为行号）替代。

    Args:
        jsonl_path: ``*-instruct-sft.jsonl`` 路径。
        max_questions: 采样上限（None 表示全量）。

    Returns:
        ``[{id, question, answer}, ...]``。
    """
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_questions is not None and len(rows) >= max_questions:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            msgs = align_to_eval_prompt(obj["messages"])
            question = next(
                (m["content"] for m in msgs if m.get("role") == "user"),
                "",
            )
            answer = ""
            for m in reversed(msgs):
                if m.get("role") == "assistant" and "<answer>" in m.get("content", ""):
                    answer = extract_answer(m["content"]) or ""
                    break
            rows.append(
                {"id": f"sr-{idx}", "question": str(question).strip(), "answer": answer}
            )
    return rows


def run_beacon_agent(model, tokenizer, retriever, question, max_turns=MAX_TURNS, topk=3,
                     max_docs_tokens=1024,
                     max_new_tokens_per_turn=MAX_NEW_TOKENS_PER_TURN) -> Dict[str, Any]:
    question = str(question).strip()
    if not question.endswith("?"):
        question += "?"

    chat_prefix = build_search_chat_prompt(question, add_generation_prompt=True)
    context_ids = tokenizer(chat_prefix, add_special_tokens=False).input_ids
    pending_ids = context_ids
    pending_regions: List[tuple] = []
    reuse_cache = False
    regions: List[tuple] = []
    queries: List[str] = []
    turns = 0

    device = next(model.parameters()).device
    # 压缩区布局必须与训练一致：训练开了子文档切分，推理也要切。
    # 只认真正的 bool——测试里的 Mock 模型会让 getattr 返回真值对象。
    _split_flag = getattr(getattr(model, "beacon_config", None), "beacon_doc_region_split", False)
    doc_region_split = _split_flag if isinstance(_split_flag, bool) else False
    question_kwargs = {}
    if getattr(getattr(model, "beacon_config", None), "beacon_question_memory_v1", False):
        question_kwargs["question_input_ids"] = torch.tensor(
            [tokenizer(question, add_special_tokens=False).input_ids], dtype=torch.long, device=device,
        )
    for _ in range(max_turns):
        ids = torch.tensor([pending_ids], dtype=torch.long, device=device)
        attn = torch.ones(1, len(pending_ids), dtype=torch.long, device=device)
        gen_ids = model.beacon_generate(
            input_ids=ids, attention_mask=attn, regions=pending_regions,
            max_new_tokens=max_new_tokens_per_turn,
            stop_texts=["</search>", "</answer>"], tokenizer=tokenizer,
            reuse_cache=reuse_cache,
            **question_kwargs,
        )
        gen_tokens = gen_ids[0].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        context_ids = context_ids + gen_tokens

        if "</search>" in gen_text:
            query = extract_search_query(gen_text)
            if not query:
                break
            queries.append(query)
            turns += 1
            retrieved = retriever.retrieve(query, topk=topk)
            docs_text = truncate_docs_by_tokens(
                format_docs_as_reference(retrieved), tokenizer, max_docs_tokens
            )
            ip = tokenizer(INFO_PREFIX, add_special_tokens=False).input_ids
            di = tokenizer(docs_text, add_special_tokens=False).input_ids
            isuf = tokenizer(INFO_SUFFIX, add_special_tokens=False).input_ids
            docs_start = len(context_ids) + len(ip)
            context_ids = context_ids + ip + di + isuf
            sub_regions = (
                split_document_regions(docs_text, tokenizer)
                if doc_region_split else [(0, len(di))]
            )
            regions.extend((docs_start + s, docs_start + e) for s, e in sub_regions)
            pending_ids = ip + di + isuf
            pending_regions = [(len(ip) + s, len(ip) + e) for s, e in sub_regions]
            reuse_cache = True
        else:
            break

    assistant_text = tokenizer.decode(
        context_ids[len(tokenizer(chat_prefix, add_special_tokens=False).input_ids):],
        skip_special_tokens=True,
    )
    prediction = answer_or_placeholder(assistant_text)
    information_tokens = sum(end - start for start, end in regions)
    ratio = model.beacon_config.beacon_ratio
    beacon_tokens = sum(max(1, (end - start + ratio - 1) // ratio) for start, end in regions)
    return {
        "prediction": prediction,
        "turns": turns,
        "queries": queries,
        "output": assistant_text,
        "regions": regions,
        "context_tokens": len(context_ids),
        "information_tokens": information_tokens,
        "beacon_tokens": beacon_tokens,
        "generated_tokens": len(tokenizer(assistant_text, add_special_tokens=False).input_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 Beacon 交互式 SearchAgent 评估")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument(
        "--questions_jsonl", type=str, default=None,
        help=(
            "从 Search-R1 jsonl 读取题目+真值（NQ+HotpotQA 训练数据），替代 HF "
            "--split；命中时忽略 --dataset_name/--dataset_config/--split"
        ),
    )
    parser.add_argument("--dataset_name", type=str, default="hotpot_qa")
    parser.add_argument("--dataset_config", type=str, default="distractor")
    parser.add_argument("--max_questions", type=int, default=100)
    parser.add_argument(
        "--max_turns", type=int, default=MAX_TURNS,
        help=f"最大轮数（默认 {MAX_TURNS}：训练轨迹最多 3 次 search + 1 次 answer）",
    )
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_docs_tokens", type=int, default=1024)
    parser.add_argument(
        "--max_new_tokens_per_turn", type=int, default=MAX_NEW_TOKENS_PER_TURN,
        help=(
            f"每轮生成上限（默认 {MAX_NEW_TOKENS_PER_TURN}：覆盖训练分布 p99≈580）；"
            "过小会让模型在吐出 </search>/</answer> 前被截断"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true",
        help="保留已有 JSONL，并跳过其中已完成的 id",
    )
    args = parser.parse_args()

    from ..models.model_loader import load_beacon_model
    from ..milestones.qwen35_text import load_text_tokenizer

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    tokenizer = load_text_tokenizer(args.model_path)
    model = load_beacon_model(args.model_path)  # 从保存的 config 读取 beacon 字段
    model.eval()
    # 打印 checkpoint 里**实际生效**的压缩区布局（不是命令行传的），
    # 便于一眼确认评测的分窗方式与训练一致。
    print(f"[beacon-eval] 压缩区布局: {model.beacon_config.describe_layout()}", flush=True)

    retriever = BM25Retriever(args.corpus_path)
    if args.questions_jsonl:
        hp = load_questions_from_searchr1_jsonl(
            args.questions_jsonl, max_questions=args.max_questions
        )
        print(f"[beacon-eval] 题目源: Search-R1 jsonl，共 {len(hp)} 题", flush=True)
    else:
        from datasets import load_dataset

        hp = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
        if args.max_questions:
            hp = hp.select(range(args.max_questions))

    completed = {}
    if args.resume and os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as existing_file:
            for line in existing_file:
                if line.strip():
                    row = json.loads(line)
                    completed[str(row.get("id"))] = row
    elif os.path.exists(args.output_path):
        os.remove(args.output_path)
    print(f"[beacon-eval] 总题数={len(hp)} 已完成={len(completed)}", flush=True)
    results = list(completed.values())
    mp = os.path.splitext(args.output_path)[0] + "_metrics.json"
    checkpointer = ProgressCheckpointer(len(hp), mp, config=vars(args))
    pbar = tqdm(hp, desc="eval", ncols=100)
    for ex in pbar:
        example_id = str(ex["id"])
        if example_id in completed:
            continue
        started_at = time.time()
        r = run_beacon_agent(model, tokenizer, retriever, ex["question"],
                             max_turns=args.max_turns, topk=args.topk,
                             max_docs_tokens=args.max_docs_tokens,
                             max_new_tokens_per_turn=args.max_new_tokens_per_turn)
        r["id"] = example_id
        r["question"] = str(ex["question"]).strip()
        r["ground_truth"] = str(ex["answer"]).strip()
        r["latency_seconds"] = round(time.time() - started_at, 4)
        results.append(r)
        append_jsonl(args.output_path, r)
        checkpointer.update(results)
        pbar.set_postfix(turns=r["turns"], pred=r["prediction"][:30])

    m = summarize_results(results)
    checkpointer.finalize(m)
    print(
        f"[beacon-eval] 总体 EM={m['overall_em']:.3f} F1={m['overall_f1']:.3f} | "
        f"格式正确率 {m['format_correct_rate']:.0%} "
        f"(子集 EM={m['formatted_em']:.3f} F1={m['formatted_f1']:.3f}) | "
        f"平均轮次 {m['average_turns']:.2f} 分布 {m['turns_histogram']}"
    )
    print(f"[beacon-eval] 结果 -> {args.output_path}\n[beacon-eval] 指标 -> {mp}")


if __name__ == "__main__":
    main()
