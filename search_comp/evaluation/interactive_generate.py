"""交互式搜索 Agent 推理。

贴近 Search-R1 原文的交互式搜索：

1. 模型生成 ``<think>...</think><search>query</search>``（遇到 ``</search>`` 停止）。
2. 提取 query，用 BM25 在线检索 top-k 文档。
3. 把 ``<information>`` 文档块追加到上下文，并记录其 token 区间为 beacon 压缩区。
4. 继续生成；最多 ``max_turns`` 轮，直到模型输出 ``</answer>`` 或耗尽轮数。

多轮生成的每个 ``<information>`` 块都是独立的 beacon 压缩区，原始文档 K/V
不参与后续解码，只保留 beacon 软提示。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from ..data.retrieval import BM25Retriever, format_docs_as_reference
from ..data.trajectory import truncate_docs_by_tokens
from ..data.trajectory import (
    INFO_PREFIX,
    INFO_SUFFIX,
    build_search_chat_prompt,
    extract_search_query,
    split_document_regions,
)
from ..models.model_loader import load_model, load_tokenizer
from .budget import MAX_NEW_TOKENS_PER_TURN, MAX_TURNS
from .em_f1 import answer_or_placeholder


def run_interactive_agent(
    model,
    tokenizer: PreTrainedTokenizer,
    retriever: BM25Retriever,
    question: str,
    max_turns: int = MAX_TURNS,
    topk: int = 3,
    max_docs_tokens: int = 1024,
    max_new_tokens_per_turn: int = MAX_NEW_TOKENS_PER_TURN,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, Any]:
    """运行单个问题的交互式搜索。

    Args:
        model: beacon 模型。
        tokenizer: tokenizer。
        retriever: BM25 检索器。
        question: 问题文本。
        max_turns: 最大搜索轮数。
        topk: 每轮检索文档数。
        max_docs_tokens: 每轮 docs 文本最大 token 数（截断）。
        max_new_tokens_per_turn: 每轮生成的最大新 token 数。
        device: 推理设备。

    Returns:
        ``{prediction, turns, queries, output}``。
    """
    question = str(question).strip()
    if not question.endswith("?"):
        question += "?"

    # 压缩区布局必须与训练一致：训练开了子文档切分，推理也要切。
    # 只认真正的 bool——测试里的 Mock 模型会让 getattr 返回真值对象。
    _split_flag = getattr(getattr(model, "beacon_config", None), "beacon_doc_region_split", False)
    doc_region_split = _split_flag if isinstance(_split_flag, bool) else False
    chat_prefix_text = build_search_chat_prompt(question, add_generation_prompt=True)
    context_ids = tokenizer(chat_prefix_text, add_special_tokens=False).input_ids
    regions: List[tuple] = []
    queries: List[str] = []
    turns_taken = 0

    model.to(device).eval()
    for _ in range(max_turns):
        input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones(
            1, len(context_ids), dtype=torch.long, device=device
        )

        gen_ids = model.beacon_generate(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            regions=regions,
            max_new_tokens=max_new_tokens_per_turn,
            stop_texts=["</search>", "</answer>"],
            tokenizer=tokenizer,
        )
        gen_tokens = gen_ids[0].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)
        context_ids = context_ids + gen_tokens

        if "</search>" in gen_text:
            query = extract_search_query(gen_text)
            if not query:
                break
            queries.append(query)
            turns_taken += 1

            # 在线检索并追加 <information> 块
            retrieved = retriever.retrieve(query, topk=topk)
            docs_text = format_docs_as_reference(retrieved)
            docs_text = truncate_docs_by_tokens(docs_text, tokenizer, max_docs_tokens)

            info_prefix_ids = tokenizer(INFO_PREFIX, add_special_tokens=False).input_ids
            docs_ids = tokenizer(docs_text, add_special_tokens=False).input_ids
            info_suffix_ids = tokenizer(INFO_SUFFIX, add_special_tokens=False).input_ids
            docs_start = len(context_ids) + len(info_prefix_ids)
            context_ids = context_ids + info_prefix_ids + docs_ids + info_suffix_ids
            sub_regions = (
                split_document_regions(docs_text, tokenizer)
                if doc_region_split else [(0, len(docs_ids))]
            )
            regions.extend((docs_start + s, docs_start + e) for s, e in sub_regions)
        else:
            # 未触发搜索（直接回答或已耗尽生成长度）
            break

    # 解码完整 assistant 输出并抽取 <answer>
    assistant_text = tokenizer.decode(
        context_ids[
            len(tokenizer(chat_prefix_text, add_special_tokens=False).input_ids) :
        ],
        skip_special_tokens=True,
    )
    prediction = answer_or_placeholder(assistant_text)
    return {
        "prediction": prediction,
        "turns": turns_taken,
        "queries": queries,
        "output": assistant_text,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="交互式搜索 Agent 推理")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--max_turns", type=int, default=MAX_TURNS)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_docs_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens_per_turn", type=int, default=MAX_NEW_TOKENS_PER_TURN)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path)
    tokenizer = load_tokenizer(args.model_path)
    retriever = BM25Retriever(args.corpus_path)
    hp = load_dataset("hotpot_qa", "distractor", split=args.split)
    if args.max_questions:
        hp = hp.select(range(args.max_questions))

    results = []
    for ex in hp:
        r = run_interactive_agent(
            model,
            tokenizer,
            retriever,
            ex["question"],
            max_turns=args.max_turns,
            topk=args.topk,
            max_docs_tokens=args.max_docs_tokens,
            max_new_tokens_per_turn=args.max_new_tokens_per_turn,
            device=device,
        )
        r["id"] = str(ex["id"])
        r["ground_truth"] = str(ex["answer"]).strip()
        results.append(r)
        print(
            f"[interactive] turns={r['turns']} pred={r['prediction'][:30]!r} "
            f"gold={r['ground_truth'][:30]!r}"
        )

    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[interactive] 生成 {len(results)} 条 -> {args.output_path}")
