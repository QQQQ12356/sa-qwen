"""Milestone-1：Qwen3.5-2B SearchAgent 检索能力探针。

在**不加 Beacon 压缩**的前提下，用原生 Qwen3.5 基础模型 + 简单 BM25 索引跑
Search-R1 风格的检索 Agent，验证基础大模型是否支持检索行为（自生成
``<search>query</search>`` → 观察 ``<information>`` → 再推理 → ``<answer>``）：

1. 将 SEARCH_INSTRUCTION 放入 system，首个 user 消息只保留问题文本。
2. 逐 token 自回归解码，遇到 ``</search>`` 或 ``</answer>`` 即停止本轮。
3. 若本轮输出含 ``<search>``，提取 query，BM25 在线检索 top-k 文档。
4. 把 ``<information>`` 文档块追加到上下文（token 区间记为压缩区）。
5. 继续生成，直到 ``</answer>`` 或达到 ``max_turns``。

把每条轨迹完整打印到终端，并存成 JSONL 便于查看模型是否学会了搜索。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from ..data.trajectory import truncate_docs_by_tokens
from ..data.retrieval import BM25Retriever, format_docs_as_reference
from ..data.trajectory import (
    INFO_PREFIX,
    INFO_SUFFIX,
    build_search_chat_prompt,
    extract_search_query,
)
from ..evaluation.budget import MAX_NEW_TOKENS_PER_TURN, MAX_TURNS
from .qwen35_native import generate_text, load_chat_model, load_tokenizer


def decode_until(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    stop_texts: List[str],
    eos_token_id: int,
    max_new_tokens: int = MAX_NEW_TOKENS_PER_TURN,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> List[int]:
    """用 ``model.generate``（KV cache）逐步解码，直到命中 ``stop_texts`` 或 eos。

    依赖 transformers 5.x 的 ``StoppingCriteria`` 语义：``generate`` 每次只
    产生一个新增 token 后立即检查回调，命中 stop 字符串即停止并保留该 token，
    等效于 beacon_generate 的分段生成语义，但享受 KV cache 加速。

    Args:
        model: Qwen3.5 原生模型。
        input_ids: ``(1, seq_len)`` 起始上下文。
        stop_texts: 文本级停止序列（如 ``["</search>", "</answer>"]``）。
        eos_token_id: eos token id。
        max_new_tokens: 最大新增 token 数。
        do_sample / temperature / top_p: 采样参数。

    Returns:
        生成的 token id 列表（含触发停止序列的 token）。
    """
    from transformers.generation import StoppingCriteria, StoppingCriteriaList

    class _SC(StoppingCriteria):
        def __init__(self, tok, stops, bl):
            self.tok = tok
            self.stops = stops
            self.bl = bl

        def __call__(self, input_ids, scores, **kw):
            text = self.tok.decode(
                input_ids[0][self.bl:].tolist(), skip_special_tokens=False
            )
            return any(s in text for s in self.stops)

    sc = _SC(tokenizer, stop_texts, input_ids.shape[-1])
    gen_cfg = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_id,
        stopping_criteria=StoppingCriteriaList([sc]),
        return_dict_in_generate=False,
    )
    if do_sample:
        gen_cfg["temperature"] = temperature
        gen_cfg["top_p"] = top_p

    with torch.no_grad():
        generated = model.generate(input_ids=input_ids, **gen_cfg)
    new_tokens = generated[0][input_ids.shape[-1]:].tolist()
    return new_tokens


def run_searchagent_probe(
    model,
    tokenizer,
    retriever: BM25Retriever,
    question: str,
    max_turns: int = MAX_TURNS,
    topk: int = 3,
    max_docs_tokens: int = 1024,
    max_new_tokens_per_turn: int = MAX_NEW_TOKENS_PER_TURN,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    verbosity: int = 2,
) -> Dict[str, Any]:
    """运行单个问题的 SearchAgent 检索探针。

    Args:
        verbosity: 0=静默，1=每轮摘要，2=每轮完整解码文本。

    Returns:
        ``{prediction, turns, queries, output, turn_texts}``。
    """
    question = str(question).strip()
    if not question.endswith("?"):
        question += "?"

    chat_prefix_text = build_search_chat_prompt(question, add_generation_prompt=True)
    context_ids = tokenizer(chat_prefix_text, add_special_tokens=False).input_ids
    context_ids = torch.tensor([context_ids], dtype=torch.long, device=next(model.parameters()).device)

    regions: List[tuple] = []
    queries: List[str] = []
    turn_texts: List[str] = []
    turns_taken = 0

    for turn in range(max_turns):
        if verbosity >= 2:
            print(f"    --- 第 {turn + 1} 轮 ---")
        gen_ids = decode_until(
            model,
            tokenizer,
            context_ids,
            stop_texts=["</search>", "</answer>"],
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens_per_turn,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        turn_texts.append(gen_text)
        if verbosity >= 2:
            print(f"      <gen> {gen_text!r}")
        context_ids = torch.cat(
            [context_ids, torch.tensor([gen_ids], dtype=torch.long, device=context_ids.device)],
            dim=1,
        )

        if "</search>" in gen_text:
            query = extract_search_query(gen_text)
            if not query:
                if verbosity >= 1:
                    print("      [probe] <search> 但无有效 query，结束")
                break
            queries.append(query)
            turns_taken += 1
            if verbosity >= 2:
                print(f"      [probe] 发起检索: query={query!r}")

            retrieved = retriever.retrieve(query, topk=topk)
            docs_text = format_docs_as_reference(retrieved)
            docs_text = truncate_docs_by_tokens(docs_text, tokenizer, max_docs_tokens)

            info_prefix_ids = tokenizer(INFO_PREFIX, add_special_tokens=False).input_ids
            docs_ids = tokenizer(docs_text, add_special_tokens=False).input_ids
            info_suffix_ids = tokenizer(INFO_SUFFIX, add_special_tokens=False).input_ids
            docs_start = context_ids.shape[1] + len(info_prefix_ids)
            context_ids = torch.cat(
                [
                    context_ids,
                    torch.tensor([info_prefix_ids], dtype=torch.long, device=context_ids.device),
                    torch.tensor([docs_ids], dtype=torch.long, device=context_ids.device),
                    torch.tensor([info_suffix_ids], dtype=torch.long, device=context_ids.device),
                ],
                dim=1,
            )
            regions.append((docs_start, docs_start + len(docs_ids)))
            if verbosity >= 2:
                print(f"      [probe] 注入 <information>（{len(docs_ids)} tokens, region={docs_start}:{docs_start + len(docs_ids)}）")
        else:
            # 直接 <answer> 或耗尽生成长度，结束
            if verbosity >= 1 and "</answer>" not in gen_text:
                print("      [probe] 未触发搜索也未给出 </answer>，结束")
            break

    assistant_text = tokenizer.decode(
        context_ids[0, len(tokenizer(chat_prefix_text, add_special_tokens=False).input_ids):].tolist(),
        skip_special_tokens=True,
    )
    from ..evaluation.em_f1 import answer_or_placeholder

    # 与 beacon 评测共用同一兜底：未闭合 <answer> 时统一记 [无作答]，
    # 而不是拿整段原始输出当预测（口径一致、prediction 字段也可读）。
    prediction = answer_or_placeholder(assistant_text)
    return {
        "prediction": prediction,
        "turns": turns_taken,
        "queries": queries,
        "output": assistant_text,
        "turn_texts": turn_texts,
        "regions": regions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 SearchAgent 检索能力探针")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_questions", type=int, default=10)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=MAX_TURNS)
    parser.add_argument("--max_docs_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens_per_turn", type=int, default=MAX_NEW_TOKENS_PER_TURN)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.model_path)
    model = load_chat_model(args.model_path)
    retriever = BM25Retriever(args.corpus_path)

    from datasets import load_dataset

    hp = load_dataset("hotpot_qa", "distractor", split=args.split)
    if args.max_questions:
        hp = hp.select(range(args.max_questions))

    print(f"\n================== Qwen3.5 SearchAgent 检索能力探针 ==================")
    print(f"模型: {args.model_path} | 语料: {args.corpus_path} | 问题数: {len(hp)} | topk={args.topk} | max_turns={args.max_turns}")
    print("====================================================================\n")

    results = []
    t0 = time.time()
    for i, ex in enumerate(hp):
        q = str(ex["question"])
        gold = str(ex["answer"]).strip()
        print(f"\n[{i + 1}/{len(hp)}] 问题: {q}")
        print(f"  期望答案: {gold}")
        r = run_searchagent_probe(
            model, tokenizer, retriever, q,
            max_turns=args.max_turns, topk=args.topk,
            max_docs_tokens=args.max_docs_tokens,
            max_new_tokens_per_turn=args.max_new_tokens_per_turn,
            do_sample=args.do_sample, temperature=args.temperature,
            verbosity=2,
        )
        r["id"] = str(ex["id"])
        r["ground_truth"] = gold
        results.append(r)

        em = "✓" if r["prediction"].strip().casefold() == gold.strip().casefold() else "✗"
        print(f"  => turns={r['turns']} queries={r['queries']!r}")
        print(f"  => 抽取答案: {r['prediction']!r}  [gold {gold!r}]  {em}")

    elapsed = time.time() - t0
    n_search = sum(1 for r in results if r["turns"] > 0)
    n_multi = sum(1 for r in results if r["turns"] > 1)
    print("\n===================== 汇总 =====================")
    print(f"总问题: {len(results)} | 触发搜索: {n_search} ({n_search / max(len(results), 1):.0%}) | 多轮搜索(≥2): {n_multi}")
    print(f"平均搜索轮次: {sum(r['turns'] for r in results) / max(len(results), 1):.2f}")
    print(f"耗时: {elapsed:.1f}s")

    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n轨迹已写入 -> {args.output_path}")


if __name__ == "__main__":
    main()
