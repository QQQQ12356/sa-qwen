"""实验：把 SearchAgent 搜索指令从 user 角色搬进 system prompt。

对照组 ``searchagent_probe`` 里的做法是把整条 Search-R1 指令（含 "Question:"
）塞进 **user** 消息；本实验把指令段独立成 **system** prompt，user 只放问题，
验证 Qwen3.5-2B 在该排版下是否仍能正确跟随指令生成 ``<search>query</search>``。

新增文件，不改动既有代码：共享常量/检索/解析逻辑均从既有模块 import，
仅 prompt 排版与评估主循环**局部复制**到这里（与 searchagent_probe 对齐），
方便独立运行、互不污染。

用法：
  python -m search_comp.milestones.searchagent_sysprompt_probe \
      --corpus_path outputs/data/hotpotqa_corpus_small.jsonl \
      --output_path outputs/results/sysprompt_probe/predictions.jsonl \
      [--style system|user]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch

from ..data.trajectory import truncate_docs_by_tokens
from ..data.retrieval import BM25Retriever, format_docs_as_reference
from ..data.trajectory import BASE_SYSTEM_PROMPT, INFO_PREFIX, INFO_SUFFIX, extract_search_query
from ..milestones.searchagent_probe import decode_until
from .qwen35_native import load_chat_model, load_tokenizer

#: 搜索指令正文（原 SEARCH_INSTRUCTION 中 "Question:" 之前的描述段）。
#: v2：明确要求 query 直接写在 <search> 标签内、不得带 "query:" 前缀或留空，
#: 以抑制基础模型"复读模板"导致的 `<search>` 双标签 / 空标签伪影。
SEARCH_INSTRUCTION_TEXT = (
    "Answer the given question. Every time you get new information, you must "
    "first conduct reasoning inside <thinking> and </thinking>. After reasoning, "
    "if you find you lack some knowledge, you may call a search engine. To do so, "
    "write only the search query directly between the tags and nothing else, for "
    "example <search>Scott Derrickson nationality</search>. Do not put any label "
    "such as \"query:\" inside, do not add empty <search></search>, and do not "
    "insert extra tags. The search engine returns the top results between "
    "<information> and </information>. You can search as many times as you want. "
    "If you find no further external knowledge needed, you can directly provide "
    "the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>."
)

#: 实验排版：指令入 system，user 仅问题。
SYSTEM_SEARCH_PROMPT = SEARCH_INSTRUCTION_TEXT


#: v3：正面例句驱动、最小禁止语（对比 v2 的多从句否定句式：v2 修复了格式但
#: 副作用是让基础模型几乎不再发起搜索）。用具体 query 例句引导格式。
SEARCH_INSTRUCTION_TEXT_V3 = (
    "Answer the given question. First, reason inside <thinking> and </thinking> "
    "every time you get new information. If you need more knowledge, search with "
    "a short factual query placed directly in <search> and </search>, like "
    "<search>Scott Derrickson nationality</search>. The results come back in "
    "<information> and </information>. You can search as many times as needed. "
    "When you have enough, give the final answer directly in <answer> and "
    "</answer>, like <answer> Beijing </answer>."
)

#: v4：在 v3 基础上，用一段紧凑的"整轮回合计"工作示例 + 显式"务必以 <answer>
#: 收尾"指令，解决基础模型"只搜不答"（v3 在 50 题里仅 24% 收敛到 </answer>）。
SEARCH_INSTRUCTION_TEXT_V4 = (
    "Answer the given question. Reasoning goes inside <thinking> and </thinking>. "
    "If you need facts, search with a short factual query placed directly between "
    "<search> and </search>, like <search>Scott Derrickson nationality</search>. "
    "Search results come back between <information> and </information>. After "
    "reading them, decide the answer, then STOP and finish with your final answer "
    "between <answer> and </answer>, like <answer> Beijing </answer>. "
    "Always end your reply with an <answer>.</answer> pair. Full worked example of "
    "one turn: Question: How tall is the Eiffel Tower? <thinking>I need the exact "
    "height of the Eiffel Tower.</thinking> <search>Eiffel Tower height</search> "
    "<information>The Eiffel Tower is 330 meters tall.</information> "
    "<thinking>The Eiffel Tower is 330 m tall.</thinking> <answer> 330 meters </answer>."
)


def build_sysprompt_chat_prompt(question: str, add_generation_prompt: bool = True) -> str:
    """system=搜索指令，user=仅问题 的 ChatML 提示。"""
    text = f"<|im_start|>system\n{SYSTEM_SEARCH_PROMPT}<|im_end|>\n"
    text += f"<|im_start|>user\nQuestion: {question}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def build_sysprompt_chat_prompt_v3(question: str, add_generation_prompt: bool = True) -> str:
    """system=搜索指令 v3（正面例句驱动），user=仅问题。"""
    text = f"<|im_start|>system\n{SEARCH_INSTRUCTION_TEXT_V3}<|im_end|>\n"
    text += f"<|im_start|>user\nQuestion: {question}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def build_sysprompt_chat_prompt_v4(question: str, add_generation_prompt: bool = True) -> str:
    """system=搜索指令 v4（含整轮回合计示例，引导收敛 <answer>）。"""
    text = f"<|im_start|>system\n{SEARCH_INSTRUCTION_TEXT_V4}<|im_end|>\n"
    text += f"<|im_start|>user\nQuestion: {question}<|im_end|>\n"
    if add_generation_prompt:
        text += "<|im_start|>assistant\n"
    return text


def _build_chat_prompt(style: str, question: str) -> str:
    """按排版构造 ChatML 提示。``user`` 风格对应原始对照组。"""
    if style == "system":
        return build_sysprompt_chat_prompt(question)
    if style == "system3":
        return build_sysprompt_chat_prompt_v3(question)
    if style == "system4":
        return build_sysprompt_chat_prompt_v4(question)
    # user 风格（对照）：指令仍放 user，仅用于对比
    text = f"<|im_start|>system\n{BASE_SYSTEM_PROMPT}<|im_end|>\n"
    text += f"<|im_start|>user\n{SEARCH_INSTRUCTION_TEXT} Question: {question}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def run_sysprompt_probe(
    model,
    tokenizer,
    retriever: BM25Retriever,
    question: str,
    style: str = "system",
    max_turns: int = 3,
    topk: int = 3,
    max_docs_tokens: int = 1024,
    max_new_tokens_per_turn: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    verbosity: int = 2,
) -> Dict[str, Any]:
    """单问题 SearchAgent 检索探针，prompt 排版由 ``style`` 决定。

    Returns:
        ``{prediction, turns, queries, output, turn_texts}``。
    """
    question = str(question).strip()
    if not question.endswith("?"):
        question += "?"

    prompt_text = _build_chat_prompt(style, question)
    context_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    context_ids = torch.tensor([context_ids], dtype=torch.long, device=next(model.parameters()).device)

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
            context_ids = torch.cat(
                [
                    context_ids,
                    torch.tensor([info_prefix_ids], dtype=torch.long, device=context_ids.device),
                    torch.tensor([docs_ids], dtype=torch.long, device=context_ids.device),
                    torch.tensor([info_suffix_ids], dtype=torch.long, device=context_ids.device),
                ],
                dim=1,
            )
            if verbosity >= 2:
                print(f"      [probe] 注入 <information>（{len(docs_ids)} tokens）")
        else:
            if verbosity >= 1 and "</answer>" not in gen_text:
                print("      [probe] 未触发搜索也未给出 </answer>，结束")
            break

    assistant_text = tokenizer.decode(
        context_ids[0, len(tokenizer(prompt_text, add_special_tokens=False).input_ids):].tolist(),
        skip_special_tokens=True,
    )
    from ..evaluation.em_f1 import extract_answer

    prediction = extract_answer(assistant_text) or assistant_text.strip()
    return {
        "prediction": prediction,
        "turns": turns_taken,
        "queries": queries,
        "output": assistant_text,
        "turn_texts": turn_texts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3.5 SearchAgent 指令入 system 探针")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_questions", type=int, default=10)
    parser.add_argument("--style", type=str, choices=["system", "system3", "system4", "user"], default="system")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_docs_tokens", type=int, default=1024)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="每轮生成上限——基础模型常在 256 token 内写不完 "
                             "think→<answer> 就截断，调大有助于收敛 </answer>")
    parser.add_argument("--verbose", type=int, default=0,
                        help="0=简洁；1=每轮摘要+query；2=完整解码文本")
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

    print("=" * 68)
    print(f"SearchAgent 指令入 system 探针 | style={args.style} | 模型={args.model_path}")
    print(f"语料={args.corpus_path} | 问题数={len(hp)} | topk={args.topk} | max_turns={args.max_turns}")
    print("=" * 68)

    results = []
    t0 = time.time()
    for i, ex in enumerate(hp):
        q = str(ex["question"])
        gold = str(ex["answer"]).strip()
        print(f"\n[{i + 1}/{len(hp)}] 问题: {q}")
        print(f"  期望答案: {gold}")
        r = run_sysprompt_probe(
            model, tokenizer, retriever, q,
            style=args.style, max_turns=args.max_turns, topk=args.topk,
            max_docs_tokens=args.max_docs_tokens,
            max_new_tokens_per_turn=args.max_new_tokens,
            do_sample=args.do_sample, temperature=args.temperature,
            verbosity=args.verbose,
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
    all_queries = [q for r in results for q in r["queries"]]
    n_q = len(all_queries)
    n_prefix = sum(1 for q in all_queries if q.strip().startswith("query:") or "\nquery:" in q)
    n_bad = sum(1 for q in all_queries if not q.strip())
    n_answered = sum(1 for r in results if "</answer>" in r["output"])
    print("\n===================== 汇总 =====================")
    print(f"总问题: {len(results)} | 触发搜索: {n_search} ({n_search / max(len(results), 1):.0%}) | 多轮搜索(≥2): {n_multi}")
    print(f"平均搜索轮次: {sum(r['turns'] for r in results) / max(len(results), 1):.2f}")
    print(f"格式体检: 总query={n_q} | 含前缀'query:' {n_prefix} ({n_prefix / max(n_q, 1):.0%}) | 空query {n_bad} | 给出</answer> {n_answered}")
    print(f"耗时: {elapsed:.1f}s")

    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n轨迹已写入 -> {args.output_path}")


if __name__ == "__main__":
    main()
