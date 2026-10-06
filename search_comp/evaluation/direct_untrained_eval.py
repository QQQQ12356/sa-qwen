"""评测**原生无训练模型**直接回答问题的能力（EM/F1 基线）。

不加载语料/检索器/Beacon，只做最朴素的事：把 ``question`` 与一条
「请最简洁地直接作答」的提示一并拼进 ChatML，用基础模型
（默认 ``Qwen/Qwen3.5-2B``）自回归生成回答。不引入 think、不使用
``<answer>`` 标签，prediction 直接取生成的原始文本。

输出字段与现有评测脚本保持一致，便于用 ``statistics.summarize_results``
直接汇总、并与 ``export_excel`` 一起对比：

- ``id`` / ``question`` / ``ground_truth``／``prediction``／``output``／``turns``
- ``latency_seconds``／``generated_tokens``／``information_tokens``（无检索恒为 0）

**不抽取 ``<answer>``、不要求 think**：prediction 直接取模型生成的原始文本，
提示词只要求最简洁地直接作答，得到的是「原生无训练模型朴素回答能力」基线。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from tqdm import tqdm

from ..milestones.qwen35_text import load_text_causal_model, load_text_tokenizer
from ..evaluation.statistics import ProgressCheckpointer, summarize_results
from ..utils.runtime import append_jsonl

#: 空白角色 system + 最简洁直接作答的用户指令（不引入 think、不要求 <answer> 标签）。
PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Answer the following question directly and concisely. "
    "Give the shortest possible answer with no reasoning or explanation.\n"
    "Question: {question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测原生无训练模型的直接回答能力")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--dataset_name", type=str, default="hotpot_qa")
    parser.add_argument("--dataset_config", type=str, default="distractor")
    parser.add_argument("--max_questions", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", action="store_true",
        help="保留已有 JSONL，并跳过其中已完成的 id",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    tokenizer = load_text_tokenizer(args.model_path)
    model = load_text_causal_model(args.model_path)

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

    print(f"\n[direct-eval] 模型={args.model_path} 题数={len(hp)} 已完成={len(completed)}", flush=True)
    results = list(completed.values())
    metric_path = os.path.splitext(args.output_path)[0] + "_metrics.json"
    checkpointer = ProgressCheckpointer(len(hp), metric_path, config=vars(args))
    progress = tqdm(hp, desc="direct-eval", ncols=100)
    for ex in progress:
        example_id = str(ex["id"])
        if example_id in completed:
            continue
        question = str(ex["question"]).strip()
        prompt = PROMPT_TEMPLATE.format(question=question)
        prefix_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        context_ids = torch.tensor([prefix_ids], dtype=torch.long, device=next(model.parameters()).device)

        started_at = time.time()
        with torch.no_grad():
            generated = model.generate(
                input_ids=context_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - started_at
        new_tokens = generated[0][context_ids.shape[-1]:]
        raw_text = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=False)

        r = {
            "id": example_id,
            "question": question,
            "ground_truth": str(ex["answer"]).strip(),
            "output": raw_text,
            "prediction": raw_text.strip(),
            "turns": 0,
            "queries": [],
            "latency_seconds": round(latency, 4),
            "generated_tokens": len(new_tokens),
            "information_tokens": 0,
        }
        results.append(r)
        append_jsonl(args.output_path, r)
        checkpointer.update(results)
        progress.set_postfix(pred=str(r["prediction"])[:24])

    metrics = summarize_results(results)
    checkpointer.finalize(metrics)

    print(f"\n=== 结果 ===")
    print(f"EM={metrics['em']:.3f}  F1={metrics['f1']:.3f}  (samples={metrics['samples']})")
    latency = metrics.get("latency_seconds", {}).get("total", 0.0)
    print(f"总耗时 {latency:.1f}s 平均每题 {latency / max(metrics['samples'], 1):.2f}s")
    print(f"结果 -> {args.output_path}\n指标 -> {metric_path}")


if __name__ == "__main__":
    main()