"""SFT 后复测：同一 v3/ user prompt 排版下，测 base 与 SFT 模型的 <answer> 收敛。

新增文件，不改动既有代码。与 :mod:`searchagent_sysprompt_probe` 共用
``run_sysprompt_probe`` 的检索循环，仅替换模型加载方式：

- **base（HF id，多模态 checkpoint）**：用 ``qwen35_text`` 的文本主干重装
  （与 native_trainer 一致，纯文本因果 LM）。
- **SFT 模型（本地目录，已纯文本因果 LM）**：``AutoModelForCausalLM`` 直接加载。

用法::

    python -m search_comp.evaluation.native_sft_reeval \\
        --model_path Qwen/Qwen3.5-2B \\
        --style system3 --max_questions 50 --output_path out_a.jsonl
    python -m search_comp.evaluation.native_sft_reeval \\
        --model_path outputs/models/native_qwen3_sft_small_v1/final \\
        --style system3 --max_questions 50 --output_path out_b.jsonl
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from ..data.retrieval import BM25Retriever
from ..milestones.searchagent_sysprompt_probe import run_sysprompt_probe
from .em_f1 import compute_metrics
from .statistics import ProgressCheckpointer, summarize_results


def load_for_eval(model_path: str):
    """按模型类型返回 (tokenizer, model)。统一 bf16 + auto device。"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    archs = getattr(cfg, "architectures", []) or []
    if "Qwen3_5ForCausalLM" in archs:
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        return tok, model.eval()
    # base 多模态 checkpoint：重装文本主干为因果 LM
    from ..milestones.qwen35_text import load_text_causal_model, load_text_tokenizer

    return load_text_tokenizer(model_path), load_text_causal_model(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT 后复测（搜索/<answer> 收敛）")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--style", type=str, choices=["system", "system3", "system4", "user"], default="system3")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_questions", type=int, default=50)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=3)
    parser.add_argument("--max_docs_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=384)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.manual_seed(42)

    tokenizer, model = load_for_eval(args.model_path)
    retriever = BM25Retriever(args.corpus_path)

    from datasets import load_dataset

    hp = load_dataset("hotpot_qa", "distractor", split=args.split).select(range(args.max_questions))

    print(f"=== SFT 复测 style={args.style} model={args.model_path} n={len(hp)} ===")
    results = []
    metric_path = os.path.splitext(args.output_path)[0] + "_metrics.json"
    checkpointer = ProgressCheckpointer(len(hp), metric_path, config=vars(args))
    for i, ex in enumerate(hp):
        r = run_sysprompt_probe(
            model, tokenizer, retriever, str(ex["question"]),
            style=args.style, max_turns=args.max_turns, topk=args.topk,
            max_docs_tokens=args.max_docs_tokens,
            max_new_tokens_per_turn=args.max_new_tokens,
            do_sample=False, temperature=1.0, verbosity=0,
        )
        r["id"] = str(ex["id"])
        r["ground_truth"] = str(ex["answer"]).strip()
        results.append(r)
        checkpointer.update(results)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics = compute_metrics([(r["id"], r["prediction"], r["ground_truth"]) for r in results])
    ns = sum(1 for r in results if r["turns"] > 0)
    na = sum(1 for r in results if "<answer>" in r["output"])
    nq = sum(len(r["queries"]) for r in results)
    nempty = sum(1 for q in (q for r in results for q in r["queries"]) if not q.strip())
    print(f"EM={metrics['em']:.3f} F1={metrics['f1']:.3f} 搜索={ns}({ns/len(results):.0%}) "
          f"<answer>={na} 总query={nq} 空query={nempty} avgTurns={sum(r['turns'] for r in results)/len(results):.2f}")
    checkpointer.finalize(summarize_results(results))
    print(f"-> {args.output_path}")
    print(f"[sft-reeval] 指标 -> {metric_path}")


if __name__ == "__main__":
    main()