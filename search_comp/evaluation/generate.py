"""Beacon 模式推理生成。

流程：
1. 加载 beacon 模型、tokenizer、BM25 检索器。
2. 对每个问题在线检索 top-k 文档，构造与训练一致的 prompt。
3. tokenize 并定位文档压缩区。
4. ``model.beacon_generate`` 先编码 question+文档为 beacon K/V，再自回归生成答案。
5. 抽取 ``<answer>`` 内容，输出 ``(id, prediction, ground_truth)``。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from ..models.model_loader import load_model, load_tokenizer
from .em_f1 import extract_answer
from ..data.retrieval import BM25Retriever, format_docs_as_reference

#: 与训练一致的 prompt 模板
PROMPT_TEMPLATE = (
    "Answer the given question with some potentially useful context. "
    "Show your reasoning in <think> </think> tags and return the final answer "
    "in <answer> </answer> tags.\n"
    "Question: {question}\n"
    "<information>\n"
    "{docs}\n"
    "</information>"
)


def build_prompt(question: str, docs_text: str) -> str:
    """构造推理 prompt（与训练一致）。"""
    return PROMPT_TEMPLATE.format(question=question, docs=docs_text)


def locate_compress_region(
    tokenizer: PreTrainedTokenizer, question: str, docs_text: str, max_length: int
) -> Dict[str, Any]:
    """tokenize prompt 并定位文档压缩区。

    Args:
        tokenizer: tokenizer。
        question: 问题。
        docs_text: 检索到的文档文本。
        max_length: 最大长度。

    Returns:
        ``{input_ids, attention_mask, compress_start, compress_end}``。
    """
    prompt = build_prompt(question, docs_text)
    chat_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(chat_input, add_special_tokens=False).input_ids[:max_length]

    # 压缩区定位：用 docs 在 chat_input 中的真实位置，避免把模板尾部算入
    docs_pos = chat_input.index(docs_text)
    before_docs = chat_input[:docs_pos]
    compress_start = len(tokenizer(before_docs, add_special_tokens=False).input_ids)
    docs_tokens = tokenizer(docs_text, add_special_tokens=False).input_ids
    compress_end = compress_start + len(docs_tokens)
    compress_end = min(compress_end, len(ids))

    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        "compress_start": compress_start,
        "compress_end": compress_end,
    }


def generate_answers(
    model,
    tokenizer: PreTrainedTokenizer,
    retriever: BM25Retriever,
    dataset: Any,
    topk: int,
    max_new_tokens: int,
    max_length: int,
    do_sample: bool,
    temperature: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """对数据集逐条生成答案。

    Args:
        model: beacon 模型（已 load）。
        tokenizer: tokenizer。
        retriever: BM25 检索器。
        dataset: HuggingFace 数据集。
        topk: 检索文档数。
        max_new_tokens / max_length / do_sample / temperature: 生成参数。
        device: 推理设备。

    Returns:
        ``[{id, question, prediction, ground_truth, docs}]`` 列表。
    """
    model.to(device).eval()
    results: List[Dict[str, Any]] = []
    with torch.no_grad():
        for ex in dataset:
            question = str(ex["question"]).strip()
            if question.endswith("?"):
                question = question[:-1].strip()
            gold = str(ex["answer"]).strip()

            try:
                retrieved = retriever.retrieve(question + "?", topk=topk)
            except Exception as exc:  # noqa: BLE001
                print(f"[generate] 检索失败 {question}: {exc}")
                results.append(
                    {
                        "id": str(ex["id"]),
                        "question": question,
                        "prediction": "",
                        "ground_truth": gold,
                        "docs": "",
                    }
                )
                continue

            docs_text = format_docs_as_reference(retrieved)
            inputs = locate_compress_region(
                tokenizer, question + "?", docs_text, max_length
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            compress_start = inputs["compress_start"]
            compress_end = inputs["compress_end"]

            gen_ids = model.beacon_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                compress_start=compress_start,
                compress_end=compress_end,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            prediction = extract_answer(gen_text) or gen_text.strip()
            results.append(
                {
                    "id": str(ex["id"]),
                    "question": question,
                    "prediction": prediction,
                    "ground_truth": gold,
                    "docs": docs_text[:200],
                }
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beacon 模式推理生成")
    parser.add_argument("--model_path", type=str, required=True, help="beacon 模型路径")
    parser.add_argument(
        "--corpus_path", type=str, required=True, help="语料 JSONL 路径"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="结果 JSONL 输出路径"
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path)
    tokenizer = load_tokenizer(args.model_path)
    retriever = BM25Retriever(args.corpus_path)
    hp = load_dataset("hotpot_qa", "distractor", split=args.split)
    if args.max_questions:
        hp = hp.select(range(args.max_questions))

    results = generate_answers(
        model,
        tokenizer,
        retriever,
        hp,
        topk=args.topk,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        device=device,
    )
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[generate] 生成 {len(results)} 条 -> {args.output_path}")
