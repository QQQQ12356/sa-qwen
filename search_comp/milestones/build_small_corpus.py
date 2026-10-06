"""Milestone-1：构建小规模 HotpotQA 检索语料。

为快速验证 Qwen3.5 SearchAgent 检索能力，从 HotpotQA(distractor) 的 train /
validation 前 N 条样本抽取 ``context`` 段落，构建一个小规模 BM25 语料
（默认数百到上千段落）。

复用 ``search_comp.data.retrieval.build_corpus_from_hotpotqa`` 做段落级去重，
并保证与数据构造同 seed 的样本对齐（gold 支撑文档大概率在语料内）。

用法::

    python -m search_comp.milestones.build_small_corpus \
        --output_path outputs/data/hotpotqa_corpus_small.jsonl \
        --max_per_split 200 --seed 42
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset

from ..data.retrieval import build_corpus_from_hotpotqa


def build(
    output_path: str,
    splits: str = "train,validation",
    max_per_split: int = 200,
    seed: int = 42,
) -> None:
    """构建小规模语料。

    Args:
        output_path: 输出 JSONL 路径。
        splits: 逗号分隔的 split 名。
        max_per_split: 每个 split 最多取前 N 条样本的 context。
        seed: 采样 seed（与数据构造保持一致）。
    """
    ds_list = []
    for split in [s.strip() for s in splits.split(",") if s.strip()]:
        ds = load_dataset("hotpot_qa", "distractor", split=split)
        print(f"[build_small_corpus] {split}: 原始样本 {len(ds)}")
        if max_per_split:
            ds = ds.shuffle(seed=seed).select(range(min(len(ds), max_per_split)))
            print(
                f"[build_small_corpus] {split}: 取前 {len(ds)} 条构建语料上下文"
            )
        ds_list.append(ds)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    build_corpus_from_hotpotqa(ds_list, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建小规模 HotpotQA 检索语料")
    parser.add_argument("--output_path", type=str, default="outputs/data/hotpotqa_corpus_small.jsonl")
    parser.add_argument("--splits", type=str, default="train,validation")
    parser.add_argument("--max_per_split", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.output_path, args.splits, args.max_per_split, args.seed)