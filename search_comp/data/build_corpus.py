"""从 HotpotQA 构建检索语料（JSONL）。

用法：``python -m search_comp.data.build_corpus --output_path outputs/data/hotpotqa_corpus.jsonl --max_questions 1000``
"""

from __future__ import annotations

import argparse

from datasets import load_dataset

from .retrieval import build_corpus_from_hotpotqa


def main() -> None:
    """构建语料入口。

    注意：语料中的金标准支撑文档必须覆盖后续数据构建采样的问题，
    因此这里与数据构建脚本使用**相同 seed 的 shuffle** 采样（默认 42），
    保证 ``shuffle(seed).select(N)`` 采到的问题其支撑文档在语料内。
    """
    parser = argparse.ArgumentParser(description="从 HotpotQA 构建检索语料")
    parser.add_argument(
        "--output_path", type=str, required=True, help="语料 JSONL 输出路径"
    )
    parser.add_argument(
        "--splits", type=str, default="train,validation", help="使用的 split，逗号分隔"
    )
    parser.add_argument(
        "--max_per_split", type=int, default=None, help="每个 split 最多处理的样本数"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="采样种子（需与数据构建一致）"
    )
    args = parser.parse_args()

    datasets = []
    for split in args.splits.split(","):
        hp = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
        if args.max_per_split:
            hp = hp.shuffle(seed=args.seed).select(
                range(min(len(hp), args.max_per_split))
            )
        datasets.append(hp)
        print(f"[build_corpus] split {split}: {len(hp)} 条")

    build_corpus_from_hotpotqa(datasets, args.output_path)


if __name__ == "__main__":
    main()
