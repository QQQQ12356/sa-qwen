"""评估入口：读取生成结果，计算 EM/F1 指标并保存。

用法：``python -m search_comp.evaluation.evaluate --result_path ... --metric_path ...``
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .em_f1 import compute_metrics


def load_results(result_path: str) -> List[Dict[str, Any]]:
    """加载生成结果 JSONL。

    Args:
        result_path: JSONL 路径（每行 ``{id, question, prediction, ground_truth}``）。

    Returns:
        结果列表。

    Raises:
        FileNotFoundError: 文件不存在时抛出。
    """
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"结果文件不存在: {result_path}")
    results: List[Dict[str, Any]] = []
    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def evaluate_result_file(result_path: str) -> Dict[str, float]:
    """从结果文件计算 EM/F1。

    Args:
        result_path: 生成结果 JSONL 路径。

    Returns:
        指标 dict ``{"em", "f1", "valid_count"}``。
    """
    results = load_results(result_path)
    pairs = [
        (r.get("id", ""), r.get("prediction", ""), r.get("ground_truth", ""))
        for r in results
    ]
    metrics = compute_metrics(pairs)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 EM/F1")
    parser.add_argument("--result_path", type=str, required=True, help="生成结果 JSONL")
    parser.add_argument(
        "--metric_path", type=str, default=None, help="指标 JSON 输出路径"
    )
    args = parser.parse_args()

    metrics = evaluate_result_file(args.result_path)
    print(
        f"[evaluate] EM={metrics['em']:.4f} F1={metrics['f1']:.4f} "
        f"有效样本={metrics['valid_count']}"
    )

    if args.metric_path:
        os.makedirs(os.path.dirname(args.metric_path), exist_ok=True)
        with open(args.metric_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[evaluate] 指标已保存到 {args.metric_path}")
