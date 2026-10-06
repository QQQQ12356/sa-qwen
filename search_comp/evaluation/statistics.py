"""汇总 SearchAgent 预测文件的质量、搜索行为、耗时与压缩统计。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from collections import Counter
from typing import Any, Iterable

from .em_f1 import NO_ANSWER, compute_metrics, extract_answer
from ..utils.runtime import write_json

#: 视为「未按格式作答」的占位预测（协议层面模型未闭合 <answer>）
NO_ANSWER_PLACEHOLDERS = {"", NO_ANSWER}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} 不是合法 JSONL") from error
    return records


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(index, 0)]


def _is_format_correct(row: dict[str, Any]) -> bool:
    """判断该样本是否按协议闭合了 ``<answer>...</answer>``。

    优先用原始输出 ``output`` 判定（两条评测路径都会写入）；
    兼容缺失 ``output`` 的旧记录，退化为检查预测占位符。
    """
    output = row.get("output")
    if output is not None:
        return extract_answer(str(output)) is not None
    return str(row.get("prediction", "")).strip() not in NO_ANSWER_PLACEHOLDERS


def summarize_results(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """计算统一统计；缺失的可选字段不会影响 EM/F1。

    指标分两层：

    - **总体**（``overall_em`` / ``overall_f1``）：全部样本参与，
      未按格式作答的样本计 0 分，反映端到端真实效果；
    - **格式正确子集**（``formatted_em`` / ``formatted_f1``）：仅统计
      闭合了 ``<answer>...</answer>`` 的样本，衡量「会作答时的质量」，
      与 ``format_correct_rate``（格式正确率）配套。
    """
    rows = list(records)
    scored = [
        (str(row.get("id", "")), str(row.get("prediction", "")), str(row.get("ground_truth", "")))
        for row in rows
        if row.get("ground_truth") is not None
    ]
    formatted = [
        (str(row.get("id", "")), str(row.get("prediction", "")), str(row.get("ground_truth", "")))
        for row in rows
        if row.get("ground_truth") is not None and _is_format_correct(row)
    ]
    overall = compute_metrics(scored)
    formatted_quality = compute_metrics(formatted)
    turns = [int(row.get("turns", 0) or 0) for row in rows]
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    information_tokens = [int(row.get("information_tokens", 0) or 0) for row in rows]
    beacon_tokens = [int(row.get("beacon_tokens", 0) or 0) for row in rows]
    total_information = sum(information_tokens)
    total_beacons = sum(beacon_tokens)
    format_correct = sum(1 for row in rows if _is_format_correct(row))
    turns_histogram = {
        str(turn): count
        for turn, count in sorted(Counter(turns).items())
    }
    return {
        # 兼容别名：em / f1 即总体指标
        "em": overall["em"],
        "f1": overall["f1"],
        "valid_count": overall["valid_count"],
        "overall_em": overall["em"],
        "overall_f1": overall["f1"],
        "format_correct_samples": format_correct,
        "format_correct_rate": format_correct / len(rows) if rows else 0.0,
        "formatted_em": formatted_quality["em"],
        "formatted_f1": formatted_quality["f1"],
        "formatted_count": formatted_quality["valid_count"],
        "samples": len(rows),
        "answered": format_correct,
        "answer_rate": format_correct / len(rows) if rows else 0.0,
        "search_samples": sum(turn > 0 for turn in turns),
        "search_rate": sum(turn > 0 for turn in turns) / len(rows) if rows else 0.0,
        "multi_turn_samples": sum(turn >= 2 for turn in turns),
        "average_turns": mean(turns) if turns else 0.0,
        "turns_histogram": turns_histogram,
        "latency_seconds": {
            "total": sum(latencies),
            "mean": mean(latencies) if latencies else 0.0,
            "median": median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
        },
        "information_tokens": total_information,
        "beacon_tokens": total_beacons,
        "effective_information_compression_ratio": (
            total_information / total_beacons if total_beacons else None
        ),
    }


def summarize_file(path: str | Path) -> dict[str, Any]:
    summary = summarize_results(load_jsonl(path))
    summary["result_path"] = str(path)
    return summary


class ProgressCheckpointer:
    """评测进度检查点：每跨过一个进度阈值计算一次当前指标并落盘/打印。

    以「已完成样本数 / 总题数」作为进度变量。每处理到 ``interval``（默认 20%）
    的整数倍，就用当前已收集的结果运行 ``summarize_results``，把检查点追加进
    ``progress_metrics`` 列表写入指标文件（额外字段），并把一行摘要打印到
    stdout —— 被 shell 的 ``run_logged``（tee）同时写入日志文件与终端。

    Args:
        total: 总题数，用于计算进度。
        metric_path: 指标 JSON 路径（通常是 ``<output>_metrics.json``）。
        interval: 进度阈值间隔，须在 ``(0, 1)``。默认取环境变量
            ``EVAL_PROGRESS_INTERVAL``（例如 ``0.25``），否则 0.2。
        config: 评测配置（可选），写入指标文件的 ``evaluation_config`` 字段。

    Usage::

        cp = ProgressCheckpointer(total=len(hp), metric_path=metric_path,
                                  config=vars(args))
        for ex in hp:
            ...
            results.append(r)
            cp.update(results)
        cp.finalize(summarize_results(results))
    """

    def __init__(
        self,
        total: int,
        metric_path: str | Path,
        interval: float | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if interval is None:
            interval = float(os.environ.get("EVAL_PROGRESS_INTERVAL", "0.2"))
        if not 0.0 < float(interval) < 1.0:
            raise ValueError(f"进度阈值 interval 需在 (0,1) 之间，实际为 {interval}")
        self.total = int(total)
        self.metric_path = str(metric_path)
        self.interval = float(interval)
        self.config = config or {}
        # 第一个阈值以「处理数」对齐到 interval 的最小整数。
        self._next_at = max(1, math.ceil(self.total * self.interval))
        self.records: list[dict[str, Any]] = []

    def update(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """用当前已处理结果尝试推进检查点；跨过阈值时计算、落盘并打印。

        Args:
            results: 当前已处理（含 resume 的已完成）样本列表。

        Returns:
            生成的检查点记录；未跨过阈值时返回 None。
        """
        processed = len(results)
        if self.total == 0 or processed < self._next_at:
            return None
        record = self._build_record(results)
        self.records.append(record)
        self._write()
        self._print(record)
        # 已到末尾则推进到永不触发；否则推进一个 interval。
        if processed >= self.total:
            self._next_at = self.total + 1
        else:
            self._next_at = processed + max(1, math.ceil(self.total * self.interval))
        return record

    def finalize(self, final_metrics: dict[str, Any]) -> None:
        """写入最终指标文件：完整指标 + 全部进度检查点 + 配置。"""
        payload = {
            **final_metrics,
            "progress_metrics": self.records,
            "evaluation_config": self.config,
        }
        write_json(self.metric_path, payload)

    def _build_record(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = summarize_results(results)
        processed = len(results)
        return {
            "progress": len(results) / self.total if self.total else 0.0,
            "processed": processed,
            "total": self.total,
            "em": metrics["overall_em"],
            "f1": metrics["overall_f1"],
            "format_correct_rate": metrics["format_correct_rate"],
            "formatted_em": metrics["formatted_em"],
            "formatted_f1": metrics["formatted_f1"],
            "search_rate": metrics["search_rate"],
            "average_turns": metrics["average_turns"],
            "samples": metrics["samples"],
        }

    def _write(self) -> None:
        payload = {"progress_metrics": self.records, "evaluation_config": self.config}
        write_json(self.metric_path, payload)

    @staticmethod
    def _print(record: dict[str, Any]) -> None:
        print(
            f"[progress] 进度 {record['progress']:.0%} "
            f"({record['processed']}/{record['total']}) | "
            f"EM={record['em']:.3f} F1={record['f1']:.3f} | "
            f"格式正确率={record['format_correct_rate']:.0%} "
            f"(子集 EM={record['formatted_em']:.3f} F1={record['formatted_f1']:.3f}) | "
            f"搜索率={record['search_rate']:.0%} 平均轮次={record['average_turns']:.2f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 SearchAgent 评测 JSONL")
    parser.add_argument(
        "--result_path", action="append", required=True,
        help="预测 JSONL，可重复传入以比较多个实验",
    )
    parser.add_argument("--output_path", required=True, help="汇总 JSON 输出路径")
    args = parser.parse_args()

    summaries = [summarize_file(path) for path in args.result_path]
    payload = {"experiments": summaries}
    write_json(args.output_path, payload)
    for summary in summaries:
        print(
            f"[stats] {summary['result_path']} samples={summary['samples']} "
            f"总体EM={summary['overall_em']:.4f} 总体F1={summary['overall_f1']:.4f} | "
            f"格式正确率={summary['format_correct_rate']:.1%} "
            f"子集EM={summary['formatted_em']:.4f} 子集F1={summary['formatted_f1']:.4f} | "
            f"search={summary['search_rate']:.1%} turns={summary['average_turns']:.2f} "
            f"turns分布={summary['turns_histogram']} "
            f"compression={summary['effective_information_compression_ratio']}",
            flush=True,
        )
    print(f"[stats] 汇总已保存 -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
