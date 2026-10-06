"""把评测指标 JSON（``*_metrics.json``，即 ``tmp.json`` 格式）导出为 Excel 对比表。

输入是评测脚本写出的指标文件（``{**metrics, "evaluation_config": ...}``），
每个 JSON 一行，字段包括：测试样本数、总体 EM/F1、格式正确率、
格式正确子集 EM/F1、检索轮次与压缩比。

用法::

    python -m search_comp.evaluation.export_excel \
        --result_path a/metrics.json --result_path b/metrics.json \
        --output_path outputs/results/summary.xlsx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

#: Excel 列（指标字段名与 summarize_results 输出保持一致）
EXPORT_COLUMNS = {
    "result_path": "结果文件",
    "model_path": "模型",
    "samples": "测试样本数",
    "overall_em": "总体EM",
    "overall_f1": "总体F1",
    "format_correct_rate": "格式正确率",
    "formatted_em": "格式正确EM",
    "formatted_f1": "格式正确F1",
    "average_turns": "平均检索轮次",
    "effective_information_compression_ratio": "压缩比",
}

#: 旧版指标文件缺新字段时的回退映射
_LEGACY_ALIASES = {
    "overall_em": "em",
    "overall_f1": "f1",
    "samples": "valid_count",
}


def load_metrics(path: str | Path) -> dict:
    """读取一个指标 JSON；缺 ``evaluation_config`` 也能导出。"""
    with Path(path).open("r", encoding="utf-8") as input_file:
        metrics = json.load(input_file)
    if not isinstance(metrics, dict):
        raise ValueError(f"{path} 不是指标 JSON（顶层应为对象）")
    for new_field, legacy_field in _LEGACY_ALIASES.items():
        if new_field not in metrics and legacy_field in metrics:
            metrics[new_field] = metrics[legacy_field]
    metrics.setdefault("result_path", str(path))
    metrics.setdefault("model_path", metrics.get("evaluation_config", {}).get("model_path"))
    return metrics


def export_excel(result_paths: list[str], output_path: str) -> pd.DataFrame:
    """汇总各指标 JSON 并写出 Excel；返回汇总 DataFrame。"""
    if not result_paths:
        raise ValueError("至少需要一个 --result_path")
    frame = pd.DataFrame([load_metrics(path) for path in result_paths])
    frame = frame.reindex(columns=list(EXPORT_COLUMNS)).rename(columns=EXPORT_COLUMNS)
    frame.to_excel(output_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总评测指标 JSON 并导出 Excel")
    parser.add_argument(
        "--result_path", action="append", required=True,
        help="指标 JSON（*_metrics.json / tmp.json 格式），可重复传入",
    )
    parser.add_argument("--output_path", required=True, help="Excel 输出路径（.xlsx）")
    args = parser.parse_args()

    frame = export_excel(args.result_path, args.output_path)
    print(frame.to_string(index=False))
    print(f"[export-excel] 已保存 -> {args.output_path}")


if __name__ == "__main__":
    main()
