import json

import pytest

from search_comp.evaluation.statistics import summarize_results
from search_comp.utils.runtime import load_yaml_config, prepare_run_artifacts


def test_yaml_overrides_support_nested_values(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "learning_rate: 5.0e-5\nbeacon:\n  beacon_ratio: 16\n",
        encoding="utf-8",
    )

    config = load_yaml_config(
        str(config_path),
        ["learning_rate=1.0e-5", "beacon.beacon_ratio=32", "use_lora=true"],
    )

    assert config["learning_rate"] == pytest.approx(1.0e-5)
    assert config["beacon"]["beacon_ratio"] == 32
    assert config["use_lora"] is True


def test_prepare_run_artifacts_records_resolved_config(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["trainer", "--config", "train.yaml"])
    run_dir = prepare_run_artifacts(
        tmp_path / "run",
        {"exp_name": "smoke"},
        "train.yaml",
        ["exp_name=smoke"],
    )

    config = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert config["exp_name"] == "smoke"
    assert metadata["overrides"] == ["exp_name=smoke"]
    assert metadata["command"].startswith("trainer")


def test_statistics_cover_quality_search_latency_and_compression():
    summary = summarize_results(
        [
            {
                "id": "1",
                "prediction": "Paris",
                "ground_truth": "Paris",
                "output": "think...<answer>Paris</answer>",
                "turns": 1,
                "latency_seconds": 2.0,
                "information_tokens": 64,
                "beacon_tokens": 4,
            },
            {
                "id": "2",
                "prediction": "[无作答]",
                "ground_truth": "Rome",
                "output": "think...<search>rome</search>（未闭合 answer）",
                "turns": 2,
                "latency_seconds": 4.0,
                "information_tokens": 32,
                "beacon_tokens": 2,
            },
            {
                "id": "3",
                "prediction": "Big Ben",
                "ground_truth": "Ben",
                "output": "<answer>Big Ben</answer>",
                "turns": 0,
                "latency_seconds": 1.0,
                "information_tokens": 0,
                "beacon_tokens": 0,
            },
        ]
    )

    # 总体指标：全部样本（未闭合 <answer> 的样本按 0 分计）
    assert summary["overall_em"] == pytest.approx(1 / 3)
    assert summary["overall_f1"] == pytest.approx((1.0 + 0.0 + 2 / 3) / 3)
    # 兼容别名：em / f1 与总体一致
    assert summary["em"] == summary["overall_em"]
    assert summary["f1"] == summary["overall_f1"]
    # 格式正确率：3 条中 2 条闭合了 <answer>...</answer>
    assert summary["format_correct_samples"] == 2
    assert summary["format_correct_rate"] == pytest.approx(2 / 3)
    # 仅格式正确样本的 EM/F1
    assert summary["formatted_em"] == pytest.approx(0.5)
    assert summary["formatted_f1"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert summary["formatted_count"] == 2
    # 检索轮次：均值与分布
    assert summary["search_rate"] == pytest.approx(2 / 3)
    assert summary["multi_turn_samples"] == 1
    assert summary["average_turns"] == pytest.approx(1.0)
    assert summary["turns_histogram"] == {"0": 1, "1": 1, "2": 1}
    # 行为与压缩统计
    assert summary["latency_seconds"]["mean"] == pytest.approx(7 / 3)
    assert summary["effective_information_compression_ratio"] == pytest.approx(
        (64 + 32) / 6
    )
    # answer_rate 与格式正确率同义
    assert summary["answer_rate"] == summary["format_correct_rate"]


def test_export_excel_writes_requested_columns(tmp_path):
    from search_comp.evaluation.export_excel import export_excel

    result_path = tmp_path / "metrics.json"
    result_path.write_text(
        json.dumps(
            {
                "samples": 100,
                "overall_em": 0.43,
                "overall_f1": 0.52,
                "format_correct_rate": 0.677,
                "formatted_em": 0.635,
                "formatted_f1": 0.77,
                "average_turns": 1.43,
                "effective_information_compression_ratio": 16.0,
                "evaluation_config": {"model_path": "outputs/models/demo/final"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "summary.xlsx"

    frame = export_excel([str(result_path)], str(output_path))

    assert list(frame.columns) == [
        "结果文件", "模型", "测试样本数", "总体EM", "总体F1", "格式正确率",
        "格式正确EM", "格式正确F1", "平均检索轮次", "压缩比",
    ]
    row = frame.iloc[0]
    assert row["结果文件"] == str(result_path)
    assert row["模型"] == "outputs/models/demo/final"
    assert row["测试样本数"] == 100
    assert row["总体EM"] == pytest.approx(0.43)
    assert row["格式正确率"] == pytest.approx(0.677)
    assert row["格式正确EM"] == pytest.approx(0.635)
    assert row["平均检索轮次"] == pytest.approx(1.43)
    assert row["压缩比"] == pytest.approx(16.0)
    assert output_path.exists()


def test_invalid_override_is_rejected(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    with pytest.raises(ValueError, match="key=value"):
        load_yaml_config(str(config_path), ["seed"])
