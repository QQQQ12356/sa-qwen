"""EM/F1 指标的单元测试。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_comp.evaluation.em_f1 import (
    compute_em,
    compute_f1,
    compute_metrics,
    extract_answer,
    normalize_answer,
)


def test_extract_answer_last_tag():
    """应提取最后一个 <answer> 标签的内容。"""
    text = "<think>reason</think><answer> Paris </answer> and <answer>Rome</answer>"
    assert extract_answer(text) == "Rome"


def test_extract_answer_none():
    """无 <answer> 标签时返回 None。"""
    assert extract_answer("<think>no answer</think>") is None


def test_normalize_answer():
    """标准化：小写、去冠词、去标点、压缩空格。"""
    assert normalize_answer("  The  Paris,  France! ") == "paris france"


def test_compute_em():
    """精确匹配：标准化后一致为 1，否则 0。"""
    assert compute_em("Paris", "Paris") == 1.0
    assert compute_em("the Paris", "Paris") == 1.0  # 去冠词
    assert compute_em("Paris", "Rome") == 0.0


def test_compute_f1_perfect_and_zero():
    """F1：完全一致为 1，完全不同为 0。"""
    assert compute_f1("Paris is the capital", "Paris is the capital") == 1.0
    assert compute_f1("Paris", "Rome") == 0.0


def test_compute_metrics_empty():
    """空列表应返回 0 指标。"""
    m = compute_metrics([])
    assert m["em"] == 0.0 and m["valid_count"] == 0
