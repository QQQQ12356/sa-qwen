"""未闭合 ``<answer>`` 时的预测兜底：所有评测路径必须一致。

beacon / plain（probe）/ native 三条路径都要在模型没闭合 ``<answer>`` 时写入同一个
占位符 ``[无作答]``，而不是拿整段原始输出当预测——否则 prediction 字段不可读，且原文
里的字符串可能意外命中金标准，让不同路径的口径对不上。
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from search_comp.evaluation.beacon_interactive_eval import run_beacon_agent
from search_comp.evaluation.em_f1 import NO_ANSWER, answer_or_placeholder, extract_answer
from search_comp.evaluation.statistics import NO_ANSWER_PLACEHOLDERS, summarize_results
from search_comp.milestones.searchagent_probe import run_searchagent_probe

_UNCLOSED_TEXT = "<thinking>I could not find it</thinking>\n<search>query</search"


class _CharacterTokenizer:
    """字符级 tokenizer，够驱动 decode/encode 与停止条件。"""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


class _UnclosedAnswerModel:
    """generate 恒定吐出一段不含 </answer> 的文本。"""

    def generate(self, input_ids=None, **kwargs):
        new_tokens = torch.tensor([ord(c) for c in _UNCLOSED_TEXT], dtype=torch.long)
        return torch.cat([input_ids[0], new_tokens]).unsqueeze(0)

    def parameters(self):
        return iter([torch.zeros(1)])


def test_answer_or_placeholder_extracts_or_falls_back():
    assert answer_or_placeholder("<answer> Beijing </answer>") == "Beijing"
    assert answer_or_placeholder("<answer>a</answer> then <answer> b </answer>") == "b"
    assert answer_or_placeholder(_UNCLOSED_TEXT) == NO_ANSWER
    assert answer_or_placeholder("") == NO_ANSWER
    # 只有 <answer> 开标签时也算未作答
    assert answer_or_placeholder("<answer>Beijing") == NO_ANSWER


def test_placeholder_is_registered_as_unanswered():
    assert NO_ANSWER in NO_ANSWER_PLACEHOLDERS
    assert extract_answer("<answer>x</answer>") == "x"


def test_plain_probe_uses_placeholder_when_answer_unclosed():
    tokenizer = _CharacterTokenizer()
    retriever = Mock()

    result = run_searchagent_probe(
        _UnclosedAnswerModel(), tokenizer, retriever, "Question",
        max_turns=1, verbosity=0,
    )

    assert result["prediction"] == NO_ANSWER
    # 原始输出仍完整保留在 output 字段，供人工排查。
    assert "</thinking>" in result["output"]
    assert retriever.retrieve.call_count == 0  # 未闭合 </search>，不触发检索


def test_beacon_agent_uses_placeholder_when_answer_unclosed():
    tokenizer = _CharacterTokenizer()
    model = Mock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.beacon_config.beacon_ratio = 2
    model.beacon_config.beacon_question_memory_v1 = False
    model.beacon_generate.return_value = torch.tensor(
        [tokenizer(_UNCLOSED_TEXT).input_ids]
    )

    result = run_beacon_agent(model, tokenizer, Mock(), "Question", max_turns=1)

    assert result["prediction"] == NO_ANSWER
    assert "</thinking>" in result["output"]


def test_both_paths_report_same_format_metric_for_unclosed_answer():
    """同一份「未作答」轨迹，两条路径的格式正确率与 EM 口径必须一致。"""
    record = {
        "id": "q1", "question": "q?", "ground_truth": "Beijing",
        "prediction": NO_ANSWER, "output": _UNCLOSED_TEXT, "turns": 0,
    }
    summary = summarize_results([record])
    assert summary["format_correct_rate"] == 0.0
    assert summary["overall_em"] == 0.0
    assert summary["overall_f1"] == 0.0
    # output 字段决定格式判定，与 prediction 用哪个占位符无关。
    assert summary["samples"] == 1
