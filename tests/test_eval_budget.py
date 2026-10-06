"""评测生成预算与训练分布对齐的回归测试。

训练轨迹里 assistant 段 p99≈580 token、最多 3 次 search + 1 次 answer；评测若沿用
更小的预算（256 token / 3 轮）会让约 12%~14% 的题在结构上不可能答对。这里固定
共享常量，并确保 beacon / plain / native 三条评测路径与启动脚本都取自同一处。
"""

import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from search_comp.data.retrieval import format_docs_as_reference
from search_comp.evaluation.beacon_interactive_eval import run_beacon_agent
from search_comp.evaluation.budget import MAX_NEW_TOKENS_PER_TURN, MAX_TURNS
from search_comp.evaluation.interactive_generate import run_interactive_agent
from search_comp.milestones.searchagent_probe import decode_until, run_searchagent_probe

_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
_BUDGET_SCRIPTS = (
    "00_searchagent_probe.sh",
    "21_plain_eval.sh",
    "23_beacon_eval.sh",
    "28_plain_untrained_interactive_eval.sh",
)


def test_budget_covers_training_distribution():
    # 训练轨迹 assistant 段 p99≈580 token、最多 3 次 search + 1 次 answer。
    assert MAX_NEW_TOKENS_PER_TURN >= 580
    assert MAX_TURNS >= 4


def test_agent_function_defaults_use_shared_budget():
    import inspect

    beacon = inspect.signature(run_beacon_agent).parameters
    assert beacon["max_turns"].default == MAX_TURNS
    assert beacon["max_new_tokens_per_turn"].default == MAX_NEW_TOKENS_PER_TURN
    probe = inspect.signature(run_searchagent_probe).parameters
    assert probe["max_turns"].default == MAX_TURNS
    assert probe["max_new_tokens_per_turn"].default == MAX_NEW_TOKENS_PER_TURN
    assert (
        inspect.signature(decode_until).parameters["max_new_tokens"].default
        == MAX_NEW_TOKENS_PER_TURN
    )
    interactive = inspect.signature(run_interactive_agent).parameters
    assert interactive["max_turns"].default == MAX_TURNS
    assert interactive["max_new_tokens_per_turn"].default == MAX_NEW_TOKENS_PER_TURN


def test_launch_scripts_use_shared_budget():
    for name in _BUDGET_SCRIPTS:
        with open(os.path.join(_SCRIPTS_DIR, name), "r", encoding="utf-8") as f:
            script = f.read()
        assert re.search(r"MAX_TURNS=\$\{MAX_TURNS:-" + str(MAX_TURNS) + r"\}", script), name
        assert re.search(
            r"MAX_NEW_TOKENS_PER_TURN=\$\{MAX_NEW_TOKENS_PER_TURN:-"
            + str(MAX_NEW_TOKENS_PER_TURN)
            + r"\}",
            script,
        ), name


class _CharacterTokenizer:
    """把每个字符当成一个 token，便于断言传给 beacon_generate 的参数。"""

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[ord(character) for character in text])

    def decode(self, tokens, **kwargs):
        return "".join(chr(token) for token in tokens)


def test_beacon_agent_forwards_budget_to_generation():
    tokenizer = _CharacterTokenizer()
    model = Mock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.beacon_config.beacon_ratio = 2
    model.beacon_config.beacon_question_memory_v1 = False
    model.beacon_generate.side_effect = [torch.tensor([tokenizer("<answer>done</answer>").input_ids])]
    retriever = Mock()

    run_beacon_agent(model, tokenizer, retriever, "Question")

    assert model.beacon_generate.call_args.kwargs["max_new_tokens"] == MAX_NEW_TOKENS_PER_TURN


def test_beacon_agent_uses_unified_doc_format():
    # 评测喂给模型的检索文档必须与训练数据同格式（Doc N + 标题行，无 ID/Score）。
    tokenizer = _CharacterTokenizer()
    model = Mock()
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])
    model.beacon_config.beacon_ratio = 2
    model.beacon_config.beacon_question_memory_v1 = False
    outputs = ["<search>query</search>", "<answer>done</answer>"]
    model.beacon_generate.side_effect = [
        torch.tensor([tokenizer(text).input_ids]) for text in outputs
    ]
    retriever = Mock()
    retrieved = [{"title": "Result", "text": "Result\nRetrieved content"}]
    retriever.retrieve.return_value = retrieved

    run_beacon_agent(model, tokenizer, retriever, "Question")

    info_input = tokenizer.decode(model.beacon_generate.call_args_list[1].kwargs["input_ids"][0].tolist())
    assert format_docs_as_reference(retrieved) in info_input
    # INFO_PREFIX/INFO_SUFFIX 自带空行，块内即统一格式。
    assert "Doc 1 Result\nRetrieved content</information>" in info_input
    assert "(Title:" not in info_input and "Score:" not in info_input
