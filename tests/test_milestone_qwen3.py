"""Qwen3.5 迁移里程碑的烟雾测试。

覆盖已交付并验证的路径（不依赖尚未完成的混合 Beacon 训练）：

1. Qwen3.5 tokenizer 能渲染 `` thinking`` / `` response`` 推理特殊 token 与
   SearchAgent ``<search>`` 指令模板。
2. 轨迹序列构建（think→search→<information>→answer）的压缩区 / 损失区定位。
3. ``extract_search_query`` / ``extract_answer`` / EM/F1 指标。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from search_comp.data.trajectory import (
    SEARCH_INSTRUCTION,
    SYSTEM_PROMPT,
    build_search_chat_prompt,
    extract_search_query,
)
from search_comp.evaluation.em_f1 import (
    compute_metrics,
    extract_answer,
    normalize_answer,
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B", local_files_only=True)
    except OSError:
        pytest.skip("未缓存 Qwen3.5 tokenizer；单元测试不联网下载")


def test_search_instruction_is_system_and_user_is_question_only(tokenizer):
    question = "Who founded Google?"
    text = build_search_chat_prompt(question, add_generation_prompt=True)
    assert f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>" in text
    assert f"<|im_start|>user\n{question}<|im_end|>" in text
    user_text = text.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    assert user_text == question
    assert SEARCH_INSTRUCTION not in user_text
    ids = tokenizer(text, add_special_tokens=False).input_ids
    assert tokenizer.eos_token_id in (248046,)  # eos = <|im_end|>
    dec = tokenizer.decode(ids, skip_special_tokens=False)
    assert question in dec






def test_extract_and_metrics():
    assert extract_search_query("a<search>foo bar</search>b") == "foo bar"
    assert extract_search_query("no search") == ""
    # last <answer> wins
    assert extract_answer("x<answer>A</answer><answer>B</answer>") == "B"
    assert normalize_answer("  The  Paris, France  ") == "paris france"
    m = compute_metrics([("1", "Paris", "paris"), ("2", "Rome", "Tokyo")])
    assert m["em"] == 0.5  # 1/2 exact
    assert 0.0 < m["f1"] < 1.0
    assert m["valid_count"] == 2


