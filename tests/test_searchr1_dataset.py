"""Search-R1 messages 格式数据集的单元测试（Qwen3.5 tokenizer）。

覆盖：messages → input_ids/labels/regions 的映射，特别是
- Qwen3.5 多模态 chat 模板对 [system] 前缀的增量编码修复；
- ``<information>`` 文档块 → 压缩区（labels=-100）；
- assistant 片段 → 有监督（labels=自身 id）。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from search_comp.data.searchr1_dataset import SearchR1Collator, SearchR1SFTDataset
from search_comp.data.trajectory import INFO_PREFIX, INFO_SUFFIX


def _write_sample(path):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer the question. Question: Who founded Google?"},
        {"role": "assistant", "content": "<thinking>\nI need to search.\n</thinking>\n<search>Google founders</search>"},
        {"role": "user", "content": "<information>[Document 1] Google was founded by Larry Page and Sergey Brin.</information>"},
        {"role": "assistant", "content": "<thinking>\nNow I know.\n</thinking>\n<answer> Larry Page and Sergey Brin </answer>"},
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"messages": msgs}) + "\n")


@pytest.fixture(scope="module")
def tokenizer():
    from search_comp.milestones.qwen35_native import load_tokenizer

    tok = load_tokenizer("Qwen/Qwen3.5-2B")
    tok.pad_token = tok.eos_token
    return tok


def test_searchr1_dataset_regions_and_labels(tokenizer, tmp_path):
    data = tmp_path / "sft.jsonl"
    _write_sample(str(data))
    ds = SearchR1SFTDataset(str(data), tokenizer)
    assert len(ds) == 1
    s = ds[0]
    assert s["question"] == "Answer the question. Question: Who founded Google?"
    original_batch = SearchR1Collator(tokenizer)([s])
    assert "question_input_ids" not in original_batch
    question_batch = SearchR1Collator(tokenizer, question_memory_v1=True)([s])
    assert tokenizer.decode(question_batch["question_input_ids"][0]) == s["question"]

    # 至少一个 <information> 压缩区
    assert len(s["regions"]) >= 1
    for rs, re in s["regions"]:
        assert rs < re <= len(s["input_ids"])
        # 压缩区只覆盖文档正文，不包含 <information> 标签。
        region_text = tokenizer.decode(s["input_ids"][rs:re])
        assert "Document 1" in region_text
        assert "information" not in region_text
        # 压缩区内标签均为 -100（文档不计损失）
        assert all(l == -100 for l in s["labels"][rs:re])

    # 有监督 token（assistant 生成片段）存在
    sup = [i for i, l in enumerate(s["labels"]) if l != -100]
    assert len(sup) > 0
    # 两个 assistant 消息
    assert s["n_turns"] == 2
    assert s["n_searches"] == 1

    full_text = tokenizer.decode(s["input_ids"])
    assert full_text.count("<|im_start|>assistant") == 1
    assert INFO_PREFIX in full_text
    assert INFO_SUFFIX in full_text
    assert "<thinking>\nNow I know." in full_text


def test_searchr1_dataset_never_truncates_complete_trajectory(tokenizer, tmp_path):
    data = tmp_path / "sft.jsonl"
    _write_sample(str(data))
    ds = SearchR1SFTDataset(str(data), tokenizer)
    sample = ds[0]
    assert len(sample["input_ids"]) > 10
    assert sample["labels"][-1] != -100


def test_collator_bs1_only(tokenizer):
    with pytest.raises(ValueError):
        SearchR1Collator(tokenizer)([{}, {}])


def test_dataset_warns_on_legacy_document_format(tokenizer, tmp_path, capsys):
    """旧格式文档块要显式告警，避免静默训练出与评测不一致的格式。"""
    from search_comp.data.searchr1_dataset import SearchR1SFTDataset

    legacy = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Question: q?"},
            {"role": "assistant", "content": "<answer>a</answer>"},
            {
                "role": "user",
                "content": '<information>[Document 1] (ID: 1, Score: 0.5)\n"T"\nT body.\n</information>',
            },
        ]
    }
    unified = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Question: q?"},
            {"role": "assistant", "content": "<answer>a</answer>"},
            {"role": "user", "content": "<information>Doc 1 T\nT body.</information>"},
        ]
    }

    def _write(name, rows):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    SearchR1SFTDataset(str(_write("legacy.jsonl", [legacy])), tokenizer)
    assert "旧格式文档块" in capsys.readouterr().out

    SearchR1SFTDataset(str(_write("unified.jsonl", [unified])), tokenizer)
    assert "旧格式文档块" not in capsys.readouterr().out
