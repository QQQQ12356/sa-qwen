"""``normalize_docs`` 数据规范化脚本的单元测试。

覆盖两种样本结构（Search-R1 ``messages`` / 交互式 ``turns``）、输出文件的幂等性、
以及「输入输出不能是同一文件」的保护。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from search_comp.data.build_corpus_searchr1 import build_corpus_from_searchr1_jsonl
from search_comp.data.normalize_docs import normalize_jsonl, normalize_sample
from search_comp.data.retrieval import format_docs_as_reference
from search_comp.data.trajectory import INFORMATION_PATTERN, format_document_blocks

_MESSAGES_SAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Question: what car?"},
        {"role": "assistant", "content": "<thinking>x</thinking>\n<search>car</search>"},
        {
            "role": "user",
            "content": (
                "<information>[Document 1] (ID: 1160246, Score: 0.861)\n"
                '"DeLorean time machine"\n'
                "DeLorean time machine The DeLorean time machine is a fictional car.\n"
                "</information>"
            ),
        },
    ]
}

_TURNS_SAMPLE = {
    "id": "q1",
    "question": "what car?",
    "answer": "DeLorean",
    "turns": [
        {"query": "car", "docs": "Doc 1 (Title: DeLorean time machine) DeLorean time machine\n"
                                 "DeLorean time machine The DeLorean time machine is a car."}
    ],
    "thinks": ["x"],
    "final_think": "y",
}


def _write(tmp_path, rows, name="in.jsonl"):
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_messages_sample_is_rendered_in_unified_format():
    normalized, changed = normalize_sample(_MESSAGES_SAMPLE)
    assert changed == 1
    info = next(m["content"] for m in normalized["messages"] if "<information>" in m["content"])
    # 正文原样保留（含开头重复的标题），只去掉 [Document N] (ID:, Score:) 元信息。
    assert info == (
        "<information>Doc 1 DeLorean time machine\n"
        "DeLorean time machine The DeLorean time machine is a fictional car.</information>"
    )
    # 非检索消息原样保留（不改并入参）
    assert normalized["messages"][0] is _MESSAGES_SAMPLE["messages"][0]


def test_turns_sample_is_rendered_in_unified_format():
    normalized, changed = normalize_sample(_TURNS_SAMPLE)
    assert changed == 1
    assert normalized["turns"][0]["docs"] == (
        "Doc 1 DeLorean time machine\nDeLorean time machine The DeLorean time machine is a car."
    )


def test_unknown_sample_shape_is_left_untouched():
    sample = {"foo": "bar"}
    normalized, changed = normalize_sample(sample)
    assert changed == 0
    assert normalized is sample


def test_normalize_jsonl_is_idempotent(tmp_path):
    source = _write(tmp_path, [_MESSAGES_SAMPLE, _TURNS_SAMPLE])
    first = tmp_path / "norm1.jsonl"
    second = tmp_path / "norm2.jsonl"

    stats = normalize_jsonl(str(source), str(first))
    assert stats["lines"] == 2
    assert stats["changed_samples"] == 2
    assert stats["changed_blocks"] == 2
    assert stats["docs"] == 2

    # 对已规范化的文件再跑一次：不再改写，输出逐字节相同。
    again = normalize_jsonl(str(first), str(second))
    assert again["changed_samples"] == 0
    assert first.read_bytes() == second.read_bytes()


def test_normalize_jsonl_rejects_same_input_and_output(tmp_path):
    source = _write(tmp_path, [_MESSAGES_SAMPLE])
    with pytest.raises(ValueError, match="不能与 --input_path 相同"):
        normalize_jsonl(str(source), str(source))


def test_normalize_jsonl_output_matches_training_renderer(tmp_path):
    # 规范化后的 <information> 块内容必须与评测侧渲染函数逐字一致。
    source = _write(tmp_path, [_MESSAGES_SAMPLE])
    out = tmp_path / "norm.jsonl"
    normalize_jsonl(str(source), str(out))

    rendered = _read(out)[0]["messages"][3]["content"]
    matched = INFORMATION_PATTERN.fullmatch(rendered)
    assert matched is not None
    docs = [{"title": "DeLorean time machine",
             "text": "DeLorean time machine\n"
                     "DeLorean time machine The DeLorean time machine is a fictional car."}]
    assert matched.group(1) == format_document_blocks(docs)


def test_training_text_equals_eval_render_after_normalization(tmp_path):
    """训练侧文档文本 == 评测侧渲染：这是本次统一格式要保证的核心不变量。

    走完整链路：原始轨迹 -> 规范化 -> 训练装载取到的 ``<information>`` 块内容；
    以及 规范化文件 -> 构建语料 -> 评测侧 ``format_docs_as_reference`` 渲染。
    """
    source = _write(tmp_path, [_MESSAGES_SAMPLE])
    normalized = tmp_path / "norm.jsonl"
    normalize_jsonl(str(source), str(normalized))

    corpus = tmp_path / "corpus.jsonl"
    build_corpus_from_searchr1_jsonl(str(normalized), str(corpus))

    # 训练侧：装载时取块内文本作为压缩区。
    block = _read(normalized)[0]["messages"][3]["content"]
    training_docs_text = INFORMATION_PATTERN.fullmatch(block).group(1)

    # 评测侧：从语料检索到同一批文档后渲染（顺序一致时文本必须逐字相同）。
    eval_docs_text = format_docs_as_reference(_read(corpus))
    assert eval_docs_text == training_docs_text
