"""build_corpus_searchr1 语料抽取的单元测试。

覆盖从 Search-R1 ``<information>`` 块解析 ``{id, title, text}`` 文档、
跨样本去重、以及与既有 hotpotqa 语料的 ``merge_base`` 合并。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from search_comp.data.build_corpus_searchr1 import (
    build_corpus_from_searchr1_jsonl,
    parse_document_blocks,
)

# 与真实 Search-R1 数据格式一致的样本文档块（NQ 数值 ID + 带引号标题）。
_SAMPLE_INFO = (
    "<information>[Document 1] (ID: 1160246, Score: 0.861)\n"
    '"DeLorean time machine"\n'
    "DeLorean time machine The DeLorean time machine is a fictional car.\n"
    "\n"
    "[Document 2] (ID: 1192876, Score: 0.863)\n"
    '"DeLorean DMC-12"\n'
    "DeLorean DMC-12 The DeLorean is a sports car.\n"
    "</information>"
)


def test_parse_document_blocks_basic():
    docs = parse_document_blocks(_SAMPLE_INFO)
    assert len(docs) == 2

    d0 = docs[0]
    assert d0["id"] == "1160246"
    assert d0["title"] == "DeLorean time machine"
    # 保留原文格式：标题行 + 正文行（正文以重复标题开头）。
    assert d0["text"].startswith("DeLorean time machine\nDeLorean time machine The DeLorean")

    d1 = docs[1]
    assert d1["id"] == "1192876"
    assert d1["title"] == "DeLorean DMC-12"


def test_parse_document_blocks_handles_empty_info():
    assert parse_document_blocks("no document here") == []
    assert parse_document_blocks("") == []


def test_build_corpus_from_searchr1_dedup(tmp_path):
    # 两条样本引用同一文档 -> 语料去重后只有 1 个文档。
    line = {"messages": [{"role": "user", "content": _SAMPLE_INFO}]}
    data = tmp_path / "train.jsonl"
    out = tmp_path / "corpus.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")
        f.write(json.dumps(line) + "\n")

    count = build_corpus_from_searchr1_jsonl(str(data), str(out))
    assert count == 2
    with open(out, "r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    assert len(rows) == 2  # 两个不同文档
    ids = {r["id"] for r in rows}
    assert ids == {"1160246", "1192876"}


def test_build_corpus_merge_base(tmp_path):
    line = {"messages": [{"role": "user", "content": _SAMPLE_INFO}]}
    data = tmp_path / "train.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")

    # 既有语料：含一个与 jsonl 重复的 title+text + 一个不重复的。
    base = tmp_path / "base.jsonl"
    with open(base, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "id": "DeLorean time machine|||DeLorean",
            "title": "DeLorean time machine",
            # 与 parser 产出的 text 完全一致，用于验证去重。
            "text": "DeLorean time machine\nDeLorean time machine The DeLorean time machine is a fictional car.",
        }) + "\n")
        f.write(json.dumps({
            "id": "Existing|||other",
            "title": "Existing",
            "text": "Existing\nSome other doc.",
        }) + "\n")

    out = tmp_path / "merged.jsonl"
    count = build_corpus_from_searchr1_jsonl(str(data), str(out), merge_base=str(base))
    with open(out, "r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f if x.strip()]
    # 去重后：jsonl 2 文档 + base 中 1 个不重复 = 3
    assert count == 3
    assert len(rows) == 3