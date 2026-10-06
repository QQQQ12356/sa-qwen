"""beacon_interactive_eval 从 Search-R1 jsonl 读题目+真值的单元测试。

覆盖 ``load_questions_from_searchr1_jsonl``：
- 裸问题（协议已剥除）；
- 真值来自最后那个 ``<answer>``；
- id 形如 ``sr-N``（jsonl 无原生 id）；
- ``max_questions`` 采样上限。
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from search_comp.data.trajectory import SEARCH_INSTRUCTION
from search_comp.evaluation.beacon_interactive_eval import load_questions_from_searchr1_jsonl


def _sample(q: str, answer: str):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful and harmless assistant."},
            {"role": "user", "content": f"{SEARCH_INSTRUCTION} Question: {q}"},
            {"role": "assistant", "content": "<thinking>\nneed search\n</thinking>\n<search>some query</search>"},
            {"role": "user", "content": "<information>[Document 1] some docs.</information>"},
            {"role": "assistant", "content": f"<thinking>\nnow I know\n</thinking>\n<answer> {answer} </answer>"},
        ]
    }


def test_load_questions_returns_bare_question_answer_id(tmp_path):
    data = tmp_path / "train.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        f.write(json.dumps(_sample("Who founded Google?", "Larry Page and Sergey Brin")) + "\n")
    rows = load_questions_from_searchr1_jsonl(str(data))
    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == "Who founded Google?"
    assert row["answer"] == "Larry Page and Sergey Brin"
    assert row["id"] == "sr-0"


def test_load_questions_respects_max_questions(tmp_path):
    data = tmp_path / "train.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps(_sample(f"Q{i}?", f"A{i}")) + "\n")
    rows = load_questions_from_searchr1_jsonl(str(data), max_questions=2)
    assert len(rows) == 2
    assert rows[0]["id"] == "sr-0"
    assert rows[1]["id"] == "sr-1"


def test_load_questions_ids_numbered_across_calls(tmp_path):
    data = tmp_path / "train.jsonl"
    with open(data, "w", encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps(_sample(f"Q{i}?", f"A{i}")) + "\n")
    rows = load_questions_from_searchr1_jsonl(str(data))
    assert [r["id"] for r in rows] == ["sr-0", "sr-1", "sr-2"]