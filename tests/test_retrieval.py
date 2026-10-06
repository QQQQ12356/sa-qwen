"""BM25 检索器的单元测试。"""

import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
from search_comp.data.retrieval import BM25Retriever


def _make_corpus(path):
    docs = [
        {"id": "1", "title": "Paris", "text": "Paris is the capital of France."},
        {"id": "2", "title": "Rome", "text": "Rome is the capital of Italy."},
        {"id": "3", "title": "Beijing", "text": "Beijing is the capital of China."},
    ]
    with open(path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")


def test_bm25_retrieve_topk():
    """BM25 应检索到与 query 最相关的文档。"""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        _make_corpus(tmp.name)
    retriever = BM25Retriever(tmp.name)
    results = retriever.retrieve("capital of France", topk=1)
    assert results[0]["title"] == "Paris"
    os.unlink(tmp.name)


def test_bm25_missing_corpus_raises():
    """语料不存在时应抛出友好错误。"""
    import pytest

    with pytest.raises(FileNotFoundError):
        BM25Retriever("/nonexistent/corpus.jsonl")
