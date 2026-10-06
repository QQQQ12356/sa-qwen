"""BM25 在线检索器。

基于 ``rank_bm25`` 的纯 Python 实现，无需外部服务。用于：
- 训练阶段离线预检索每个问题的 top-k 文档（生成静态 SFT 数据）。
- 测试阶段对每个问题在线检索 top-k 文档（推理时实时检索）。

语料来自 HotpotQA 的 ``context`` 字段（按 (title, sentences) 去重后的段落）。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from .trajectory import format_document_blocks

_DEFAULT_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _default_tokenize(text: str) -> List[str]:
    """默认分词：小写并抽取字母数字 token。"""
    return _DEFAULT_TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """基于 BM25Okapi 的稀疏检索器。

    Args:
        corpus_path: JSONL 语料路径，每行 ``{"id", "title", "text"}``。
        tokenize_fn: 分词函数，默认英文小写词元。
    """

    def __init__(self, corpus_path: str, tokenize_fn=_default_tokenize):
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"语料文件不存在: {corpus_path}")
        self.tokenize_fn = tokenize_fn
        self.docs: List[Dict[str, Any]] = []
        self._tokenized: List[List[str]] = []
        self._id_to_idx: Dict[str, int] = {}
        self._load(corpus_path)

    def _load(self, corpus_path: str) -> None:
        """加载语料并构建 BM25 索引。

        Args:
            corpus_path: JSONL 语料路径。
        """
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    idx = len(self.docs)
                    self.docs.append(doc)
                    self._id_to_idx[doc["id"]] = idx
                    self._tokenized.append(self.tokenize_fn(doc["text"]))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"加载语料 {corpus_path} 失败: {exc}") from exc

        if not self.docs:
            raise RuntimeError(f"语料为空: {corpus_path}")
        self.bm25 = BM25Okapi(self._tokenized)

    def retrieve(self, query: str, topk: int = 10) -> List[Dict[str, Any]]:
        """检索与 query 最相关的 top-k 文档。

        Args:
            query: 查询文本（通常是问题）。
            topk: 返回的文档数。

        Returns:
            ``[{id, title, text, score}]``，按分数降序。
        """
        query_tokens = self.tokenize_fn(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[::-1][:topk]
        results = []
        for idx in top_indices:
            doc = self.docs[idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc.get("title", ""),
                    "text": doc.get("text", ""),
                    "score": float(scores[idx]),
                }
            )
        return results


def build_corpus_from_hotpotqa(
    datasets: list,
    output_path: str,
) -> None:
    """从 HotpotQA 数据集构建去重段落语料并保存为 JSONL。

    每个段落的 ``id`` 为 ``title|||first_sentence``（同一标题下按首句去重），
    ``text`` 为 ``title\\n<句子>``（与 Search-R1 的 corpus 格式兼容）。

    Args:
        datasets: HotpotQA 的 dataset split 列表（含 ``context`` 字段）。
        output_path: 输出 JSONL 路径。

    Raises:
        OSError: 输出目录写入失败时抛出。
    """
    seen: Dict[str, Dict[str, str]] = {}
    for ds in datasets:
        for ex in ds:
            titles = ex["context"]["title"]
            sentences_list = ex["context"]["sentences"]
            for title, sentences in zip(titles, sentences_list):
                text = "\n".join(sentences)
                doc_id = f"{title}|||{sentences[0] if sentences else ''}"
                if doc_id not in seen:
                    seen[doc_id] = {
                        "id": doc_id,
                        "title": title,
                        "text": f"{title}\n{text}",
                    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for doc in seen.values():
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise OSError(f"写入语料 {output_path} 失败: {exc}") from exc

    print(f"[build_corpus] 语料段落数: {len(seen)} -> {output_path}")


def format_docs_as_reference(retrieved: List[Dict[str, Any]]) -> str:
    """把检索结果格式化为 ``<information>`` 块内的文本。

    渲染规则见 :func:`search_comp.data.trajectory.format_document_blocks`：每篇文档
    为 ``Doc <序号> <标题>`` + 正文（正文以标题开头时去重），文档之间空行分隔，
    不输出 ``(ID: ..., Score: ...)`` / ``(Title: ...)`` 检索元信息。训练数据里的
    ``<information>`` 块使用同一渲染函数，保证训练与评测逐字一致。

    Args:
        retrieved: ``retrieve()`` 返回的结果列表（``{title, text, ...}``）。

    Returns:
        格式化后的文档文本（不含 ``<information>`` 标签）。
    """
    return format_document_blocks(retrieved)
