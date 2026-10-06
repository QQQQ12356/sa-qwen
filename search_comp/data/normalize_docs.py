"""把训练轨迹里的检索文档重写为统一格式（``Doc N <标题>`` + 正文）。

训练侧（Search-R1 轨迹的 ``<information>`` 块、交互式轨迹的 ``turns[].docs``）与
评测侧（BM25 在线检索结果）必须把检索文档渲染成**同一种文本**，否则模型在评测时
看到的是训练分布外的格式。渲染规则见
:func:`search_comp.data.trajectory.format_document_blocks`：每篇文档为
``Doc <序号> <标题>`` + 正文（正文原样保留，仅把开头**连续重复**的标题折叠成一个，
至少保留一次），文档之间空行分隔，不再输出
``[Document N] (ID: ..., Score: ...)`` 或 ``(Title: ...)`` 检索元信息。

本脚本把**已有数据文件**改写成该格式，输出到新文件（不覆盖输入，幂等可重复执行）：

    python -m search_comp.data.normalize_docs \\
        --input_path outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl \\
        --output_path outputs/data/searchr1/qwen3-4b-instruct-sft-normalized.jsonl \\
        --tokenizer Qwen/Qwen3.5-2B

改写完成后把训练配置指过去即可（语料无需重建：评测侧渲染时使用同一函数）：

    bash scripts/24_beacon_train_searchr1.sh configs/train/beacon_qwen35_searchr1.yaml \\
        --set train_data_path=outputs/data/searchr1/qwen3-4b-instruct-sft-normalized.jsonl

支持两种样本结构，按字段自动分派：

- ``{"messages": [...]}``（Search-R1 SFT 轨迹）：重写 ``<information>`` 块内容。
- ``{"turns": [{"docs": ...}, ...]}``（交互式轨迹）：重写 ``turns[].docs``。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from .trajectory import INFORMATION_PATTERN, format_document_blocks, parse_document_blocks


def _normalize_docs_text(docs_text: str) -> str:
    """把一段文档文本重写为统一格式；无法解析出文档时原样返回。"""
    docs = parse_document_blocks(docs_text)
    if not docs:
        return docs_text
    return format_document_blocks(docs)


def normalize_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """重写 ``messages`` 中 ``<information>`` 块的文档。

    Returns:
        ``(新 messages, 改写块数)``；未改动的消息原样复用（不改并入参）。
    """
    changed = 0
    result: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "") or ""
        matched = INFORMATION_PATTERN.fullmatch(content)
        if matched is None:
            result.append(msg)
            continue
        rendered = _normalize_docs_text(matched.group(1))
        if rendered == matched.group(1):
            result.append(msg)
            continue
        changed += 1
        result.append({**msg, "content": f"<information>{rendered}</information>"})
    return result, changed


def normalize_turns(turns: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """重写交互式轨迹 ``turns[].docs`` 的文档。

    Returns:
        ``(新 turns, 改写条数)``；未改动的 turn 原样复用。
    """
    changed = 0
    result: List[Dict[str, Any]] = []
    for turn in turns:
        docs_text = turn.get("docs")
        if not isinstance(docs_text, str) or not docs_text.strip():
            result.append(turn)
            continue
        rendered = _normalize_docs_text(docs_text)
        if rendered == docs_text:
            result.append(turn)
            continue
        changed += 1
        result.append({**turn, "docs": rendered})
    return result, changed


def normalize_sample(sample: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """按样本结构分派到 :func:`normalize_messages` 或 :func:`normalize_turns`。

    Returns:
        ``(新样本, 改写条数)``；无法识别的结构原样返回、改写数为 0。
    """
    if isinstance(sample.get("messages"), list):
        messages, changed = normalize_messages(sample["messages"])
        return ({**sample, "messages": messages} if changed else sample), changed
    if isinstance(sample.get("turns"), list):
        turns, changed = normalize_turns(sample["turns"])
        return ({**sample, "turns": turns} if changed else sample), changed
    return sample, 0


def normalize_jsonl(
    input_path: str,
    output_path: str,
    limit: Optional[int] = None,
    tokenizer=None,
) -> Dict[str, Any]:
    """把 JSONL 里的检索文档重写为统一格式，写入新文件。

    Args:
        input_path: 源 JSONL 路径。
        output_path: 输出 JSONL 路径（必须与 ``input_path`` 不同）。
        limit: 只处理前 N 行（调试用）；None 表示全量。
        tokenizer: 可选的 HuggingFace tokenizer，用于统计改写前后的 token 数。

    Returns:
        统计字典：``lines`` / ``changed_samples`` / ``changed_blocks`` / ``docs`` /
        ``chars_before`` / ``chars_after``（有 tokenizer 时另有 ``tokens_before`` /
        ``tokens_after``）。
    """
    if os.path.abspath(input_path) == os.path.abspath(output_path):
        raise ValueError(
            "--output_path 不能与 --input_path 相同：请输出到新文件，避免覆盖原始数据"
        )

    stats: Dict[str, Any] = {
        "lines": 0,
        "changed_samples": 0,
        "changed_blocks": 0,
        "docs": 0,
        "chars_before": 0,
        "chars_after": 0,
        "tokens_before": 0,
        "tokens_after": 0,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as source, open(
        output_path, "w", encoding="utf-8"
    ) as sink:
        for line in source:
            if limit is not None and stats["lines"] >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            sample = json.loads(stripped)
            before = json.dumps(sample, ensure_ascii=False)
            normalized, changed = normalize_sample(sample)
            after = json.dumps(normalized, ensure_ascii=False)

            stats["lines"] += 1
            if changed:
                stats["changed_samples"] += 1
                stats["changed_blocks"] += changed
            stats["docs"] += _count_docs(normalized)
            stats["chars_before"] += len(before)
            stats["chars_after"] += len(after)
            if tokenizer is not None:
                stats["tokens_before"] += _count_tokens(tokenizer, before)
                stats["tokens_after"] += _count_tokens(tokenizer, after)
            sink.write(after + "\n")
    return stats


def _count_docs(sample: Dict[str, Any]) -> int:
    """统计样本里的文档篇数（用于核对改写没有丢文档）。"""
    if isinstance(sample.get("messages"), list):
        return sum(
            len(parse_document_blocks(msg.get("content", "") or ""))
            for msg in sample["messages"]
        )
    if isinstance(sample.get("turns"), list):
        return sum(
            len(parse_document_blocks(turn.get("docs", "") or ""))
            for turn in sample["turns"]
        )
    return 0


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把训练数据里的检索文档重写为统一格式（Doc N + 标题 + 正文）"
    )
    parser.add_argument("--input_path", type=str, required=True, help="源 JSONL 路径")
    parser.add_argument(
        "--output_path", type=str, required=True, help="输出 JSONL 路径（不能与输入相同）"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="可选：模型/tokenizer 路径，用于统计改写前后的 token 数",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="只处理前 N 行（调试用）"
    )
    args = parser.parse_args()

    tokenizer = None
    if args.tokenizer:
        from ..milestones.qwen35_text import load_text_tokenizer

        tokenizer = load_text_tokenizer(args.tokenizer)

    stats = normalize_jsonl(
        args.input_path, args.output_path, limit=args.limit, tokenizer=tokenizer
    )
    print(
        f"[normalize_docs] 行数={stats['lines']} 改写样本={stats['changed_samples']} "
        f"改写块={stats['changed_blocks']} 文档数={stats['docs']}"
    )
    print(
        f"[normalize_docs] 字符数 {stats['chars_before']} -> {stats['chars_after']}"
        + (
            f" | token 数 {stats['tokens_before']} -> {stats['tokens_after']}"
            if tokenizer is not None
            else ""
        )
    )
    print(f"[normalize_docs] 输出 -> {args.output_path}")
    print(
        "[normalize_docs] 下一步：训练时指向新文件\n"
        f"    --set train_data_path={args.output_path}\n"
        "[normalize_docs] 语料无需重建：评测侧渲染使用同一函数。"
    )


if __name__ == "__main__":
    main()
