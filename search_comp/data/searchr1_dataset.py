"""Search-R1 SFT 数据集（messages 格式）与 collator。

从搜索/检索增强模型的 SFT 轨迹（``messages`` 对话格式）直接构造
beacon 训练样本，**不再**用规则生成轨迹。数据来源为 Search-R1 官方发布的
``{backbone}-instruct-sft.jsonl``（见 :data:`SEARCH_R1_DATA_HINT`），默认放在：

    outputs/data/searchr1/{backbone}-instruct-sft.jsonl

该文件不在本仓库内（约 60MB），首次使用请按提示下载，或直接把
``train_data_path`` 指向本地已有路径。

每行形如（Search-R1 原文格式，含系统 / 用户 / 助手多轮）::

    {"messages": [
        {"role": "system", "content": "You are a helpful and harmless assistant."},
        {"role": "user", "content": "Answer the given question. ... Question: <q>?"},
        {"role": "assistant", "content": "<thinking> ... </thinking>\\n<search> <query> </search>"},
        {"role": "user", "content": "<information> [Document 1] ... </information>"},
        {"role": "assistant", "content": "<thinking> ... </thinking>\\n<answer> <answer> </answer>"},
    ]}

设计要点：

- 训练/评测统一为裸检索协议：初始 system/user prompt 使用 ChatML；assistant 输出与
  后续 ``<information>`` 检索结果直接拼接，不额外插入 user/assistant ChatML 边界。
- **loss（labels）**：只对 ``role == "assistant"`` 的 token 计算（think / search /
  answer 均为模型输出，整体作为 SFT 目标）。
- **压缩区（regions）**：每个裸 ``<information>`` 块的文档正文对应一个压缩区；
  XML 标签保留在上下文内，但不参与 Beacon 压缩。
- 样本结构多变（turn 数、文档长度都不同），且需逐样本对齐压缩区与 loss 区间，
  因此**仅支持 batch_size=1**（用梯度累积扩大有效 batch）。

其他字段（非 ``<information>`` 的 user / system / 助手消息）token 保持原样但不
计损失、不压缩。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from .trajectory import (
    INFORMATION_PATTERN,
    INFO_PREFIX,
    INFO_SUFFIX,
    SEARCH_INSTRUCTION,
    SYSTEM_PROMPT,
    split_document_regions,
)
from ..loss_segments import SEGMENT_IGNORE, LossSegmentConfig, classify_assistant_text

#: 训练数据不在仓库内时的获取提示（拼进报错信息，方便新用户自助解决）。
SEARCH_R1_DATA_HINT = (
    "Search-R1 SFT 轨迹（*-instruct-sft.jsonl）不属于本仓库，请先下载：\n"
    "  huggingface-cli download --repo-type dataset PeterJinGo/nq_hotpotqa_train \\\n"
    "      --include '*instruct-sft.jsonl' --local-dir outputs/data/searchr1\n"
    "或把配置里的 train_data_path 指向已有文件的绝对路径：\n"
    "  bash scripts/24_beacon_train_searchr1.sh configs/train/beacon_qwen35_searchr1.yaml \\\n"
    "      --set train_data_path=/path/to/qwen3-4b-instruct-sft.jsonl"
)

#: 标记文档块的 user 消息前缀（用于识别压缩区）
_INFO_MARKER = "<information>"
_SEARCH_PATTERN = re.compile(r"<search>")
_ANSWER_PATTERN = re.compile(r"<answer>")
_INFORMATION_PATTERN = INFORMATION_PATTERN  # 与 trajectory 共用同一匹配（向后兼容别名）


def align_to_eval_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 Search-R1 官方提示词对齐到评测的 ``build_search_chat_prompt``。

    评测（``search_comp.data.trajectory.build_search_chat_prompt``）的初始上下文是：:

         system = BASE + 完整搜索协议  |  user = 裸问题

    而 Search-R1 官方轨迹数据正好相反：system 只有角色说明，搜索协议被塞进首个
    user 消息（``SEARCH_INSTRUCTION + " Question: <q>?"``）。为了让**训练看到的提示
    与评测完全一致**（train/eval 严格对齐），本函数在装载时改写：

    1. system 内容改为 ``SYSTEM_PROMPT``（角色说明 + ``\\n\\n`` + 完整协议）。
    2. 首个 user 剥离协议前缀，只保留裸问题。

    已对全量数据校验：首个 user 恒以 ``SEARCH_INSTRUCTION`` 开头，其后紧随
    ``Question: ``。改写只影响输入上下文，不影响 loss（system/user 本就以 -100 掩码）。

    Returns:
        改写后的 messages 列表（改动处返回新 dict，不改并入参）。
    """
    if not messages or messages[0].get("role") != "system":
        return messages
    result = [dict(messages[0], content=SYSTEM_PROMPT)]
    if len(messages) > 1 and messages[1].get("role") == "user":
        question = messages[1]["content"]
        if question.startswith(SEARCH_INSTRUCTION):
            question = question[len(SEARCH_INSTRUCTION):]
        question = question.strip().removeprefix("Question:").strip()
        result.append(dict(messages[1], content=question))
        result.extend(messages[2:])
    else:
        result.extend(messages[1:])
    return result


#: 向后兼容别名（旧名）。
_align_to_eval_prompt = align_to_eval_prompt

#: 旧格式文档头（``[Document N] (ID: ..., Score: ...)``）与评测统一格式不一致。
_LEGACY_DOC_MARKER = re.compile(r"\[Document\s+\d+\]\s*\(ID:")

#: 评测侧渲染与训练数据不一致时的修复提示。
LEGACY_DOC_FORMAT_HINT = (
    "检测到旧格式检索文档（[Document N] (ID: ..., Score: ...)）：评测侧的 "
    "<information> 块现在是统一格式（Doc N + 标题行 + 去重正文，无 ID/Score），"
    "继续用旧文件训练会让模型看到的文档与评测不一致。请先规范化数据：\n"
    "  bash scripts/29_normalize_docs.sh\n"
    "  --set train_data_path=outputs/data/searchr1/qwen3-4b-instruct-sft-normalized.jsonl"
)


def _warn_if_legacy_document_format(
    samples: List[Dict[str, Any]], data_path: str
) -> None:
    """训练数据仍是旧文档格式时打印一次显式告警（不阻断，便于复现历史结果）。"""
    legacy_blocks = sum(
        1
        for sample in samples
        for msg in sample.get("messages", [])
        if _LEGACY_DOC_MARKER.search(msg.get("content", "") or "")
    )
    if legacy_blocks:
        print(
            f"[searchr1] 警告：{data_path} 中有 {legacy_blocks} 个旧格式文档块。\n"
            f"{LEGACY_DOC_FORMAT_HINT}",
            flush=True,
        )


class SearchR1SFTDataset(Dataset):
    """把 Search-R1 ``messages`` 格式的 SFT 轨迹转成 beacon 训练样本。

    Args:
        data_path: JSONL 路径，每行 ``{messages: [{role, content}, ...]}``。
        tokenizer: HuggingFace tokenizer（Qwen3.5 系列）。
        loss_segments: 分段损失配置；为 None 或全默认时不生成 ``loss_segment_ids``，
            保持与历史行为一致（逐 token 等权）。
        doc_region_split: 是否把每个 ``<information>`` 块按子文档切成多个压缩区。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        loss_segments: Optional[LossSegmentConfig] = None,
        doc_region_split: bool = False,
    ):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}\n{SEARCH_R1_DATA_HINT}")
        self.tokenizer = tokenizer
        self.doc_region_split = doc_region_split
        # 全默认权重等价于不加权，此时连分类都不做（省一次带 offset 的分词）
        self.loss_segments = None if (loss_segments is None or loss_segments.is_default) else loss_segments
        self.samples: List[Dict[str, Any]] = []
        try:
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        # 规范化：若整行已是 dict 且含 "messages"
                        if "messages" not in obj:
                            raise ValueError(f"缺少 messages 字段: {line[:80]}")
                        # 训练/评测严格对齐：把协议移进 system、首个 user 只留裸问题。
                        obj["messages"] = align_to_eval_prompt(obj["messages"])
                        self.samples.append(obj)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"加载数据 {data_path} 失败: {exc}") from exc
        if not self.samples:
            raise RuntimeError(f"数据为空: {data_path}")
        _warn_if_legacy_document_format(self.samples, data_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """把一条 messages 会话编码为 ``input_ids`` / ``labels`` / ``regions``。

        Returns:
            dict：
            - ``input_ids``: 拼接后的 token id 列表。
            - ``labels``: loss 掩码，assistant token 处为自身 id，其余 -100。
            - ``regions``: 各 ``<information>`` 文档块的 token 区间。
            - ``loss_segment_ids``: 与 ``input_ids`` 等长的片段类别 id；未启用
              分段损失时为 None。
            - ``n_turns`` / ``n_searches``: 用于统计（交互轮数、搜索次数）。
        """
        msgs = sample["messages"]
        if not msgs or msgs[0].get("role") != "system":
            raise ValueError(f"messages 应以 system 消息开头: {sample.get('id', '?')}")

        input_ids: List[int] = []
        labels: List[int] = []
        segment_ids: List[int] = []
        regions: List[Tuple[int, int]] = []
        n_searches = 0
        question_parts = []

        seen_assistant = False
        for msg_idx, msg in enumerate(msgs):
            role = msg.get("role")
            content = msg.get("content", "") or ""
            if role == "assistant":
                if not seen_assistant:
                    assistant_prefix_ids = self.tokenizer(
                        "<|im_start|>assistant\n", add_special_tokens=False
                    ).input_ids
                    input_ids.extend(assistant_prefix_ids)
                    labels.extend([-100] * len(assistant_prefix_ids))
                    segment_ids.extend([SEGMENT_IGNORE] * len(assistant_prefix_ids))
                seen_assistant = True
                seg_ids = self.tokenizer(content, add_special_tokens=False).input_ids
                # 模型输出整体作为 SFT 目标（thinking + search + answer）。
                labels.extend(seg_ids)
                segment_ids.extend(self._segment_ids_for(content, seg_ids))
                n_searches += len(_SEARCH_PATTERN.findall(content))
            elif role == "user" and seen_assistant:
                info_match = _INFORMATION_PATTERN.fullmatch(content)
                if info_match is None:
                    raise ValueError(
                        "Search-R1 首个 assistant 消息后的 user 消息必须是 "
                        f"完整 <information> 块: {sample.get('id', msg_idx)}"
                    )
                docs_text = info_match.group(1)
                docs_ids = self.tokenizer(docs_text, add_special_tokens=False).input_ids
                prefix_ids = self.tokenizer(INFO_PREFIX, add_special_tokens=False).input_ids
                suffix_ids = self.tokenizer(INFO_SUFFIX, add_special_tokens=False).input_ids
                start = len(input_ids) + len(prefix_ids)
                if self.doc_region_split:
                    for sub_start, sub_end in split_document_regions(docs_text, self.tokenizer):
                        regions.append((start + sub_start, start + sub_end))
                else:
                    regions.append((start, start + len(docs_ids)))
                seg_ids = prefix_ids + docs_ids + suffix_ids
                labels.extend([-100] * len(seg_ids))
                segment_ids.extend([SEGMENT_IGNORE] * len(seg_ids))
            else:
                # 初始 system / user prompt 使用 ChatML，与评测初始上下文完全一致。
                if seen_assistant:
                    raise ValueError(
                        f"assistant 后不支持 role={role!r} 的非检索消息: {sample.get('id', msg_idx)}"
                    )
                if role == "user":
                    question_parts.append(content)
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                seg_ids = self.tokenizer(text, add_special_tokens=False).input_ids
                labels.extend([-100] * len(seg_ids))
                segment_ids.extend([SEGMENT_IGNORE] * len(seg_ids))

            input_ids.extend(seg_ids)

        n_turns = sum(1 for m in msgs if m.get("role") == "assistant")
        return {
            "input_ids": input_ids,
            "question": "\n".join(question_parts),
            "labels": labels,
            "regions": regions,
            "loss_segment_ids": segment_ids if self.loss_segments is not None else None,
            "n_turns": n_turns,
            "n_searches": n_searches,
        }

    def _segment_ids_for(self, content: str, seg_ids: List[int]) -> List[int]:
        """把一段 assistant 文本按 think/search/answer 标签与正文分类。

        Args:
            content: assistant 消息原文。
            seg_ids: 该消息的 token id 列表（用于对齐长度与兜底）。

        Returns:
            与 ``seg_ids`` 等长的类别 id 列表；未启用分段损失时为全
            :data:`SEGMENT_IGNORE`。
        """
        if self.loss_segments is None:
            return [SEGMENT_IGNORE] * len(seg_ids)
        classified = classify_assistant_text(content, self.tokenizer)
        if len(classified) != len(seg_ids):
            # 分词器在带/不带 offset_mapping 时切分不一致，退回等权而不是错位
            return [SEGMENT_IGNORE] * len(seg_ids)
        return classified

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._tokenize(self.samples[idx])


class SearchR1Collator:
    """Search-R1 数据 collator（仅支持 batch_size=1）。

    Args:
        tokenizer: HuggingFace tokenizer（用于 pad_token_id，其实 bs=1 不会 pad）。
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, question_memory_v1: bool = False):
        self.tokenizer = tokenizer
        self.question_memory_v1 = question_memory_v1
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer 缺少 pad_token，请先设置 pad_token")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """打包一个样本。

        Returns:
            dict：``input_ids``(1,L), ``attention_mask``(1,L), ``labels``(1,L),
            ``regions``（压缩区 token 区间列表）。
        """
        if len(features) != 1:
            raise ValueError(
                f"Search-R1 数据仅支持 batch_size=1（当前 {len(features)}）。"
                "请用 grad_accum_steps 扩大有效 batch。"
            )
        f = features[0]
        input_ids = torch.tensor([f["input_ids"]], dtype=torch.long)
        labels = torch.tensor([f["labels"]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "regions": f["regions"],
            **({"question_input_ids": torch.tensor([
                self.tokenizer(f["question"], add_special_tokens=False).input_ids
            ], dtype=torch.long)} if self.question_memory_v1 else {}),
            **({"loss_segment_ids": torch.tensor(
                [f["loss_segment_ids"]], dtype=torch.long
            )} if f.get("loss_segment_ids") is not None else {}),
        }


def require_searchr1_mode(data_mode: str) -> None:
    """训练数据统一为 Search-R1 messages；不再接受旧轨迹格式。"""
    if data_mode != "searchr1":
        raise ValueError(f"不支持 data_mode={data_mode!r}；当前仅支持 searchr1")
