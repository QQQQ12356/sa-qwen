"""分段损失：按片段类别控制「哪些 token 计损失、各占多少权重」。

Search-R1 / 交互式轨迹里一条 assistant 输出被拆成若干片段，各类片段的 token 数
差异极大（实测 4000 条轨迹：``<thinking>`` 占监督 token 的 87%，``<answer>``
只占 1%）。默认的全局 token 平均会让 answer 的梯度被 thinking 长度稀释。本模块
提供按类别开关 + 加权的机制。

片段类别（:data:`SEGMENT_NAMES`，下标即 :data:`SegmentId`）：

===============  ==========================================================
``think_tag``     ``<thinking>`` / ``</thinking>`` 标签本身
``think_content`` 标签之间的思考正文
``search_tag``    ``<search>`` / ``</search>`` 标签本身
``search_content``标签之间的检索 query
``answer_tag``    ``<answer>`` / ``</answer>`` 标签本身
``answer_content``标签之间的最终答案
``other``         其余被监督的 assistant token（标点、换行等）
===============  ==========================================================

权重语义：每个监督 token 的交叉熵乘以它所属类别的权重，分母同步换成权重和，
即 ``L = Σ w_i·CE_i / Σ w_i``。因此**全默认（都 enabled、weight=1.0）时与不加权
逐位等价**，不会改变有效学习率；把某类 ``enabled`` 设为 false 等价于 weight=0，
该类完全不产生梯度也不占分母。

注意权重是**逐 token 乘子**，不是类别级的份额。实测 answer 只占 1% 的 token，
若想让它在损失里占到 ~10%，需要 weight 约 10~50（见 docs 里的算式）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence

import torch

#: 片段类别名（下标即 token 的类别 id）。
SEGMENT_NAMES: tuple = (
    "think_tag",
    "think_content",
    "search_tag",
    "search_content",
    "answer_tag",
    "answer_content",
    "other",
)

#: 不属于任何被监督类别的占位 id（未监督 token / beacon / 提示词）。
SEGMENT_IGNORE: int = -1

#: 类别名 -> 类别 id。
SEGMENT_IDS: Dict[str, int] = {name: index for index, name in enumerate(SEGMENT_NAMES)}

#: 匹配 `<thinking>` / `<think>` / `<search>` / `<answer>` 的开闭标签。
_TAG_PATTERN = re.compile(r"</?(thinking|think|search|answer)>")

#: 标签名 -> 类别前缀（`<think>` 与 `<thinking>` 归为同一类）。
_TAG_KIND = {"thinking": "think", "think": "think", "search": "search", "answer": "answer"}


# ======================================================================
# 配置
# ======================================================================
@dataclass
class SegmentWeight:
    """单个片段类别的开关与权重。

    Args:
        enabled: 是否对该类别的 token 计算损失。false 等价于 ``weight=0``。
        weight: 逐 token 乘子，必须非负。
    """

    enabled: bool = True
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"片段权重不能为负，收到 {self.weight}")
        if not self.enabled and self.weight != 0:
            # 归一化：关闭即权重 0，避免「关了但权重非零」这种自相矛盾的配置
            self.weight = 0.0

    @property
    def effective_weight(self) -> float:
        """实际生效的权重（未启用时为 0）。"""
        return float(self.weight) if self.enabled else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"enabled": bool(self.enabled), "weight": float(self.weight)}

    @classmethod
    def from_dict(cls, data: Any) -> "SegmentWeight":
        """从 dict / bool / 数字构造，便于 YAML 简写。"""
        if isinstance(data, SegmentWeight):
            return data
        if isinstance(data, bool):
            return cls(enabled=data, weight=1.0 if data else 0.0)
        if isinstance(data, (int, float)):
            return cls(enabled=float(data) != 0, weight=float(data))
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(f"片段权重必须是 dict / bool / 数字，收到 {type(data).__name__}")
        unknown = set(data) - {"enabled", "weight"}
        if unknown:
            raise ValueError(f"未知的片段权重字段: {sorted(unknown)}")
        return cls(
            enabled=bool(data.get("enabled", True)),
            weight=float(data.get("weight", 1.0)),
        )


@dataclass
class LossSegmentConfig:
    """各类片段的损失开关与权重。

    每个字段都是一个 :class:`SegmentWeight`；字段名即 :data:`SEGMENT_NAMES` 的
    成员。默认全部启用且权重 1.0，此时损失与不加权完全一致。
    """

    think_tag: SegmentWeight = field(default_factory=SegmentWeight)
    think_content: SegmentWeight = field(default_factory=SegmentWeight)
    search_tag: SegmentWeight = field(default_factory=SegmentWeight)
    search_content: SegmentWeight = field(default_factory=SegmentWeight)
    answer_tag: SegmentWeight = field(default_factory=SegmentWeight)
    answer_content: SegmentWeight = field(default_factory=SegmentWeight)
    other: SegmentWeight = field(default_factory=SegmentWeight)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        for name in SEGMENT_NAMES:
            value = getattr(self, name)
            if not isinstance(value, SegmentWeight):
                setattr(self, name, SegmentWeight.from_dict(value))

    @property
    def is_default(self) -> bool:
        """是否等价于「不加权」——全启用且权重全 1。"""
        return all(
            getattr(self, name).enabled and getattr(self, name).weight == 1.0
            for name in SEGMENT_NAMES
        )

    def weight_table(self) -> torch.Tensor:
        """返回 ``(num_segments,)`` 的 float32 权重表，下标为类别 id。"""
        return torch.tensor(
            [getattr(self, name).effective_weight for name in SEGMENT_NAMES],
            dtype=torch.float32,
        )

    def enabled_names(self) -> List[str]:
        """当前参与损失的类别名。"""
        return [name for name in SEGMENT_NAMES if getattr(self, name).enabled]

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in SEGMENT_NAMES}

    @classmethod
    def from_dict(cls, data: Any) -> "LossSegmentConfig":
        """从 YAML/JSON dict 构造；允许只写部分类别，未写的用默认值。"""
        if isinstance(data, LossSegmentConfig):
            return data
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ValueError(f"loss_segments 必须是映射，收到 {type(data).__name__}")
        unknown = set(data) - set(SEGMENT_NAMES)
        if unknown:
            raise ValueError(
                f"未知的片段类别: {sorted(unknown)}；可选 {list(SEGMENT_NAMES)}"
            )
        return cls(**{name: SegmentWeight.from_dict(data[name]) for name in data})


def field_default(field_info) -> Any:
    """取 dataclass 字段的默认值，兼容 ``default_factory``。"""
    from dataclasses import MISSING

    if field_info.default is not MISSING:
        return field_info.default
    return field_info.default_factory()


def segment_field_names() -> Sequence[str]:
    """:class:`LossSegmentConfig` 的字段名（与 :data:`SEGMENT_NAMES` 一致）。"""
    return tuple(f.name for f in fields(LossSegmentConfig))


# ======================================================================
# 文本 -> 类别
# ======================================================================
def classify_assistant_text(text: str, tokenizer) -> List[int]:
    """把一段 assistant 文本的每个 token 归到一个片段类别。

    用 tokenizer 的 ``offset_mapping`` 把 token 对齐到字符区间，再按标签正则切分：
    标签本身归 ``*_tag``，开标签到对应闭标签之间的内容归 ``*_content``，其余归
    ``other``。token 的类别按其**起始字符**所在区间判定。

    Args:
        text: 一段 assistant 输出（如 ``<thinking>...</thinking><search>q</search>``）。
        tokenizer: HuggingFace 分词器；需要支持 ``return_offsets_mapping``。

    Returns:
        与 ``tokenizer(text)["input_ids"]`` 等长的类别 id 列表（取值见
        :data:`SEGMENT_NAMES` 的下标）。若分词器不支持 offset mapping，则整段
        退化为 ``other``——保证 ``labels != -100`` 的位置总有合法类别。
    """
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    token_count = len(encoded["input_ids"])
    if offsets is None:
        return [SEGMENT_IDS["other"]] * token_count

    char_labels = _char_labels(text)
    other_id = SEGMENT_IDS["other"]
    text_length = len(text)
    result: List[int] = []
    for start, _end in offsets:
        if start is None or start >= text_length:
            result.append(other_id)
        else:
            result.append(SEGMENT_IDS[char_labels[start]])
    return result


def _char_labels(text: str) -> List[str]:
    """逐字符标注类别名，供 token 按起始字符查表。"""
    labels = ["other"] * len(text)
    cursor = 0
    current = "other"
    for match in _TAG_PATTERN.finditer(text):
        _fill(labels, cursor, match.start(), current)
        kind = _TAG_KIND[match.group(1)]
        _fill(labels, match.start(), match.end(), f"{kind}_tag")
        # 开标签进入内容态，闭标签回到 other
        current = "other" if match.group(0).startswith("</") else f"{kind}_content"
        cursor = match.end()
    _fill(labels, cursor, len(text), current)
    return labels


def _fill(labels: List[str], start: int, end: int, value: str) -> None:
    for index in range(start, end):
        labels[index] = value


def segment_ids_to_weights(
    segment_ids: Optional[torch.Tensor],
    loss_segments: LossSegmentConfig,
) -> Optional[torch.Tensor]:
    """把类别 id 张量转成逐位置 float 权重；``SEGMENT_IGNORE`` 位置权重为 0。

    Args:
        segment_ids: ``(batch, seq_len)`` 的类别 id；None 表示不启用加权。
        loss_segments: 片段权重配置。

    Returns:
        同形状的 float32 权重张量；``segment_ids`` 为 None 时返回 None。
    """
    if segment_ids is None:
        return None
    table = loss_segments.weight_table().to(segment_ids.device)
    weights = table[segment_ids.clamp(min=0)]
    return weights * (segment_ids >= 0).to(weights.dtype)
