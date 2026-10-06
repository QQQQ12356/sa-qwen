"""Beacon 模型共用的工具函数：损失计算与张量切片/拼接。

包含:
- :func:`compute_loss`：带 -100 掩码的交叉熵损失（仅计算答案部分）。
- :func:`slice_tensor` / :func:`cat_tensor`：在序列维度上安全地切片/拼接张量，
  用于从模型输出中按 beacon_indices 抽取 beacon 的 K/V。
- :func:`init_beacon_params`：初始化 beacon 嵌入与投影参数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def beacon_intersect_order(raw_length: int, ratio: int, device: torch.device) -> torch.Tensor:
    """将 chunk 末尾的 Beacon 重排到每 ratio 个原始 token 后，尾片也保留 Beacon。"""
    beacon_size = (raw_length + ratio - 1) // ratio
    raw_indices = torch.arange(raw_length, device=device)
    raw_positions = raw_indices + raw_indices // ratio
    beacon_mask = torch.ones(raw_length + beacon_size, dtype=torch.bool, device=device)
    beacon_mask[raw_positions] = False
    order = torch.empty(raw_length + beacon_size, dtype=torch.long, device=device)
    order[raw_positions] = raw_indices
    order[beacon_mask] = torch.arange(raw_length, raw_length + beacon_size, device=device)
    return order


# ----------------------------------------------------------------------
# 损失计算
# ----------------------------------------------------------------------
def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    shift: bool = True,
    num_items_in_batch: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算带掩码的交叉熵损失。

    ``labels`` 中值为 -100 的位置不参与损失计算（用于忽略指令与文档部分，
    只计算答案部分的损失）。

    Args:
        logits: 形状为 ``(batch, seq_len, vocab_size)`` 的模型输出 logits。
        labels: 形状为 ``(batch, seq_len)`` 的目标 token id，-100 表示忽略。
        shift: 是否对 logits/labels 错位一位（预测下一个 token）。
        num_items_in_batch: 归一化用的有效 token 总数；若为 None 则用实际有效数。

    Returns:
        ``(loss, token_loss)``：标量损失与逐样本的平均 token 损失。
    """
    if shift:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    vocab_size = logits.size(-1)
    flat_logits = logits.view(-1, vocab_size)
    flat_labels = labels.view(-1)

    token_loss = F.cross_entropy(
        flat_logits, flat_labels, reduction="none", ignore_index=-100
    )
    valid_mask = flat_labels != -100
    valid_num = valid_mask.sum()

    if valid_num.item() == 0:
        # 整条序列没有有效标签时返回 0 损失，避免 NaN
        loss = flat_logits.sum() * 0.0
    else:
        if num_items_in_batch is None:
            num_items_in_batch = valid_num
        loss = token_loss.sum() / num_items_in_batch

    # 还原为 (batch, seq_len) 的逐位置损失，便于日志统计
    token_loss = token_loss.view(logits.size(0), logits.size(1))
    return loss, token_loss


# ----------------------------------------------------------------------
# 张量工具（在序列维度 dim=2 上操作 K/V cache）
# ----------------------------------------------------------------------
def slice_tensor(
    x: Optional[torch.Tensor],
    start: Optional[int] = None,
    end: Optional[int] = None,
    index: Optional[torch.Tensor] = None,
    dim: int = 2,
) -> Optional[torch.Tensor]:
    """在 ``dim`` 维度上切片张量，兼容 None 输入与 bool 索引。

    用于从完整的 K/V 中抽出 beacon 位置（``beacon_indices == 1``）的 K/V。

    Args:
        x: 输入张量，可为 None（返回 None）。
        start / end: 切片区间（前闭后开）。
        index: bool 张量索引（与 start/end 互斥，优先级最高）。
        dim: 切片维度，K/V cache 的序列维度为 2。

    Returns:
        切片后的张量；输入为 None 或空切片时返回 None。
    """
    if x is None:
        return None
    if (
        end == 0
        or (start is not None and start == end)
        or (start is not None and start == x.shape[dim])
    ):
        return None
    if index is not None:
        if dim == 2:
            return x[:, :, index]
        return x[:, index] if dim == 1 else x[index]
    if dim == 2:
        return x[:, :, start:end]
    if dim == 1:
        return x[:, start:end]
    return x[start:end]


def cat_tensor(tensors: list, dim: int = 2) -> Optional[torch.Tensor]:
    """拼接一批张量，自动忽略 None；全为 None 时返回 None。"""
    valid = [t for t in tensors if t is not None]
    if len(valid) > 1:
        return torch.cat(valid, dim=dim)
    return valid[0] if len(valid) == 1 else None


# ----------------------------------------------------------------------
# beacon 参数初始化
# ----------------------------------------------------------------------
def init_beacon_params(model: nn.Module, missing_keys: Optional[list] = None) -> None:
    """初始化 beacon 相关参数。

    - ``beacon_embed_tokens``：从 eos/bos 的 embedding 复制初始化。
    - 每层注意力 ``beacon_q/k/v/o_proj``：从原始投影权重复制初始化。

    Args:
        model: 已加载的 Beacon 因果模型（带 beacon 参数但未初始化）。
        missing_keys: ``from_pretrained`` 返回的 ``loading_info["missing_keys"]``。
            只初始化出现在该列表中的 beacon 参数（即 checkpoint 中缺失的参数）。
            为 None 时初始化全部（用于直接构造模型）。

    Notes:
        关键：**不能无条件覆盖**。重新加载微调模型时，beacon 参数已在 checkpoint
        中，若此处用原始投影覆盖会丢失训练成果（见 code review C-1）。
    """
    config = model.config
    beacon_cfg = model.beacon_config
    embed_tokens: nn.Embedding = model.model.embed_tokens
    beacon_embed: nn.Embedding = model.model.beacon_embed_tokens

    def _needs_init(param_name: str) -> bool:
        """参数是否缺失（不在 checkpoint 中）需要初始化。"""
        if missing_keys is None:
            return True
        # 检查权重与偏置任一缺失
        return any(param_name in k for k in missing_keys)

    # 1. 初始化 beacon 嵌入
    if _needs_init("model.beacon_embed_tokens.weight"):
        if beacon_cfg.beacon_embed_init == "eos":
            source_id = config.eos_token_id
        else:
            source_id = config.bos_token_id
        if source_id is not None and source_id < embed_tokens.num_embeddings:
            with torch.no_grad():
                beacon_embed.weight.data.copy_(embed_tokens.weight.data[source_id])
        else:
            with torch.no_grad():
                beacon_embed.weight.data.normal_(mean=0.0, std=config.hidden_size**-0.5)

    # 2. 初始化每层的 beacon 投影
    params = beacon_cfg.beacon_param.split()
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        with torch.no_grad():
            for p in params:
                beacon_proj = getattr(attn, f"beacon_{p}_proj", None)
                ordinal_proj = getattr(attn, f"{p}_proj", None)
                if beacon_proj is None or ordinal_proj is None:
                    continue
                if _needs_init(
                    f"model.layers.{layer_idx}.self_attn.beacon_{p}_proj.weight"
                ):
                    beacon_proj.weight.data.copy_(ordinal_proj.weight.data)
                    if ordinal_proj.bias is not None and beacon_proj.bias is not None:
                        beacon_proj.bias.data.copy_(ordinal_proj.bias.data)
