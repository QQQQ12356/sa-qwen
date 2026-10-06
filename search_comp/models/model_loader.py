"""模型加载/保存工具。

负责：
- 按配置加载 Qwen3 / Qwen3.5 基础模型并挂载 beacon 参数。
- 保存带 beacon 参数的模型（config.json 中含 beacon_* 字段）。
- 参数冻结（可选：只训练 beacon 参数）。
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel

from .beacon_config import BeaconConfig
DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"


def load_model(model_name_or_path=DEFAULT_MODEL, beacon_config=None,
               torch_dtype=torch.bfloat16, attn_implementation="eager",
               device_map="auto", trust_remote_code=False):
    """按 checkpoint model_type 分派；原生和已训练 Beacon 均可加载。"""
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if config.model_type == "qwen3":
        from .beacon_qwen3 import load_beacon_qwen3
        loader = load_beacon_qwen3
    elif config.model_type in {"qwen3_5", "qwen3_5_text"}:
        from .beacon_qwen35 import load_beacon_qwen3_5
        loader = load_beacon_qwen3_5
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}，支持 Qwen3 / Qwen3.5")
    return loader(model_name_or_path, beacon_config=beacon_config,
                  torch_dtype=torch_dtype, device_map=device_map)


load_beacon_model = load_model


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = False):
    """加载 tokenizer，并确保已知特殊 token 存在。"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def freeze_llm_except_beacon(model: PreTrainedModel) -> None:
    """冻结 LLM 主体参数，只训练 beacon 参数（beacon_embed + beacon_*_proj）。

    Args:
        model: 已挂载 beacon 参数的模型。
    """
    for name, param in model.named_parameters():
        is_beacon = ("beacon" in name) or (name == "lm_head.weight")
        param.requires_grad = is_beacon


def get_trainable_param_stats(model: PreTrainedModel) -> dict:
    """统计可训练参数数量，返回 dict。"""
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return {"total_params": total, "trainable_params": trainable}


def save_model(model: PreTrainedModel, tokenizer, output_dir: str) -> None:
    """保存模型、tokenizer 与配置到目录。

    Args:
        model: beacon 模型。
        tokenizer: tokenizer。
        output_dir: 输出目录（不存在则创建）。

    Raises:
        OSError: 目录写入失败时抛出。
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)
    except Exception as exc:  # noqa: BLE001
        raise OSError(f"保存模型到 {output_dir} 失败: {exc}") from exc
