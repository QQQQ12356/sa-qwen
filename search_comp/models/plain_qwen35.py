"""Qwen3.5 **纯文本 SFT** 的建模入口（无 Beacon、不压缩）。

和 Beacon 路径各用一个建模模块相对应，本模块是「普通因果推理 SFT 定制全链接口」：

- Beacon 侧：:mod:`search_comp.models.beacon_qwen35`，在窗口上挂压缩参数。
- 本模块：直接把 ``Qwen/Qwen3.5-4B``（多模态 ``ForConditionalGeneration``）的
  文本主干装载为 ``Qwen3_5ForCausalLM``，无任何压缩参数，走标准因果 LM 前向。

权重装载逻辑复用 :func:`search_comp.milestones.qwen35_text.load_text_causal_model`
（视觉权重被剔除、键做 ``model.language_model.*`` -> ``model.*`` 重映射），这里只提供
给 SFT 训练/评测统一的薄封装，避免各训练/评测入口各自散落加载逻辑。

```python
from search_comp.models.plain_qwen35 import load_sft_model, load_sft_tokenizer

model = load_sft_model("Qwen/Qwen3.5-4B")
tokenizer = load_sft_tokenizer("Qwen/Qwen3.5-4B")
```
"""

from __future__ import annotations

from typing import Optional

import torch

#: 纯文本 SFT 模型的架构名（写入 checkpoint 的 config.json，供评测识别）。
PLAIN_SFT_ARCH = "Qwen3_5ForCausalLM"

#: 默认基础 checkpoint。
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def load_sft_model(
    model_name_or_path: str = DEFAULT_MODEL,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    """装载用于 SFT 的 Qwen3.5 纯文本因果 LM。

    Args:
        model_name_or_path: HF 模型 ID 或本地目录。可以是基础多模态 checkpoint，
            也可以是已合并好的纯文本 SFT checkpoint。
        torch_dtype: 权重 dtype（默认 bf16）。
        device_map: HF device_map。
        trust_remote_code: 是否允许远端代码。

    Returns:
        ``Qwen3_5ForCausalLM`` 实例（已 ``eval()``）。
    """
    from ..milestones.qwen35_text import load_text_causal_model

    return load_text_causal_model(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )


def load_sft_tokenizer(model_name_or_path: str = DEFAULT_MODEL):
    """装载与纯文本模型共享的 tokenizer。"""
    from ..milestones.qwen35_text import load_text_tokenizer

    return load_text_tokenizer(model_name_or_path)


# 便捷别名：与 Beacon 侧 ``load_beacon_qwen3_5`` 命名对齐。
load_plain_qwen3_5 = load_sft_model

__all__ = [
    "PLAIN_SFT_ARCH",
    "DEFAULT_MODEL",
    "load_sft_model",
    "load_sft_tokenizer",
    "load_plain_qwen3_5",
]