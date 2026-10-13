"""Qwen3.5 基础大模型的原生（未加 Beacon）加载与文本生成。

Milestone-1 能力探针使用：仅用基础模型 + 标准 ``generate`` 验证 SearchAgent
（think → <search> → observe → <answer>）检索行为，不涉及 Beacon 压缩。

注意：``Qwen/Qwen3.5-4B`` 是**多模态**架构 ``Qwen3_5ForConditionalGeneration``
（含 vision encoder、linear_attention + full_attention 混合层、mRoPE），
因此用 :class:`AutoModelForImageTextToText` 加载；纯文本输入时跳过视觉分支。

本模块与 ``model_loader``（Beacon 微调模型）解耦，便于 Milestone-1 快速探测。
"""

from __future__ import annotations

import torch

#: 待加载的基础模型标识。
QWEN35_4B = "Qwen/Qwen3.5-4B"
QWEN35_2B = "Qwen/Qwen3.5-2B"  # 旧调用兼容；默认使用 4B

#: 需要显式关闭的第三方依赖打印（并发 tokenizer 无关）。
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_tokenizer(model_name_or_path: str = QWEN35_4B):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_chat_model(
    model_name_or_path: str = QWEN35_4B,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
):
    """加载 Qwen3.5 原生模型（文本生成入口）。

    ``Qwen3_5ForConditionalGeneration`` 是视觉-语言模型，用
    ``AutoModelForImageTextToText`` 加载；纯文本 SearchAgent 探测时只喂文本。

    Args:
        model_name_or_path: HF 模型 ID 或本地目录。
        torch_dtype: 推断 dtype（transformers>=5 传 ``dtype``）。
        device_map: HF device_map（"auto" 用 GPU）。

    Returns:
        已 ``eval()`` 并在 ``torch.no_grad()`` 下可 ``generate`` 的模型。
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if config.model_type == "qwen3":
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=torch_dtype, device_map=device_map,
        ).eval()

    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model


def generate_text(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
    stopping_criteria=None,
) -> str:
    """对给定 ``input_ids`` 生成文本。

    Args:
        input_ids: ``(1, seq_len)`` 长整型 token 张量（含 chat 模板前缀）。
        stopping_criteria: 额外的 text 停止标准（见
            :class:`KeywordStoppingCriteria`）。
        max_new_tokens / do_sample / temperature / top_p: 生成超参数。

    Returns:
        ``generate`` 后相对输入新增部分的解码文本（``skip_special_tokens=False``，
        保留 ``<search>`` / `` thinking`` 等特殊 token 以便解析）。
    """
    kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    if stopping_criteria is not None:
        kwargs["stopping_criteria"] = stopping_criteria

    import torch

    with torch.no_grad():
        generated = model.generate(**kwargs)
    # 只取新增部分
    new_tokens = generated[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens.tolist(), skip_special_tokens=False)