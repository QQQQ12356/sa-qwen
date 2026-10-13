"""加载 Qwen3.5 的**纯文本**因果 LM（``Qwen3_5ForCausalLM``）。

基础 checkpoint ``Qwen/Qwen3.5-4B`` 是**多模态** ``Qwen3_5ForConditionalGeneration``，
其文本主干嵌套在 ``model.language_model`` 下。本模块把文本权重重新装载进
``Qwen3_5ForCausalLM``（负词表、仅文本、无 vision encoder），用于 SFT 训练
与 Beacon 移植的基线：

- :func:`text_state_dict_keys`：列出 checkpoint 中 ``model.language_model.*``
  与 ``lm_head`` 相关的键，映射为 ``Qwen3_5ForCausalLM`` 的 ``model.*``。
- :func:`load_text_for_causal_lm`：构造并装载纯文本模型。

注意：视觉权重（``model.visual.*``、``language_model.patch_merger.*``）被忽略。
"""

from __future__ import annotations

import os

import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_IGNORE_PREFIXES = ("model.visual.", "model.language_model.patch_merger.", "^mtp")


def _remap_key(key: str) -> str | None:
    """把多维模型的文本权重键映射到纯文本 ForCausalLM 命名。

    ``model.language_model.layers.N.xxx`` -> ``model.layers.N.xxx``，
    ``model.language_model.embed_tokens`` -> ``model.embed_tokens``，
    ``model.language_model.norm`` -> ``model.norm``，
    ``model.language_model.rotary_emb`` -> ``model.rotary_emb``。
    视觉/merger 键返回 None（丢弃）。
    """
    if key.startswith("model.visual.") or "patch_merger" in key:
        return None
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    return key


def load_text_causal_model(
    model_name_or_path: str = "Qwen/Qwen3.5-4B",
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
):
    """装载 Qwen3.5 纯文本因果 LM。

    从 ``Qwen3_5ForConditionalGeneration`` checkpoint 提取文本主干权重，
    装入 ``Qwen3_5ForCausalLM``。**纯文本模型，可用于标准 SFT 训练。**

    Args:
        model_name_or_path: HF 模型 ID 或本地目录。
        torch_dtype: 参数 dtype。
        device_map: HF device_map。
        trust_remote_code: 是否允许远端代码。

    Returns:
        ``Qwen3_5ForCausalLM`` 实例（已 ``eval()``）。
    """
    from transformers import (
        AutoConfig,
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    if "BeaconQwen3ForCausalLM" in (config.architectures or []):
        raise ValueError("Beacon checkpoint 请使用 search_comp.models.model_loader.load_model")
    if config.model_type == "qwen3":
        from transformers import Qwen3ForCausalLM
        return Qwen3ForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map=device_map,
            trust_remote_code=trust_remote_code,
        ).eval()

    # 已经是纯文本 checkpoint（原生 SFT 保存的 Qwen3_5ForCausalLM）时没有
    # text_config，直接加载即可——否则下面的权重重映射无从谈起。
    if not hasattr(config, "text_config"):
        # Beacon checkpoint 同样是纯文本布局（beacon 超参写在 config 里），
        # 但它必须经 load_beacon_qwen3_5 重建压缩参数；在此放行会静默丢弃
        # 训练好的 beacon 权重，得到一个「看起来正常」的无压缩模型。
        if "BeaconQwen3_5ForCausalLM" in (getattr(config, "architectures", None) or []):
            raise ValueError(
                f"{model_name_or_path} 是 Beacon checkpoint，请用 "
                "search_comp.models.model_loader.load_model 加载。"
            )
        model = Qwen3_5ForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        return model.eval()

    text_config = config.text_config

    # 构造纯文本模型
    model = Qwen3_5ForCausalLM(text_config)
    model.resize_token_embeddings(text_config.vocab_size)

    # 读取 checkpoint 权重（只取文本部分）
    src = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, device_map="cpu"
    )
    remapped = {}
    for k, v in src.state_dict().items():
        nk = _remap_key(k)
        if nk is None:
            continue
        if nk not in model.state_dict():
            print(f"[loader] 忽略多余键: {k}")
            continue
        remapped[nk] = v

    model.load_state_dict(remapped, strict=False)
    del src
    model = model.to(torch_dtype)
    if device_map == "auto":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return model


def load_text_tokenizer(model_name_or_path: str = "Qwen/Qwen3.5-4B"):
    """装载 Qwen3.5 tokenizer（与纯文本模型共享）。"""
    from .qwen35_native import load_tokenizer

    return load_tokenizer(model_name_or_path)