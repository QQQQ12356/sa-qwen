"""Qwen3 原生纯文本模型加载（无 Beacon）。"""
import torch

DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
PLAIN_SFT_ARCH = "Qwen3ForCausalLM"


def load_plain_qwen3(model_name_or_path=DEFAULT_MODEL, torch_dtype=torch.bfloat16,
                     device_map="auto", trust_remote_code=True):
    from transformers import AutoConfig, Qwen3ForCausalLM
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if "BeaconQwen3ForCausalLM" in (config.architectures or []):
        raise ValueError("Beacon checkpoint 请使用 load_beacon_model")
    return Qwen3ForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, device_map=device_map,
        trust_remote_code=trust_remote_code,
    ).eval()


def load_sft_model(model_name_or_path=DEFAULT_MODEL, torch_dtype=torch.bfloat16,
                   device_map="auto", trust_remote_code=True):
    """保留统一 SFT API，按模型类型分派到两个独立的架构加载模块。"""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if config.model_type == "qwen3":
        loader = load_plain_qwen3
    elif config.model_type in {"qwen3_5", "qwen3_5_text"}:
        from .plain_qwen35 import load_sft_model as loader
    else:
        raise ValueError(f"不支持的纯文本模型类型: {config.model_type}")
    return loader(model_name_or_path, torch_dtype, device_map, trust_remote_code)


def load_sft_tokenizer(model_name_or_path=DEFAULT_MODEL):
    from .model_loader import load_tokenizer
    return load_tokenizer(model_name_or_path)
