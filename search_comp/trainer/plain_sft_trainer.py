"""Qwen3.5 纯文本 SFT 训练器（无 Beacon、不压缩）。

专为普通 SFT 定制全链接口：模型只需学会
``think → <search>query</search> → 观察 <information> → <answer>`` 行为。

- 数据：Search-R1 官方 ``messages`` 格式 SFT 轨迹
  （``outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl``），复用
  :class:`search_comp.data.searchr1_dataset.SearchR1SFTDataset`，
  **不截断**，loss 只对 assistant 输出计算。
- 模型：经 :func:`search_comp.models.plain_qwen3.load_sft_model` 装载的
  ``Qwen3_5ForCausalLM`` 纯文本模型，无任何压缩/Beacon 参数。
- 训练：transformers 标准 :class:`~transformers.Trainer`，自带 tqdm 进度条、
  loss 实时显示、checkpoint 与日志，参考 Beacon 训练脚本 `beacon_trainer.py`
  的组织方式。

``use_lora: true`` 时冻结基础权重只训练低秩适配器，保存的是 adapter；
训练结束后自动用 :mod:`search_comp.trainer.merge_lora` 合并到 ``final_merged/``
（Beacon 版对应 :mod:`search_comp.trainer.merge_beacon_lora`）。

用法::

    python -m search_comp.trainer.plain_sft_trainer --config configs/train/qwen35_plain_sft.yaml
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from ..utils.runtime import (
    count_parameters,
    load_yaml_config,
    prepare_run_artifacts,
    require_keys,
    resolve_experiment_dir,
    write_json,
)


class PlainSFTDataset(Dataset):
    """把 Search-R1 ``messages`` 轨迹包装成标准 ``(input_ids, labels)`` 数据集。

    复用 :class:`search_comp.data.searchr1_dataset.SearchR1SFTDataset` 的
    tokenize/labels 逻辑（assistant 输出作为 SFT 目标），但丢弃其 ``regions``
    （Beacon 压缩区信息在此处无用）。轨迹保留完整路径、不截断。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 8192):
        from ..data.searchr1_dataset import SearchR1SFTDataset

        self._inner = SearchR1SFTDataset(data_path, tokenizer)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: int):
        sample = self._inner[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }


def collate_sft_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Search-R1 轨迹 collator（仅支持 batch_size=1）。

    轨迹长度差异大且不截断，因此和 Beacon/Search-R1 路径一样固定 ``bs=1``，
    用 ``grad_accum_steps`` 扩大有效 batch。只返回模型入参，丢弃 regions。
    """
    if len(batch) != 1:
        raise ValueError(
            f"Search-R1 数据仅支持 batch_size=1（当前 {len(batch)}）。"
            "请用 grad_accum_steps 扩大有效 batch。"
            "多卡可见时会放大 batch——单卡训练请设 CUDA_VISIBLE_DEVICES=0。"
        )
    sample = batch[0]
    input_ids = sample["input_ids"].unsqueeze(0)
    labels = sample["labels"].unsqueeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


# LoRA 默认目标模块：标准 Qwen/Llama 式注意力 + MLP 投影。
# 可在 config 里用 ``lora_target_modules`` 覆盖，例如仅适配 GatedDeltaNet
# 窗口内输入投影（in_proj_qkv / in_proj_a / in_proj_b）。
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def apply_lora(model, cfg: Dict[str, Any]):
    """挂上 LoRA adapter，冻结基础权重只训练低秩分支。"""
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", DEFAULT_LORA_TARGET_MODULES),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def main_train(
    config_path: str,
    overrides=(),
    resume_from_checkpoint: str | None = None,
) -> None:
    from ..models.plain_qwen3 import load_sft_model, load_sft_tokenizer

    cfg = load_yaml_config(config_path, overrides)
    from ..data.searchr1_dataset import require_searchr1_mode
    require_searchr1_mode(cfg.get("data_mode", "searchr1"))
    require_keys(cfg, ("model_name_or_path", "train_data_path", "output_dir", "exp_name"))

    tokenizer = load_sft_tokenizer(cfg["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    model = load_sft_model(cfg["model_name_or_path"])

    use_lora = cfg.get("use_lora", False)
    if use_lora:
        model = apply_lora(model, cfg)

    max_length = cfg.get("max_length", 8192)
    train_ds = PlainSFTDataset(cfg["train_data_path"], tokenizer, max_length)
    eval_ds = (
        PlainSFTDataset(cfg["val_data_path"], tokenizer, max_length)
        if cfg.get("val_data_path")
        else None
    )

    exp_dir = str(resolve_experiment_dir(cfg))
    prepare_run_artifacts(exp_dir, cfg, config_path, overrides)
    ckpt_dir = os.path.join(exp_dir, "final")

    use_8bit = cfg.get("optim", "adamw_torch") == "adamw_bnb_8bit"
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=cfg.get("per_device_batch_size", 1),
        gradient_accumulation_steps=cfg.get("grad_accum_steps", 8),
        learning_rate=cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "linear"),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        num_train_epochs=cfg.get("num_epochs", 1),
        max_steps=cfg.get("max_train_steps", -1) or -1,
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        logging_steps=cfg.get("logging_steps", cfg.get("log_freq_steps", 5)),
        save_steps=cfg.get("save_freq_steps", 200),
        save_total_limit=2,
        save_strategy="steps",
        prediction_loss_only=True,
        remove_unused_columns=False,
        bf16=cfg.get("use_bf16", True),
        optim="adamw_bnb_8bit" if use_8bit else cfg.get("optim", "adamw_torch"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        report_to="none",
        disable_tqdm=False,
        seed=cfg.get("seed", 42),
        logging_first_step=True,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps", cfg.get("save_freq_steps", 200)),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        run_name=cfg["exp_name"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collate_sft_batch,
    )
    parameter_stats = count_parameters(model)
    write_json(
        os.path.join(exp_dir, "dataset_summary.json"),
        {
            "data": cfg["train_data_path"],
            "use_lora": use_lora,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds is not None else 0,
            **parameter_stats,
        },
    )
    print(
        f"\n[plain-train] 样本={len(train_ds)} LoRA={use_lora} "
        f"可训练参数={parameter_stats['trainable_parameters'] / 1e6:.2f}M 输出={exp_dir}\n",
        flush=True,
    )
    started_at = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    summary = {
        "use_lora": use_lora,
        "total_steps": trainer.state.global_step,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "final_model_path": ckpt_dir,
        "train_metrics": train_result.metrics,
        **parameter_stats,
    }
    write_json(os.path.join(exp_dir, "train_summary.json"), summary)
    if use_lora:
        print(
            f"[plain-train] 已保存 LoRA adapter -> {ckpt_dir}\n"
            "[plain-train] 开始在 CPU 上合并 LoRA adapter",
            flush=True,
        )
        merged_dir = f"{ckpt_dir}_merged"
        subprocess.run(
            [
                sys.executable, "-m", "search_comp.trainer.merge_lora",
                "--base_model_path", cfg["model_name_or_path"],
                "--adapter_path", ckpt_dir,
                "--output_path", merged_dir,
                "--device-map", "cpu",
            ],
            check=True,
        )
        summary["final_adapter_path"] = ckpt_dir
        summary["final_model_path"] = merged_dir
        write_json(os.path.join(exp_dir, "train_summary.json"), summary)
        print(f"[plain-train] 已保存合并模型 -> {merged_dir}", flush=True)
    else:
        print(f"[plain-train] 已保存最终模型 -> {ckpt_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5 纯文本搜索轨迹 SFT（无 Beacon）")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="覆盖 YAML 参数，可重复，例如 --set learning_rate=1e-5",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Trainer checkpoint 路径；支持断点续训",
    )
    args = parser.parse_args()
    main_train(args.config, args.overrides, args.resume_from_checkpoint)
