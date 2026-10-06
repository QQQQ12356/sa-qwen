"""Qwen3.5 原生（无 Beacon）搜索轨迹 SFT 训练入口。

基于 **transformers 标准 Trainer** 做搜索轨迹监督微调：模型只需学会
``think → <search>query</search> → 观察 <information> → <answer>`` 行为。

训练数据仅支持 Search-R1 messages，不截断完整搜索轨迹。

满足「训练过程可视化到终端、参考标准 Trainer 训练方式」。

用法::

    python -m search_comp.trainer.native_trainer --config configs/train/native_qwen35.yaml
"""

from __future__ import annotations

import json
import os
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
from ..utils.trainer_callbacks import JsonlMetricsCallback



#: 受支持的训练数据格式，与 ``configs/train`` 下的 ``data_mode`` 取值一致。
SUPPORTED_DATA_MODES = ("searchr1",)






def collate_for_searchr1(batch: List[Dict[str, Any]]):
    """Search-R1 轨迹的 collator（仅支持 batch_size=1）。

    :class:`~search_comp.data.searchr1_dataset.SearchR1SFTDataset` 返回的是
    Python list，并附带 ``regions`` / ``n_turns`` / ``n_searches`` 等统计键。
    这里转成模型可接受的 tensor 并**只保留模型入参**——多余的键会被 Trainer
    透传给 ``model.forward()`` 而报错。

    轨迹长度差异大且不做截断，因此和 Beacon 路径一样固定 ``bs=1``，
    用 ``grad_accum_steps`` 扩大有效 batch。
    """
    if len(batch) != 1:
        raise ValueError(
            f"Search-R1 数据仅支持 batch_size=1（当前 {len(batch)}）。"
            "请用 grad_accum_steps 扩大有效 batch。"
            "注意 DataLoader 的 batch 是 per_device_batch_size × 可见 GPU 数，"
            "多卡可见时会在这里放大——单卡训练请设 CUDA_VISIBLE_DEVICES=0。"
        )
    sample = batch[0]
    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long)
    labels = torch.tensor([sample["labels"]], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }


def build_dataset(
    data_mode: str, data_path: str, tokenizer, max_length: int = 8192
) -> Dataset:
    """按 ``data_mode`` 选择训练数据集。

    Args:
        data_mode: ``searchr1``。
        data_path: JSONL 数据路径。
        tokenizer: HuggingFace tokenizer。
        max_length: 兼容旧调用签名；Search-R1 不截断，因此不使用。

    Raises:
        ValueError: ``data_mode`` 不是受支持的值。
    """
    if data_mode == "searchr1":
        from ..data.searchr1_dataset import SearchR1SFTDataset

        return SearchR1SFTDataset(data_path, tokenizer)
    raise ValueError(
        f"未知 data_mode={data_mode!r}，可选: {', '.join(SUPPORTED_DATA_MODES)}"
    )


def build_collator(data_mode: str):
    """按 ``data_mode`` 选择数据 collator。

    Raises:
        ValueError: ``data_mode`` 不是受支持的值。
    """
    if data_mode == "searchr1":
        return collate_for_searchr1
    raise ValueError(
        f"未知 data_mode={data_mode!r}，可选: {', '.join(SUPPORTED_DATA_MODES)}"
    )


def apply_lora(model, cfg: Dict[str, Any]):
    """给模型挂上 LoRA adapter，冻结基础权重只训练低秩分支。"""
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
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
    from ..milestones.qwen35_text import load_text_tokenizer

    cfg = load_yaml_config(config_path, overrides)
    from ..data.searchr1_dataset import require_searchr1_mode
    require_searchr1_mode(cfg.get("data_mode", "searchr1"))
    require_keys(cfg, ("model_name_or_path", "train_data_path", "output_dir", "exp_name"))

    tokenizer = load_text_tokenizer(cfg["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    from ..milestones.qwen35_text import load_text_causal_model

    model = load_text_causal_model(cfg["model_name_or_path"])

    use_lora = cfg.get("use_lora", False)
    if use_lora:
        model = apply_lora(model, cfg)

    data_mode = cfg.get("data_mode", "searchr1")
    max_length = cfg.get("max_length", 8192)
    train_ds = build_dataset(data_mode, cfg["train_data_path"], tokenizer, max_length)
    eval_ds = (
        build_dataset(data_mode, cfg["val_data_path"], tokenizer, max_length)
        if cfg.get("val_data_path")
        else None
    )
    collator = build_collator(data_mode)

    exp_dir = str(resolve_experiment_dir(cfg))
    prepare_run_artifacts(exp_dir, cfg, config_path, overrides)
    ckpt_dir = os.path.join(exp_dir, "final")

    # 24GB 单卡微调 2B 模型：用 8-bit Adam 优化器 + 梯度检查点以控制显存
    use_8bit = cfg.get("optim", "adamw_torch") == "adamw_bnb_8bit"
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        # LoRA 冻结了 embedding，梯度检查点需要输入带 requires_grad 才能反传。
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
        include_num_input_tokens_seen=True,
        length_column_name="n_tokens",
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
        # 传入 processor 才会把 tokenizer 写进每个 checkpoint，merge_lora 才能
        # 直接从 checkpoint-* 目录取 tokenizer（否则只有 final/ 能合并）。
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[JsonlMetricsCallback(exp_dir, append=resume_from_checkpoint is not None)],
    )
    parameter_stats = count_parameters(model)
    write_json(
        os.path.join(exp_dir, "dataset_summary.json"),
        {
            "data_mode": data_mode,
            "use_lora": use_lora,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds is not None else 0,
            **parameter_stats,
        },
    )
    print(
        f"\n[native-train] 数据={data_mode} 样本={len(train_ds)} LoRA={use_lora} "
        f"可训练参数={parameter_stats['trainable_parameters'] / 1e6:.2f}M 输出={exp_dir}\n",
        flush=True,
    )
    started_at = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    summary = {
        "data_mode": data_mode,
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
            f"[native-train] 已保存 LoRA adapter -> {ckpt_dir}\n"
            f"[native-train] 评测前请先合并: python -m search_comp.trainer.merge_lora "
            f"--base_model_path {cfg['model_name_or_path']} "
            f"--adapter_path {ckpt_dir} --output_path {ckpt_dir}_merged",
            flush=True,
        )
    else:
        print(f"[native-train] 已保存最终模型 -> {ckpt_dir}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3.5 原生搜索轨迹 SFT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="覆盖 YAML 参数，可重复，例如 --set learning_rate=1e-5",
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="Trainer checkpoint 路径；原生训练支持完整断点续训",
    )
    args = parser.parse_args()
    main_train(args.config, args.overrides, args.resume_from_checkpoint)
