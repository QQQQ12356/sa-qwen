"""Qwen3 / Qwen3.5 Beacon 搜索轨迹 SFT 训练入口（transformers.Trainer，实时进度）。

在交互式搜索轨迹上训练 Beacon 压缩模型：

- 只压缩检索到的 ``<information>`` 文档块（``regions``），文档 K/V 压缩为 beacon。
- 损失只在模型生成片段（think / <search> / <answer>）上计算，不含检索内容。
- 用 ``transformers.Trainer`` 驱动训练：实时 tqdm 进度条、按 ``logging_steps`` 打印
  损失/学习率/耗时，``max_steps`` 天然支持"按训练步数训练"。

注意：Trainer 默认保存只写 LoRA adapter，无法被 ``load_beacon_qwen3_5`` 直接加载，
因此关闭 Trainer 自带保存（``save_strategy="no"``），改由自定义 callback 合并保存为
标准 ``BeaconQwen3_5ForCausalLM`` checkpoint（``checkpoint-<N>`` / ``final``）。

用法::

    python -m search_comp.trainer.beacon_trainer --config configs/train/beacon_qwen35_searchr1.yaml
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback, TrainingArguments

from ..models.beacon_config import BeaconConfig
from ..milestones.qwen35_text import load_text_tokenizer
from ..utils.runtime import (
    count_parameters,
    load_yaml_config,
    prepare_run_artifacts,
    require_keys,
    resolve_experiment_dir,
    write_json,
)
from ..utils.trainer_callbacks import JsonlMetricsCallback
from .beacon_distillation_trainer import BeaconDistillationTrainer


def _cuda_memory_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_available": True,
        "device": device,
        "device_name": torch.cuda.get_device_name(device),
        "free_bytes": free_bytes,
        "total_bytes": total_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _sample_length_summary(dataset, sample_size: int, seed: int, beacon_window: int = None) -> dict:
    sample_size = min(max(sample_size, 0), len(dataset))
    if sample_size == 0:
        return {"sampled": 0}
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_size].tolist()
    lengths = []
    supervised = []
    information = []
    regions = []
    for index in indices:
        item = dataset[index]
        lengths.append(len(item["input_ids"]))
        supervised.append(sum(label != -100 for label in item["labels"]))
        sample_regions = item.get("regions", [])
        regions.extend(end - start for start, end in sample_regions)
        information.append(sum(end - start for start, end in sample_regions))

    def percentile(values, fraction):
        ordered = sorted(values)
        position = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
        return ordered[position]

    summary = {
        "sampled": sample_size,
        "tokens": {
            "min": min(lengths), "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95), "p99": percentile(lengths, 0.99),
            "max": max(lengths),
        },
        "supervised_tokens": {
            "p50": percentile(supervised, 0.50),
            "p95": percentile(supervised, 0.95), "max": max(supervised),
        },
        "information_tokens": {
            "p50": percentile(information, 0.50),
            "p95": percentile(information, 0.95), "max": max(information),
        },
    }
    if beacon_window and regions:
        # 仅 window 模式（beacon_continuation_per_beacon=false）适用：压缩区长度 >
        # beacon_window 才会产生第 2 个窗口，也就是续写监督的前提。beacon 模式下每个窗口
        # 本身就是 chunk 边界，长度 >= 2 的压缩区都有监督，该比值不再反映实际覆盖。
        multi_window = sum(1 for size in regions if size > beacon_window)
        summary["compress_regions"] = {
            "count": len(regions),
            "p50": percentile(regions, 0.50),
            "p95": percentile(regions, 0.95),
            "max": max(regions),
            "gt_beacon_window": multi_window,
            "gt_beacon_window_ratio": round(multi_window / len(regions), 4),
        }
    return summary


class _SaveBeaconCheckpoints(TrainerCallback):
    """按步数保存 LoRA adapter 与 Beacon 参数，不触碰正在训练的模型。

    Args:
        exp_dir: 输出目录（``output_dir/models/<exp_name>``）。
        save_freq: 每多少优化器步保存一次。
        model: 训练中的模型（可能是 PEFT 包装）。
        tokenizer: 用于随 checkpoint 保存。
        use_lora: 是否 LoRA。
        max_steps: 按步数训练的终止步数；为 None 时按 epoch 终止。
    """

    def __init__(self, exp_dir: str, save_freq: int, model, tokenizer,
                 use_lora: bool, max_steps) -> None:
        self.exp_dir = exp_dir
        self.save_freq = save_freq
        self.model = model
        self.tokenizer = tokenizer
        self.use_lora = use_lora
        self.max_steps = max_steps

    def on_step_end(self, args, state, control, **kwargs):
        gs = state.global_step
        if gs > 0 and gs % self.save_freq == 0:
            ckpt = os.path.join(self.exp_dir, f"checkpoint-{gs}")
            _save_training_checkpoint(self.model, self.tokenizer, ckpt, self.use_lora)
            print(f"[beacon] 已保存 adapter/state -> {ckpt}", flush=True)
        if self.max_steps is not None and gs >= self.max_steps:
            control.should_training_stop = True  # 精确在 max_steps 处停止

    def on_train_end(self, args, state, control, **kwargs):
        final_dir = os.path.join(self.exp_dir, "final_adapter")
        _save_training_checkpoint(self.model, self.tokenizer, final_dir, self.use_lora)
        print(f"[beacon] 训练结束，adapter/state 已保存到 {final_dir}", flush=True)


def configure_whole_trajectory(cfg: dict) -> None:
    """SFT 不按长度切窗；旧 YAML 的窗口选项不再限制轨迹计算。"""
    beacon = cfg.setdefault("beacon", {})
    beacon.update(
        beacon_information_mode="whole",
        beacon_window_enabled=False,
        beacon_doc_region_split=False,
        beacon_keep_window=0,
    )


def main_train(
    config_path: str,
    overrides=(),
) -> None:
    from ..models.model_loader import load_beacon_model

    cfg = load_yaml_config(config_path, overrides)
    configure_whole_trajectory(cfg)
    shuffle_train_data = cfg.setdefault("shuffle_train_data", True)
    if not isinstance(shuffle_train_data, bool):
        raise ValueError("shuffle_train_data 必须为 bool（true 或 false）")
    from ..data.searchr1_dataset import require_searchr1_mode
    require_searchr1_mode(cfg.get("data_mode", "searchr1"))
    require_keys(cfg, ("model_name_or_path", "train_data_path", "output_dir", "exp_name"))
    if cfg.get("beacon", {}).get("beacon_output_distill_weight", 0) > 0:
        cfg["distill_teacher_model_name_or_path"] = (
            cfg.get("distill_teacher_model_name_or_path") or cfg["model_name_or_path"]
        )
    exp_dir = str(resolve_experiment_dir(cfg))
    prepare_run_artifacts(exp_dir, cfg, config_path, overrides)

    torch.manual_seed(cfg.get("seed", 42))

    tokenizer = load_text_tokenizer(cfg["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token

    beacon_cfg = BeaconConfig.from_dict(cfg.get("beacon", {}))
    print(f"[beacon] SFT 已取消长度分窗：{beacon_cfg.describe_layout()}", flush=True)
    model = load_beacon_model(cfg["model_name_or_path"], beacon_config=beacon_cfg)
    model.train()
    if cfg.get("gradient_checkpointing", False):
        print(
            "[beacon] 警告：混合 Beacon 使用显式跨窗状态，当前不启用 Trainer 的层级重计算；"
            "该配置仅保留在运行记录中。",
            flush=True,
        )

    # 可选 LoRA：冻结基础模型，仅在注意力/MLP 投影上训练低秩适配器，
    # beacon 参数保持全量可训练。显著降低优化器/梯度显存（约 -10GB）。
    use_lora = cfg.get("use_lora", False)
    if use_lora:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            # 目标模块从 config 读取（lora_target_modules），默认标准注意力+MLP 投影；
            # 如需仅适配窗口内 GatedDeltaNet 输入投影，可设为
            #   [in_proj_qkv, in_proj_a, in_proj_b]
            target_modules=cfg.get(
                "lora_target_modules",
                [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        # beacon 投影（beacon_*_proj / beacon_embed_tokens）保持全量可训练。
        # 关键：get_peft_model 会把所有基础参数 requires_grad=False（含 beacon），
        # 若不重新开启，loss 将不连接任何可训练参数，backward 报

        for name, param in model.named_parameters():
            if "beacon" in name:
                param.requires_grad = True
        beacon_trainable = [
            n for n, p in model.named_parameters() if "beacon" in n and p.requires_grad
        ]
        if not beacon_trainable:
            raise RuntimeError(
                "LoRA 模式下未找到任何 require_grad 的 beacon 压缩参数；"
                "请检查 beacon_config.enable_beacon 与模型结构。"
            )
        model.print_trainable_parameters()

    data_mode = "searchr1"
    loss_segments = beacon_cfg.loss_segments
    if not loss_segments.is_default:
        print(
            "[beacon] 分段损失已启用："
            + ", ".join(
                f"{name}={getattr(loss_segments, name).effective_weight:g}"
                for name in loss_segments.enabled_names()
            ),
            flush=True,
        )
    from ..data.searchr1_dataset import SearchR1Collator, SearchR1SFTDataset

    train_ds = SearchR1SFTDataset(
        cfg["train_data_path"], tokenizer, loss_segments=loss_segments,
        doc_region_split=beacon_cfg.beacon_doc_region_split,
        return_loss_segment_ids=beacon_cfg.beacon_output_distill_strategy == "stratified",
    )
    collator = SearchR1Collator(tokenizer, question_memory_v1=beacon_cfg.beacon_question_memory_v1)
    eval_ds = (
        SearchR1SFTDataset(
            cfg["val_data_path"], tokenizer, loss_segments=loss_segments,
            doc_region_split=beacon_cfg.beacon_doc_region_split,
            return_loss_segment_ids=beacon_cfg.beacon_output_distill_strategy == "stratified",
        )
        if cfg.get("val_data_path") else None
    )
    print(f"[beacon] data_mode=searchr1，{len(train_ds)} 条 Search-R1 SFT 轨迹", flush=True)

    parameter_stats = count_parameters(model)
    n_train = parameter_stats["trainable_parameters"]
    print(
        f"[beacon] 可训练参数 {n_train/1e6:.2f}M / 总 "
        f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M",
        flush=True,
    )

    grad_accum = cfg.get("grad_accum_steps", 8)
    save_freq = cfg.get("save_freq_steps", 200)
    logging_steps = cfg.get("logging_steps", 5)
    max_train_steps = cfg.get("max_train_steps", None)

    if max_train_steps is not None:
        max_train_steps = int(max_train_steps)
        print(f"[beacon] 按训练步数训练：max_train_steps={max_train_steps} "
              f"(steps/epoch≈{len(train_ds)//grad_accum})", flush=True)
    else:
        print(f"[beacon] 按 epoch 训练：num_epochs={cfg.get('num_epochs', 1)} "
              f"(steps/epoch≈{len(train_ds)//grad_accum})", flush=True)

    args = TrainingArguments(
        output_dir=exp_dir,
        per_device_train_batch_size=1,          # collator 强制 bs=1，通过grad_accum 扩大有效 batch
        gradient_accumulation_steps=grad_accum,
        learning_rate=cfg.get("learning_rate", 5e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        # 二选一：max_steps>0 则按步数训练（覆盖 num_epochs）；否则按 epoch
        max_steps=max_train_steps if max_train_steps is not None else -1,
        num_train_epochs=cfg.get("num_epochs", 1),
        logging_steps=logging_steps,
        logging_first_step=True,
        include_num_input_tokens_seen=True,
        log_level="info",
        save_strategy="no",                     # 由 _SaveBeaconCheckpoints 保存 adapter/state
        report_to=[],                            # 不写 wandb/tensorboard
        remove_unused_columns=False,             # 保留 collator 的 regions 透传给 forward
        seed=cfg.get("seed", 42),
        train_sampling_strategy="random" if shuffle_train_data else "sequential",
        disable_tqdm=False,                      # 实时进度条
        fp16=False,
        bf16=cfg.get("use_bf16", True),
        optim=cfg.get("optim", "adamw_torch"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "constant"),
        warmup_steps=cfg.get("warmup_steps", 0),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps", save_freq),
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        run_name=cfg["exp_name"],
    )

    teacher = None
    if beacon_cfg.beacon_output_distill_weight > 0:
        from ..milestones.qwen35_text import load_text_causal_model

        teacher_path = cfg.get("distill_teacher_model_name_or_path") or cfg["model_name_or_path"]
        if not teacher_path:
            raise ValueError("输出蒸馏需要配置 distill_teacher_model_name_or_path")
        teacher_device = cfg.get("distill_teacher_device", "auto")
        if teacher_device not in {"auto", "cpu"}:
            raise ValueError("distill_teacher_device 必须为 auto 或 cpu")
        # 两者须有相同 token ID 语义；仅检查 vocab_size 无法发现等长的错位词表。
        teacher_tokenizer = load_text_tokenizer(teacher_path)
        if teacher_tokenizer.get_vocab() != tokenizer.get_vocab():
            raise ValueError("蒸馏 teacher 与 student 的 tokenizer 词表不一致")
        teacher = load_text_causal_model(
            teacher_path,
            torch_dtype=next(model.parameters()).dtype,
            device_map=teacher_device,
        )
        print(f"[beacon] 冻结原生输出蒸馏 teacher: {teacher_path}", flush=True)

    trainer = BeaconDistillationTrainer(
        model=model,
        distillation_teacher=teacher,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        # 每个 log_step 的损失行会额外打印 ce_loss / cont_loss / readout_loss
        # （超类只是 Trainer 的子类，其余行为不变）。
        callbacks=[
            _SaveBeaconCheckpoints(exp_dir, save_freq, model, tokenizer, use_lora, max_train_steps),
            JsonlMetricsCallback(exp_dir),
        ],
    )

    t0 = time.time()
    length_summary = _sample_length_summary(
        train_ds,
        int(cfg.get("length_diagnostics_samples", 256)),
        int(cfg.get("seed", 42)),
        beacon_window=(beacon_cfg.beacon_window if beacon_cfg.beacon_information_mode == "window" else None),
    )
    write_json(
        os.path.join(exp_dir, "dataset_summary.json"),
        {
            "data_mode": data_mode,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds) if eval_ds is not None else 0,
            "length_diagnostics": length_summary,
            **parameter_stats,
        },
    )
    print(f"[beacon] 长度抽样统计: {length_summary}", flush=True)
    print(
        f"[beacon] 数据顺序={'随机打乱' if shuffle_train_data else 'JSONL 原始顺序'}；"
        f"seed={cfg.get('seed', 42)}，开始训练 ...",
        flush=True,
    )
    try:
        train_result = trainer.train()
    except torch.OutOfMemoryError as error:
        failure = {
            "error_type": type(error).__name__,
            "error": str(error),
            "cuda_memory": _cuda_memory_snapshot(),
            "recommendations": [
                "减小 beacon.beacon_loss_chunk_size（例如 64 -> 32）",
                "启用 beacon.beacon_cpu_offload_activations=true",
                "减小 beacon.beacon_window 或 LoRA rank",
                "确认 GPU 没有其他进程占用显存",
            ],
        }
        write_json(Path(exp_dir) / "failure.json", failure)
        print(f"[beacon] CUDA OOM，诊断已保存到 {exp_dir}/failure.json", flush=True)
        raise

    final_adapter_dir = os.path.join(exp_dir, "final_adapter")
    final_dir = os.path.join(exp_dir, "final")
    if use_lora:
        _merge_final_adapter_in_subprocess(
            cfg["model_name_or_path"], final_adapter_dir, final_dir, cfg.get("beacon", {})
        )
    else:
        _save_training_checkpoint(model, tokenizer, final_dir, use_lora=False)
    write_json(
        os.path.join(exp_dir, "train_summary.json"),
        {
            "total_steps": trainer.state.global_step,
            "elapsed_seconds": round(time.time() - t0, 3),
            "final_model_path": final_dir,
            "train_metrics": train_result.metrics,
            **parameter_stats,
        },
    )
    print(f"[beacon] 训练完成，训练步数={trainer.state.global_step}，耗时 {time.time()-t0:.1f}s，"
          f"模型保存到 {final_dir}")


def _save_training_checkpoint(model, tokenizer, save_dir: str, use_lora: bool) -> None:
    """保存可恢复的 LoRA adapter 与 Beacon 参数 state dict。"""
    os.makedirs(save_dir, exist_ok=True)
    if use_lora:
        model.save_pretrained(save_dir)
        beacon_state = {
            name.removeprefix("base_model.model."): param.detach().cpu()
            for name, param in model.named_parameters()
            if "beacon" in name
        }
        torch.save(beacon_state, os.path.join(save_dir, "beacon_state.pt"))
    else:
        model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def _merge_final_adapter_in_subprocess(
    base_model_path: str, adapter_path: str, output_path: str, beacon_config: dict
) -> None:
    """在独立 Python 进程中加载 adapter 后合并，避免修改 live PeftModel。"""
    command = [
        sys.executable,
        "-m",
        "search_comp.trainer.merge_beacon_lora",
        "--base_model_path",
        base_model_path,
        "--adapter_path",
        adapter_path,
        "--output_path",
        output_path,
        "--beacon_config_json",
        json.dumps(beacon_config),
    ]
    print(f"[beacon] 在独立进程合并 final adapter -> {output_path}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3 / Qwen3.5 Beacon 搜索轨迹 SFT")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        help="覆盖 YAML 参数，可重复，例如 --set beacon.beacon_ratio=32",
    )
    args = parser.parse_args()
    main_train(args.config, args.overrides)
