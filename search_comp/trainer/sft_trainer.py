"""基于 Accelerate 的 Search-R1 Beacon SFT 训练入口。

只接受 messages 格式，逐样本保留完整轨迹，用梯度累积扩大有效 batch。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from ..data.searchr1_dataset import SearchR1Collator, SearchR1SFTDataset, require_searchr1_mode
from ..models.beacon_config import BeaconConfig
from ..models.model_loader import (
    freeze_llm_except_beacon,
    get_trainable_param_stats,
    load_model,
    load_tokenizer,
    save_model,
)


@dataclass
class TrainerConfig:
    """SFT 训练超参数。"""

    #: 模型与数据
    model_name_or_path: str = "Qwen/Qwen3-4B-Thinking-2507"
    train_data_path: str = ""
    val_data_path: Optional[str] = ""
    max_length: int = 8192
    #: 仅支持 Search-R1 messages 数据。
    data_mode: str = "searchr1"
    #: 训练
    per_device_batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    max_steps: Optional[int] = None
    #: 模型
    freeze_llm: bool = False
    gradient_checkpointing: bool = True
    use_bf16: bool = True
    #: 日志与保存
    exp_name: str = "beacon_sft"
    output_dir: str = "outputs"
    save_freq_steps: int = 500
    log_freq_steps: int = 10
    seed: int = 42
    #: 其他配置（原样透传记录）
    extra: Dict[str, Any] = field(default_factory=dict)


class BeaconSFTTrainer:
    """Beacon SFT 训练器。

    Args:
        config: :class:`TrainerConfig`。
    """

    def __init__(self, config: TrainerConfig):
        require_searchr1_mode(config.data_mode)
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision="bf16" if config.use_bf16 else "no",
            gradient_accumulation_steps=config.grad_accum_steps,
        )
        self.device = self.accelerator.device

        # 记录参数规模
        self.accelerator.print(
            f"训练设备: {self.accelerator.device}，进程数: {self.accelerator.num_processes}"
        )

        self._setup_dirs()
        self._record_config()

    # ------------------------------------------------------------------
    # 目录与配置记录
    # ------------------------------------------------------------------
    def _setup_dirs(self) -> None:
        """创建输出目录。"""
        self.exp_dir = os.path.join(
            self.config.output_dir, "models", self.config.exp_name
        )
        os.makedirs(self.exp_dir, exist_ok=True)

    def _record_config(self) -> None:
        """把训练超参数写入 ``config.json``（每次训练记录一次，便于对齐）。"""
        cfg = self.config.__dict__.copy()
        cfg["num_gpus"] = self.accelerator.num_processes
        cfg["device_map"] = (
            str(self.accelerator.device)
            if self.accelerator.num_processes == 1
            else f"cuda:{self.accelerator.process_index}"
        )
        path = os.path.join(self.exp_dir, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        if self.accelerator.is_main_process:
            print(f"[trainer] 超参数已记录到 {path}")

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Any]:
        """执行训练主循环。

        Returns:
            训练统计信息（总步数、总时长、最佳验证 loss 等）。
        """
        cfg = self.config

        # ---- 加载模型与数据 ----
        beacon_cfg = None
        if cfg.extra.get("beacon"):
            beacon_cfg = BeaconConfig.from_dict(cfg.extra["beacon"])
        model = load_model(cfg.model_name_or_path, beacon_config=beacon_cfg)
        tokenizer = load_tokenizer(cfg.model_name_or_path)

        if cfg.freeze_llm:
            freeze_llm_except_beacon(model)
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False

        stats = get_trainable_param_stats(model)
        self.accelerator.print(
            f"[trainer] 总参数 {stats['total_params']/1e6:.1f}M，可训练参数 {stats['trainable_params']/1e6:.1f}M"
        )

        if cfg.per_device_batch_size != 1:
            raise ValueError("显式 Beacon 状态前向要求 per_device_batch_size=1；请用 grad_accum_steps 增大有效 batch")
        train_ds = SearchR1SFTDataset(
            cfg.train_data_path, tokenizer,
            doc_region_split=model.beacon_config.beacon_doc_region_split,
        )
        collator = SearchR1Collator(
            tokenizer, question_memory_v1=model.beacon_config.beacon_question_memory_v1,
        )
        batch_size = 1

        dataloader = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=collator,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        # ---- 优化器与调度器 ----
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        total_steps = (
            cfg.max_steps
            if cfg.max_steps
            else (len(dataloader) * cfg.num_epochs // cfg.grad_accum_steps)
        )
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        model, optimizer, dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, dataloader, scheduler
        )

        # beacon 多窗口前向 + 梯度检查点会重复使用参数，DDP 下需启用静态图模式
        if cfg.gradient_checkpointing and hasattr(model, "_set_static_graph"):
            model._set_static_graph()
            self.accelerator.print(
                "[trainer] 已启用 DDP 静态图模式（支持 beacon 多窗口 + 梯度检查点）"
            )

        # ---- 训练循环 ----
        model.train()
        global_step = 0
        running_loss = 0.0
        start_time = time.time()
        steps_in_epoch = max(len(dataloader) // cfg.grad_accum_steps, 1)

        for epoch in range(cfg.num_epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(dataloader):
                with self.accelerator.accumulate(model):
                    loss = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        compress_regions=batch["regions"],
                        question_input_ids=batch.get("question_input_ids"),
                    ).loss
                    self.accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                running_loss += loss.detach().float().item()
                epoch_loss += loss.detach().float().item()

                if (
                    self.accelerator.is_main_process
                    and (step + 1) % cfg.log_freq_steps == 0
                ):
                    avg = running_loss / cfg.log_freq_steps
                    lr = scheduler.get_last_lr()[0]
                    self.accelerator.print(
                        f"[train] epoch {epoch+1}/{cfg.num_epochs} step {global_step+1} "
                        f"loss {avg:.4f} lr {lr:.2e}"
                    )
                    running_loss = 0.0

                global_step += 1

                # 保存检查点
                if (step + 1) % cfg.save_freq_steps == 0:
                    self._save_checkpoint(model, tokenizer, global_step)

                if cfg.max_steps is not None and global_step >= cfg.max_steps:
                    break

            avg_epoch_loss = epoch_loss / max(steps_in_epoch, 1)
            self.accelerator.print(
                f"[train] epoch {epoch+1} 完成，平均 loss {avg_epoch_loss:.4f}"
            )
            if cfg.max_steps is not None and global_step >= cfg.max_steps:
                break

        # ---- 保存最终模型 ----
        self._save_checkpoint(model, tokenizer, global_step, is_final=True)

        elapsed = time.time() - start_time
        result = {
            "exp_name": cfg.exp_name,
            "total_steps": global_step,
            "elapsed_seconds": round(elapsed, 1),
            "final_loss": round(avg_epoch_loss, 4),
        }
        if self.accelerator.is_main_process:
            with open(
                os.path.join(self.exp_dir, "train_summary.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        return result

    # ------------------------------------------------------------------
    def _print_memory(self, tag: str) -> None:
        """打印当前 GPU 内存占用。"""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.memory_reserved() / 1e9
            self.accelerator.print(
                f"[mem][{tag}] allocated={alloc:.2f}GB reserved={total:.2f}GB"
            )

    # ------------------------------------------------------------------
    def _save_checkpoint(
        self, model, tokenizer, step: int, is_final: bool = False
    ) -> None:
        """保存模型检查点（只在主进程执行）。

        Args:
            model: accelerate 包装后的模型。
            tokenizer: tokenizer。
            step: 全局步数。
            is_final: 是否最终保存。
        """
        if not self.accelerator.is_main_process:
            return
        unwrapped = self.accelerator.unwrap_model(model)
        sub_dir = "final" if is_final else f"checkpoint-{step}"
        save_dir = os.path.join(self.exp_dir, sub_dir)
        try:
            save_model(unwrapped, tokenizer, save_dir)
            print(f"[trainer] 已保存模型到 {save_dir}")
        except OSError as exc:
            print(f"[trainer] 保存模型失败: {exc}")


def main_train(config_path: str) -> Dict[str, Any]:
    """从 YAML 配置启动训练。

    Args:
        config_path: YAML 配置路径。

    Returns:
        训练统计信息。
    """
    import yaml

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # beacon 配置放入 extra 透传
    beacon_cfg = raw.pop("beacon", None)
    extra = raw.pop("extra", {})
    if beacon_cfg:
        extra["beacon"] = beacon_cfg
    raw["extra"] = extra

    valid_fields = set(TrainerConfig.__dataclass_fields__.keys())
    cfg = TrainerConfig(**{k: v for k, v in raw.items() if k in valid_fields})
    # 记录完整原始配置
    cfg.extra = {**cfg.extra, "raw_config": raw}

    trainer = BeaconSFTTrainer(cfg)
    return trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Beacon RAG SFT 训练")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置路径")
    args = parser.parse_args()
    result = main_train(args.config)
    print(f"[main] 训练完成: {result}")
