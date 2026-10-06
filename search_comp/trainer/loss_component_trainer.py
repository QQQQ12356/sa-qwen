"""把复合损失的各个组成分量并入 Trainer 的日志。

Beacon 的损失包含主 CE、续写、读出蒸馏和输出 KL，但 ``Trainer`` 默认只记录
``outputs.loss`` 一个标量，看不出各项各占多少。模型在
:attr:`~search_comp.models.beacon_qwen3.BeaconQwen3_5ForCausalLM._last_loss_parts`
里按名留痕（detach 后的标量），本模块的 :class:`LossComponentTrainer` 把它按
**log 窗口**对 micro-batch 取算术平均后写进 ``logs``：终端打印行、
``trainer_metrics.jsonl``、``trainer_state_summary.json`` 的 ``log_history`` 三处都有。

未激活的项恒为 ``0``，因此日志的键集合稳定，不随配置变化。分量以张量形式累加、
只在 log 时取值，不额外触发 GPU 同步。只在训练日志行（含 ``loss`` 键）注入，
eval 行（``eval_loss``）不注入；eval 前向也不计入窗口。

分量直接从 ``model`` 上按属性名读取：``PeftModel.__getattr__`` 会逐层代理到内层
模型，因此 LoRA 包装下读到的是同一个字典。这条假设由
``tests/test_loss_component_logging.py`` 用**真实** PeftModel 钉住 —— 不要自己
沿 ``base_model`` 手工解包，``LoraModel.base_model`` 指向的是文本主干
（``Qwen3_5TextModel``）而不是被包装的 CausalLM。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from transformers import Trainer

#: 模型上存放复合损失分量的属性名（与 beacon 模型的约定）。
LOSS_PARTS_ATTRIBUTE = "_last_loss_parts"


class LossComponentTrainer(Trainer):
    """在训练日志行里额外打印复合损失的每个分量。

    用法与 :class:`transformers.Trainer` 完全一致，直接替换即可。模型没有
    :data:`LOSS_PARTS_ATTRIBUTE`（原生 / 纯文本模型）时行为与父类完全相同。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._loss_part_totals: Dict[str, torch.Tensor] = {}
        self._loss_part_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        # eval 前向（model.eval()）不污染下一个训练日志窗口。
        parts = getattr(model, LOSS_PARTS_ATTRIBUTE, None) if model.training else None
        if parts:
            for name, value in parts.items():
                total = self._loss_part_totals.get(name)
                self._loss_part_totals[name] = value if total is None else total + value
            self._loss_part_count += 1
        return loss

    def log(self, logs: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """把窗口内的分量均值写进 ``logs`` 并清零，再交给父类打印。"""
        # 训练日志行带 loss；eval 行只有 eval_loss，不应混入训练分量。
        if self._loss_part_count and "loss" in logs:
            for name, total in self._loss_part_totals.items():
                logs[name] = float(total) / self._loss_part_count
            self._loss_part_totals = {}
            self._loss_part_count = 0
        return super().log(logs, *args, **kwargs)
