"""只在压缩后 keep 段的有效生成位置蒸馏，冻结教师保留完整原文。"""

import torch

from .loss_component_trainer import LossComponentTrainer


def select_distillation_positions(labels, regions, tokens):
    """返回原始序列中的预测行号 i（其监督目标为 labels[i+1]）。

    相邻子文档之间没有 keep 段，不产生读出。每个压缩后的 keep 段最多
    选择 tokens 个有效位置，防止长思考片段支配 KL。
    """
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("Beacon 输出蒸馏仅支持 batch_size=1")
    selected = []
    for index, (_, end) in enumerate(regions):
        next_start = regions[index + 1][0] if index + 1 < len(regions) else labels.shape[1]
        stop = min(next_start, labels.shape[1] - 1)
        if stop <= end:
            continue
        valid = (labels[0, end + 1:stop + 1] != -100).nonzero(as_tuple=True)[0]
        selected.append(valid[:tokens] + end)
    return torch.cat(selected) if selected else labels.new_empty(0)


class BeaconDistillationTrainer(LossComponentTrainer):
    """训练用独立教师；教师不注册到 student，不进入优化器或 checkpoint。"""

    def __init__(self, *args, distillation_teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_teacher = distillation_teacher
        if self.model.beacon_config.beacon_output_distill_weight > 0:
            if distillation_teacher is None:
                raise ValueError("启用输出蒸馏必须提供冻结的原生 teacher")
            if distillation_teacher.config.vocab_size != self.model.config.vocab_size:
                raise ValueError("teacher 和 student 必须使用同一词表")
            distillation_teacher.requires_grad_(False)
            distillation_teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        config = model.beacon_config
        if model.training and config.beacon_output_distill_weight > 0:
            inputs = dict(inputs)
            positions = select_distillation_positions(
                inputs["labels"], inputs["regions"], config.beacon_output_distill_tokens,
            )
            inputs["distill_positions"] = positions
            if positions.numel():
                teacher = self.distillation_teacher
                teacher.eval()
                device = next(teacher.parameters()).device
                attention_mask = inputs.get("attention_mask")
                with torch.no_grad():
                    targets = teacher(
                        input_ids=inputs["input_ids"].to(device),
                        attention_mask=attention_mask.to(device) if attention_mask is not None else None,
                        logits_to_keep=positions.to(device),
                        use_cache=False,
                    ).logits
                inputs["teacher_logits"] = targets.detach().to(inputs["input_ids"].device)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch,
        )
