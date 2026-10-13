"""对齐同一轨迹的有效生成位置，冻结教师保留完整原文。"""

import torch

from .loss_component_trainer import LossComponentTrainer


def select_distillation_positions(labels, regions, tokens=8, strategy="prefix",
                                  loss_segment_ids=None, loss_segments=None):
    """返回原轨迹预测行 i，目标是 labels[i+1]。

    all 覆盖所有有效 keep 预测行（包括首次检索前）；prefix / stratified
    按每个压缩后的 keep 段限制预算。压缩正文和无效目标均不参与 KL。
    """
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("Beacon 输出蒸馏仅支持 batch_size=1")
    if strategy not in {"all", "prefix", "stratified"}:
        raise ValueError(f"未知的输出蒸馏采样策略: {strategy}")
    valid = labels[0, 1:] != -100
    if loss_segment_ids is not None and loss_segments is not None:
        from ..loss_segments import segment_ids_to_weights
        valid = valid & (segment_ids_to_weights(loss_segment_ids, loss_segments)[0, 1:] > 0)
    for start, end in regions:
        valid[start:end] = False
    if strategy == "all":
        return valid.nonzero(as_tuple=True)[0]
    if tokens <= 0:
        raise ValueError("蒸馏采样预算必须为正数")
    selected = []
    for index, (_, end) in enumerate(regions):
        stop = min(regions[index + 1][0] if index + 1 < len(regions) else labels.shape[1],
                   labels.shape[1] - 1)
        rows = valid[end:stop].nonzero(as_tuple=True)[0] + end
        if strategy == "prefix":
            selected.append(rows[:tokens])
        else:
            if loss_segment_ids is None:
                raise ValueError("stratified 蒸馏需要 loss_segment_ids")
            from ..loss_segments import SEGMENT_IDS
            kinds = loss_segment_ids[0, rows + 1]
            think = rows[kinds == SEGMENT_IDS["think_content"]]
            # 思考开头和均匀覆盖各自形成队列，再与 search/answer 轮流分预算。
            early = think[:min(tokens, (think.numel() + 1) // 2)]
            uniform = think[torch.linspace(0, think.numel() - 1, min(max(2, (tokens + 3) // 4 + 1), think.numel()),
                                          device=think.device).long()] if think.numel() else think
            queues = [early.tolist(), uniform.tolist(),
                      rows[kinds == SEGMENT_IDS["search_content"]].tolist(),
                      rows[kinds == SEGMENT_IDS["answer_content"]].tolist()]
            chosen = set()
            while len(chosen) < tokens and any(queues):
                for queue in queues:
                    while queue and queue[0] in chosen:
                        queue.pop(0)
                    if queue and len(chosen) < tokens:
                        chosen.add(queue.pop(0))
            # 正文不足时再用标签/other 填满相同预算。
            for row in rows.tolist():
                if len(chosen) >= tokens:
                    break
                chosen.add(row)
            selected.append(rows.new_tensor(sorted(chosen)))
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
                strategy=config.beacon_output_distill_strategy,
                loss_segment_ids=inputs.get("loss_segment_ids"), loss_segments=config.loss_segments,
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
