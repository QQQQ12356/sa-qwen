"""架构无关的 Beacon KV 管理、损失和生成逻辑。"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from .beacon_config import BeaconConfig
from .modeling_utils import beacon_intersect_order, cat_tensor, slice_tensor
from ..loss_segments import SEGMENT_IGNORE, segment_ids_to_weights

class BeaconMemory:
    """仅管理全注意力 KV 与窗口计划；Qwen3.5 子类额外维护线性状态。"""

    def __init__(self, model, beacon_config: BeaconConfig):
        self.model = model
        self.text = model.model
        self.config = model.config
        self.beacon = beacon_config
        self.layer_types = getattr(self.config, "layer_types", ["full_attention"] * self.config.num_hidden_layers)
        self.num_layers = self.config.num_hidden_layers
        # 每层 full-attn 缓存 (K, V)，RoPE 之前
        self._cache: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = [
            (None, None) for _ in range(self.num_layers)
        ]
        self.reset()

    def reset(self):
        self._pos = 0
        self._seg_idx = 0
        self._segments: List[Tuple[int, int, str]] = []
        self._cache = [(None, None) for _ in range(self.num_layers)]
        self._batch_loss = None
        self._valid_num = None
        self._store = None
        self._step_beacon_indices = None
        self._question = None
        self._question_input_ids = None
        self._readout_losses = []
        self._distill_positions = None
        self._teacher_logits = None
        self._distill_losses = []
        # 续写监督：全局移位的 token id、当前窗口的读出位置与监督标签、跨窗口损失累积
        self._cont_losses: List[torch.Tensor] = []
        self._shifted_ids: Optional[torch.Tensor] = None
        self._step_cont_rows: Optional[torch.Tensor] = None
        self._step_cont_labels: Optional[torch.Tensor] = None
        # 分段损失：全序列逐位置权重，以及当前窗口的切片
        self._seq_weights: Optional[torch.Tensor] = None
        self._step_weights: Optional[torch.Tensor] = None

    @property
    def finish(self) -> bool:
        return self._pos >= self._seq_len

    # ------------------------------------------------------------------
    def prepare(self, input_ids, labels, regions, loss_segment_ids=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("Beacon 需要非空 input_ids，形状为 (1, length)")
        self._segments = []
        self._seq_len = input_ids.shape[1]
        self._device = input_ids.device
        self._input_ids = input_ids
        # labels 全局预移位：位置 i 监督 token i+1（跨窗口因果监督）
        if labels is not None:
            labels = torch.cat([labels[:, 1:], labels.new_full((labels.shape[0], 1), -100)], dim=1)
        self._labels = labels
        # 续写损失的目标：全局移位的原始 token id（位置 i 的目标是 token i+1）。
        # 只在训练（labels 非 None）且开启时构造，推理路径完全不做额外工作。
        if labels is not None and self.beacon.beacon_information_loss is not False and self.beacon.beacon_continuation_loss_weight > 0:
            self._shifted_ids = torch.cat(
                [input_ids[:, 1:], input_ids.new_full((input_ids.shape[0], 1), -100)], dim=1
            )
        # 片段类别描述的是「被监督的那个 token」，因此与 labels 同步移位，
        # 再按配置查表转成逐位置权重（SEGMENT_IGNORE 位置权重 0）。
        if loss_segment_ids is not None:
            loss_segment_ids = torch.cat(
                [
                    loss_segment_ids[:, 1:],
                    loss_segment_ids.new_full((loss_segment_ids.shape[0], 1), SEGMENT_IGNORE),
                ],
                dim=1,
            )
        self._seq_weights = segment_ids_to_weights(loss_segment_ids, self.beacon.loss_segments)

        regions = [(int(s), int(e)) for s, e in regions]
        cursor = 0
        for s, e in regions:
            if not (cursor <= s <= e <= self._seq_len):
                raise ValueError(f"压缩区必须升序、不重叠且在序列范围内: {(s, e)}")
            if s > cursor:
                self._segments.append((cursor, s, "keep"))
            if e > s:
                self._segments.append((s, e, "compress"))
            cursor = e
        if cursor < self._seq_len:
            self._segments.append((cursor, self._seq_len, "keep"))
        self._seg_idx = 0

    # ------------------------------------------------------------------
    def step(self):
        while self._seg_idx < len(self._segments) and self._segments[self._seg_idx][1] <= self._pos:
            self._seg_idx += 1
        if self._pos >= self._seq_len:
            raise RuntimeError("序列已处理完毕")

        seg_start, seg_end, mode = self._segments[self._seg_idx]
        window = self.beacon.beacon_window if self.beacon.beacon_window_enabled else None
        start = self._pos

        if mode == "keep":
            # keep 段切窗粒度不影响结果（全部 K/V 提交、位置全局单调、状态跨窗携带），
            # 只影响前向次数与峰值显存：None 跟随 window（历史行为）、0 整段不切、
            # >0 为上限（防长序列 O(L²) 注意力显存）。
            keep_window = self.beacon.beacon_keep_window
            if keep_window is None:
                keep_window = window if window is not None else seg_end - start
            elif keep_window == 0:
                keep_window = seg_end - start
            end = min(start + keep_window, seg_end)
            beacon_size = 0
            store = "cache"
        else:
            end = seg_end if self.beacon.beacon_information_mode == "whole" else min(start + window, seg_end)
            remaining = end - start
            beacon_size = max(1, (remaining + self.beacon.beacon_ratio - 1) // self.beacon.beacon_ratio)
            store = "beacon"

        input_ids = self._input_ids[:, start:end]
        labels = self._labels[:, start:end] if self._labels is not None else None
        weights = self._seq_weights[:, start:end] if self._seq_weights is not None else None

        if beacon_size > 0:
            input_ids = torch.cat(
                [input_ids, input_ids.new_full((input_ids.shape[0], beacon_size), self.config.vocab_size)], dim=1
            )
            if labels is not None:
                labels = torch.cat([labels, labels.new_full((labels.shape[0], beacon_size), -100)], dim=1)
            if weights is not None:
                # beacon 位置不参与损失，权重置 0（与 labels=-100 对齐）
                weights = torch.cat(
                    [weights, weights.new_zeros((weights.shape[0], beacon_size))], dim=1
                )

        cur_len = input_ids.shape[1]
        if beacon_size > 0:
            self._step_beacon_indices = torch.cat(
                [torch.zeros(cur_len - beacon_size, dtype=torch.long, device=self._device),
                 torch.ones(beacon_size, dtype=torch.long, device=self._device)]
            )
        else:
            self._step_beacon_indices = None

        if beacon_size > 0 and self.beacon.beacon_pos == "intersect":
            order = beacon_intersect_order(end - start, self.beacon.beacon_ratio, self._device)
            input_ids = input_ids[:, order]
            self._step_beacon_indices = self._step_beacon_indices[order]
            if labels is not None:
                labels = labels[:, order]
            if weights is not None:
                weights = weights[:, order]
        self._step_weights = weights

        # 续写监督：读出位置与目标都在「置换 + beacon 补齐」之后按窗口局部坐标确定。
        cont = self._build_cont_rows_labels(input_ids, mode, start, end, seg_start, seg_end)
        self._step_cont_rows, self._step_cont_labels = cont if cont is not None else (None, None)

        # past：full-attn 层 4 元组
        mem_size = 0
        past = []
        for layer_idx in range(self.num_layers):
            if self.layer_types[layer_idx] == "full_attention":
                ck, cv = self._cache[layer_idx]
                if ck is not None:
                    mem_size = ck.shape[2]
                past.append((layer_idx, (ck, cv, beacon_size, self._step_beacon_indices)))
        # mem_size 取任一 full-attn 层缓存长度（每层长度一致）
        for layer_idx in range(self.num_layers):
            if self.layer_types[layer_idx] == "full_attention":
                ck, _ = self._cache[layer_idx]
                if ck is not None:
                    mem_size = ck.shape[2]
                break

        # 位置：全局单调（含 beacon 占位），用于 mRoPE
        total = mem_size + cur_len
        cos, sin = self.model._rope_for(total, self._device)

        attn_mask = self._make_4d_causal_mask(1, cur_len, mem_size, self._device)

        self._store = store
        self._pos = end
        return input_ids, labels, attn_mask, (cos, sin), past

    # ------------------------------------------------------------------
    def _build_cont_rows_labels(self, win_ids, mode, start, end, seg_start, seg_end):
        """构造当前窗口的续写监督 ``(rows, labels)``；无监督点时返回 ``None``。

        ``rows`` 与 ``labels`` 形状均为 ``(bsz, n)``：取 ``hidden`` 中 ``rows`` 指定的行
        去预测 ``labels`` 里的目标 token。两种模式的读出点不同：

        - window 模式（append）：压缩段**非首窗**窗头的 ``beacon_continuation_tokens``
          个 token，分别由窗头第 ``0..k-1`` 行的隐状态预测。首窗窗头能看到本段更早的
          原始 token，故只监督非首窗；预测依赖已提交的压缩记忆和当前窗原文前缀。
          开启 ``beacon_doc_region_split`` 后压缩段即单篇子文档，非首窗等价于
          「同一篇文档的续写窗」。
        - beacon 模式：每个 chunk 的 beacon 监督其后 ``k`` 个 token，``k`` 个目标**共用**
          beacon 那一行的隐状态（见 :meth:`_per_beacon_cont_rows_labels`）。
        """
        if (mode != "compress"
                or self.beacon.beacon_information_loss is False
                or self.beacon.beacon_continuation_loss_weight <= 0
                or self._shifted_ids is None):
            return None
        if self.beacon.beacon_information_mode == "whole":
            # Original rows remain causally before their targets even after insertion.
            count = end - start - 1
            if count <= 0:
                return None
            original_rows = (self._step_beacon_indices == 0).nonzero(as_tuple=True)[0]
            rows = original_rows[:count].expand(win_ids.shape[0], -1)
            return rows, self._input_ids[:, start + 1:end]
        if self.beacon.beacon_continuation_per_beacon:
            return self._per_beacon_cont_rows_labels(win_ids, start, end, seg_end)
        if self.beacon.beacon_pos != "append" or start <= seg_start:
            return None
        # 位置 i 的目标是 i+1，不能把 seg_end 的文档外 token 当作续写。
        stop = min(start + self.beacon.beacon_continuation_tokens, end, seg_end - 1)
        if stop <= start:
            return None
        rows = torch.arange(stop - start, device=self._device).expand(win_ids.shape[0], -1)
        return rows, self._shifted_ids[:, start:stop]

    def _per_beacon_cont_rows_labels(self, win_ids, start, end, seg_end):
        """窗内每个 chunk 的 beacon 监督紧随其后的 ``k`` 个 token，共用该 beacon 那一行。

        ``beacon_window == beacon_ratio`` 时窗内只有一个 chunk、beacon 位于窗末，目标正好是
        下一窗首 k 个 token，与历史行为逐点一致。``beacon_window > beacon_ratio`` 时窗内有
        多个 chunk，逐个读出（要求 ``beacon_pos="intersect"`` —— 只有它把 beacon 放在各自
        chunk 之后）。后者窗内 chunk 之间**不互相隔离**：后一个 chunk 的 token 既读得到更早
        chunk 的原始 token，也读得到它们的 beacon（full-attention 层靠整窗因果掩码，
        linear-attention 层是无掩码的循环扫描，掩码挡不住，见
        ``tests/test_beacon_window_visibility.py``）。隔离只发生在**窗口边界**：窗末只提交
        beacon、丢弃原始 K/V，于是下一个窗口的 token 左侧只有已提交的 beacon。

        目标一律钳制在 ``seg_end`` 内，跨段（进入 keep 或下一篇子文档）不监督，因此不会
        制造伪续写；不足 k 个的位置补 -100（与 labels 的忽略语义一致，不参与归一化）。某个
        beacon 无有效目标（如段尾那个）就跳过，全窗都无目标时返回 ``None`` —— 保持「无监督
        点就不进 ``_cont_losses``」的语义，避免零值稀释跨窗均值。注意「压缩段 == 子文档」
        成立的前提是 ``beacon_doc_region_split=True``。
        """
        is_beacon = self._step_beacon_indices
        if is_beacon is None:
            return None
        beacon_rows = (is_beacon > 0).nonzero(as_tuple=True)[0]

        ratio = self.beacon.beacon_ratio
        tokens = self.beacon.beacon_continuation_tokens
        win_len = end - start
        if self.beacon.beacon_pos != "intersect" and win_len > ratio:
            # 只有 intersect 把 beacon 放在各自 chunk 之后。append 布局下 beacon 全贴在窗末，
            # 行号与 chunk 无关，套用下面的映射会把读出点排到目标 token **之后**，监督退化成
            # 「照着原文抄」。配置校验在构造期已拦下，这里再兜一层：``BeaconConfig`` 的校验
            # 只在 ``__post_init__`` 跑一次，构造后改动配置对象的代码绕过它。
            raise ValueError(
                "按 chunk 逐个读出续写要求 beacon_pos='intersect'（当前 "
                f"{self.beacon.beacon_pos}，窗内 {win_len} token > ratio {ratio}）"
            )
        batch = win_ids.shape[0]
        rows, targets = [], []
        for offset, row in enumerate(beacon_rows.tolist()):
            # 第 offset 个 beacon 关闭第 offset 个 chunk；末端在窗内的原始偏移即
            # ``min((offset+1)*ratio, win_len)``（末个 chunk 可能不满一整片）。
            chunk_end = min((offset + 1) * ratio, win_len)
            target_start = start + chunk_end
            stop = min(target_start + tokens, seg_end)
            if stop <= target_start:
                continue
            rows.append(win_ids.new_full((batch, tokens), int(row)))
            target = self._input_ids.new_full((batch, tokens), -100)
            target[:, :stop - target_start] = self._input_ids[:, target_start:stop]
            targets.append(target)
        if not rows:
            return None
        return torch.cat(rows, dim=1), torch.cat(targets, dim=1)

    # ------------------------------------------------------------------
    def update_memory(self, new_past):
        """new_past: list of (layer_idx, (key, value, beacon_size, indices)) 增量、RoPE 前。"""
        for layer_idx, (key, value, _bs, _indices) in new_past:
            ck, cv = self._cache[layer_idx]
            if self._store == "beacon":
                sel = self._step_beacon_indices.bool()
                bk = slice_tensor(key, index=sel, dim=2)
                bv = slice_tensor(value, index=sel, dim=2)
                self._cache[layer_idx] = (cat_tensor([ck, bk], dim=2), cat_tensor([cv, bv], dim=2))
            else:
                self._cache[layer_idx] = (cat_tensor([ck, key], dim=2), cat_tensor([cv, value], dim=2))

    # ------------------------------------------------------------------
    def update_loss(self, batch_loss, valid_num):
        if self._batch_loss is None:
            self._batch_loss = batch_loss * valid_num
            self._valid_num = valid_num
        else:
            self._batch_loss = self._batch_loss + batch_loss * valid_num
            self._valid_num = self._valid_num + valid_num

    def output(self):
        if self._batch_loss is None:
            return None
        total = self._valid_num.sum()
        if total.item() == 0:
            return self._batch_loss.sum() * 0.0
        return self._batch_loss.sum() / total

    # ------------------------------------------------------------------
    def _make_4d_causal_mask(self, batch, q_len, mem_size, device):
        min_value = torch.finfo(torch.float32).min
        full_len = mem_size + q_len
        mask = torch.full((batch, 1, q_len, full_len), min_value, dtype=torch.float32, device=device)
        if mem_size > 0:
            mask[:, :, :, :mem_size] = 0.0
        causal = torch.tril(torch.ones(q_len, q_len, device=device, dtype=torch.bool))
        cur = torch.where(causal, torch.zeros((), device=device), torch.full((), min_value, device=device))
        mask[:, :, :, mem_size:] = cur.unsqueeze(0).unsqueeze(0)
        return mask

class BeaconCausalLMMixin:
    """两种架构共用的损失、窗口调度和增量生成接口。"""

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                compress_regions=None, compress_start=None, compress_end=None,
                regions=None, question_input_ids=None, loss_segment_ids=None,
                distill_positions=None, teacher_logits=None, **kwargs):
        if not self.beacon_config.enable_beacon:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        if compress_regions is None and compress_start is not None and compress_end is not None:
            compress_regions = [(int(compress_start), int(compress_end))]
        if compress_regions is None:
            # collator 返回的 batch 键名可能是 regions（Trainer 经 remove_unused_columns=False 透传）
            compress_regions = regions
        if compress_regions is None:
            raise ValueError("beacon 模式必须提供 compress_regions 或 compress_start/compress_end")
        use_cpu_offload = (
            self.training
            and labels is not None
            and self.beacon_config.beacon_cpu_offload_activations
            and (
                self.beacon_config.beacon_cpu_offload_threshold == 0
                or input_ids.shape[1] >= self.beacon_config.beacon_cpu_offload_threshold
            )
        )
        if use_cpu_offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=input_ids.is_cuda):
                loss, logits = self._beacon_forward(
                    input_ids, labels, compress_regions, question_input_ids, loss_segment_ids,
                    distill_positions, teacher_logits,
                )
        else:
            loss, logits = self._beacon_forward(
                input_ids, labels, compress_regions, question_input_ids, loss_segment_ids,
                distill_positions, teacher_logits,
            )
        # 返回 ModelOutput，兼容 transformers.Trainer（取 outputs["loss"]）
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def _embed(self, input_ids, beacon_size):
        """分开嵌入普通 token 与 beacon token（beacon id == vocab_size）。"""
        if beacon_size > 0:
            bi = self._mem._step_beacon_indices
            cur = input_ids.shape[1]
            bi = bi[-cur:] if bi is not None else None
            if bi is not None and bi.any():
                ordinal_ids = input_ids[:, bi == 0]
                beacon_ids = input_ids[:, bi > 0]
                ordinal_emb = self.model.embed_tokens(ordinal_ids)
                beacon_emb = self.model.beacon_embed_tokens(beacon_ids - self.config.vocab_size)
                emb = beacon_emb.new_zeros(*input_ids.shape, beacon_emb.shape[-1])
                emb[:, bi == 0] = ordinal_emb
                emb[:, bi > 0] = beacon_emb
                return emb
        return self.model.embed_tokens(input_ids)

    def _native_forward(
        self,
        win_ids,
        win_labels,
        attn_mask,
        position_embeddings,
        past,
        last_logits_only=False,
    ):
        """单窗口前向。past: list of (layer_idx, 4元组)。返回 (new_past, logits)。"""
        new_past, hidden = self._window_hidden(win_ids, win_labels, attn_mask, position_embeddings, past)
        hidden = self.model.norm(hidden)
        logits = None

        if win_labels is not None:
            # 仅在 keep 段读出：此前压缩窗的 Writer 和 Beacon K/V 已经提交。
            # 使用主轨迹的同一 teacher-forced 前缀，不创建或提交额外读出状态。
            if self._mem._teacher_logits is not None and self._mem._store == "cache":
                start = self._mem._pos - win_ids.shape[1]
                positions = self._mem._distill_positions
                selected = (positions >= start) & (positions < self._mem._pos)
                if selected.any():
                    kd_hidden = hidden[:, positions[selected] - start]
                    kd_teacher = self._mem._teacher_logits[:, selected]
                    kd_loss = self._output_distillation_loss(kd_hidden, kd_teacher)
                    self._mem._distill_losses.append((kd_loss, kd_hidden.shape[1]))
            valid_num = (win_labels != -100).sum(-1)
            if self.beacon_config.beacon_ce_loss_weight > 0 and valid_num.sum().item() > 0:
                batch_loss, weight_sum = self._sparse_window_loss(
                    hidden, win_labels, valid_num, self._mem._step_weights
                )
                self._mem.update_loss(batch_loss, weight_sum)
            # 续写损失：独立项（不进主 CE），读出点由 _step_cont_rows 指定（beacon 模式下
            # k 个目标共用 beacon 那一行）。decode 路径 win_labels 为 None，因此不会读到
            # prefill 遗留的 _step_cont_labels。
            cont_labels = self._mem._step_cont_labels
            if cont_labels is not None:
                cont_hidden = hidden.gather(
                    1, self._mem._step_cont_rows.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
                )
                cont_valid = (cont_labels != -100).sum(-1)
                cont_loss, _ = self._sparse_window_loss(cont_hidden, cont_labels, cont_valid, None)
                self._mem._cont_losses.append(cont_loss.mean())
        elif last_logits_only:
            logits = self.lm_head(hidden[:, -1:, :]).float()
        else:
            logits = self.lm_head(hidden).float()
        return new_past, logits

    def _output_distillation_loss(self, hidden, teacher_logits):
        """KL(teacher || student)，按读出 token 平均、按 LM-head chunk 重计算。"""
        temperature = self.beacon_config.beacon_output_distill_temperature
        chunk_size = self.beacon_config.beacon_loss_chunk_size

        def chunk_loss(current_hidden, targets):
            student_log_probs = F.log_softmax(self.lm_head(current_hidden).float() / temperature, dim=-1)
            teacher_log_probs = F.log_softmax(targets.float() / temperature, dim=-1)
            return F.kl_div(
                student_log_probs, teacher_log_probs, log_target=True, reduction="sum",
            ) * temperature ** 2

        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, hidden.shape[1], chunk_size):
            current_hidden = hidden[:, start:start + chunk_size]
            targets = teacher_logits[:, start:start + chunk_size].detach()
            if self.beacon_config.beacon_checkpoint_loss and self.training and current_hidden.requires_grad:
                value = torch.utils.checkpoint.checkpoint(
                    chunk_loss, current_hidden, targets, use_reentrant=False,
                )
            else:
                value = chunk_loss(current_hidden, targets)
            total = total + value
        return total / (hidden.shape[0] * hidden.shape[1])

    def _sparse_window_loss(self, hidden, labels, valid_num, weights=None):
        """只为非 ``-100`` 位置计算分块词表交叉熵（可选按片段类别加权）。

        Args:
            hidden: ``(bsz, cur, hidden)`` 当前窗口隐状态。
            labels: ``(bsz, cur)`` 已移位的目标 token，-100 表示不监督。
            valid_num: ``(bsz,)`` 每个样本的监督 token 数（无权重时的归一化分母）。
            weights: ``(bsz, cur)`` 逐位置权重，或 None（等权，与历史行为一致）。

        Returns:
            ``(batch_loss, weight_sum)``：每个样本的加权平均损失，以及对应的权重和
            （作为跨窗口聚合的归一化分母）。
        """
        chunk_size = self.beacon_config.beacon_loss_chunk_size
        checkpoint_loss = self.beacon_config.beacon_checkpoint_loss and self.training
        batch_losses = []
        weight_sums = []
        for batch_idx in range(hidden.shape[0]):
            selected_mask = labels[batch_idx] != -100
            selected_hidden = hidden[batch_idx, selected_mask]
            selected_labels = labels[batch_idx, selected_mask]
            selected_weights = weights[batch_idx, selected_mask] if weights is not None else None
            # 分母 = 权重和（等权时退化为监督 token 数），保证加权不改变损失量纲
            denominator = (
                selected_weights.sum()
                if selected_weights is not None
                else valid_num[batch_idx].to(torch.float32)
            )
            loss_sum = hidden.new_zeros((), dtype=torch.float32)
            for start in range(0, selected_hidden.shape[0], chunk_size):
                chunk_hidden = selected_hidden[start:start + chunk_size]
                chunk_labels = selected_labels[start:start + chunk_size]
                chunk_weights = (
                    selected_weights[start:start + chunk_size]
                    if selected_weights is not None
                    else None
                )

                def loss_function(current_hidden, current_labels, current_weights):
                    chunk_logits = self.lm_head(current_hidden).float()
                    token_loss = F.cross_entropy(chunk_logits, current_labels, reduction="none")
                    if current_weights is None:
                        return token_loss.sum()
                    return (token_loss * current_weights).sum()

                if checkpoint_loss and chunk_hidden.requires_grad:
                    chunk_loss = torch.utils.checkpoint.checkpoint(
                        loss_function,
                        chunk_hidden,
                        chunk_labels,
                        chunk_weights,
                        use_reentrant=False,
                    )
                else:
                    chunk_loss = loss_function(chunk_hidden, chunk_labels, chunk_weights)
                loss_sum = loss_sum + chunk_loss
            batch_losses.append(loss_sum / denominator.clamp(min=1e-6))
            weight_sums.append(denominator)
        return torch.stack(batch_losses), torch.stack(weight_sums)

    def _prepare_question(self, question_input_ids):
        if not self.beacon_config.beacon_question_memory_v1:
            return
        if self._mem._question is not None:
            if question_input_ids is not None:
                question_input_ids = question_input_ids[:, :self.beacon_config.beacon_question_max_tokens]
                if not torch.equal(question_input_ids.to(self._mem._question_input_ids.device), self._mem._question_input_ids):
                    raise ValueError("复用缓存时不能切换 question，请使用 reuse_cache=False")
            return
        if question_input_ids is None or question_input_ids.ndim != 2 or question_input_ids.shape[0] != 1 or question_input_ids.shape[1] == 0:
            raise ValueError("question memory v1 首次调用需要非空 question_input_ids (1, length)")
        question_input_ids = question_input_ids[:, :self.beacon_config.beacon_question_max_tokens]
        self._mem._question_input_ids = question_input_ids.detach().clone()
        self._mem._question = self.model.embed_tokens(question_input_ids.to(self.model.embed_tokens.weight.device))

    def _beacon_forward(self, input_ids, labels, regions, question_input_ids=None,
                        loss_segment_ids=None, distill_positions=None, teacher_logits=None):
        self._mem = self.memory_class(self, self.beacon_config)
        self._prepare_question(question_input_ids)
        self._mem.prepare(input_ids, labels, regions, loss_segment_ids)
        if self.training and labels is not None and self.beacon_config.beacon_output_distill_weight > 0:
            if distill_positions is None:
                raise ValueError("输出蒸馏训练需要 BeaconDistillationTrainer 提供 distill_positions")
            positions = distill_positions.to(input_ids.device)
            if positions.ndim != 1 or positions.dtype != torch.long:
                raise ValueError("distill_positions 必须是 1D long tensor")
            if positions.numel():
                if (positions < 0).any() or (positions >= input_ids.shape[1] - 1).any():
                    raise ValueError("蒸馏读出位置超出有效预测范围")
                if not (positions[1:] > positions[:-1]).all():
                    raise ValueError("蒸馏读出位置必须严格递增")
                for start, end in regions:
                    if ((positions >= start) & (positions < end)).any():
                        raise ValueError("蒸馏只能从 keep 段读出，不能读取未提交的压缩正文")
                if (self._mem._labels[:, positions] == -100).any():
                    raise ValueError("蒸馏读出位置必须对应有效监督 token")
                expected = (input_ids.shape[0], positions.numel(), self.config.vocab_size)
                if teacher_logits is None or tuple(teacher_logits.shape) != expected:
                    raise ValueError(f"teacher_logits 形状必须为 {expected}")
                self._mem._distill_positions = positions
                self._mem._teacher_logits = teacher_logits.detach().to(input_ids.device)

        last_logits = None
        while not self._mem.finish:
            win_ids, win_labels, attn_mask, position_embeddings, past = self._mem.step()
            new_past, _logits = self._native_forward(win_ids, win_labels, attn_mask, position_embeddings, past)
            last_logits = _logits
            self._mem.update_memory(new_past)

        ce_loss = self._mem.output()
        readout = None
        if self._mem._readout_losses:
            readout = torch.stack(self._mem._readout_losses).mean()
            self._mem._readout_losses = []
        continuation = None
        if self._mem._cont_losses:
            continuation = torch.stack(self._mem._cont_losses).mean()
            self._mem._cont_losses = []

        # 权重是「该项在总损失里的份额」：按本次**实际出现**的项的权重和归一。``ce_loss``
        # 为 None（本批 labels 全 -100，没有任何 CE 监督）时它不占份额，否则续写梯度会被
        # w_ce 白白稀释。于是只有 CE 时 L = ce（改 w_ce 不动尺度），两项都在时才按比例加权。
        # 权重为 0 的项不参与归一；readout / KD 是独立的蒸馏加项，不进分母。
        weighted = []
        if ce_loss is not None and self.beacon_config.beacon_ce_loss_weight > 0:
            weighted.append(("ce_loss", self.beacon_config.beacon_ce_loss_weight, ce_loss))
        if continuation is not None and self.beacon_config.beacon_continuation_loss_weight > 0:
            weighted.append(
                ("cont_loss", self.beacon_config.beacon_continuation_loss_weight, continuation)
            )
        if weighted:
            denominator = sum(weight for _, weight, _ in weighted)
            loss = sum(weight * value for _, weight, value in weighted) / denominator
            # 逐项留痕：分量之和恒等于总损失，每个分量都能被日志读出。
            parts_by_name = {
                name: (weight / denominator) * value for name, weight, value in weighted
            }
        else:
            # 本次 CE 与续写均缺席或关闭：给一个保持计算图的零，再加入蒸馏损失。
            loss = self.model.beacon_embed_tokens.weight.sum() * 0.0
            parts_by_name = {}
        if readout is not None:
            readout_term = self.beacon_config.beacon_readout_distill_weight * readout
            loss = loss + readout_term
            parts_by_name["readout_loss"] = readout_term
        if self._mem._teacher_logits is not None:
            consumed = sum(count for _, count in self._mem._distill_losses)
            if consumed != self._mem._distill_positions.numel():
                raise RuntimeError("teacher/student KL token 数不一致，存在未读出的位置")
        if self._mem._distill_losses:
            kd = sum(value * count for value, count in self._mem._distill_losses)
            kd = kd / sum(count for _, count in self._mem._distill_losses)
            kd_term = self.beacon_config.beacon_output_distill_weight * kd
            loss = loss + kd_term
            parts_by_name["kd_loss"] = kd_term
            self._mem._distill_losses = []
        self._mem._teacher_logits = None
        # 未激活的项恒为 0、键序恒定，保证日志不随配置变化。
        zero = loss.detach() * 0.0
        loss_parts = {
            name: parts_by_name.get(name, zero)
            for name in ("ce_loss", "cont_loss", "readout_loss", "kd_loss")
        }
        self._last_loss_parts = {name: value.detach() for name, value in loss_parts.items()}
        return loss, last_logits

    def _mem_full_cache_past(self):
        """构造 decode 用的 full-attn 层 past 4 元组列表。"""
        past = []
        for idx in range(self.config.num_hidden_layers):
            if self._mem.layer_types[idx] == "full_attention":
                ck, cv = self._mem._cache[idx]
                past.append((idx, (ck, cv, 0, None)))
        return past

    def _mem_cache_len(self) -> int:
        """full-attn 层缓存长度（用于位置编码/掩码）。"""
        for idx in range(self.config.num_hidden_layers):
            if self._mem.layer_types[idx] == "full_attention":
                ck, _ = self._mem._cache[idx]
                if ck is not None:
                    return ck.shape[2]
        return 0

    def prefill_and_get_cache(self, input_ids, regions, return_last_logits=False,
                              reuse_cache=False, question_input_ids=None):
        """把整条上下文（question + 各检索文档块）编码为 beacon K/V。

        ``reuse_cache=True`` 时保留已有持久状态，只处理新增的 ``input_ids``；
        此时 ``regions`` 使用相对于新增输入的 token 区间。

        Returns:
            ``return_last_logits=False`` 时返回 :class:`_Qwen3BeaconMemory`（其
            持久缓存已就绪）；``True`` 时返回
            ``(memory, last_logits)``。
        """
        if reuse_cache:
            if not hasattr(self, "_mem"):
                raise ValueError("复用缓存前必须先执行首次 prefill")
            self._mem._pos = 0
            self._mem._seg_idx = 0
            self._mem._segments = []
        else:
            self._mem = self.memory_class(self, self.beacon_config)
        self._prepare_question(question_input_ids)
        self._mem.prepare(input_ids, None, regions)

        last_logits = None
        while not self._mem.finish:
            win_ids, _, attn_mask, position_embeddings, past = self._mem.step()
            new_past, logits = self._native_forward(
                win_ids,
                None,
                attn_mask,
                position_embeddings,
                past,
                last_logits_only=True,
            )
            self._mem.update_memory(new_past)
            # 每个窗口末位 logits = 对下一 token 的预测；prefill 结束即首个新 token 的预测
            last_logits = logits[:, -1, :]

        if return_last_logits:
            return self._mem, last_logits
        return self._mem

    def decode_step(self, last_token_ids):
        """单步自回归解码：对 ``last_token_ids`` (bsz,1) 取下一个 token 的 logits。"""
        mem_size = self._mem_cache_len()
        total = mem_size + 1
        device = last_token_ids.device
        cos, sin = self._rope_for(total, device)
        attn_mask = self._mem._make_4d_causal_mask(1, 1, mem_size, device)
        past = self._mem_full_cache_past()

        self._mem._store = "cache"
        new_past, logits = self._native_forward(
            last_token_ids,
            None,
            attn_mask,
            (cos, sin),
            past,
            last_logits_only=True,
        )
        self._mem.update_memory(new_past)
        return logits[:, -1, :]

    def beacon_generate(self, input_ids, attention_mask=None, regions=None,
                        compress_start=None, compress_end=None, max_new_tokens=256,
                        do_sample=False, temperature=1.0, top_p=1.0,
                        eos_token_ids=None, stop_texts=None, tokenizer=None,
                        reuse_cache=False, question_input_ids=None):
        """Beacon 模式生成：先编码上下文为 beacon K/V，再自回归生成答案。

        ``regions`` 指定
        各 ``<information>`` 检索文档块，其余（question/模板尾部/生成）为 keep。
        ``reuse_cache=True`` 时只传入新增输入和相对压缩区，接续已有缓存生成。
        """
        if regions is None and compress_start is not None and compress_end is not None:
            regions = [(int(compress_start), int(compress_end))]
        if regions is None:
            raise ValueError("必须提供 regions 或 compress_start/compress_end")

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self._beacon_generate_loop(
                    input_ids, regions, max_new_tokens, do_sample, temperature, top_p,
                    eos_token_ids, stop_texts, tokenizer, reuse_cache, question_input_ids,
                )
        finally:
            self.train(was_training)

    def _beacon_generate_loop(self, input_ids, regions, max_new_tokens, do_sample,
                              temperature, top_p, eos_token_ids, stop_texts, tokenizer,
                              reuse_cache=False, question_input_ids=None):
        _mem, logits = self.prefill_and_get_cache(
            input_ids, regions, return_last_logits=True, reuse_cache=reuse_cache,
            question_input_ids=question_input_ids,
        )
        eos = eos_token_ids if eos_token_ids is not None else [self.config.eos_token_id]
        generated = []

        for _ in range(max_new_tokens):
            if do_sample:
                logits = logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    logits = logits.masked_fill(
                        remove.scatter(-1, sorted_indices, remove), float("-inf")
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated.append(next_token)
            logits = self.decode_step(next_token)
            if next_token.item() in eos:
                break
            if stop_texts and tokenizer is not None:
                gen_text = tokenizer.decode(
                    torch.cat(generated, dim=1)[0].tolist(), skip_special_tokens=False
                )
                if any(st in gen_text for st in stop_texts):
                    break

        if not generated:
            return torch.empty((1, 0), dtype=torch.long, device=input_ids.device)
        return torch.cat(generated, dim=1)
