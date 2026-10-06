"""Activate Beacon 压缩机制在 Qwen3.5 上的移植（只压缩检索到的内容）。

背景：``Qwen/Qwen3.5-2B`` 是混合架构 —— 24 层中仅 6 层（``layer_types`` 为
``full_attention``）带标准 QKV 注意力缓存，其余 18 层为 ``linear_attention``
（GatedDeltaNet，固定尺寸循环状态）。

压缩和缓存语义：

- **只压缩检索内容**：由 ``regions``（各 ``<information>`` 块 token 区间）指定，
  window 模式按窗口处理，whole 模式每个 information 一次前向；append 在末尾
  追加 beacon，intersect 每 ``beacon_ratio`` 个原文 token 插入一个 beacon。
- **损失可配置**：主 labels 监督 assistant 内容；可另启用 information 续写损失，
  原有线性 readout 和输出 KL 蒸馏接口保留。
- **混合层都只提交 Beacon 载体**：full_attention 层只持久化 beacon K/V；
  linear_attention 层把压缩窗的原生 DeltaNet/卷积状态视为临时 reader 状态，
  窗结束后丢弃，并仅由 beacon 激活经可训练 writer 重建固定尺寸循环状态。

``enable_beacon=False`` 退化为原生 ``Qwen3_5ForCausalLM`` 前向。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5ForCausalLM,
    apply_rotary_pos_emb,
)

from .beacon_common import BeaconMemory, BeaconCausalLMMixin
from .beacon_config import BeaconConfig
from .beacon_question_memory import QuestionMemoryMixin

class BeaconLinearStateWriter(nn.Module):
    """仅从 Beacon 激活更新 GatedDeltaNet 的持久循环状态。"""

    def __init__(self, config, rank: int):
        super().__init__()
        self.num_heads = config.linear_num_value_heads
        self.key_dim = config.linear_key_head_dim
        self.value_dim = config.linear_value_head_dim
        self.norm = nn.LayerNorm(config.hidden_size)
        self.down = nn.Linear(config.hidden_size, rank, bias=False)
        self.up = nn.Linear(
            rank,
            self.num_heads * (self.key_dim + self.value_dim + 2),
        )
        with torch.no_grad():
            self.up.bias.zero_()
            self.up.bias[-2 * self.num_heads:-self.num_heads].fill_(4.0)

    def forward(self, beacons, previous):
        batch = beacons.shape[0]
        projected = self.up(F.silu(self.down(self.norm(beacons))))
        key, value, decay, beta = projected.split(
            [
                self.num_heads * self.key_dim,
                self.num_heads * self.value_dim,
                self.num_heads,
                self.num_heads,
            ],
            dim=-1,
        )
        key = key.reshape(batch, -1, self.num_heads, self.key_dim).float()
        key = key * torch.rsqrt(key.square().sum(-1, keepdim=True) + 1e-6)
        value = value.reshape(batch, -1, self.num_heads, self.value_dim).float()
        if previous is None:
            state = torch.zeros(
                batch,
                self.num_heads,
                self.key_dim,
                self.value_dim,
                device=beacons.device,
                dtype=torch.float32,
            )
        else:
            state = previous.float().clone()
        for token_idx in range(beacons.shape[1]):
            state = state * decay[:, token_idx].float().sigmoid()[..., None, None]
            token_key = key[:, token_idx]
            prediction = (state * token_key.unsqueeze(-1)).sum(-2)
            update = (value[:, token_idx] - prediction) * beta[:, token_idx].float().sigmoid().unsqueeze(-1)
            state = state + token_key.unsqueeze(-1) * update.unsqueeze(-2)
        return state

# ======================================================================
# Beacon Attention（full_attention 层）
# ======================================================================
class QuestionBeaconLinearStateWriter(QuestionMemoryMixin, BeaconLinearStateWriter):
    def __init__(self, config, rank):
        super().__init__(config, rank)
        self.init_question_memory(config.hidden_size, rank)

class BeaconQwen3Attention(Qwen3_5Attention):
    """带 beacon 投影切换的 Qwen3.5 注意力（仅 full_attention 层使用）。

    ``q_proj`` 输出 ``num_heads * head_dim * 2``（query + 门控），beacon_q_proj
    同宽，门控随 query 一起被 ``torch.where`` 切换。K/V 缓存保存 **RoPE 之前**
    的 key，每个窗口对 ``past + current`` 整体重施加 mRoPE。
    """

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.beacon_param = getattr(config, "beacon_param", "").split()

        if "q" in self.beacon_param:
            self.beacon_q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias
            )
            self.beacon_q_proj.weight.data.zero_()
        if "k" in self.beacon_param:
            self.beacon_k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.beacon_k_proj.weight.data.zero_()
        if "v" in self.beacon_param:
            self.beacon_v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.beacon_v_proj.weight.data.zero_()
        if "o" in self.beacon_param:
            self.beacon_o_proj = nn.Linear(
                self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
            )
            self.beacon_o_proj.weight.data.zero_()

    # ------------------------------------------------------------------
    def _qkv(self, hidden_states, beacon_size, beacon_indices):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cur = hidden_states.shape[1]
        bi = beacon_indices[-cur:] if beacon_size > 0 else None

        if beacon_size > 0 and "q" in self.beacon_param:
            oq = self.q_proj(hidden_states)
            bq = self.beacon_q_proj(hidden_states)
            qg = torch.where((bi == 0)[:, None], oq, bq)
        else:
            qg = self.q_proj(hidden_states)
        query_states, gate = torch.chunk(qg.view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)

        if beacon_size > 0 and "k" in self.beacon_param:
            ok = self.k_proj(hidden_states)
            bk = self.beacon_k_proj(hidden_states)
            key_states = torch.where((bi == 0)[:, None], ok, bk)
        else:
            key_states = self.k_proj(hidden_states)
        if beacon_size > 0 and "v" in self.beacon_param:
            ov = self.v_proj(hidden_states)
            bv = self.beacon_v_proj(hidden_states)
            value_states = torch.where((bi == 0)[:, None], ov, bv)
        else:
            value_states = self.v_proj(hidden_states)

        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states, gate, input_shape

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs,
    ):
        """分派：beacon 窗口路径 vs 原生前向。

        - 传 ``past_key_value``（4 元组）时走 beacon 窗口路径（由 BeaconMemory 驱动）。
        - 否则委托原生 ``Qwen3_5Attention.forward``（``enable_beacon=False`` 时
          整条序列正常前向，保证与原生等价）。
        """
        if isinstance(past_key_value, tuple) and len(past_key_value) == 4:
            return self._beacon_forward(hidden_states, position_embeddings, attention_mask, past_key_value)
        return super().forward(
            hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
        )

    def _beacon_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Tuple,
    ) -> Tuple[torch.Tensor, Tuple]:
        """窗口前向。

        Args:
            hidden_states: ``(bsz, cur+beacon, hidden)`` 当前窗口。
            position_embeddings: ``(cos, sin)``，覆盖 ``[0, mem+cur+beacon)`` 全长。
            attention_mask: 4D 因果掩码 ``(bsz, 1, cur+beacon, mem+cur+beacon)``。
            past_key_value: 4 元组 ``(key, value, beacon_size, beacon_indices)``，
                key/value 为 **RoPE 之前** 的历史 K/V。

        Returns:
            ``(attn_output, new_past)``；new_past = 当前**增量**、RoPE 之前的
            ``(key, value, beacon_size, beacon_indices)``。
        """
        past_key, past_value, beacon_size, beacon_indices = past_key_value
        cur_len = hidden_states.shape[1]

        query_states, key_states, value_states, gate, input_shape = self._qkv(
            hidden_states, beacon_size, beacon_indices
        )

        # 增量、RoPE 之前的 K/V（供记忆缓存，下个窗口整体重施加 RoPE）
        new_past = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        cos, sin = position_embeddings
        # query 用当前窗口位置（末尾 cur_len 位），key 用全局位置（全长）。
        # apply_rotary_pos_emb 内部会自行 unsqueeze，这里无需额外扩展。
        q_cos = cos[:, -cur_len:, :]
        q_sin = sin[:, -cur_len:, :]
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, q_cos, q_sin)
        key_states, _ = apply_rotary_pos_emb(key_states, key_states, cos, sin)

        # 组扩展
        g = self.num_key_value_groups
        if g > 1:
            key_states = key_states.repeat_interleave(g, dim=1)
            value_states = value_states.repeat_interleave(g, dim=1)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(gate)
        if "o" in self.beacon_param and beacon_size > 0:
            oo = self.o_proj(attn_output)
            bo = self.beacon_o_proj(attn_output)
            bi = beacon_indices[-cur_len:]
            attn_output = torch.where((bi == 0)[:, None], oo, bo)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output, new_past

# ======================================================================
# Beacon 窗口状态机（只压缩检索内容，线性层显式管理压缩状态）
# ======================================================================
class _Qwen3BeaconMemory(BeaconMemory):
    """Qwen3.5 额外管理线性层的循环状态和卷积状态。"""
    def reset(self):
        super().reset()
        self._linear_recurrent = {}
        self._linear_conv = {}

# ======================================================================
# ForCausalLM
# ======================================================================
class BeaconQwen3_5ForCausalLM(BeaconCausalLMMixin, Qwen3_5ForCausalLM):
    """Qwen3.5 混合注意力 Beacon 模型，支持可选 information 续写监督。"""

    memory_class = _Qwen3BeaconMemory

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    # 损失已按当前样本归一化，没有使用 Trainer 的 num_items_in_batch。
    accepts_loss_kwargs = False

    def __init__(self, config):
        super().__init__(config)
        self.beacon_config = BeaconConfig.from_model_config(config)
        self.beacon_config.merge_into_config(config)
        self.model.beacon_embed_tokens = nn.Embedding(1, config.hidden_size)
        self.model.beacon_embed_tokens.weight.data.zero_()
        self.model.beacon_linear_writers = nn.ModuleDict()
        for idx, layer in enumerate(self.model.layers):
            layer_type = getattr(config, "layer_types", [None] * config.num_hidden_layers)[idx]
            if layer_type == "full_attention":
                layer.self_attn = BeaconQwen3Attention(config, idx)
            elif layer_type == "linear_attention":
                writer_type = QuestionBeaconLinearStateWriter if self.beacon_config.beacon_question_memory_v1 else BeaconLinearStateWriter
                self.model.beacon_linear_writers[str(idx)] = writer_type(
                    config,
                    self.beacon_config.beacon_linear_writer_rank,
                )
        self.post_init()
        self._init_beacon_params()
        #: 最近一次前向的复合损失分量（CE / continuation / readout / output KL），
        #: 已 detach，供 :class:`~search_comp.trainer.loss_component_trainer.LossComponentTrainer`
        #: 在 log_step 打印。不参与求导，也不触发 GPU 同步。
        self._last_loss_parts: Dict[str, torch.Tensor] = {}

    def _init_beacon_params(self):
        cfg = self.config
        src_id = getattr(cfg, f"{self.beacon_config.beacon_embed_init}_token_id", None)
        with torch.no_grad():
            if src_id is not None and src_id < self.model.embed_tokens.num_embeddings:
                self.model.beacon_embed_tokens.weight.data.copy_(self.model.embed_tokens.weight.data[src_id])
            else:
                self.model.beacon_embed_tokens.weight.data.normal_(0, cfg.hidden_size ** -0.5)
        params = self.beacon_config.beacon_param.split()
        for writer in self.model.beacon_linear_writers.values():
            with torch.no_grad():
                writer.up.bias.zero_()
                writer.up.bias[-2 * writer.num_heads:-writer.num_heads].fill_(4.0)
                if isinstance(writer, QuestionBeaconLinearStateWriter):
                    writer.question_up.weight.zero_()
        for idx, layer in enumerate(self.model.layers):
            if getattr(cfg, "layer_types", [None] * cfg.num_hidden_layers)[idx] != "full_attention":
                continue
            attn = layer.self_attn
            with torch.no_grad():
                for p in params:
                    b = getattr(attn, f"beacon_{p}_proj", None)
                    o = getattr(attn, f"{p}_proj", None)
                    if b is None or o is None:
                        continue
                    b.weight.data.copy_(o.weight.data)
                    if b.bias is not None and o.bias is not None:
                        b.bias.data.copy_(o.bias.data)

    def set_beacon_config(self, beacon_config: BeaconConfig):
        if beacon_config.beacon_param != self.beacon_config.beacon_param:
            raise ValueError("beacon_param 改变模型结构，必须在构造模型前配置")
        if beacon_config.beacon_question_memory_v1 != self.beacon_config.beacon_question_memory_v1:
            raise ValueError("beacon_question_memory_v1 改变模型结构，必须在构造模型前配置")
        for writer in self.model.beacon_linear_writers.values():
            if writer.down.out_features != beacon_config.beacon_linear_writer_rank:
                raise ValueError(
                    "checkpoint 的 beacon_linear_writer_rank 与请求配置不一致: "
                    f"{writer.down.out_features} != {beacon_config.beacon_linear_writer_rank}"
                )
        self.beacon_config = beacon_config
        beacon_config.merge_into_config(self.config)
        return self

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _window_hidden(self, win_ids, win_labels, attn_mask, position_embeddings, past):
        beacon_size = past[0][1][2] if past else 0
        emb = self._embed(win_ids, beacon_size)

        hidden = emb
        new_past = []
        for idx, layer in enumerate(self.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            if self._mem.layer_types[idx] == "full_attention":
                pkv = next(pv for li, pv in past if li == idx)
                out, npv = layer.self_attn(
                    hidden, position_embeddings=position_embeddings,
                    attention_mask=attn_mask, past_key_value=pkv,
                )
                new_past.append((idx, npv))
            else:
                previous_recurrent = self._mem._linear_recurrent.get(idx)
                previous_conv = self._mem._linear_conv.get(idx)
                reader_initial_recurrent = previous_recurrent
                reader_initial_conv = previous_conv
                if self._mem._store == "beacon":
                    reader_initial_recurrent = previous_recurrent.clone() if previous_recurrent is not None else None
                    reader_initial_conv = previous_conv.clone() if previous_conv is not None else None
                out, reader_recurrent, reader_conv = self._linear_attention_forward(
                    layer.linear_attn,
                    hidden,
                    reader_initial_recurrent,
                    reader_initial_conv,
                )
            hidden = residual + out
            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden
            if self._mem.layer_types[idx] == "linear_attention":
                if self._mem._store == "beacon":
                    beacon_mask = self._mem._step_beacon_indices.bool()
                    beacon_hidden = hidden[:, beacon_mask]
                    writer = self.model.beacon_linear_writers[str(idx)]
                    if self.beacon_config.beacon_question_memory_v1:
                        written = writer(beacon_hidden, previous_recurrent, self._mem._question)
                        self._mem._linear_recurrent[idx] = written
                        if self.training and win_labels is not None and self.beacon_config.beacon_readout_distill_weight > 0:
                            self._mem._readout_losses.append(
                                writer.readout_loss(written, reader_recurrent, self._mem._question)
                            )
                    else:
                        self._mem._linear_recurrent[idx] = writer(beacon_hidden, previous_recurrent)
                    self._mem._linear_conv.pop(idx, None)
                else:
                    self._mem._linear_recurrent[idx] = reader_recurrent
                    self._mem._linear_conv[idx] = reader_conv

        return new_past, hidden

    @staticmethod
    def _linear_attention_forward(module, hidden, previous_recurrent, previous_conv):
        """Qwen3.5 GatedDeltaNet 的显式、可微分状态前向。"""
        batch, length, _ = hidden.shape
        projected = module.in_proj_qkv(hidden).transpose(1, 2)
        if previous_conv is not None:
            conv_input = torch.cat([previous_conv, projected], dim=-1)
        else:
            conv_input = projected
        padded = F.pad(conv_input, (module.conv_kernel_size - conv_input.shape[-1], 0))
        next_conv = padded[..., -module.conv_kernel_size:]
        if module.causal_conv1d_fn is not None:
            mixed = module.causal_conv1d_fn(
                x=conv_input,
                weight=module.conv1d.weight.squeeze(1),
                bias=module.conv1d.bias,
                activation=module.activation,
            )
        else:
            mixed = F.silu(module.conv1d(conv_input)[:, :, :conv_input.shape[-1]])
        mixed = mixed[:, :, -length:].transpose(1, 2)

        query, key, value = mixed.split([module.key_dim, module.key_dim, module.value_dim], dim=-1)
        query = query.reshape(batch, length, -1, module.head_k_dim)
        key = key.reshape(batch, length, -1, module.head_k_dim)
        value = value.reshape(batch, length, -1, module.head_v_dim)
        repeat = module.num_v_heads // module.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        decay = -module.A_log.float().exp() * F.softplus(
            module.in_proj_a(hidden).float() + module.dt_bias
        )
        output, next_recurrent = module.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=decay,
            beta=module.in_proj_b(hidden).sigmoid(),
            initial_state=previous_recurrent,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        gate = module.in_proj_z(hidden).reshape(-1, module.head_v_dim)
        output = module.norm(output.reshape(-1, module.head_v_dim), gate)
        output = module.out_proj(output.reshape(batch, length, module.value_dim))
        return output, next_recurrent, next_conv

    # ------------------------------------------------------------------
    # 生成 / 推理

    def _rope_for(self, total, device):
        """计算 mRoPE cos/sin，覆盖 ``[0, total)`` 全局文本位置。"""
        pos_ids = torch.arange(total, device=device).expand(3, total).unsqueeze(1)
        ref = torch.zeros(
            1, 1, self.config.hidden_size,
            device=device, dtype=self.model.embed_tokens.weight.dtype,
        )
        return self.model.rotary_emb(ref, pos_ids)

def load_beacon_qwen3_5(model_name_or_path, beacon_config=None,
                        torch_dtype=torch.bfloat16, device_map="auto"):
    """装载 BeaconQwen3_5ForCausalLM。

    两种输入形态：
    1. ``architectures`` 含 ``BeaconQwen3_5ForCausalLM``（如训练后 ``_save_model`` 保存的
       checkpoint）：直接 ``from_pretrained`` 加载，保留训练好的 beacon 压缩参数。
    2. 原始 HF 多维包装器（``Qwen3_5ForConditionalGeneration``）：提取文本主干权重后构建，
       beacon 参数按配置重新初始化。
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if "BeaconQwen3_5ForCausalLM" in (cfg.architectures or []):
        if not hasattr(cfg, "beacon_linear_writer_rank"):
            raise RuntimeError(
                "该 checkpoint 来自旧版 Qwen3.5 Beacon：它保留了原始 information 的 "
                "linear-attention DynamicCache，且不含已训练的 BeaconLinearStateWriter。"
                "为避免静默使用随机 writer，必须用当前实现重新训练 Beacon 参数。"
            )
        model = BeaconQwen3_5ForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, device_map="cpu"
        )
        if beacon_config is not None:
            model.set_beacon_config(beacon_config)
        if device_map == "auto":
            model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        elif device_map is not None:
            model = model.to(device_map)
        model = model.to(torch_dtype)
        model.eval()
        return model

    from ..milestones.qwen35_text import load_text_causal_model

    base = load_text_causal_model(model_name_or_path, torch_dtype=torch_dtype, device_map="cpu")
    text_config = base.config
    if beacon_config is not None:
        beacon_config.merge_into_config(text_config)
    model = BeaconQwen3_5ForCausalLM(text_config)
    model.load_state_dict(base.state_dict(), strict=False)
    # 基础 checkpoint 不含 Beacon 权重：现在才从已加载的预训练权重初始化。
    # 已训练 Beacon checkpoint 的分支不执行此操作。
    model._init_beacon_params()
    if device_map == "auto":
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    elif device_map is not None:
        model = model.to(device_map)
    model = model.to(torch_dtype)
    if beacon_config is not None:
        model.set_beacon_config(beacon_config)
    model.eval()
    return model
