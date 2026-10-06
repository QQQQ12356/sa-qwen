"""Qwen3 全注意力 Beacon 模型；无 GatedDeltaNet Reader/Writer。"""
from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3ForCausalLM, apply_rotary_pos_emb
from .beacon_config import BeaconConfig
from .beacon_common import BeaconMemory, BeaconCausalLMMixin

class BeaconQwen3Attention(Qwen3Attention):
    """Qwen3 全注意力，缓存 RoPE 前 K/V，并为 beacon 切换投影。"""

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
                self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
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
        query_states = qg

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

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(key_states.view(hidden_shape)).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states, input_shape

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
        - 否则委托原生 ``Qwen3Attention.forward``（``enable_beacon=False`` 时
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

        query_states, key_states, value_states, input_shape = self._qkv(
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
        if "o" in self.beacon_param and beacon_size > 0:
            oo = self.o_proj(attn_output)
            bo = self.beacon_o_proj(attn_output)
            bi = beacon_indices[-cur_len:]
            attn_output = torch.where((bi == 0)[:, None], oo, bo)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output, new_past

class BeaconQwen3ForCausalLM(BeaconCausalLMMixin, Qwen3ForCausalLM):
    memory_class = BeaconMemory
    accepts_loss_kwargs = False

    def __init__(self, config):
        # 原生 Qwen3 checkpoint 未包含 Beacon 配置时采用当前暂停策略。
        defaults = {
            "beacon_information_mode": "whole",
            "beacon_window_enabled": False,
            "beacon_doc_region_split": False,
            "beacon_question_memory_v1": False,
            "beacon_readout_distill_weight": 0.0,
            "beacon_keep_window": 1024,
        }
        for name, value in defaults.items():
            if not hasattr(config, name):
                setattr(config, name, value)
        if any(kind != "full_attention" for kind in config.layer_types):
            raise ValueError("Qwen3 Beacon 当前仅支持全注意力层")
        super().__init__(config)
        self.beacon_config = BeaconConfig.from_model_config(config)
        self.beacon_config.merge_into_config(config)
        if self.beacon_config.beacon_question_memory_v1:
            raise ValueError("Qwen3 没有线性状态，不支持 beacon_question_memory_v1")
        self.model.beacon_embed_tokens = nn.Embedding(1, config.hidden_size)
        for idx, layer in enumerate(self.model.layers):
            layer.self_attn = BeaconQwen3Attention(config, idx)
        self.post_init()
        self._init_beacon_params()
        self._last_loss_parts = {}

    def _init_beacon_params(self):
        source = getattr(self.config, f"{self.beacon_config.beacon_embed_init}_token_id", None)
        if isinstance(source, (list, tuple)):
            source = source[0]
        with torch.no_grad():
            if source is not None and 0 <= source < self.config.vocab_size:
                self.model.beacon_embed_tokens.weight.copy_(self.model.embed_tokens.weight[source])
            else:
                self.model.beacon_embed_tokens.weight.normal_(0, self.config.hidden_size ** -0.5)
            for layer in self.model.layers:
                for name in self.beacon_config.beacon_param.split():
                    beacon = getattr(layer.self_attn, f"beacon_{name}_proj")
                    base = getattr(layer.self_attn, f"{name}_proj")
                    beacon.weight.copy_(base.weight)
                    if beacon.bias is not None:
                        beacon.bias.copy_(base.bias)

    def set_beacon_config(self, config):
        if config.beacon_question_memory_v1:
            raise ValueError("Qwen3 不支持线性 question memory")
        if config.beacon_param != self.beacon_config.beacon_param:
            raise ValueError("beacon_param 改变模型结构，必须在加载前配置")
        self.beacon_config = config
        config.merge_into_config(self.config)
        return self

    def _rope_for(self, total, device):
        positions = torch.arange(total, device=device).unsqueeze(0)
        ref = self.model.embed_tokens.weight.new_zeros(1, 1, self.config.hidden_size)
        return self.model.rotary_emb(ref, positions)

    def _window_hidden(self, win_ids, win_labels, attn_mask, position_embeddings, past):
        hidden = self._embed(win_ids, past[0][1][2])
        new_past = []
        for idx, layer in enumerate(self.model.layers):
            residual = hidden
            out, current = layer.self_attn(
                layer.input_layernorm(hidden), position_embeddings=position_embeddings,
                attention_mask=attn_mask, past_key_value=past[idx][1],
            )
            hidden = residual + out
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            new_past.append((idx, current))
        return new_past, hidden


def load_beacon_qwen3(model_name_or_path="Qwen/Qwen3-4B-Thinking-2507", beacon_config=None,
                      torch_dtype=torch.bfloat16, device_map="auto"):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name_or_path)
    trained = "BeaconQwen3ForCausalLM" in (config.architectures or [])
    if beacon_config is not None and not trained:
        beacon_config.merge_into_config(config)
    model = BeaconQwen3ForCausalLM.from_pretrained(
        model_name_or_path, config=config, torch_dtype=torch_dtype,
        attn_implementation="eager",
    )
    if not trained:
        # 此时基础权重已就位，仅初始化基础 checkpoint 中缺失的 beacon 参数。
        model._init_beacon_params()
    if beacon_config is not None:
        model.set_beacon_config(beacon_config)
    target = ("cuda" if torch.cuda.is_available() else "cpu") if device_map == "auto" else device_map
    if target is not None:
        model.to(target)
    return model.eval()
