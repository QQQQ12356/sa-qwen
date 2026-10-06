"""分段损失（loss_segments）的单元测试。

覆盖三层：
1. 文本分类：``<thinking>`` / ``<search>`` / ``<answer>`` 的标签与正文被正确区分；
2. 配置：默认等价于不加权、部分覆盖、YAML 简写、非法值拒绝、随 checkpoint 往返；
3. 损失：加权 CE 与手算一致、权重 0 的位置不产生梯度、全默认时与历史路径逐位一致。
"""

import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5

from search_comp.data.searchr1_dataset import SearchR1Collator, SearchR1SFTDataset
from search_comp.data.trajectory import INFO_PREFIX, INFO_SUFFIX, build_search_chat_prompt
from search_comp.loss_segments import (
    SEGMENT_IDS,
    SEGMENT_IGNORE,
    SEGMENT_NAMES,
    LossSegmentConfig,
    SegmentWeight,
    classify_assistant_text,
    segment_ids_to_weights,
)
from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen35 import BeaconQwen3_5ForCausalLM


def _tiny_model(loss_segments=None):
    """两层（linear + full）微型 beacon 模型，窗口 4 / 压缩率 2。"""
    layer_types = ["linear_attention", "full_attention"]
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        layer_types=layer_types,
        max_position_embeddings=256,
        eos_token_id=2,
        pad_token_id=0,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [1, 1, 0],
        },
    )
    BeaconConfig(
        beacon_window=4,
        beacon_stride=4,
        beacon_ratio=2,
        beacon_linear_writer_rank=8,
        loss_segments=loss_segments or LossSegmentConfig(),
    ).merge_into_config(config)
    with patch.multiple(
        modeling_qwen3_5,
        chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None,
        causal_conv1d_fn=None,
        FusedRMSNormGated=None,
    ):
        return BeaconQwen3_5ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def tokenizer():
    from search_comp.milestones.qwen35_native import load_tokenizer

    tok = load_tokenizer("Qwen/Qwen3.5-2B")
    tok.pad_token = tok.eos_token
    return tok


# ======================================================================
# 1. 文本分类
# ======================================================================
def test_classifier_splits_tags_content_and_other(tokenizer):
    text = "<thinking>need the capital</thinking>\n<search>capital of France</search>"
    ids = tokenizer(text, add_special_tokens=False).input_ids
    classes = classify_assistant_text(text, tokenizer)
    assert len(classes) == len(ids)

    names = [SEGMENT_NAMES[index] for index in classes]
    # 标签与正文必须落在不同类别，且换行归 other
    assert "think_tag" in names and "think_content" in names
    assert "search_tag" in names and "search_content" in names
    assert "other" in names
    assert set(names) <= set(SEGMENT_NAMES)
    # 正文 token 的数量必须严格少于标签 token（证明真的做了区分）
    assert names.count("think_content") < names.count("think_tag") + names.count("think_content")


def test_classifier_handles_answer_and_think_alias(tokenizer):
    for text, kind in (
        ("<answer> Paris </answer>", "answer"),
        ("<think>reasoning</think>", "think"),
        ("<thinking>reasoning</thinking>", "think"),
    ):
        classes = classify_assistant_text(text, tokenizer)
        names = [SEGMENT_NAMES[index] for index in classes]
        assert f"{kind}_tag" in names
        assert f"{kind}_content" in names


def test_classifier_falls_back_without_offset_mapping():
    class _NoOffsets:
        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text.split())))}

    classes = classify_assistant_text("<answer> Paris </answer>", _NoOffsets())
    assert classes == [SEGMENT_IDS["other"]] * 3


def test_segment_ids_to_weights_zeroes_ignored_positions():
    config = LossSegmentConfig.from_dict({"answer_content": {"enabled": True, "weight": 3.0}})
    segment_ids = torch.tensor([[
        SEGMENT_IGNORE,
        SEGMENT_IDS["answer_content"],
        SEGMENT_IDS["think_content"],
    ]])
    weights = segment_ids_to_weights(segment_ids, config)
    assert weights.tolist() == [[0.0, 3.0, 1.0]]
    assert segment_ids_to_weights(None, config) is None


# ======================================================================
# 2. 配置
# ======================================================================
def test_default_config_is_identity():
    config = LossSegmentConfig()
    assert config.is_default
    assert config.weight_table().tolist() == [1.0] * len(SEGMENT_NAMES)
    assert config.enabled_names() == list(SEGMENT_NAMES)


def test_config_accepts_partial_and_shorthand():
    config = LossSegmentConfig.from_dict(
        {"answer_content": {"enabled": True, "weight": 5.0}, "think_content": False}
    )
    assert config.answer_content == SegmentWeight(enabled=True, weight=5.0)
    # bool 简写：False 等价于 weight 0
    assert config.think_content.enabled is False
    assert config.think_content.effective_weight == 0.0
    # 未提及的类别保持默认
    assert config.search_content == SegmentWeight()
    assert not config.is_default


def test_config_rejects_unknown_and_negative():
    with pytest.raises(ValueError, match="未知的片段类别"):
        LossSegmentConfig.from_dict({"thnik_content": True})
    with pytest.raises(ValueError, match="不能为负"):
        SegmentWeight(enabled=True, weight=-1.0)
    with pytest.raises(ValueError, match="未知的片段权重字段"):
        SegmentWeight.from_dict({"enabled": True, "weigth": 2.0})


def test_config_roundtrips_through_model_config():
    segments = LossSegmentConfig.from_dict(
        {"answer_content": {"enabled": True, "weight": 4.0}, "think_content": False}
    )
    original = BeaconConfig(loss_segments=segments)
    model_config = type("Cfg", (), {})()
    original.merge_into_config(model_config)
    restored = BeaconConfig.from_model_config(model_config)
    assert restored.loss_segments.to_dict() == segments.to_dict()
    # JSON 可序列化（会随 checkpoint 的 config.json 落盘）
    assert json.loads(json.dumps(original.to_dict()))["loss_segments"]["think_content"]["weight"] == 0.0


def test_legacy_model_config_without_loss_segments_falls_back_to_default():
    """旧 checkpoint 的 config.json 里没有 loss_segments，必须回落到默认配置。"""
    model_config = type("Cfg", (), {"beacon_window": 1024, "beacon_stride": 1024})()
    restored = BeaconConfig.from_model_config(model_config)
    assert restored.loss_segments.is_default


# ======================================================================
# 3. 损失
# ======================================================================
def _window_loss(model, hidden, labels, weights):
    valid_num = (labels != -100).sum(-1)
    return model._sparse_window_loss(hidden, labels, valid_num, weights)


def test_sparse_window_loss_matches_manual_weighted_ce():
    torch.manual_seed(0)
    model = _tiny_model()
    hidden = torch.randn(1, 6, 32, requires_grad=True)
    labels = torch.tensor([[3, 4, -100, 7, 8, -100]])
    weights = torch.tensor([[1.0, 1.0, 0.0, 2.0, 3.0, 0.0]])

    loss, weight_sum = _window_loss(model, hidden, labels, weights)

    with torch.no_grad():
        logits = model.lm_head(hidden).float()
    token_loss = F.cross_entropy(logits[0], labels[0], reduction="none")
    mask = labels[0] != -100
    expected = (token_loss[mask] * weights[0][mask]).sum() / weights[0][mask].sum()
    torch.testing.assert_close(loss[0], expected)
    torch.testing.assert_close(weight_sum[0], weights[0][mask].sum())


def test_sparse_window_loss_without_weights_is_plain_mean():
    torch.manual_seed(0)
    model = _tiny_model()
    hidden = torch.randn(1, 6, 32, requires_grad=True)
    labels = torch.tensor([[3, 4, -100, 7, 8, -100]])

    loss, weight_sum = _window_loss(model, hidden, labels, None)

    with torch.no_grad():
        logits = model.lm_head(hidden).float()
    token_loss = F.cross_entropy(logits[0], labels[0], reduction="none")
    mask = labels[0] != -100
    torch.testing.assert_close(loss[0], token_loss[mask].mean())
    torch.testing.assert_close(weight_sum[0], mask.sum().float())


def test_zero_weight_positions_receive_no_gradient():
    torch.manual_seed(0)
    model = _tiny_model()
    hidden = torch.randn(1, 6, 32, requires_grad=True)
    labels = torch.tensor([[3, 4, -100, 7, 8, -100]])
    weights = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

    loss, _ = _window_loss(model, hidden, labels, weights)
    grad = torch.autograd.grad(loss.sum(), hidden)[0]

    # 权重为 0 的位置不产生梯度（位置 2 是 -100，未监督）
    for index in (1, 2, 3, 5):
        assert grad[0, index].abs().sum() == 0, index
    # 有权重的位置必须有非零梯度
    for index in (0, 4):
        assert grad[0, index].abs().sum() > 0, index


def test_weights_are_relative_and_renormalised():
    """权重是相对量：``L = Σw·CE/Σw``，所以 (w_i/Σw) 还原后各位置梯度与不加权一致。

    这条性质保证调大某类权重**不会整体放大损失/有效学习率**，只改变类别间的份额。
    """
    torch.manual_seed(0)
    model = _tiny_model()
    labels = torch.tensor([[3, 4, -100, 7, 8, -100]])
    hidden_base = torch.randn(1, 6, 32)

    def grads_for(weights):
        hidden = hidden_base.clone().requires_grad_(True)
        loss, _ = _window_loss(model, hidden, labels, weights)
        return torch.autograd.grad(loss.sum(), hidden)[0]

    unweighted = grads_for(None)
    weights = torch.tensor([[1.0, 4.0, 0.0, 2.0, 0.5, 0.0]])
    weighted = grads_for(weights)

    mask = labels[0] != -100
    denominator = weights[0][mask].sum()
    # 把两侧的分母都还原掉，剩下的应当都是原始 ∂CE_i/∂h_i
    for index in torch.nonzero(mask).flatten().tolist():
        recovered = weighted[0, index] * denominator / weights[0, index]
        reference = unweighted[0, index] * mask.sum()
        torch.testing.assert_close(recovered, reference, rtol=1e-5, atol=1e-7)


# ----------------------------------------------------------------------
# 整条前向：默认权重必须与历史路径逐位一致
# ----------------------------------------------------------------------
_IDS = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])
_REGIONS = [(3, 7)]
_LABELS = torch.full((1, 12), -100, dtype=torch.long)
_LABELS[0, 8] = int(_IDS[0, 9])
_LABELS[0, 9] = int(_IDS[0, 10])
_LABELS[0, 10] = int(_IDS[0, 11])


def _segment_ids():
    ids = torch.full((1, 12), SEGMENT_IGNORE, dtype=torch.long)
    ids[0, 8] = SEGMENT_IDS["think_content"]
    ids[0, 9] = SEGMENT_IDS["answer_content"]
    ids[0, 10] = SEGMENT_IDS["think_content"]
    return ids


def test_forward_without_segments_equals_default_weighted_forward():
    model = _tiny_model()
    plain = model(input_ids=_IDS, labels=_LABELS.clone(), regions=_REGIONS).loss
    with_segments = model(
        input_ids=_IDS, labels=_LABELS.clone(), regions=_REGIONS,
        loss_segment_ids=_segment_ids(),
    ).loss
    # 全默认权重等价于不加权
    torch.testing.assert_close(plain, with_segments, rtol=0, atol=0)


def test_forward_upweighting_answer_changes_loss():
    plain_model = _tiny_model()
    plain = plain_model(input_ids=_IDS, labels=_LABELS.clone(), regions=_REGIONS).loss

    weighted_model = _tiny_model(
        LossSegmentConfig.from_dict({"answer_content": {"enabled": True, "weight": 8.0}})
    )
    weighted = weighted_model(
        input_ids=_IDS, labels=_LABELS.clone(), regions=_REGIONS,
        loss_segment_ids=_segment_ids(),
    ).loss
    assert not torch.allclose(plain, weighted)


def test_forward_disabling_every_segment_yields_zero_loss():
    model = _tiny_model(
        LossSegmentConfig.from_dict({name: False for name in SEGMENT_NAMES})
    )
    loss = model(
        input_ids=_IDS, labels=_LABELS.clone(), regions=_REGIONS,
        loss_segment_ids=_segment_ids(),
    ).loss
    assert loss.item() == 0.0


def test_weighted_loss_backward_in_training_mode():
    """训练态会走 beacon_checkpoint_loss 的分块重算路径，权重需一并传入。"""
    model = _tiny_model(
        LossSegmentConfig.from_dict({"answer_content": {"enabled": True, "weight": 3.0}})
    ).train()
    loss = model(
        input_ids=_IDS, labels=_LABELS.clone(), regions=_REGIONS,
        loss_segment_ids=_segment_ids(),
    ).loss
    loss.backward()
    writer = model.model.beacon_linear_writers["0"]
    assert writer.down.weight.grad is not None
    assert torch.isfinite(writer.down.weight.grad).all()


# ======================================================================
# 4. 数据集接线
# ======================================================================
def _write_sample(path):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Answer the question. Question: Who founded Google?"},
        {
            "role": "assistant",
            "content": "<thinking>\nI need to search.\n</thinking>\n<search>Google founders</search>",
        },
        {
            "role": "user",
            "content": f"{INFO_PREFIX}[Document 1] Google was founded by Larry Page.{INFO_SUFFIX}",
        },
        {
            "role": "assistant",
            "content": "<thinking>\nNow I know.\n</thinking>\n<answer> Larry Page </answer>",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"messages": msgs}) + "\n")


def test_searchr1_dataset_omits_segments_by_default(tokenizer, tmp_path):
    data = tmp_path / "sft.jsonl"
    _write_sample(str(data))
    sample = SearchR1SFTDataset(str(data), tokenizer)[0]
    assert sample["loss_segment_ids"] is None
    batch = SearchR1Collator(tokenizer)([sample])
    assert "loss_segment_ids" not in batch


def test_searchr1_dataset_segment_ids_align_with_labels(tokenizer, tmp_path):
    data = tmp_path / "sft.jsonl"
    _write_sample(str(data))
    segments = LossSegmentConfig.from_dict({"answer_content": {"enabled": True, "weight": 5.0}})
    ds = SearchR1SFTDataset(str(data), tokenizer, loss_segments=segments)
    sample = ds[0]

    ids = sample["loss_segment_ids"]
    assert ids is not None
    assert len(ids) == len(sample["input_ids"])
    # 每个被监督 token 都必须有合法类别；未监督 token 必须是 IGNORE
    for token_index, (label, segment) in enumerate(zip(sample["labels"], ids)):
        if label == -100:
            assert segment == SEGMENT_IGNORE, token_index
        else:
            assert 0 <= segment < len(SEGMENT_NAMES), token_index
    # 答案正文必须被识别出来
    assert SEGMENT_IDS["answer_content"] in ids
    assert SEGMENT_IDS["think_content"] in ids
    # 压缩区（文档）不参与损失，也不该带类别
    for start, end in sample["regions"]:
        assert all(value == SEGMENT_IGNORE for value in ids[start:end])

    batch = SearchR1Collator(tokenizer)([sample])
    assert batch["loss_segment_ids"].shape == batch["labels"].shape


