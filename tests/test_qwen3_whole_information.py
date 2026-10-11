"""在 CPU 小模型上验证架构分派、整段压缩、损失与缓存生命周期。"""
from unittest.mock import patch

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen3 import BeaconQwen3ForCausalLM
from search_comp.models.model_loader import load_model
from test_beacon_qwen35_linear_memory import _tiny_model


def tiny_qwen3():
    config = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=256, eos_token_id=2, pad_token_id=0,
        attention_dropout=0.0, attention_bias=True,
    )
    BeaconConfig(beacon_window=4, beacon_stride=4, beacon_ratio=2).merge_into_config(config)
    return BeaconQwen3ForCausalLM(config).eval()


@pytest.fixture(params=['qwen3', 'qwen35'])
def model(request):
    torch.manual_seed(9)
    return tiny_qwen3() if request.param == 'qwen3' else _tiny_model()


def whole_config(model, position, loss):
    config = BeaconConfig.from_dict(model.beacon_config.to_dict())
    config.beacon_information_mode = 'whole'
    config.beacon_information_loss = loss
    config.beacon_pos = position
    config.beacon_continuation_loss_weight = 0.5
    config.__post_init__()
    return config


@pytest.mark.parametrize('position', ['append', 'intersect'])
@pytest.mark.parametrize('length', [1, 2, 7, 12])
def test_whole_information_is_one_forward_and_commits_only_beacons(model, position, length):
    model.set_beacon_config(whole_config(model, position, False))
    ids = torch.arange(10, 10 + length + 4).unsqueeze(0)
    with patch.object(model, '_window_hidden', wraps=model._window_hidden) as forward:
        memory = model.prefill_and_get_cache(ids, [(2, 2 + length)])
    assert forward.call_count == 3  # prefix, entire information, suffix
    expected = 4 + (length + 1) // 2
    for kind, (key, value) in zip(memory.layer_types, memory._cache):
        if kind == 'full_attention':
            assert key.shape[2] == value.shape[2] == expected
    if isinstance(model, BeaconQwen3ForCausalLM):
        assert not hasattr(memory, '_linear_recurrent')
        assert not any('writer' in name for name, _ in model.named_parameters())


@pytest.mark.parametrize('position', ['append', 'intersect'])
def test_whole_loss_is_causal_in_region_and_switch_preserves_cache(model, position):
    ids = torch.arange(10, 23).unsqueeze(0)
    labels = torch.full_like(ids, -100)
    enabled = whole_config(model, position, True)
    model.set_beacon_config(enabled)
    model._mem = model.memory_class(model, enabled)
    model._mem.prepare(ids, labels, [(2, 11)])
    model._mem.step()
    win_ids, _, _, _, _ = model._mem.step()
    rows, targets = model._mem._step_cont_rows, model._mem._step_cont_labels
    assert targets.tolist() == [ids[0, 3:11].tolist()]
    for row, target in zip(rows[0], targets[0]):
        assert win_ids[0, row] == target - 1
        assert row < (win_ids[0] == target).nonzero()[0, 0]
    model.train()
    result = model(input_ids=ids, labels=labels, regions=[(2, 11)])
    assert torch.isfinite(result.loss) and result.loss.item() > 0
    saved = [(None if k is None else k.detach().clone(), None if v is None else v.detach().clone()) for k, v in model._mem._cache]
    result.loss.backward()
    assert model.model.embed_tokens.weight.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    model.set_beacon_config(whole_config(model, position, False))
    disabled = model(input_ids=ids, labels=labels, regions=[(2, 11)])
    assert disabled.loss.item() == 0
    for (old_k, old_v), (new_k, new_v) in zip(saved, model._mem._cache):
        if old_k is not None:
            torch.testing.assert_close(old_k, new_k)
            torch.testing.assert_close(old_v, new_v)
    assert not model._mem._cont_losses


def test_qwen3_without_compression_matches_native_and_decode():
    model = tiny_qwen3()
    native = Qwen3ForCausalLM(model.config).eval()
    native.load_state_dict(model.state_dict(), strict=False)
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    with torch.no_grad():
        _, logits = model.prefill_and_get_cache(ids, [], return_last_logits=True)
        torch.testing.assert_close(logits, native(ids).logits[:, -1], atol=1e-6, rtol=1e-5)
        token = torch.tensor([[16]])
        step = model.decode_step(token)
        torch.testing.assert_close(step, native(torch.cat([ids, token], 1)).logits[:, -1], atol=1e-6, rtol=1e-5)
        model.beacon_config.enable_beacon = False
        torch.testing.assert_close(model(ids).logits, native(ids).logits)


def test_qwen3_saved_beacon_and_base_loading(tmp_path):
    model = tiny_qwen3()
    model.set_beacon_config(whole_config(model, 'intersect', True))
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 30]])
    with torch.no_grad():
        _, expected = model.prefill_and_get_cache(ids, [(2, 7)], return_last_logits=True)
    model.save_pretrained(tmp_path / 'beacon')
    restored = load_model(tmp_path / 'beacon', torch_dtype=torch.float32, device_map='cpu')
    assert restored.beacon_config.beacon_information_mode == 'whole'
    with torch.no_grad():
        _, actual = restored.prefill_and_get_cache(ids, [(2, 7)], return_last_logits=True)
    torch.testing.assert_close(expected, actual)
    native = Qwen3ForCausalLM(model.config)
    native.save_pretrained(tmp_path / 'base')
    loaded = load_model(tmp_path / 'base', torch_dtype=torch.float32, device_map='cpu')
    for layer in loaded.model.layers:
        torch.testing.assert_close(layer.self_attn.q_proj.weight, layer.self_attn.beacon_q_proj.weight)
        torch.testing.assert_close(layer.self_attn.q_proj.bias, layer.self_attn.beacon_q_proj.bias)


@pytest.mark.parametrize('position', ['append', 'intersect'])
def test_whole_prefill_reuse_matches_full_prefill(model, position):
    model.set_beacon_config(whole_config(model, position, False))
    ids = torch.arange(10, 23).unsqueeze(0)
    with torch.no_grad():
        _, expected = model.prefill_and_get_cache(ids, [(2, 9)], return_last_logits=True)
        model.prefill_and_get_cache(ids[:, :9], [(2, 9)])
        _, actual = model.prefill_and_get_cache(ids[:, 9:], [], reuse_cache=True, return_last_logits=True)
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=1e-5)


def test_whole_config_rejects_document_splitting():
    with pytest.raises(ValueError, match='子文档'):
        BeaconConfig(beacon_information_mode='whole', beacon_doc_region_split=True)


@pytest.mark.parametrize('position', ['append', 'intersect'])
def test_answer_loss_trains_qwen3_beacon_kv_projections(position):
    model = tiny_qwen3().train()
    model.set_beacon_config(whole_config(model, position, False))
    ids = torch.arange(10, 23).unsqueeze(0)
    labels = torch.full_like(ids, -100)
    labels[:, 10:] = ids[:, 10:]
    result = model(input_ids=ids, labels=labels, regions=[(2, 9)])
    result.loss.backward()
    assert model.model.beacon_embed_tokens.weight.grad.abs().sum() > 0
    for layer in model.model.layers:
        assert layer.self_attn.beacon_k_proj.weight.grad.abs().sum() > 0
        assert layer.self_attn.beacon_v_proj.weight.grad.abs().sum() > 0


def test_qwen3_lora_merge_preserves_beacon_outputs(tmp_path):
    peft = pytest.importorskip('peft')
    model = tiny_qwen3()
    model.set_beacon_config(whole_config(model, 'intersect', False))
    lora = peft.get_peft_model(model, peft.LoraConfig(
        task_type='CAUSAL_LM', r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'],
    )).eval()
    # Simulate trained nonzero adapters without running training.
    with torch.no_grad():
        for name, parameter in lora.named_parameters():
            if 'lora_B' in name:
                parameter.fill_(0.01)
        ids = torch.arange(10, 23).unsqueeze(0)
        _, expected = lora.base_model.model.prefill_and_get_cache(ids, [(2, 9)], return_last_logits=True)
        merged = lora.merge_and_unload().eval()
        merged.save_pretrained(tmp_path)
        restored = load_model(tmp_path, torch_dtype=torch.float32, device_map='cpu')
        _, actual = restored.prefill_and_get_cache(ids, [(2, 9)], return_last_logits=True)
    torch.testing.assert_close(expected, actual, atol=2e-6, rtol=1e-5)


def test_whole_ratio_need_not_divide_legacy_window():
    model = tiny_qwen3()
    config = whole_config(model, 'intersect', False)
    config.beacon_ratio = 3
    config.__post_init__()
    model.set_beacon_config(config)
    ids = torch.tensor([[10, 11, 12, 13]])
    memory = model.prefill_and_get_cache(ids, [(0, 4)])
    assert memory._cache[0][0].shape[2] == 2
    assert memory._step_beacon_indices.tolist() == [0, 0, 0, 1, 0, 1]


def test_paused_windows_ignore_window_and_stride_for_information_and_keep():
    model = tiny_qwen3()
    config = whole_config(model, 'intersect', False)
    config.beacon_window_enabled = False
    config.beacon_window = 0
    config.beacon_stride = -1  # Paused legacy parameters are not consulted.
    config.beacon_keep_window = None
    config.beacon_readout_distill_weight = 0.0
    config.__post_init__()
    model.set_beacon_config(config)
    ids = torch.arange(10, 40).unsqueeze(0)
    with patch.object(model, '_window_hidden', wraps=model._window_hidden) as forward:
        memory = model.prefill_and_get_cache(ids, [(10, 24)])
    assert forward.call_count == 3
    assert memory._cache[0][0].shape[2] == 10 + 7 + 6
    assert '暂停' in config.describe_layout()


def test_training_disables_all_length_windows_for_two_information_rounds(model):
    from search_comp.trainer.beacon_trainer import configure_whole_trajectory

    cfg = {"beacon": model.beacon_config.to_dict()}
    cfg["beacon"].update(beacon_window=0, beacon_stride=-1, beacon_keep_window=1,
                         beacon_doc_region_split=True)
    configure_whole_trajectory(cfg)
    config = BeaconConfig.from_dict(cfg["beacon"])
    model.set_beacon_config(config)
    ids = torch.arange(10, 40).unsqueeze(0)
    with patch.object(model, '_window_hidden', wraps=model._window_hidden) as forward:
        model.prefill_and_get_cache(ids, [(5, 13), (18, 26)])
    assert forward.call_count == 5  # three whole keep segments and two information blocks
    assert not config.beacon_window_enabled
    assert config.beacon_keep_window == 0
    assert not config.beacon_doc_region_split


@pytest.mark.parametrize('position', ['append', 'intersect'])
def test_qwen3_discards_exact_original_information_kv_before_decoding(position):
    model = tiny_qwen3()
    config = whole_config(model, position, False)
    config.beacon_window_enabled = False
    config.beacon_readout_distill_weight = 0.0
    model.set_beacon_config(config)
    ids = torch.arange(10, 20).unsqueeze(0)
    recorded = []
    original = model._window_hidden

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        recorded.append((result[0], model._mem._step_beacon_indices))
        return result

    with torch.no_grad(), patch.object(model, '_window_hidden', side_effect=record):
        memory = model.prefill_and_get_cache(ids, [(2, 10)])
    prefix, _ = recorded[0]
    compressed, indices = recorded[1]
    assert indices.sum().item() == 4
    for (layer, (key, value, _, _)), (_, (pk, pv, _, _)) in zip(compressed, prefix):
        # Full current information K/V exist only transiently inside its forward.
        assert key.shape[2] == 12
        retained_key, retained_value = memory._cache[layer]
        torch.testing.assert_close(retained_key, torch.cat([pk, key[:, :, indices.bool()]], dim=2))
        torch.testing.assert_close(retained_value, torch.cat([pv, value[:, :, indices.bool()]], dim=2))
        assert retained_key.shape[2] == 6
    with torch.no_grad(), patch.object(model.model.layers[0].self_attn, '_beacon_forward',
                                      wraps=model.model.layers[0].self_attn._beacon_forward) as attention:
        model.decode_step(torch.tensor([[40]]))
    decode_past = attention.call_args.args[3]
    assert decode_past[0].shape[2] == 6  # Two keep tokens plus four beacons; no raw document KV.


def test_new_native_qwen3_defaults_pause_unused_features():
    from transformers import Qwen3Config
    config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, eos_token_id=2)
    model = BeaconQwen3ForCausalLM(config)
    assert model.beacon_config.beacon_information_mode == 'whole'
    assert not model.beacon_config.beacon_window_enabled
    assert not model.beacon_config.beacon_question_memory_v1
    assert not model.beacon_config.beacon_doc_region_split
    assert model.beacon_config.beacon_readout_distill_weight == 0.0


def test_plain_qwen3_checkpoint_loads_with_paused_beacon_defaults(tmp_path):
    native_config = Qwen3Config(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, eos_token_id=2,
    )
    Qwen3ForCausalLM(native_config).save_pretrained(tmp_path)
    loaded = load_model(tmp_path, torch_dtype=torch.float32, device_map='cpu')
    assert not loaded.beacon_config.beacon_window_enabled
    assert loaded.beacon_config.beacon_readout_distill_weight == 0.0
    assert loaded.beacon_config.beacon_information_mode == 'whole'
    torch.testing.assert_close(loaded.model.layers[0].self_attn.beacon_k_proj.weight,
                               loaded.model.layers[0].self_attn.k_proj.weight)
    with torch.no_grad():
        memory = loaded.prefill_and_get_cache(torch.arange(10, 20).unsqueeze(0), [(2, 10)])
    assert memory._cache[0][0].shape[2] == 3  # Two keep tokens plus ceil(8/64) beacon.
