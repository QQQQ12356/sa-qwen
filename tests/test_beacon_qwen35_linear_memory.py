from unittest.mock import patch
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pytest
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5

from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen35 import BeaconQwen3_5ForCausalLM


def _tiny_model(beacon_pos="append", layer_types=None, question_memory_v1=False):
    if layer_types is None:
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
        beacon_pos=beacon_pos,
        beacon_question_memory_v1=question_memory_v1,
    ).merge_into_config(config)
    with patch.multiple(
        modeling_qwen3_5,
        chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None,
        causal_conv1d_fn=None,
        FusedRMSNormGated=None,
    ):
        return BeaconQwen3_5ForCausalLM(config).eval()


def test_compressed_region_commits_only_beacon_linear_state():
    torch.manual_seed(7)
    model = _tiny_model()
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
    memory = model.prefill_and_get_cache(ids, regions=[(2, 6)])

    assert set(memory._linear_recurrent) == {0}
    assert memory._linear_recurrent[0].shape == (1, 2, 8, 8)
    assert 0 in memory._linear_conv

    compressed_only = model.prefill_and_get_cache(ids[:, :6], regions=[(2, 6)])
    assert 0 not in compressed_only._linear_conv


@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
@pytest.mark.parametrize("information_mode", ["window", "whole"])
def test_native_linear_state_survives_compression(monkeypatch, beacon_pos, information_mode):
    model = _tiny_model(beacon_pos)
    cfg = BeaconConfig.from_dict({
        **model.beacon_config.to_dict(),
        "beacon_preserve_native_linear_state": True,
        "beacon_information_mode": information_mode,
    })
    model.set_beacon_config(cfg)
    states = []
    native_reader = model._linear_attention_forward

    def record_reader(module, hidden, previous_recurrent, previous_conv):
        if states:
            torch.testing.assert_close(previous_recurrent, states[-1][0])
            torch.testing.assert_close(previous_conv, states[-1][1])
        result = native_reader(module, hidden, previous_recurrent, previous_conv)
        states.append(result[1:])
        return result

    monkeypatch.setattr(model, "_linear_attention_forward", record_reader)
    with patch.object(model.model.beacon_linear_writers["0"], "forward", side_effect=AssertionError("writer must be bypassed")):
        ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24]])
        memory = model.prefill_and_get_cache(ids, regions=[(2, 7)])
        assert memory._linear_recurrent[0] is states[-1][0]
        assert memory._linear_conv[0] is states[-1][1]
        # 两个 keep token + ceil(5/2) 个 Beacon，原文 KV 仍被丢弃。
        assert memory._cache[1][0].shape[2] == 5
        assert torch.isfinite(model.decode_step(torch.tensor([[30]]))).all()


def test_native_linear_state_has_document_gradient_without_writer():
    model = _tiny_model().train()
    model.beacon_config.beacon_preserve_native_linear_state = True
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
    labels = torch.full_like(ids, -100)
    labels[:, -2:] = ids[:, -2:]
    output = model(input_ids=ids, labels=labels, compress_regions=[(2, 6)])
    output.loss.backward()
    assert model.model.embed_tokens.weight.grad[20:24].abs().sum() > 0
    assert all(p.grad is None for p in model.model.beacon_linear_writers.parameters())


def test_native_linear_state_config_roundtrip(tmp_path):
    model = _tiny_model()
    model.set_beacon_config(BeaconConfig.from_dict({
        **model.beacon_config.to_dict(), "beacon_preserve_native_linear_state": True,
    }))
    model.save_pretrained(tmp_path)
    restored = BeaconQwen3_5ForCausalLM.from_pretrained(tmp_path)
    assert restored.beacon_config.beacon_preserve_native_linear_state is True
    assert BeaconConfig().beacon_preserve_native_linear_state is False
    with pytest.raises(ValueError, match="beacon_preserve_native_linear_state"):
        BeaconConfig(beacon_preserve_native_linear_state="false")


@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
def test_linear_writer_receives_gradient_from_post_compression_tokens(beacon_pos):
    torch.manual_seed(7)
    model = _tiny_model(beacon_pos).train()
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
    labels = ids.clone()
    output = model(input_ids=ids, labels=labels, compress_regions=[(2, 6)])
    output.loss.backward()

    writer = model.model.beacon_linear_writers["0"]
    assert writer.up.weight.grad is not None
    assert writer.up.weight.grad.abs().sum() > 0


def test_sparse_chunked_loss_matches_dense_cross_entropy():
    torch.manual_seed(7)
    model = _tiny_model().train()
    model.beacon_config.beacon_loss_chunk_size = 2
    hidden = torch.randn(1, 5, model.config.hidden_size, requires_grad=True)
    labels = torch.tensor([[10, -100, 20, 21, -100]])
    valid_num = (labels != -100).sum(-1)

    sparse, _weight_sum = model._sparse_window_loss(hidden, labels, valid_num)
    sparse = sparse.mean()
    selected = labels[0] != -100
    dense = F.cross_entropy(
        model.lm_head(hidden[0, selected]).float(),
        labels[0, selected],
    )

    torch.testing.assert_close(sparse, dense, atol=1e-6, rtol=1e-5)
    sparse.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_training_does_not_materialize_return_logits():
    torch.manual_seed(7)
    model = _tiny_model().train()
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
    labels = ids.clone()
    output = model(input_ids=ids, labels=labels, compress_regions=[(2, 6)])

    assert output.logits is None
    assert torch.isfinite(output.loss)


def test_training_lm_head_never_exceeds_loss_chunk_size():
    torch.manual_seed(7)
    model = _tiny_model().train()
    model.beacon_config.beacon_loss_chunk_size = 2
    projected_lengths = []

    def record_shape(_module, inputs):
        projected_lengths.append(inputs[0].shape[-2])

    handle = model.lm_head.register_forward_pre_hook(record_shape)
    try:
        ids = torch.tensor([[10, 11, 20, 21, 22, 23, 30, 31]])
        labels = torch.tensor([[10, 11, -100, -100, -100, -100, 30, 31]])
        output = model(input_ids=ids, labels=labels, compress_regions=[(2, 6)])
        output.loss.backward()
    finally:
        handle.remove()

    assert projected_lengths
    assert max(projected_lengths) <= 2


def test_intersect_discards_reader_state_after_every_chunk(monkeypatch):
    model = _tiny_model("intersect")
    reader_inputs = []
    writer_outputs = []
    native_reader = model._linear_attention_forward

    def record_reader(module, hidden, previous_recurrent, previous_conv):
        reader_inputs.append((hidden.shape[1], previous_recurrent, previous_conv))
        return native_reader(module, hidden, previous_recurrent, previous_conv)

    monkeypatch.setattr(model, "_linear_attention_forward", record_reader)
    handle = model.model.beacon_linear_writers["0"].register_forward_hook(
        lambda module, inputs, output: writer_outputs.append(output)
    )
    try:
        ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24]])
        memory = model.prefill_and_get_cache(ids, regions=[(2, 7)])
    finally:
        handle.remove()

    assert [entry[0] for entry in reader_inputs] == [2, 6, 2]
    assert len(writer_outputs) == 2
    torch.testing.assert_close(reader_inputs[2][1], writer_outputs[0])
    assert reader_inputs[2][1].data_ptr() != writer_outputs[0].data_ptr()
    assert reader_inputs[2][2] is None
    assert memory._linear_recurrent[0] is writer_outputs[-1]
    assert 0 not in memory._linear_conv
    assert memory._cache[1][0].shape[2] == 5


def test_intersect_first_beacon_is_independent_of_future_slice():
    torch.manual_seed(7)
    model = _tiny_model("intersect")
    first = model.prefill_and_get_cache(
        torch.tensor([[10, 11, 20, 21, 22, 23]]), regions=[(2, 6)]
    )
    second = model.prefill_and_get_cache(
        torch.tensor([[10, 11, 20, 21, 40, 41]]), regions=[(2, 6)]
    )
    for first_cache, second_cache in zip(first._cache[1], second._cache[1]):
        torch.testing.assert_close(first_cache[:, :, :3], second_cache[:, :, :3])
    logits = model.decode_step(torch.tensor([[30]]))
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
@torch.no_grad()
def test_compressed_reader_state_poison_cannot_escape_chunk(monkeypatch, beacon_pos):
    torch.manual_seed(7)
    model = _tiny_model(beacon_pos, ["linear_attention", "full_attention"] * 2)
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31]])
    baseline, baseline_logits = model.prefill_and_get_cache(ids, regions=[(2, 11)], return_last_logits=True)
    baseline_decode = model.decode_step(torch.tensor([[32]]))
    native_reader = model._linear_attention_forward
    poisoned_layers = []

    def poison_reader(module, hidden, previous_recurrent, previous_conv):
        result, reader_recurrent, reader_conv = native_reader(module, hidden, previous_recurrent, previous_conv)
        if model._mem._store == "beacon":
            layer_idx = next(idx for idx, layer in enumerate(model.model.layers) if getattr(layer, "linear_attn", None) is module)
            persistent = model._mem._linear_recurrent[layer_idx]
            expected = persistent.clone()
            assert previous_recurrent.data_ptr() != persistent.data_ptr()
            previous_recurrent.fill_(float("nan"))
            torch.testing.assert_close(persistent, expected)
            if previous_conv is not None:
                persistent_conv = model._mem._linear_conv[layer_idx]
                expected_conv = persistent_conv.clone()
                assert previous_conv.data_ptr() != persistent_conv.data_ptr()
                previous_conv.fill_(float("nan"))
                torch.testing.assert_close(persistent_conv, expected_conv)
            poisoned_layers.append(layer_idx)
            return result, torch.full_like(reader_recurrent, float("nan")), torch.full_like(reader_conv, float("nan"))
        return result, reader_recurrent, reader_conv

    monkeypatch.setattr(model, "_linear_attention_forward", poison_reader)
    actual, actual_logits = model.prefill_and_get_cache(ids, regions=[(2, 11)], return_last_logits=True)
    torch.testing.assert_close(actual_logits, baseline_logits)
    torch.testing.assert_close(model.decode_step(torch.tensor([[32]])), baseline_decode)
    assert poisoned_layers == [0, 2] * 3
    for layer_idx in (0, 2):
        torch.testing.assert_close(actual._linear_recurrent[layer_idx], baseline._linear_recurrent[layer_idx])
        torch.testing.assert_close(actual._linear_conv[layer_idx], baseline._linear_conv[layer_idx])
    for layer_idx in (1, 3):
        for actual_cache, baseline_cache in zip(actual._cache[layer_idx], baseline._cache[layer_idx]):
            torch.testing.assert_close(actual_cache, baseline_cache)


@torch.no_grad()
def test_linear_persistent_state_depends_on_documents_only_through_beacons(monkeypatch):
    torch.manual_seed(7)
    model = _tiny_model("intersect", ["linear_attention", "full_attention"] * 2)
    for writer in model.model.beacon_linear_writers.values():
        native_writer = writer.forward

        def fixed_beacon_writer(beacons, previous, original=native_writer):
            assert beacons.shape[1] == 2
            return original(torch.zeros_like(beacons), previous)

        monkeypatch.setattr(writer, "forward", fixed_beacon_writer)

    first = model.prefill_and_get_cache(
        torch.tensor([[10, 11, 20, 21, 22, 23]]), regions=[(2, 6)]
    )
    second = model.prefill_and_get_cache(
        torch.tensor([[10, 11, 40, 41, 42, 43]]), regions=[(2, 6)]
    )
    for layer_idx in (0, 2):
        torch.testing.assert_close(first._linear_recurrent[layer_idx], second._linear_recurrent[layer_idx])
    assert first._linear_conv == second._linear_conv == {}


@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
def test_reader_state_copy_preserves_cross_chunk_writer_gradients(beacon_pos):
    torch.manual_seed(7)
    model = _tiny_model(beacon_pos).train()
    committed_states = []

    def record_state(module, inputs, output):
        output.retain_grad()
        committed_states.append(output)

    handle = model.model.beacon_linear_writers["0"].register_forward_hook(record_state)
    try:
        ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31]])
        labels = torch.full_like(ids, -100)
        labels[:, 10:] = ids[:, 10:]
        output = model(input_ids=ids, labels=labels, compress_regions=[(2, 10)])
        output.loss.backward()
    finally:
        handle.remove()

    assert len(committed_states) == 2
    for state in committed_states:
        assert state.grad is not None
        assert torch.isfinite(state.grad).all()
        assert state.grad.abs().sum() > 0


def _assert_memory_close(actual, expected):
    for actual_layer, expected_layer in zip(actual._cache, expected._cache):
        for actual_tensor, expected_tensor in zip(actual_layer, expected_layer):
            if expected_tensor is None:
                assert actual_tensor is None
            else:
                torch.testing.assert_close(actual_tensor, expected_tensor)
    for state_name in ("_linear_recurrent", "_linear_conv"):
        actual_states = getattr(actual, state_name)
        expected_states = getattr(expected, state_name)
        assert actual_states.keys() == expected_states.keys()
        for layer_idx in expected_states:
            torch.testing.assert_close(actual_states[layer_idx], expected_states[layer_idx])


@torch.no_grad()
@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
def test_incremental_retrieval_matches_full_history_prefill(beacon_pos):
    torch.manual_seed(7)
    model = _tiny_model(beacon_pos)
    reference = _tiny_model(beacon_pos)
    reference.load_state_dict(model.state_dict())
    history = torch.tensor([[10, 11, 12]])
    regions = []
    memory = model.prefill_and_get_cache(history, regions)

    for document_length in (9, 0, 3):
        generated = torch.tensor([[30, 31]])
        for token in generated.unbind(dim=1):
            model.decode_step(token.unsqueeze(1))
        history = torch.cat([history, generated], dim=1)
        addition = torch.tensor([[32] + list(range(40, 40 + document_length)) + [60, 61]])
        regions.append((history.shape[1] + 1, history.shape[1] + 1 + document_length))
        history = torch.cat([history, addition], dim=1)
        with patch.object(model, "_native_forward", wraps=model._native_forward) as forward:
            actual_memory, actual_logits = model.prefill_and_get_cache(
                addition, [(1, 1 + document_length)], return_last_logits=True,
                reuse_cache=True,
            )
        assert actual_memory is memory
        processed_tokens = sum(call.args[0].shape[1] for call in forward.call_args_list)
        assert processed_tokens == addition.shape[1] + (document_length + 1) // 2
        expected_memory, expected_logits = reference.prefill_and_get_cache(
            history, regions, return_last_logits=True,
        )
        _assert_memory_close(actual_memory, expected_memory)
        torch.testing.assert_close(actual_logits, expected_logits)

    fresh = torch.tensor([[70, 71]])
    actual_memory, actual_logits = model.prefill_and_get_cache(fresh, [], return_last_logits=True)
    expected_memory, expected_logits = reference.prefill_and_get_cache(fresh, [], return_last_logits=True)
    assert actual_memory is not memory
    _assert_memory_close(actual_memory, expected_memory)
    torch.testing.assert_close(actual_logits, expected_logits)


@torch.no_grad()
@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
@pytest.mark.parametrize("stop_kind", ["eos", "text", "limit"])
def test_generation_commits_final_token_before_cache_reuse(beacon_pos, stop_kind):
    torch.manual_seed(7)
    model = _tiny_model(beacon_pos)
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 30]])
    regions = [(2, 7)]
    _, initial_logits = model.prefill_and_get_cache(ids, regions, return_last_logits=True)
    eos_ids = [initial_logits.argmax(dim=-1).item()] if stop_kind == "eos" else []
    tokenizer = SimpleNamespace(
        decode=lambda tokens, **kwargs: "</search>" if len(tokens) >= 2 else "search",
    )
    generated = model.beacon_generate(
        ids, regions=regions, max_new_tokens=3, eos_token_ids=eos_ids,
        stop_texts=["</search>"] if stop_kind == "text" else None, tokenizer=tokenizer,
    )
    assert generated.shape[1] == {"eos": 1, "text": 2, "limit": 3}[stop_kind]
    generated_memory = model._mem
    expected_memory = model.prefill_and_get_cache(torch.cat([ids, generated], dim=1), regions)
    _assert_memory_close(generated_memory, expected_memory)

    model._mem = generated_memory
    addition = torch.tensor([[31, 40, 41, 42, 43, 44, 32]])
    next_generated = model.beacon_generate(
        addition, regions=[(1, 6)], reuse_cache=True, max_new_tokens=2, eos_token_ids=[],
    )
    incremental_memory = model._mem
    assert incremental_memory is generated_memory
    full_history = torch.cat([ids, generated, addition], dim=1)
    offset = ids.shape[1] + generated.shape[1]
    expected_generated = model.beacon_generate(
        full_history, regions=regions + [(offset + 1, offset + 6)],
        max_new_tokens=2, eos_token_ids=[],
    )
    torch.testing.assert_close(next_generated, expected_generated)
    _assert_memory_close(incremental_memory, model._mem)


def test_cache_reuse_requires_initial_prefill():
    model = _tiny_model()
    with pytest.raises(ValueError, match="首次 prefill"):
        model.prefill_and_get_cache(torch.tensor([[10]]), [], reuse_cache=True)
