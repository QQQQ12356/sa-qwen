import copy
import os

import pytest
import torch

from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen35 import BeaconQwen3_5ForCausalLM, BeaconLinearStateWriter
from test_beacon_qwen35_linear_memory import _tiny_model


DEVICE = os.environ.get("BEACON_TEST_DEVICE", "cpu")


def make_model(position="append"):
    torch.manual_seed(7)
    return _tiny_model(position, question_memory_v1=True).to(DEVICE)


def inputs():
    return (
        torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 30, 31]], device=DEVICE),
        torch.tensor([[10, 11]], device=DEVICE),
    )


@pytest.mark.parametrize("position", ["append", "intersect"])
def test_v1_backward_and_single_device(position):
    model = make_model(position).train()
    ids, question = inputs()
    labels = ids.clone()
    labels[:, :8] = -100
    output = model(input_ids=ids, labels=labels, regions=[(2, 8)], question_input_ids=question)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    writer = model.model.beacon_linear_writers["0"]
    for parameter in (writer.up.weight, writer.question_up.weight, writer.importance.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert model._mem._readout_losses == []
    if DEVICE == "cuda":
        assert torch.cuda.device_count() == 1
        assert output.loss.is_cuda


def test_disabled_has_original_writer_and_ignores_question():
    torch.manual_seed(7)
    model = _tiny_model().to(DEVICE)
    assert type(model.model.beacon_linear_writers["0"]) is BeaconLinearStateWriter
    assert not any("question" in name or "importance" in name for name in model.state_dict())
    ids, question = inputs()
    with torch.no_grad():
        first = model(input_ids=ids, regions=[(2, 8)]).logits
        second = model(input_ids=ids, regions=[(2, 8)], question_input_ids=question).logits
    torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("position", ["append", "intersect"])
def test_incremental_matches_full_and_resets_question(position):
    model = make_model(position)
    ids, question = inputs()
    with torch.no_grad():
        full, full_logits = model.prefill_and_get_cache(ids, [(2, 8)], True, question_input_ids=question)
        model.prefill_and_get_cache(ids[:, :2], [], question_input_ids=question)
        incremental, logits = model.prefill_and_get_cache(ids[:, 2:], [(0, 6)], True, reuse_cache=True)
        torch.testing.assert_close(logits, full_logits)
        torch.testing.assert_close(incremental._linear_recurrent[0], full._linear_recurrent[0])
        for expected, actual in zip(full._cache[1], incremental._cache[1]):
            torch.testing.assert_close(actual, expected)
        other_question = question + 1
        with pytest.raises(ValueError, match="切换"):
            model.prefill_and_get_cache(ids[:, :2], [], reuse_cache=True, question_input_ids=other_question)
        fresh = model.prefill_and_get_cache(ids[:, :2], [], question_input_ids=other_question)
        torch.testing.assert_close(fresh._question, model.model.embed_tokens(other_question))


def test_question_is_required_bounded_and_affects_writer():
    model = make_model()
    ids, question = inputs()
    with pytest.raises(ValueError, match="question_input_ids"):
        model.prefill_and_get_cache(ids, [(2, 8)])
    model.beacon_config.beacon_question_max_tokens = 1
    first = model.prefill_and_get_cache(ids[:, :8], [(2, 8)], question_input_ids=question)
    assert first._question.shape[1] == 1
    second = model.prefill_and_get_cache(ids[:, :8], [(2, 8)], question_input_ids=question + 1)
    assert not torch.equal(first._linear_recurrent[0], second._linear_recurrent[0])
    assert 0 not in second._linear_conv


def test_teacher_is_detached_and_distillation_is_training_only():
    model = make_model().train()
    writer = model.model.beacon_linear_writers["0"]
    student = torch.randn(1, 2, 8, 8, device=DEVICE, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    question = torch.randn(1, 2, 32, device=DEVICE, requires_grad=True)
    loss = writer.readout_loss(student, teacher, question)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None
    assert question.grad is None
    ids, question_ids = inputs()
    baseline = copy.deepcopy(model)
    baseline.beacon_config.beacon_readout_distill_weight = 0
    args = dict(input_ids=ids, labels=ids, regions=[(2, 8)], question_input_ids=question_ids)
    assert model(**args).loss > baseline(**args).loss
    model.eval()
    baseline.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**args).loss, baseline(**args).loss)


@pytest.mark.parametrize("position", ["append", "intersect"])
def test_reader_state_cannot_bypass_beacons(position, monkeypatch):
    model = make_model(position)
    ids, question = inputs()
    with torch.no_grad():
        original = model.prefill_and_get_cache(ids[:, :8], [(2, 8)], question_input_ids=question)
    reader = model._linear_attention_forward

    def poison(*args):
        output, state, conv = reader(*args)
        return output, torch.full_like(state, 999), torch.full_like(conv, 999)

    with torch.no_grad():
        model.prefill_and_get_cache(ids[:, :2], [], question_input_ids=question)
    monkeypatch.setattr(model, "_linear_attention_forward", poison)
    with torch.no_grad():
        poisoned = model.prefill_and_get_cache(ids[:, 2:8], [(0, 6)], reuse_cache=True)
    torch.testing.assert_close(original._linear_recurrent[0], poisoned._linear_recurrent[0])
    assert 0 not in poisoned._linear_conv


def test_checkpoint_roundtrip_and_structure_guard(tmp_path):
    model = make_model()
    ids, question = inputs()
    model.save_pretrained(tmp_path)
    restored = BeaconQwen3_5ForCausalLM.from_pretrained(tmp_path).to(DEVICE).eval()
    assert restored.beacon_config.beacon_question_memory_v1
    with torch.no_grad():
        expected = model(input_ids=ids, regions=[(2, 8)], question_input_ids=question).logits
        actual = restored(input_ids=ids, regions=[(2, 8)], question_input_ids=question).logits
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="构造"):
        model.set_beacon_config(BeaconConfig(beacon_linear_writer_rank=8))


@pytest.mark.skipif(DEVICE != "cuda", reason="single-GPU BF16 smoke")
def test_bf16_single_gpu_optimizer_step():
    model = make_model().to(torch.bfloat16).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids, question = inputs()
    for _ in range(2):
        optimizer.zero_grad()
        output = model(input_ids=ids, labels=ids, regions=[(2, 8)], question_input_ids=question)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        optimizer.step()
    assert model._mem._linear_recurrent[0].dtype == torch.float32


@pytest.mark.parametrize("settings", [
    {"beacon_question_max_tokens": 0},
    {"beacon_readout_distill_weight": -1},
    {"beacon_readout_distill_weight": float("nan")},
])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        BeaconConfig(**settings)


def test_decay_budget_is_independent_of_beacon_count():
    writer = make_model().model.beacon_linear_writers["0"]
    with torch.no_grad():
        writer.up.weight.zero_()
        writer.up.bias.zero_()
        writer.up.bias[:16].reshape(2, 8)[:, 0] = 1
        writer.up.bias[16:32] = 1
        writer.up.bias[32:34] = 4
        writer.up.bias[34:] = -100
        writer.importance.weight.zero_()
        writer.importance.bias.zero_()
        previous = torch.zeros(1, 2, 8, 8, device=DEVICE)
        previous[:, :, 1] = 1
        beacon = torch.randn(1, 1, 32, device=DEVICE)
        question = torch.randn(1, 2, 32, device=DEVICE)
        once = writer(beacon, previous, question)
        repeated = writer(beacon.expand(-1, 8, -1), previous, question)
    torch.testing.assert_close(once, repeated)
    assert once[:, :, 1].mean() < 1


def test_zero_novelty_leaves_previous_state_unchanged():
    writer = make_model().model.beacon_linear_writers["0"]
    with torch.no_grad():
        writer.up.weight.zero_()
        writer.up.bias.zero_()
        writer.up.bias[:16].reshape(2, 8)[:, 0] = 1
        writer.up.bias[16:32] = 1
        previous = torch.ones(1, 2, 8, 8, device=DEVICE)
        actual = writer(torch.randn(1, 4, 32, device=DEVICE), previous, torch.randn(1, 2, 32, device=DEVICE))
    torch.testing.assert_close(actual, previous, rtol=0, atol=0)


@pytest.mark.skipif(not os.environ.get("BEACON_FULL_MODEL"), reason="opt-in pretrained single-GPU smoke")
def test_pretrained_single_gpu_smoke():
    import json

    from search_comp.milestones.qwen35_text import load_text_tokenizer
    from search_comp.models.beacon_qwen35 import load_beacon_qwen3_5

    assert DEVICE == "cuda" and torch.cuda.device_count() == 1
    model_path = os.environ["BEACON_FULL_MODEL"]
    config = BeaconConfig(
        beacon_window=32, beacon_stride=32, beacon_ratio=8,
        beacon_question_memory_v1=True, beacon_linear_writer_rank=16,
    )
    model = load_beacon_qwen3_5(model_path, beacon_config=config).train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("beacon" in name)
    tokenizer = load_text_tokenizer(model_path)
    question_ids = tokenizer("Where was Ada Lovelace born?", add_special_tokens=False).input_ids
    document_ids = tokenizer("Ada Lovelace was born in London, England, in 1815.", add_special_tokens=False).input_ids
    answer_ids = tokenizer("The answer is London.", add_special_tokens=False).input_ids
    ids = torch.tensor([question_ids + document_ids + answer_ids], device="cuda")
    question = torch.tensor([question_ids], device="cuda")
    regions = [(len(question_ids), len(question_ids) + len(document_ids))]
    labels = ids.clone()
    labels[:, :regions[0][1]] = -100
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    loss = model(input_ids=ids, labels=labels, regions=regions, question_input_ids=question).loss
    assert torch.isfinite(loss)
    loss.backward()
    for writer in model.model.beacon_linear_writers.values():
        assert writer.question_up.weight.grad is not None
        assert torch.isfinite(writer.question_up.weight.grad).all()
        assert writer.question_up.weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    generated = model.beacon_generate(ids[:, :regions[0][1]], regions=regions, question_input_ids=question, max_new_tokens=2)
    assert generated.shape[1] > 0
    print(json.dumps({
        "model": model_path, "gpu": torch.cuda.get_device_name(0),
        "visible_gpus": torch.cuda.device_count(), "dtype": str(next(model.parameters()).dtype),
        "loss": loss.item(), "generated_tokens": generated.shape[1],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }))
