"""CPU 小模型回归：只验证监督、状态和梯度，不执行训练/优化器更新。"""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3_5ForCausalLM, TrainingArguments

from test_beacon_qwen35_linear_memory import _tiny_model, _assert_memory_close
from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen35 import load_beacon_qwen3_5, _Qwen3BeaconMemory
from search_comp.trainer.beacon_distillation_trainer import (
    BeaconDistillationTrainer, select_distillation_positions,
)


def _models(tmp_path, position="append"):
    torch.manual_seed(7)
    student = _tiny_model(position).train()
    student.beacon_config.beacon_output_distill_weight = 0.1
    student.beacon_config.beacon_output_distill_tokens = 2
    student.beacon_config.beacon_output_distill_strategy = "prefix"
    teacher = Qwen3_5ForCausalLM(student.config).eval()
    trainer = BeaconDistillationTrainer(
        model=student, distillation_teacher=teacher,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[], gradient_accumulation_steps=4,
        ),
    )
    return student, teacher, trainer


def _batch():
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 30, 31, 32]])
    labels = torch.full_like(ids, -100)
    labels[:, 9:] = ids[:, 9:]
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, regions=[(2, 8)])


@pytest.mark.parametrize("source", ["eos", "bos"])
def test_base_loader_initializes_beacons_after_loading_pretrained_weights(source):
    config = _tiny_model().config
    config.architectures = ["Qwen3_5ForCausalLM"]
    config.bos_token_id = 1
    base = Qwen3_5ForCausalLM(config)
    with torch.no_grad():
        base.model.embed_tokens.weight.fill_(0.25)
        base.model.embed_tokens.weight[1].fill_(0.75)
        for name in ("q", "k", "v"):
            getattr(base.model.layers[1].self_attn, f"{name}_proj").weight.fill_(0.5)
    beacon = BeaconConfig.from_model_config(config)
    beacon.beacon_embed_init = source
    with patch("transformers.AutoConfig.from_pretrained", return_value=config), patch(
        "search_comp.milestones.qwen35_text.load_text_causal_model", return_value=base,
    ):
        loaded = load_beacon_qwen3_5("mock", beacon, torch_dtype=torch.float32, device_map="cpu")
    for name in ("q", "k", "v"):
        attention = loaded.model.layers[1].self_attn
        torch.testing.assert_close(
            getattr(attention, f"beacon_{name}_proj").weight, getattr(attention, f"{name}_proj").weight,
        )
    token = config.eos_token_id if source == "eos" else config.bos_token_id
    torch.testing.assert_close(loaded.model.beacon_embed_tokens.weight[0], base.model.embed_tokens.weight[token])


def test_loading_beacon_checkpoint_preserves_learned_initialization(tmp_path):
    model = _tiny_model()
    with torch.no_grad():
        model.model.beacon_embed_tokens.weight.fill_(0.7)
        model.model.layers[1].self_attn.beacon_q_proj.weight.fill_(0.6)
    model.save_pretrained(tmp_path)
    restored = load_beacon_qwen3_5(tmp_path, torch_dtype=torch.float32, device_map="cpu")
    torch.testing.assert_close(restored.model.beacon_embed_tokens.weight, model.model.beacon_embed_tokens.weight)
    torch.testing.assert_close(
        restored.model.layers[1].self_attn.beacon_q_proj.weight, model.model.layers[1].self_attn.beacon_q_proj.weight,
    )


@pytest.mark.parametrize("length,expected", [(6, [25]), (5, [])])
def test_window_continuation_does_not_predict_outside_document(length, expected):
    model = _tiny_model()
    model.beacon_config.beacon_continuation_loss_weight = 0.1
    model.beacon_config.beacon_continuation_tokens = 4
    ids = torch.tensor([[10, 11] + list(range(20, 20 + length)) + [30, 31]])
    memory = _Qwen3BeaconMemory(model, model.beacon_config)
    memory.prepare(ids, torch.full_like(ids, -100), [(2, 2 + length)])
    targets = []
    while not memory.finish:
        memory.step()
        if memory._step_cont_labels is not None:
            targets.extend(memory._step_cont_labels[0].tolist())
    assert targets == expected


def test_readout_selection_skips_adjacent_documents_and_caps_each_keep_segment():
    labels = torch.arange(20).unsqueeze(0)
    labels[:, 8:10] = -100
    positions = select_distillation_positions(labels, [(2, 5), (5, 8), (13, 16)], 2)
    assert positions.tolist() == [9, 10, 16, 17]
    assert select_distillation_positions(labels, [], 2).numel() == 0


def test_output_kl_matches_reference_and_detaches_teacher():
    model = _tiny_model().train()
    model.beacon_config.beacon_output_distill_temperature = 2.0
    model.beacon_config.beacon_loss_chunk_size = 2
    hidden = torch.randn(1, 5, 32, requires_grad=True)
    targets = torch.randn(1, 5, 128, requires_grad=True)
    actual = model._output_distillation_loss(hidden, targets)
    expected = F.kl_div(
        F.log_softmax(model.lm_head(hidden).float() / 2, dim=-1),
        F.log_softmax(targets.detach() / 2, dim=-1), log_target=True, reduction="sum",
    ) * 4 / 5
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert hidden.grad.abs().sum() > 0
    assert targets.grad is None


@pytest.mark.parametrize("position", ["append", "intersect"])
def test_post_commit_kl_reaches_all_writers_without_changing_memory(tmp_path, monkeypatch, position):
    student, teacher, trainer = _models(tmp_path, position)
    reference = _tiny_model(position).train()
    reference.load_state_dict(student.state_dict())
    batch = _batch()
    reference_output = reference(**batch)
    reference_memory = reference._mem
    states = []

    def record_state(module, inputs, output):
        output.retain_grad()
        states.append(output)

    handle = student.model.beacon_linear_writers["0"].register_forward_hook(record_state)
    teacher_inputs = []
    teacher_handle = teacher.register_forward_pre_hook(
        lambda module, args, kwargs: teacher_inputs.append(kwargs), with_kwargs=True,
    )
    # 去掉主 CE 梯度，以确定当前 Writer 的梯度确实来自提交后的 KL。
    monkeypatch.setattr(student, "_sparse_window_loss", lambda hidden, labels, valid, weights=None: (
        hidden.sum(dim=(1, 2)) * 0, valid.float(),
    ))
    try:
        loss = trainer.compute_loss(student, batch)
        memory = student._mem
        loss.backward()
    finally:
        handle.remove()
        teacher_handle.remove()
    assert len(states) == 2
    assert all(state.grad is not None and state.grad.abs().sum() > 0 for state in states)
    assert student.model.layers[1].self_attn.beacon_k_proj.weight.grad.abs().sum() > 0
    assert loss.item() > 0
    assert student._last_loss_parts["kd_loss"].item() == pytest.approx(loss.item())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    assert teacher_inputs[0]["logits_to_keep"].tolist() == [8, 9]
    assert teacher_inputs[0]["use_cache"] is False
    assert teacher_inputs[0]["input_ids"].shape == batch["input_ids"].shape
    _assert_memory_close(memory, reference_memory)
    assert memory._teacher_logits is None
    assert "teacher_logits" not in batch
    assert not any("teacher" in name for name in student.state_dict())
    assert torch.isfinite(reference_output.loss)


def test_teacher_not_called_without_readout_or_during_eval(tmp_path):
    student, teacher, trainer = _models(tmp_path)
    batch = _batch()
    with patch.object(teacher, "forward", wraps=teacher.forward) as forward:
        no_documents = dict(batch, regions=[])
        trainer.compute_loss(student, no_documents)
        assert forward.call_count == 0
        trainer.compute_loss(student, batch)
        assert forward.call_count == 1
        student.eval()
        trainer.compute_loss(student, batch)
        assert forward.call_count == 1
        assert student._last_loss_parts["kd_loss"].item() == 0


def test_trainer_uses_sample_mean_gradient_accumulation(tmp_path):
    student, _, trainer = _models(tmp_path)
    assert student.accepts_loss_kwargs is False
    assert trainer.model_accepts_loss_kwargs is False
    assert trainer.accelerator.gradient_accumulation_steps == 1
    assert trainer._get_num_items_in_batch([_batch()] * 4, torch.device("cpu")) is None
    assert trainer.optimizer is None


def test_lora_distillation_preserves_ce_and_inference_checkpoint(tmp_path):
    from peft import LoraConfig, get_peft_model

    student, teacher, _ = _models(tmp_path)
    wrapped = get_peft_model(student, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]))
    for name, parameter in wrapped.named_parameters():
        if "beacon" in name:
            parameter.requires_grad = True
    trainer = BeaconDistillationTrainer(
        model=wrapped, distillation_teacher=teacher,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    assert trainer.model_accepts_loss_kwargs is False
    loss = trainer.compute_loss(wrapped, _batch())
    parts = wrapped._last_loss_parts
    assert parts["ce_loss"] > 0 and parts["kd_loss"] > 0
    torch.testing.assert_close(loss.detach(), sum(parts.values()))
    student.beacon_config.beacon_output_distill_weight = 0
    plain = trainer.compute_loss(wrapped, _batch())
    torch.testing.assert_close(plain.detach(), parts["ce_loss"])
    student.beacon_config.beacon_output_distill_weight = 0.1
    student.set_beacon_config(student.beacon_config)
    # 蒸馏配置可保存；推理重新加载不需要 teacher。
    merged = wrapped.merge_and_unload().eval()
    merged.save_pretrained(tmp_path / "student")
    restored = load_beacon_qwen3_5(tmp_path / "student", torch_dtype=torch.float32, device_map="cpu")
    assert restored.beacon_config.beacon_output_distill_weight == 0.1
    ids = _batch()["input_ids"]
    with torch.no_grad():
        _, actual = restored.prefill_and_get_cache(ids, [(2, 8)], return_last_logits=True)
        _, expected = merged.prefill_and_get_cache(ids, [(2, 8)], return_last_logits=True)
    torch.testing.assert_close(actual, expected)


def test_distillation_rejects_raw_document_readouts(tmp_path):
    student, _, _ = _models(tmp_path)
    with pytest.raises(ValueError, match="压缩正文"):
        student(**_batch(), distill_positions=torch.tensor([3]), teacher_logits=torch.zeros(1, 1, 128))


@pytest.mark.parametrize("field,value", [
    ("beacon_output_distill_weight", -1),
    ("beacon_output_distill_temperature", 0),
    ("beacon_output_distill_tokens", 0),
])
def test_distillation_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        BeaconConfig(**{field: value})


def test_all_positions_cover_pre_search_and_every_unmasked_target():
    labels = torch.tensor([[-100, 11, 12, -100, -100, -100, 16, 17, 18, 19]])
    rows = select_distillation_positions(labels, [(3, 5)], strategy="all")
    assert rows.tolist() == [0, 1, 5, 6, 7, 8]
    assert select_distillation_positions(labels, [], strategy="all").tolist() == [0, 1, 5, 6, 7, 8]
    assert select_distillation_positions(torch.full_like(labels, -100), [], strategy="all").numel() == 0


def test_stratified_budget_and_disabled_segments():
    from search_comp.loss_segments import LossSegmentConfig, SEGMENT_IDS
    labels = torch.arange(30).unsqueeze(0)
    kinds = torch.full_like(labels, SEGMENT_IDS["think_content"])
    kinds[:, 20:23] = SEGMENT_IDS["search_content"]
    kinds[:, 23:] = SEGMENT_IDS["answer_content"]
    rows = select_distillation_positions(labels, [(2, 5)], 8, "stratified", kinds)
    assert rows.numel() == 8
    assert (rows[1:] > rows[:-1]).all()
    assert {SEGMENT_IDS[n] for n in ("think_content", "search_content", "answer_content")} <= set(kinds[0, rows + 1].tolist())
    weights = LossSegmentConfig.from_dict({"search_content": False})
    rows = select_distillation_positions(labels, [(2, 5)], strategy="all",
                                         loss_segment_ids=kinds, loss_segments=weights)
    assert not (kinds[0, rows + 1] == SEGMENT_IDS["search_content"]).any()


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_all_kl_aligns_full_multiturn_trajectory_and_preserves_ce(tmp_path, family):
    from transformers import Qwen3ForCausalLM
    from test_qwen3_whole_information import tiny_qwen3, whole_config
    student = tiny_qwen3() if family == "qwen3" else _tiny_model()
    cfg = whole_config(student, "append", False)
    cfg.beacon_output_distill_weight = 0.1
    cfg.beacon_output_distill_strategy = "all"
    cfg.beacon_keep_window = 2  # 也验证 keep 段跨窗口时的位置映射
    student.set_beacon_config(cfg)
    student.train()
    teacher_cls = Qwen3ForCausalLM if family == "qwen3" else Qwen3_5ForCausalLM
    teacher = teacher_cls(student.config)
    trainer = BeaconDistillationTrainer(model=student, distillation_teacher=teacher,
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]))
    ids = torch.arange(10, 30).unsqueeze(0)
    labels = ids.clone()
    labels[:, 0] = -100
    labels[:, 3:8] = -100
    labels[:, 12:17] = -100
    batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                 regions=[(3, 7), (12, 16)])
    student.beacon_config.beacon_output_distill_weight = 0
    ce = trainer.compute_loss(student, batch).detach()
    student.beacon_config.beacon_output_distill_weight = 0.1
    with patch.object(teacher, "forward", wraps=teacher.forward) as forward, patch.object(
        student, "_output_distillation_loss", wraps=student._output_distillation_loss,
    ) as kd:
        loss = trainer.compute_loss(student, batch)
        assert forward.call_count == 1
        kwargs = forward.call_args.kwargs
        assert torch.equal(kwargs["input_ids"], ids)
        expected_rows = (labels[0, 1:] != -100).nonzero(as_tuple=True)[0]
        assert torch.equal(kwargs["logits_to_keep"], expected_rows)
        assert sum(call.args[0].shape[1] for call in kd.call_args_list) == expected_rows.numel()
        assert not kwargs["use_cache"]
    torch.testing.assert_close(student._last_loss_parts["ce_loss"], ce)
    torch.testing.assert_close(loss.detach(), sum(student._last_loss_parts.values()))
    loss.backward()
    assert student.model.beacon_embed_tokens.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    # 没有检索文档也覆盖全部 CE token；全 mask 样本无需 teacher。
    with patch.object(teacher, "forward", wraps=teacher.forward) as forward:
        trainer.compute_loss(student, dict(batch, regions=[]))
        assert forward.call_count == 1
        trainer.compute_loss(student, dict(batch, labels=torch.full_like(labels, -100)))
        assert forward.call_count == 1


@pytest.mark.parametrize("family", ["qwen3", "qwen35"])
def test_local_plain_checkpoint_initializes_student_and_teacher(tmp_path, family):
    from transformers import Qwen3ForCausalLM
    from test_qwen3_whole_information import tiny_qwen3
    from search_comp.models.model_loader import load_beacon_model
    from search_comp.models.plain_qwen3 import load_sft_model
    template = tiny_qwen3() if family == "qwen3" else _tiny_model()
    native_cls = Qwen3ForCausalLM if family == "qwen3" else Qwen3_5ForCausalLM
    native = native_cls(template.config)
    native.save_pretrained(tmp_path / "plain")
    teacher = load_sft_model(tmp_path / "plain", torch_dtype=torch.float32, device_map="cpu")
    beacon = load_beacon_model(tmp_path / "plain", beacon_config=template.beacon_config,
                               torch_dtype=torch.float32, device_map="cpu")
    torch.testing.assert_close(teacher.model.embed_tokens.weight, native.model.embed_tokens.weight)
    torch.testing.assert_close(beacon.model.embed_tokens.weight, native.model.embed_tokens.weight)
    assert all(p.requires_grad for p in teacher.parameters())  # plain 路径支持全参数训练
    assert type(beacon) is type(template)
