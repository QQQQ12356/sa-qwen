"""第二阶段 CPU 小模型验证；不启动 Trainer.train 或优化器。"""
import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3ForCausalLM, Qwen3_5ForCausalLM, TrainingArguments

from search_comp.models.beacon_config import BeaconConfig
from search_comp.trainer.beacon_distillation_trainer import (
    BeaconDistillationTrainer, document_dependency_weights,
)
from test_beacon_qwen35_linear_memory import _tiny_model
from test_qwen3_whole_information import tiny_qwen3, whole_config


def test_js_weights_reference_deletion_mapping_and_detach():
    config = BeaconConfig(beacon_output_distill_temperature=2, beacon_loss_chunk_size=1)
    ids = torch.arange(18).unsqueeze(0)
    mask = torch.ones_like(ids)
    rows = torch.tensor([0, 6, 7, 13, 14])
    plus = torch.randn(1, 5, 7, requires_grad=True)
    minus = torch.randn(1, 4, 7, requires_grad=True)
    calls = []

    def teacher(**kwargs):
        calls.append(kwargs)
        assert not torch.is_grad_enabled()
        offset = 0 if len(calls) == 1 else 2
        return SimpleNamespace(logits=minus[:, offset:offset + 2])

    weights = document_dependency_weights(teacher, ids, mask, rows, [(3, 6), (10, 13)], plus, config)
    p = (plus[:, 1:].detach() / 2).softmax(-1)
    q = (minus.detach() / 2).softmax(-1)
    mixture = (p + q) / 2
    js = ((p * (p.log() - mixture.log())).sum(-1)
          + (q * (q.log() - mixture.log())).sum(-1)) / 2
    expected = torch.cat((torch.tensor([[0.1]]), 0.1 + js / math.log(2)), dim=1)
    torch.testing.assert_close(weights, expected)
    assert not weights.requires_grad
    assert calls[0]['input_ids'].tolist() == [list(range(3)) + list(range(6, 18))]
    assert calls[1]['input_ids'].tolist() == [list(range(10)) + list(range(13, 18))]
    assert calls[0]['logits_to_keep'].tolist() == [3, 4]
    assert calls[1]['logits_to_keep'].tolist() == [10, 11]
    assert all(c['attention_mask'].shape == c['input_ids'].shape and not c['use_cache'] for c in calls)


def test_identical_teacher_distributions_and_no_document_positions():
    config = BeaconConfig()
    logits = torch.randn(1, 2, 7)
    teacher = Mock(return_value=SimpleNamespace(logits=logits))
    weights = document_dependency_weights(teacher, torch.arange(12).unsqueeze(0), None,
                                         torch.tensor([6, 7]), [(3, 6)], logits, config)
    torch.testing.assert_close(weights, torch.full((1, 2), config.beacon_output_distill_doc_epsilon))
    teacher.reset_mock()
    weights = document_dependency_weights(teacher, torch.arange(12).unsqueeze(0), None,
                                         torch.tensor([0, 1]), [(3, 6)], logits, config)
    teacher.assert_not_called()
    assert (weights == config.beacon_output_distill_doc_epsilon).all()


@pytest.mark.parametrize('family', ['qwen3', 'qwen35'])
@pytest.mark.parametrize('position', ['append', 'intersect'])
def test_weighted_multiturn_loss_global_normalization_and_gradients(tmp_path, family, position):
    student = tiny_qwen3() if family == 'qwen3' else _tiny_model()
    cfg = whole_config(student, position, False)
    cfg.beacon_output_distill_weight = 0.3
    cfg.beacon_output_distill_strategy = 'all'
    cfg.beacon_output_distill_doc_weighting = True
    cfg.beacon_output_distill_temperature = 2
    cfg.beacon_keep_window = 2
    cfg.beacon_loss_chunk_size = 1
    student.set_beacon_config(cfg)
    student.train()
    teacher_cls = Qwen3ForCausalLM if family == 'qwen3' else Qwen3_5ForCausalLM
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
    cfg.beacon_output_distill_weight = 0
    ce = trainer.compute_loss(student, batch).detach()
    cfg.beacon_output_distill_weight = 0.3
    with patch.object(teacher, 'forward', wraps=teacher.forward) as forward, patch.object(
        student, '_output_distillation_loss', wraps=student._output_distillation_loss,
    ) as kd:
        loss = trainer.compute_loss(student, batch)
        assert forward.call_count == 3
        hidden = torch.cat([call.args[0] for call in kd.call_args_list], dim=1)
        targets = torch.cat([call.args[1] for call in kd.call_args_list], dim=1)
        weights = torch.cat([call.args[2] for call in kd.call_args_list], dim=1)
        assert not targets.requires_grad and not weights.requires_grad
        per_token = F.kl_div(F.log_softmax(student.lm_head(hidden).float() / 2, dim=-1),
                             F.log_softmax(targets.float() / 2, dim=-1),
                             log_target=True, reduction='none').sum(-1)
        reference = 4 * (per_token * weights).sum() / weights.sum()
        torch.testing.assert_close(student._last_loss_parts['kd_loss'], 0.3 * reference.detach())
    torch.testing.assert_close(student._last_loss_parts['ce_loss'], ce)
    torch.testing.assert_close(loss.detach(), sum(student._last_loss_parts.values()))
    loss.backward()
    assert student.model.beacon_embed_tokens.weight.grad.abs().sum() > 0
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert student._mem._distill_weights is None
    assert not any('teacher' in name for name in student.state_dict())
    # 关闭时恢复第一阶段的一次 teacher 前向。
    cfg.beacon_output_distill_doc_weighting = False
    with patch.object(teacher, 'forward', wraps=teacher.forward) as forward:
        trainer.compute_loss(student, batch)
        assert forward.call_count == 1
    # eval 和全 mask 样本均不调用 teacher。
    cfg.beacon_output_distill_doc_weighting = True
    with patch.object(teacher, 'forward', wraps=teacher.forward) as forward:
        trainer.compute_loss(student, dict(batch, labels=torch.full_like(labels, -100)))
        student.eval()
        trainer.compute_loss(student, batch)
        assert forward.call_count == 0
    restored = BeaconConfig.from_dict(cfg.to_dict())
    assert restored.beacon_output_distill_doc_weighting is True
    assert restored.beacon_output_distill_doc_epsilon == 0.1


@pytest.mark.parametrize('values', [
    {'beacon_output_distill_doc_weighting': 'true'},
    {'beacon_output_distill_doc_epsilon': 0},
    {'beacon_output_distill_doc_epsilon': -1},
    {'beacon_output_distill_doc_epsilon': float('nan')},
    {'beacon_output_distill_doc_epsilon': float('inf')},
])
def test_dependency_config_validation(values):
    with pytest.raises(ValueError):
        BeaconConfig(**values)


@pytest.mark.parametrize('checkpoint', [True, False])
def test_weighted_kl_reference_gradients_and_constant_weights(checkpoint):
    model = _tiny_model().train()
    model.beacon_config.beacon_output_distill_temperature = 2
    model.beacon_config.beacon_loss_chunk_size = 2
    model.beacon_config.beacon_checkpoint_loss = checkpoint
    hidden = torch.randn(1, 5, 32, requires_grad=True)
    targets = torch.randn(1, 5, 128, requires_grad=True)
    weights = torch.tensor([[0.1, 0.8, 0.3, 1.0, 0.2]], requires_grad=True)
    actual = model._output_distillation_loss(hidden, targets, weights)
    per_token = F.kl_div(F.log_softmax(model.lm_head(hidden).float() / 2, dim=-1),
                         F.log_softmax(targets.detach() / 2, dim=-1),
                         log_target=True, reduction='none').sum(-1)
    reference = 4 * (per_token * weights.detach()).sum() / weights.detach().sum()
    torch.testing.assert_close(actual, reference)
    actual_gradient = torch.autograd.grad(actual, hidden, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(reference, hidden)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    actual.backward()
    assert targets.grad is None and weights.grad is None
    plain = model._output_distillation_loss(hidden, targets)
    constant = model._output_distillation_loss(hidden, targets, torch.full((1, 5), 0.1))
    torch.testing.assert_close(plain, constant)
