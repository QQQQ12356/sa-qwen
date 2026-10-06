"""每个 log_step 打印复合损失的各个组成分量。

覆盖：

- :meth:`BeaconQwen3_5ForCausalLM._beacon_forward` 把复合损失拆成
  ``ce_loss`` / ``cont_loss`` / ``readout_loss`` / ``kd_loss`` 记录到 ``_last_loss_parts``，
  四项之和严格等于返回的 ``loss``；未激活的项记 0。
- :class:`LossComponentTrainer` 按 log 窗口对 micro-batch 取均值写回 ``logs``，
  打印后清零，eval 步不计入。
- 端到端：tiny 模型跑一步 ``train()``，``state.log_history`` 里带上各分量。
"""

import os
from types import SimpleNamespace

import pytest
import torch

from search_comp.trainer.loss_component_trainer import LossComponentTrainer
from test_beacon_continuation import IDS
from test_beacon_qwen35_linear_memory import _tiny_model


DEVICE = os.environ.get("BEACON_TEST_DEVICE", "cpu")

#: 前 8 个 token 不监督，监督当前窗口的后 4 个；压缩区跨两个窗口（4 token 一个窗）。
LABELS = IDS.clone()
LABELS[:, :8] = -100

#: 跨两个窗口的压缩区，触发续写监督。
REGIONS = [(2, 10)]


def _make_model(cont_weight: float = 0.5, cont_tokens: int = 2, question_memory: bool = False):
    torch.manual_seed(7)
    model = _tiny_model(question_memory_v1=question_memory)
    model.beacon_config.beacon_continuation_loss_weight = cont_weight
    model.beacon_config.beacon_continuation_tokens = cont_tokens
    return model.eval()


def _inputs():
    return {"input_ids": IDS, "labels": LABELS, "regions": REGIONS}


# ----------------------------------------------------------------------
# 模型侧：复合损失分量
# ----------------------------------------------------------------------
def test_forward_records_all_four_loss_parts():
    model = _make_model()
    output = model(**_inputs())

    assert set(model._last_loss_parts) == {"ce_loss", "cont_loss", "readout_loss", "kd_loss"}
    for value in model._last_loss_parts.values():
        assert torch.isfinite(value)


def test_loss_parts_sum_back_to_total_loss():
    model = _make_model()
    output = model(**_inputs())

    parts = model._last_loss_parts
    total = sum(parts.values())
    assert torch.allclose(total, output.loss, atol=1e-6)


def test_continuation_part_is_the_normalized_share():
    """两个分量都按同一份额缩放：ce 分量 = (w_ce/Σw)·ce，cont 分量 = (w_cont/Σw)·cont。

    w_ce=1.0、w_cont=0.5 → Σw=1.5，份额恒为 1/1.5；两者之和等于总损失。
    """
    model = _make_model(cont_weight=0.5)
    output = model(**_inputs())
    parts = model._last_loss_parts

    ce_raw = _make_model(cont_weight=0.0)(**_inputs()).loss          # 关闭续写 → 原始 ce
    cont_raw = 2 * _make_model(cont_weight=1.0)(**_inputs()).loss - ce_raw
    ce_share, cont_share = 1.0 / 1.5, 0.5 / 1.5     # w_ce/Σw 与 w_cont/Σw，两者不同

    assert parts["cont_loss"].item() > 0
    assert torch.allclose(parts["ce_loss"], ce_share * ce_raw, atol=1e-6)
    assert torch.allclose(parts["cont_loss"], cont_share * cont_raw, atol=1e-6)
    assert torch.allclose(parts["ce_loss"] + parts["cont_loss"], output.loss, atol=1e-6)


def test_inactive_parts_are_exactly_zero():
    model = _make_model(cont_weight=0.0, question_memory=False)
    model(**_inputs())

    parts = model._last_loss_parts
    assert parts["cont_loss"].item() == 0.0
    assert parts["readout_loss"].item() == 0.0
    assert parts["ce_loss"].item() > 0


def test_parts_are_detached_from_the_graph():
    model = _make_model()
    model(**_inputs())

    for value in model._last_loss_parts.values():
        assert not value.requires_grad


def test_forward_overwrites_parts_between_calls():
    model = _make_model()
    model(**_inputs())
    first = float(model._last_loss_parts["ce_loss"])

    # 监督位置全 -100：主 CE 为 0，分量应被新一次前向整体覆盖而不是累加。
    blank = LABELS.clone()
    blank[:, :8] = -100
    blank[:, 8:] = -100
    model(input_ids=IDS, labels=blank, regions=REGIONS)

    assert float(model._last_loss_parts["ce_loss"]) == 0.0
    assert first != 0.0


# ----------------------------------------------------------------------
# LoRA 包装（beacon_trainer 的 use_lora 路径）
# ----------------------------------------------------------------------
def _lora_wrapped(model):
    """用真实 PeftModel 包装，复现 ``beacon_trainer`` 的 use_lora 路径。

    真实结构与直觉不同：``LoraModel.base_model`` 是文本主干 ``Qwen3_5TextModel``，
    被包装的 CausalLM 在 ``LoraModel.model``。分量是靠属性代理读到的，这里钉住它。
    """
    from peft import LoraConfig, get_peft_model

    return get_peft_model(
        model,
        LoraConfig(r=4, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"),
    )


def test_peft_wrapper_proxies_to_the_same_parts_dict():
    model = _make_model()
    wrapped = _lora_wrapped(model)
    model(**_inputs())

    assert wrapped._last_loss_parts is model._last_loss_parts


def test_trainer_reads_parts_through_lora_wrapper(tmp_path):
    """use_lora=true 的主用例：分量必须出现在日志里。"""
    from transformers import TrainingArguments

    model = _lora_wrapped(_make_model()).train()
    trainer = LossComponentTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[],
            disable_tqdm=True, remove_unused_columns=False,
        ),
    )
    trainer.compute_loss(model, _inputs())

    logs: dict = {"loss": 1.0}
    trainer.log(logs)
    assert logs["ce_loss"] > 0
    assert {"ce_loss", "cont_loss", "readout_loss"} <= set(logs)


class _PlainLM(torch.nn.Module):
    """没有损失分量的普通模型（模拟原生 / 纯文本训练器）。"""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, input_ids=None, labels=None, **kwargs):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(loss=self.linear.weight.sum() * 0.0, logits=None)


def test_trainer_ignores_models_without_parts(tmp_path):
    from transformers import TrainingArguments

    model = _PlainLM().train()
    trainer = LossComponentTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, report_to=[], disable_tqdm=True,
        ),
    )
    trainer.compute_loss(model, _inputs())

    logs: dict = {"loss": 1.0}
    trainer.log(logs)
    assert not {"ce_loss", "cont_loss", "readout_loss"} & set(logs)


# ----------------------------------------------------------------------
# Trainer 侧：按 log 窗口平均
# ----------------------------------------------------------------------
@pytest.fixture
def trainer(tmp_path):
    from transformers import TrainingArguments

    model = _make_model()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        disable_tqdm=True,
        logging_steps=1,
        remove_unused_columns=False,
    )
    return LossComponentTrainer(model=model, args=args)


def test_trainer_averages_parts_across_micro_batches(trainer):
    model = trainer.model.train()
    for _ in range(3):
        trainer.compute_loss(model, _inputs())

    expected = float(model._last_loss_parts["ce_loss"])
    logs: dict = {"loss": 1.0}
    trainer.log(logs)

    assert logs["ce_loss"] == pytest.approx(expected)
    assert logs["cont_loss"] == pytest.approx(
        float(model._last_loss_parts["cont_loss"])
    )


def test_trainer_resets_window_after_logging(trainer):
    model = trainer.model.train()
    trainer.compute_loss(model, _inputs())
    trainer.log({"loss": 1.0})

    logs: dict = {"loss": 1.0}
    trainer.log(logs)
    assert "ce_loss" not in logs


def test_trainer_ignores_eval_steps(trainer):
    model = trainer.model.eval()
    trainer.compute_loss(model, _inputs())

    logs: dict = {"eval_loss": 1.0}
    trainer.log(logs)
    assert "ce_loss" not in logs

    # 即便窗口里还有训练分量，eval 行也不该被注入。
    model.train()
    trainer.compute_loss(model, _inputs())
    eval_logs: dict = {"eval_loss": 1.0}
    trainer.log(eval_logs)
    assert "ce_loss" not in eval_logs


def test_trainer_groups_micro_batches_by_scale(tmp_path):
    """不同量级的两批：均值应是两批真值的算术平均，而不是最后一批。"""
    from transformers import TrainingArguments

    model = _make_model().train()
    args = TrainingArguments(
        output_dir=str(tmp_path), use_cpu=True, report_to=[],
        disable_tqdm=True, remove_unused_columns=False,
    )
    trainer = LossComponentTrainer(model=model, args=args)

    trainer.compute_loss(model, _inputs())
    small = float(model._last_loss_parts["ce_loss"])
    trainer.compute_loss(model, {"input_ids": IDS, "labels": IDS.clone(), "regions": REGIONS})
    large = float(model._last_loss_parts["ce_loss"])

    logs: dict = {"loss": 1.0}
    trainer.log(logs)
    assert logs["ce_loss"] == pytest.approx((small + large) / 2)


# ----------------------------------------------------------------------
# 端到端：日志行里带上分量
# ----------------------------------------------------------------------
class _FixedDataset(torch.utils.data.Dataset):
    """固定样本，bs=1；只喂模型实际需要的键。"""

    def __init__(self, size: int = 4):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        return {"input_ids": IDS.clone(), "labels": LABELS.clone(), "regions": REGIONS}


def _identity_collator(batch):
    return batch[0]


def test_train_logs_loss_parts_end_to_end(tmp_path):
    from transformers import TrainingArguments

    model = _make_model().train()
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to=[],
        disable_tqdm=True,
        logging_steps=1,
        max_steps=2,
        # 与真实配置一致：logging_steps 跨多个 micro-batch 取均值。
        gradient_accumulation_steps=2,
        save_strategy="no",
        remove_unused_columns=False,
        learning_rate=1e-3,
    )
    trainer = LossComponentTrainer(
        model=model, args=args, train_dataset=_FixedDataset(),
        data_collator=_identity_collator,
    )
    trainer.train()

    logged = [record for record in trainer.state.log_history if "loss" in record]
    assert logged, "训练没有产生任何 loss 日志"
    assert {"ce_loss", "cont_loss", "readout_loss"} <= set(logged[-1])
    assert logged[-1]["ce_loss"] == pytest.approx(
        logged[-1]["loss"] - logged[-1]["cont_loss"] - logged[-1]["readout_loss"],
        abs=1e-4,
    )
