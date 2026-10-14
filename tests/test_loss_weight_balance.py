"""主 CE 与续写损失的权重平衡。

两项均实际参与时按权重和归一：

    L = (w_ce·ce + w_cont·cont) / (w_ce + w_cont)   [+ w_readout·readout]

只有一项实际参与时直接乘其权重；两项都出现时按份额加权平均，
同比例缩放两个权重不改变 L。``readout`` 是独立的蒸馏加项，不进分母。

各断言里的 ``ce`` / ``cont`` 都由「关掉另一半」的参考前向反解出来，不复用实现内部的值。
"""

import pytest
import torch

from search_comp.models.beacon_config import BeaconConfig
from test_beacon_qwen35_linear_memory import _tiny_model

#: 12 个 token；compress 区 (2,10) 跨两个窗，保证续写项一定出现。
IDS = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31]])
REGIONS = [(2, 10)]


def _make_model(w_ce=1.0, w_cont=0.0, w_readout=0.0, question_memory=False, train=False):
    torch.manual_seed(7)
    model = _tiny_model("intersect", question_memory_v1=question_memory)
    config = model.beacon_config
    config.beacon_window = 4
    config.beacon_stride = 4
    config.beacon_ce_loss_weight = w_ce
    config.beacon_continuation_loss_weight = w_cont
    config.beacon_continuation_tokens = 1
    config.beacon_continuation_per_beacon = True
    config.beacon_readout_distill_weight = w_readout
    # 读出蒸馏只在 training 且 question memory 开启时才产生，别的用例保持 eval 以隔离 dropout
    return model.train() if train else model


def _loss(model, regions=REGIONS, labels=None, question=None):
    labels = IDS.clone() if labels is None else labels
    torch.manual_seed(7)   # 钉死前向里任何随机性，保证各次比较只差权重
    with torch.no_grad():
        return model(
            input_ids=IDS, labels=labels, regions=regions, question_input_ids=question
        ).loss


def _ce_and_cont(question=None, **kwargs):
    """反解出未加权的 ce 与 cont。

    w_cont=0 时续写目标根本不构造 → L 就是 ce；w_ce=w_cont=1 时 L=(ce+cont)/2。
    续写项不改变 hidden，所以两次前向的 ce 相同。
    """
    ce = _loss(_make_model(w_ce=1.0, w_cont=0.0, **kwargs), regions=REGIONS, question=question)
    half = _loss(_make_model(w_ce=1.0, w_cont=1.0, **kwargs), regions=REGIONS, question=question)
    return ce, 2.0 * half - ce


def test_only_ce_uses_its_weight_directly():
    """续写关闭时 L = w_ce·ce，包括用户使用的 0.1 / 0 配置。"""
    reference = _loss(_make_model(w_ce=1.0, w_cont=0.0))
    for w_ce in (0.1, 0.3, 1.0, 7.5):
        assert torch.allclose(_loss(_make_model(w_ce=w_ce, w_cont=0.0)), w_ce * reference, atol=1e-6)


def test_weight_ratio_is_scale_invariant():
    """同比例缩放两个权重不改变 L —— 这正是「只改比例、不动总尺度」。"""
    base = _loss(_make_model(w_ce=0.3, w_cont=0.7))
    for scale in (2.0, 10.0):
        assert torch.allclose(
            _loss(_make_model(w_ce=0.3 * scale, w_cont=0.7 * scale)), base, atol=1e-6
        )


def test_half_half_is_the_plain_average():
    """0.5 / 0.5 → L 就是两项的算术平均。"""
    ce, cont = _ce_and_cont()
    assert torch.allclose(_loss(_make_model(w_ce=0.5, w_cont=0.5)), (ce + cont) / 2, atol=1e-6)


def test_three_seven_matches_the_weighted_average():
    """0.3 / 0.7 → L = 0.3·ce + 0.7·cont。"""
    ce, cont = _ce_and_cont()
    assert torch.allclose(
        _loss(_make_model(w_ce=0.3, w_cont=0.7)), 0.3 * ce + 0.7 * cont, atol=1e-6
    )


def test_unnormalized_weights_still_get_normalized():
    """1.0 / 0.1（和为 1.1）→ 分母按实际出现项求和，不是把两项直接相加。"""
    ce, cont = _ce_and_cont()
    assert torch.allclose(
        _loss(_make_model(w_ce=1.0, w_cont=0.1)), (ce + 0.1 * cont) / 1.1, atol=1e-6
    )


def test_missing_continuation_uses_ce_weight_directly():
    """没有压缩段就没有续写项，L = w_ce·ce。"""
    reference = _loss(_make_model(w_ce=1.0, w_cont=0.0), regions=[])
    for w_ce, w_cont in ((1.0, 0.0), (0.3, 0.7), (0.5, 0.5)):
        actual = _loss(_make_model(w_ce=w_ce, w_cont=w_cont), regions=[])
        assert torch.allclose(actual, w_ce * reference, atol=1e-6)


def test_readout_term_stays_out_of_the_denominator():
    """readout 独立加项：L = (w_ce·ce + w_cont·cont)/(w_ce+w_cont) + w_readout·readout。"""
    question = torch.tensor([[7, 8, 9]])
    model_kwargs = {"question_memory": True, "train": True}

    ce, cont = _ce_and_cont(question=question, **model_kwargs)
    bare = _loss(_make_model(w_ce=1.0, w_cont=0.0, w_readout=0.0, **model_kwargs), question=question)
    with_readout = _loss(
        _make_model(w_ce=1.0, w_cont=0.0, w_readout=2.0, **model_kwargs), question=question
    )
    readout = (with_readout - bare) / 2.0     # w_readout=0.1 已由 _make_model 覆盖为 2.0
    assert readout.item() > 0, "读出蒸馏没有触发，本用例会退化成恒真断言"

    mixed = _loss(
        _make_model(w_ce=0.5, w_cont=0.5, w_readout=2.0, **model_kwargs), question=question
    )
    assert torch.allclose(mixed, (0.5 * ce + 0.5 * cont) + 2.0 * readout, atol=1e-6)


def test_absent_ce_uses_continuation_weight_directly():
    """labels 全 -100 → 只有续写监督，L = w_cont·cont，不受 w_ce 影响。

    （分量之和 == 总损失由 ``test_loss_component_logging`` 覆盖，不在此重复。）
    """
    all_masked = torch.full_like(IDS, -100)
    half = _loss(_make_model(w_ce=1.0, w_cont=0.5), labels=all_masked)
    full = _loss(_make_model(w_ce=1.0, w_cont=1.0), labels=all_masked)
    bigger_ce = _loss(_make_model(w_ce=10.0, w_cont=0.5), labels=all_masked)

    assert half.item() > 0
    assert torch.allclose(half, 0.5 * full, atol=1e-6)
    assert torch.allclose(half, bigger_ce, atol=1e-6), "CE 缺席时 w_ce 不应稀释续写损失"


def test_zero_ce_loss_weight_is_supported():
    config = BeaconConfig(beacon_ce_loss_weight=0.0)
    assert BeaconConfig.from_dict(config.to_dict()).beacon_ce_loss_weight == 0.0


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_ce_loss_weight_must_be_finite_and_nonnegative(weight):
    with pytest.raises(ValueError, match="beacon_ce_loss_weight"):
        BeaconConfig(beacon_ce_loss_weight=weight)


def test_zero_ce_weight_keeps_only_continuation():
    _, cont = _ce_and_cont()
    model = _make_model(w_ce=0.0, w_cont=0.7)
    torch.testing.assert_close(_loss(model), 0.7 * cont)
    assert model._last_loss_parts["ce_loss"].item() == 0.0


def test_default_ce_loss_weight_is_one():
    """旧 checkpoint 的 config.json 没有该字段时回落到 1.0，行为不变。"""
    assert BeaconConfig().beacon_ce_loss_weight == 1.0
