"""子文档压缩区切分 + 续写损失。

覆盖：

- :func:`split_document_regions` 的边界定位、单文档与降级路径；
- window 模式续写损失：只有一个窗口的压缩区是**严格 no-op**；跨两个窗口时才产生损失与梯度；
- beacon 模式续写损失（``beacon_continuation_per_beacon``）：
  ``beacon_window == beacon_ratio`` 时每个窗口恰好一个 chunk，窗尾 beacon 的全层型 chunk
  隔离使「后续 chunk 只看得见已提交的 beacon」成立；
  ``beacon_window > beacon_ratio`` 时窗内有多个 chunk，必须配 ``beacon_pos="intersect"``
  （``append`` 布局的 beacon 全贴在窗末，无法对应各自的 chunk），此时窗内 chunk 之间**不**
  互相隔离 —— 后一个 chunk 同时读得到更早 chunk 的原始 token 与其 beacon（实测见
  ``tests/test_beacon_window_visibility.py``）。两种配置下监督点都是「每个 chunk 的 beacon
  预测紧随其后的 k 个 token」，目标只在同一压缩段内、钳制在 ``seg_end``；
- 推理路径（``labels is None``）不产生任何续写损失。
"""

import os

import pytest
import torch

from search_comp.data.trajectory import split_document_regions
from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen35 import _Qwen3BeaconMemory
from test_beacon_qwen35_linear_memory import _tiny_model


DEVICE = os.environ.get("BEACON_TEST_DEVICE", "cpu")


class _Encoded(dict):
    """既支持 ``.get`` 也支持 ``.input_ids`` 属性访问的返回对象。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _CharTokenizer:
    """字符级假分词器：token 与字符一一对应，便于精确断言区间。"""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        encoded = _Encoded(input_ids=list(range(len(text))))
        if return_offsets_mapping:
            encoded["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return encoded


class _NoOffsetTokenizer(_CharTokenizer):
    """不支持 offset mapping 的分词器（触发整块降级）。"""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return _Encoded(input_ids=list(range(len(text))))


#: 两篇子文档：``Doc 1 A`` + 空行 + ``Doc 2 B``（长度 16）。
TWO_DOCS = "Doc 1 A\n\nDoc 2 B"

SAMPLE = {
    "id": "sample-0",
    "question": "Q",
    "answer": "A",
    "turns": [{"query": "q1", "docs": TWO_DOCS}],
    "thinks": ["t1"],
    "final_think": "t2",
}


# ----------------------------------------------------------------------
# 切分
# ----------------------------------------------------------------------
def test_split_document_regions_locates_document_start():
    # 边界落在第二篇文档的首字符（9），而不是分隔空行的起点（7）。
    assert split_document_regions(TWO_DOCS, _CharTokenizer()) == [(0, 9), (9, 16)]


def test_split_document_regions_single_document_is_one_region():
    assert split_document_regions("Doc 1 only", _CharTokenizer()) == [(0, 10)]


def test_split_document_regions_falls_back_without_offset_mapping():
    assert split_document_regions(TWO_DOCS, _NoOffsetTokenizer()) == [(0, 16)]


def test_split_document_regions_empty_text():
    assert split_document_regions("", _CharTokenizer()) == []


def test_split_document_regions_covers_block_without_gaps():
    regions = split_document_regions(TWO_DOCS, _CharTokenizer())
    assert regions[0][0] == 0
    assert regions[-1][1] == len(TWO_DOCS)
    assert all(end == nxt[0] for (_, end), nxt in zip(regions, regions[1:]))






# ----------------------------------------------------------------------
# 续写损失
# ----------------------------------------------------------------------
#: 12 个 token；``beacon_window=4``、``beacon_ratio=2``（见 ``_tiny_model``）。
IDS = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31]], device=DEVICE)


def _make_model(weight: float):
    """同种子构造两个逐位相同的模型，只有续写权重不同。"""
    torch.manual_seed(7)
    model = _tiny_model()
    model.beacon_config.beacon_continuation_loss_weight = weight
    model.beacon_config.beacon_continuation_tokens = 2
    return model.eval()


def _loss(model, regions, labels):
    return model(input_ids=IDS, labels=labels, regions=regions).loss


def test_continuation_is_strict_noop_without_later_windows():
    """4 token 的压缩区只有一个窗口 → 不产生续写目标，损失逐位不变。"""
    labels = IDS.clone()
    labels[:, :8] = -100
    baseline = _loss(_make_model(0.0), [(2, 6)], labels)
    weighted = _loss(_make_model(0.5), [(2, 6)], labels)
    assert torch.equal(baseline, weighted)


def test_continuation_loss_fires_on_later_windows():
    """8 token 的压缩区跨两个窗口 → 第二窗头部被监督；总分按份额 (ce+0.5·cont)/1.5 归一。"""
    labels = IDS.clone()
    labels[:, :8] = -100
    ce = _loss(_make_model(0.0), [(2, 10)], labels)                 # 关闭 → 就是原始 ce
    cont = 2 * _loss(_make_model(1.0), [(2, 10)], labels) - ce      # (ce+cont)/2 反解
    weighted = _loss(_make_model(0.5), [(2, 10)], labels)
    assert torch.isfinite(weighted)
    assert torch.allclose(weighted, (ce + 0.5 * cont) / 1.5, atol=1e-6)


def test_continuation_gradients_reach_beacon_parameters():
    """labels 全 -100 时主 CE 为 0，梯度只能来自续写损失。"""
    model = _make_model(0.5).train()
    labels = torch.full_like(IDS, -100)
    loss = _loss(model, [(2, 10)], labels)

    assert loss.item() > 0
    loss.backward()
    for name in ("beacon_embed_tokens",):
        grad = getattr(model.model, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0


def test_adjacent_single_window_documents_never_continue():
    """两个相邻子文档区各只有 1 个窗口 → 都不监督，文档边界处不产生伪续写。"""
    labels = IDS.clone()
    labels[:, :8] = -100
    baseline = _loss(_make_model(0.0), [(2, 6), (6, 10)], labels)
    weighted = _loss(_make_model(0.5), [(2, 6), (6, 10)], labels)
    assert torch.equal(baseline, weighted)


def _supervised_window_starts(model, ids, regions):
    """逐窗调用状态机，收集真正被续写监督的窗口起点。"""
    model._mem = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem.prepare(ids, ids.clone(), regions)
    starts = []
    while not model._mem.finish:
        start = model._mem._pos
        model._mem.step()
        if model._mem._step_cont_labels is not None:
            starts.append(start)
    return starts


def test_only_later_windows_within_each_document_are_supervised():
    """(2,10) 与 (10,18) 各 2 窗 → 监督 6 与 14，**不监督**文档边界 10。"""
    ids = torch.arange(10, 28, device=DEVICE).unsqueeze(0)
    starts = _supervised_window_starts(_make_model(0.5), ids, [(2, 10), (10, 18)])
    assert starts == [6, 14]


def _count_keep_windows(model, ids, regions):
    """逐窗走状态机，统计 keep 窗口数（不跑模型前向）。"""
    model._mem = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem.prepare(ids, None, regions)
    count = 0
    while not model._mem.finish:
        model._mem.step()
        if model._mem._store == "cache":
            count += 1
    return count


ALL_KEEP_IDS = torch.arange(10, 42, device=DEVICE).unsqueeze(0)   # 32 token，全 keep


def test_keep_window_default_keeps_historical_chunking():
    """None（默认）→ keep 仍按 beacon_window 切：32/4 = 8 个窗口。"""
    assert _count_keep_windows(_make_model(0.0), ALL_KEEP_IDS, []) == 8


def test_keep_window_zero_merges_whole_segment():
    model = _make_model(0.0)
    model.beacon_config.beacon_keep_window = 0
    assert _count_keep_windows(model, ALL_KEEP_IDS, []) == 1


def test_keep_window_positive_is_a_cap():
    model = _make_model(0.0)
    model.beacon_config.beacon_keep_window = 16
    assert _count_keep_windows(model, ALL_KEEP_IDS, []) == 2


def test_keep_window_does_not_change_result():
    """keep 切窗只是计算粒度：不切窗与历史切窗的 loss 在浮点误差内一致。"""
    labels = ALL_KEEP_IDS.clone()
    chunked = _make_model(0.0)(input_ids=ALL_KEEP_IDS, labels=labels, regions=[]).loss
    merged_model = _make_model(0.0)
    merged_model.beacon_config.beacon_keep_window = 0
    merged = merged_model(input_ids=ALL_KEEP_IDS, labels=labels, regions=[]).loss
    assert torch.allclose(chunked, merged, atol=1e-5, rtol=1e-4)


def test_keep_window_does_not_disturb_continuation_targets():
    """keep 不切窗不改变压缩段布局与续写监督点。"""
    ids = torch.arange(10, 28, device=DEVICE).unsqueeze(0)
    regions = [(2, 10), (10, 18)]
    default = _supervised_window_starts(_make_model(0.5), ids, regions)
    merged_model = _make_model(0.5)
    merged_model.beacon_config.beacon_keep_window = 0
    merged = _supervised_window_starts(merged_model, ids, regions)
    assert default == merged == [6, 14]


def test_describe_layout_reports_effective_settings():
    """评测启动打印的那一行必须反映实际生效的布局，而不是默认值。"""
    split = BeaconConfig(
        beacon_window=96, beacon_stride=96, beacon_ratio=16,
        beacon_doc_region_split=True, beacon_keep_window=1024,
    ).describe_layout()
    assert "子文档独立成段" in split
    assert "每窗 6 个 beacon" in split
    assert "keep_window=1024" in split

    chunked = BeaconConfig(beacon_window=512, beacon_stride=512, beacon_ratio=16).describe_layout()
    assert "整块成段" in chunked
    assert "每窗 32 个 beacon" in chunked
    assert "keep_window=跟随 window" in chunked

    merged = BeaconConfig(
        beacon_window=512, beacon_stride=512, beacon_ratio=16, beacon_keep_window=0,
    ).describe_layout()
    assert "keep_window=整段不切" in merged


def test_prefill_produces_no_continuation_loss():
    """推理路径 labels 为 None，续写目标不构造，损失列表保持为空。"""
    model = _make_model(0.5)
    memory = model.prefill_and_get_cache(IDS, regions=[(2, 10)])
    assert memory._cont_losses == []
    assert memory._shifted_ids is None


# ----------------------------------------------------------------------
# 续写损失：按 beacon 生效（intersect 布局）
# ----------------------------------------------------------------------
def _make_per_beacon_model(weight: float, tokens: int = 1, window: int = 2, pos: str = "intersect"):
    """按 beacon 逐 chunk 监督的模型；``window == ratio``（默认）时每窗一个 chunk。

    隔离来自窗口边界而非掩码：窗末只提交 beacon、丢弃原始 K/V，于是后续 chunk 只能
    看见已提交的 beacon。这对 full-attention 与 linear-attention 层同时成立。
    ``window > ratio``（必须 ``pos="intersect"``）时窗内多个 chunk，隔离不再成立。
    """
    torch.manual_seed(7)
    model = _tiny_model(pos)
    model.beacon_config.beacon_window = window
    model.beacon_config.beacon_stride = window
    model.beacon_config.beacon_continuation_loss_weight = weight
    model.beacon_config.beacon_continuation_tokens = tokens
    model.beacon_config.beacon_continuation_per_beacon = True
    return model.eval()


def _supervised_beacon_points(model, ids, regions):
    """逐窗走状态机，返回 ``(窗起点, 读取的隐状态行, 目标 token 的全局位置)``。"""
    model._mem = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem.prepare(ids, ids.clone(), regions)
    position_of = {int(token): pos for pos, token in enumerate(ids[0].tolist())}
    points = []
    while not model._mem.finish:
        start = model._mem._pos
        model._mem.step()
        labels, rows = model._mem._step_cont_labels, model._mem._step_cont_rows
        if labels is None:
            continue
        for row, token in zip(rows[0].tolist(), labels[0].tolist()):
            # 目标不足 k 个时用 -100 补齐（与 labels 的忽略语义一致），不是监督点
            if token < 0:
                continue
            points.append((start, row, position_of[int(token)]))
    return points


def _supervised_beacon_targets(model, ids, regions):
    """``_supervised_beacon_points`` 的简化视图：只保留 ``(窗起点, 目标全局位置)``。"""
    return [(start, target) for start, _, target in _supervised_beacon_points(model, ids, regions)]


def test_per_beacon_window_gt_ratio_requires_intersect_pos():
    """window > ratio 时 append 不可用：beacon 全贴窗末，无法对应各自的 chunk 边界。"""
    with pytest.raises(ValueError, match="intersect"):
        BeaconConfig(beacon_window=4, beacon_stride=4, beacon_ratio=2,
                     beacon_pos="append", beacon_continuation_per_beacon=True)


def test_per_beacon_window_gt_ratio_allows_intersect_pos():
    """window > ratio + intersect 合法：beacon 紧跟各自 chunk，可逐 chunk 读出。"""
    config = BeaconConfig(beacon_window=4, beacon_stride=4, beacon_ratio=2,
                          beacon_pos="intersect", beacon_continuation_per_beacon=True,
                          beacon_continuation_loss_weight=0.1)
    assert "续写监督=每 beacon" in config.describe_layout()
    assert "（intersect，窗内 chunk 间不隔离）" in config.describe_layout()


def test_per_beacon_requires_window_at_least_ratio():
    """window < ratio 时一个窗口装不下一个 chunk，必须直接报错，而不是留给 assert。"""
    with pytest.raises(ValueError, match="0 < beacon_ratio <= beacon_window"):
        BeaconConfig(beacon_window=8, beacon_stride=8, beacon_ratio=16,
                     beacon_pos="intersect", beacon_continuation_per_beacon=True)


def test_per_beacon_rejects_bad_ratio_even_when_beacon_disabled():
    """beacon_ratio<=0 会绕过 enable_beacon 块里的断言，per_beacon 必须自己挡住。"""
    with pytest.raises(ValueError, match="0 < beacon_ratio <= beacon_window"):
        BeaconConfig(enable_beacon=False, beacon_window=4, beacon_stride=4, beacon_ratio=0,
                     beacon_pos="intersect", beacon_continuation_per_beacon=True)


def test_per_beacon_append_wide_window_fails_loudly_in_forward():
    """append + window>ratio 绕过构造期校验时，前向必须报错而不是静默错监督。

    这种布局下 beacon 全贴窗末，`min((offset+1)*ratio, win_len)` 会把读出点排到目标
    token 之后 —— 监督退化成「照着原文抄」，实测行 4 被要求预测它已经能看见的 token 22。
    """
    model = _make_per_beacon_model(0.5, tokens=1, window=4, pos="append")
    with pytest.raises(ValueError, match="intersect"):
        _supervised_beacon_points(model, IDS, [(2, 10)])


def test_per_beacon_supervises_token_after_every_beacon():
    """window=ratio=2 → 每窗 1 个 chunk，尾部 beacon 监督下一个 chunk 的首 token。"""
    model = _make_per_beacon_model(0.5)
    assert _supervised_beacon_targets(model, IDS, [(2, 10)]) == [(2, 4), (4, 6), (6, 8)]


def test_per_beacon_fires_for_single_window_document():
    """单窗压缩段在 window 模式下是 no-op；beacon 模式下仍监督其唯一的 chunk 边界。"""
    model = _make_per_beacon_model(0.5)
    assert _supervised_beacon_targets(model, IDS, [(2, 6)]) == [(2, 4)]


def test_per_beacon_never_continues_across_document_boundary():
    """段尾那个跨窗 beacon 只在同一压缩段内续写，不得预测下一篇文档首 token。"""
    ids = torch.arange(10, 28, device=DEVICE).unsqueeze(0)
    model = _make_per_beacon_model(0.5)
    assert _supervised_beacon_targets(model, ids, [(2, 10), (10, 18)]) == [
        (2, 4), (4, 6), (6, 8), (10, 12), (12, 14), (14, 16),
    ]


def test_per_beacon_is_layout_agnostic_when_window_equals_ratio():
    """window==ratio 时 append 与 intersect 是同一套布局，监督点必须一致。"""
    targets = []
    for beacon_pos in ("append", "intersect"):
        torch.manual_seed(7)
        model = _tiny_model(beacon_pos)
        model.beacon_config.beacon_window = 2
        model.beacon_config.beacon_stride = 2
        model.beacon_config.beacon_continuation_loss_weight = 0.5
        model.beacon_config.beacon_continuation_tokens = 1
        model.beacon_config.beacon_continuation_per_beacon = True
        targets.append(_supervised_beacon_targets(model, IDS, [(2, 10)]))
    assert targets[0] == targets[1] == [(2, 4), (4, 6), (6, 8)]


def _make_intersect_wide_model(weight: float, tokens: int = 1):
    """``beacon_window(4) > beacon_ratio(2)`` + intersect：一窗两个 chunk，逐 chunk 读出。

    窗内布局是 ``[t0, t1, B0, t2, t3, B1]`` —— beacon 恒在各自 chunk 之后，故读出点行号
    恒为 2 与 5。窗内 chunk **不互相隔离**（后一个 chunk 同时读得到前一个 chunk 的原始
    token 与其 beacon），隔离只发生在窗口边界。
    """
    return _make_per_beacon_model(weight, tokens, window=4)


def test_per_beacon_window_gt_ratio_supervises_every_chunk():
    """window=4/ratio=2 → 每窗 2 个 chunk 各有一个读出点；目标 = 紧随其后的 k 个 token。

    窗 [2,6)：B0(row2)→4、B1(row5)→6；窗 [6,10)：B0(row2)→8，B1 的目标已越段尾被丢弃。
    """
    model = _make_intersect_wide_model(0.5, tokens=1)
    assert _supervised_beacon_points(model, IDS, [(2, 10)]) == [
        (2, 2, 4), (2, 5, 6), (6, 2, 8),
    ]


def test_per_beacon_window_gt_ratio_supervises_k_tokens_per_chunk():
    """k=2：窗末那个 chunk 的目标越窗伸到下一窗头部，但一律钳制在 seg_end 内。"""
    model = _make_intersect_wide_model(0.5, tokens=2)
    assert _supervised_beacon_targets(model, IDS, [(2, 10)]) == [
        (2, 4), (2, 5), (2, 6), (2, 7), (6, 8), (6, 9),
    ]


def test_per_beacon_window_gt_ratio_pads_short_targets():
    """段尾 chunk 的目标不足 k 个时补 -100：每个 chunk 恒占 k 列，监督点只取存在的部分。"""
    model = _make_intersect_wide_model(0.5, tokens=3)
    labels_per_window = []
    model._mem = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem.prepare(IDS, IDS.clone(), [(2, 9)])
    while not model._mem.finish:
        model._mem.step()
        if model._mem._step_cont_labels is not None:
            labels_per_window.append(model._mem._step_cont_labels[0].tolist())

    # 窗 [2,6)：两个 chunk 各 3 个目标；窗 [6,9)：段尾 chunk 只剩 1 个目标，补 2 个 -100
    assert labels_per_window == [[22, 23, 24, 24, 25, 26], [26, -100, -100]]
    assert _supervised_beacon_points(model, IDS, [(2, 9)]) == [
        (2, 2, 4), (2, 2, 5), (2, 2, 6),
        (2, 5, 6), (2, 5, 7), (2, 5, 8),
        (6, 2, 8),
    ]


def test_per_beacon_window_gt_ratio_gradients_reach_beacon_parameters():
    """labels 全 -100 时主 CE 为 0，梯度只能来自逐 chunk 的续写损失。"""
    model = _make_intersect_wide_model(0.5, tokens=1).train()
    labels = torch.full_like(IDS, -100)
    loss = _loss(model, [(2, 10)], labels)

    assert loss.item() > 0
    loss.backward()
    grad = model.model.beacon_embed_tokens.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_window_equals_ratio_commits_only_the_beacon():
    """隔离机制：窗末只提交 beacon，原始文档 K/V 不进入后续 chunk 的可见范围。

    掩码做不到这件事（它只作用于 full-attention 层），所以这里断言的是缓存里
    **实际被提交的位置数**，而不是注意力可见性。
    """
    model = _make_per_beacon_model(0.0)
    mem = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem = mem
    mem.prepare(IDS, None, [(2, 10)])
    while not mem.finish:
        win_ids, _, attn_mask, pos_emb, past = mem.step()
        cur_len, store = win_ids.shape[1], mem._store
        lengths_before = [0 if ck is None else ck.shape[2] for ck, _ in mem._cache]
        new_past, _ = model._native_forward(win_ids, None, attn_mask, pos_emb, past)
        mem.update_memory(new_past)
        if store == "beacon":
            lengths_after = [0 if ck is None else ck.shape[2] for ck, _ in mem._cache]
            growth = [after - before for after, before in zip(lengths_after, lengths_before)]
            full_attn = [g for g, t in zip(growth, mem.layer_types) if t == "full_attention"]
            assert full_attn, "该模型没有 full-attention 层，无法断言"
            assert all(g == 1 for g in full_attn), f"每个压缩窗应只提交 1 个 beacon，实际 {full_attn}"
            assert cur_len == 3, f"该窗口送入了 {cur_len} 个位置，却只提交 1 个"
            break


def test_per_beacon_supervises_exactly_one_point_per_chunk():
    """window==ratio 下每个窗口只有一个 chunk：监督点必须是尾部 beacon，且不被重复计数。"""
    model = _make_per_beacon_model(0.5)
    targets = _supervised_beacon_targets(model, IDS, [(2, 10)])
    starts = [start for start, _ in targets]
    assert starts == sorted(set(starts)), "同一窗口产生了多个监督点"
    assert all(target == start + model.beacon_config.beacon_window
               for start, target in targets), "监督目标不是紧邻的下一窗口首 token"


def test_per_beacon_supervises_k_tokens_per_chunk():
    """k=2（== ratio）：每个 chunk 的 beacon 监督下一个 chunk 的全部 token。"""
    model = _make_per_beacon_model(0.5, tokens=2)
    assert _supervised_beacon_targets(model, IDS, [(2, 10)]) == [
        (2, 4), (2, 5), (4, 6), (4, 7), (6, 8), (6, 9),
    ]


def test_per_beacon_clamps_targets_to_the_segment():
    """k=3 > ratio：目标越过下一个 chunk，但一律钳制在 seg_end 内，不跨子文档。"""
    model = _make_per_beacon_model(0.5, tokens=3)
    assert _supervised_beacon_targets(model, IDS, [(2, 10)]) == [
        (2, 4), (2, 5), (2, 6),
        (4, 6), (4, 7), (4, 8),
        (6, 8), (6, 9),            # 距段尾只剩 2 个 token，被截断
    ]


def test_per_beacon_k_targets_share_one_readout_row():
    """k 个目标来自同一行隐状态：beacon 恒在末位（布局 [t, t, B] → 行 2）。"""
    model = _make_per_beacon_model(0.5, tokens=2)
    points = _supervised_beacon_points(model, IDS, [(2, 10)])
    assert points, "应当产生监督点"
    assert {row for _, row, _ in points} == {2}


def test_window_mode_k_targets_read_consecutive_rows():
    """window 模式语义不变：k 个目标分别读窗头 0..k-1 行，而不是同一行。"""
    model = _make_model(0.5)                      # window=4/ratio=2/append，k=2
    model._mem = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem.prepare(IDS, IDS.clone(), [(2, 10)])
    rows_per_window = []
    while not model._mem.finish:
        model._mem.step()
        if model._mem._step_cont_rows is not None:
            rows_per_window.append(model._mem._step_cont_rows[0].tolist())
    assert rows_per_window == [[0, 1]]


def test_per_beacon_weight_zero_builds_no_targets():
    model = _make_per_beacon_model(0.0)
    assert _supervised_beacon_targets(model, IDS, [(2, 10)]) == []


def test_intersect_without_per_beacon_keeps_no_continuation_loss():
    """intersect 布局 + 关闭 per_beacon → 逐位等价历史行为（无续写监督）。"""
    torch.manual_seed(7)
    model = _tiny_model("intersect")
    model.beacon_config.beacon_continuation_loss_weight = 0.5
    assert _supervised_beacon_targets(model, IDS, [(2, 10)]) == []


def test_per_beacon_gradients_reach_beacon_parameters():
    """labels 全 -100 时主 CE 为 0，梯度只能来自按 beacon 的续写损失。"""
    model = _make_per_beacon_model(0.5).train()
    labels = torch.full_like(IDS, -100)
    loss = _loss(model, [(2, 10)], labels)

    assert loss.item() > 0
    loss.backward()
    grad = model.model.beacon_embed_tokens.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_per_beacon_loss_contributes_to_total():
    """同一 intersect 布局下，只有续写权重不同 → 总分按份额 (ce + 0.5·cont)/1.5 归一。"""
    labels = IDS.clone()
    labels[:, :8] = -100
    ce = _loss(_make_per_beacon_model(0.0), [(2, 10)], labels)
    cont = 2 * _loss(_make_per_beacon_model(1.0), [(2, 10)], labels) - ce
    weighted = _loss(_make_per_beacon_model(0.5), [(2, 10)], labels)
    assert torch.isfinite(weighted)
    assert torch.allclose(weighted, (ce + 0.5 * cont) / 1.5, atol=1e-6)


def test_per_beacon_no_continuation_loss_on_inference():
    model = _make_per_beacon_model(0.5)
    memory = model.prefill_and_get_cache(IDS, regions=[(2, 10)])
    assert memory._cont_losses == []
    assert memory._step_cont_labels is None


def test_describe_layout_reports_continuation_granularity():
    """判定续写监督粒度必须反映在启动打印里，便于确认与训练一致。"""
    per_beacon = BeaconConfig(
        beacon_pos="intersect", beacon_window=2, beacon_stride=2, beacon_ratio=2,
        beacon_continuation_loss_weight=0.1, beacon_continuation_per_beacon=True,
        beacon_continuation_tokens=3,
    ).describe_layout()
    assert "续写监督=每 beacon 3 个" in per_beacon

    windowed = BeaconConfig(
        beacon_window=4, beacon_stride=4, beacon_ratio=2,
        beacon_continuation_loss_weight=0.1,
    ).describe_layout()
    assert "续写监督=每窗头 4 个" in windowed

    off = BeaconConfig(beacon_window=4, beacon_stride=4, beacon_ratio=2).describe_layout()
    assert "续写监督=关闭" in off

    # intersect 布局下窗头监督根本不会产生，打印不得谎报「每窗头 N 个」
    intersect = BeaconConfig(
        beacon_pos="intersect", beacon_window=4, beacon_stride=4, beacon_ratio=2,
        beacon_continuation_loss_weight=0.1,
    ).describe_layout()
    assert "续写监督=关闭（intersect 不产生窗头监督）" in intersect
