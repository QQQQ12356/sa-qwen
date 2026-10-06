"""同一 window 内 chunk 之间的可见性实测。

这是 ``beacon_window > beacon_ratio`` 下 per-beacon 续写的前提，也是「为什么
``beacon_continuation_per_beacon`` 不再要求 ``window == ratio``」的依据：

- 窗内**后一个 chunk 的 token 同时看得见更早 chunk 的原始 token 和它们的 beacon**
  —— full-attention 层靠整窗普通因果掩码（没有按 chunk 切分的屏蔽），
  linear-attention 层的 reader 是无掩码的顺序扫描（4 个原始 token + 2 个 beacon
  在一次调用里吃完，chunk 之间没有被切断）；
- 隔离只发生在**窗口边界**：窗末只提交 beacon、丢弃原始 K/V，下一窗左侧只剩已提交的
  beacon。

因此 ``window > ratio`` 时「仅凭压缩记忆预测」不成立，``describe_layout`` 会如实上报。
"""

import torch

from search_comp.models.beacon_qwen35 import _Qwen3BeaconMemory
from test_beacon_qwen35_linear_memory import _tiny_model


def _wide_model(layer_types=None):
    """window=4, ratio=2：一窗两个 chunk，intersect 布局为 [t0,t1,B0,t2,t3,B1]。"""
    model = _tiny_model("intersect", layer_types)
    model.beacon_config.beacon_window = 4
    model.beacon_config.beacon_stride = 4
    return model


def test_later_chunk_sees_both_raw_tokens_and_beacons():
    """窗内第二个 chunk 的首 token，既可见第一个 chunk 的原始 token，也可见其 beacon。"""
    model = _wide_model(["full_attention"])
    memory = _Qwen3BeaconMemory(model, model.beacon_config)
    model._mem = memory
    memory.prepare(torch.tensor([[20, 21, 22, 23]]), None, [(0, 4)])
    win_ids, _, mask, _, _ = memory.step()

    beacon_id = model.config.vocab_size
    assert win_ids.tolist() == [[20, 21, beacon_id, 22, 23, beacon_id]]
    assert memory._step_beacon_indices.tolist() == [0, 0, 1, 0, 0, 1]

    allowed = mask[0, 0] == 0
    visible = {col for col in range(allowed.shape[1]) if bool(allowed[3, col])}
    assert {0, 1} <= visible, "第二个 chunk 的 token 看不到第一个 chunk 的原始 token"
    assert 2 in visible, "第二个 chunk 的 token 看不到第一个 chunk 的 beacon"


def test_linear_reader_processes_the_whole_window_in_one_call():
    """reader 一次吃下整个窗（含 beacon），chunk 之间没有第二次调用/状态复位。"""
    # 混合层：``past`` 只带 full-attention 层，全 linear 的模型取不到 beacon_size
    model = _wide_model(["linear_attention", "full_attention"])
    lengths = []
    native_reader = model._linear_attention_forward

    def recording_reader(module, hidden, previous_recurrent, previous_conv):
        lengths.append(hidden.shape[1])
        return native_reader(module, hidden, previous_recurrent, previous_conv)

    model._linear_attention_forward = recording_reader
    model.prefill_and_get_cache(torch.tensor([[20, 21, 22, 23]]), regions=[(0, 4)])

    assert lengths == [6], f"4 个原始 token + 2 个 beacon 应在一次调用内读完，实际 {lengths}"


def test_window_boundary_keeps_only_beacons_not_raw_tokens():
    """跨窗隔离成立：8 个原始 token 压缩后持久 cache 只剩 4 个 beacon。"""
    model = _wide_model(["full_attention", "linear_attention"])
    memory = model.prefill_and_get_cache(
        torch.tensor([[20, 21, 22, 23, 24, 25, 26, 27]]), regions=[(0, 8)]
    )

    cache_k, _ = memory._cache[0]
    assert cache_k.shape[2] == 4, "两窗各 2 个 beacon，原始文档 K/V 不得留下"
