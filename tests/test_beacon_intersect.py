import pytest
import torch

from search_comp.models.beacon_config import BeaconConfig
from search_comp.models.beacon_qwen35 import _Qwen3BeaconMemory
from search_comp.models.modeling_utils import beacon_intersect_order
from test_beacon_qwen35_linear_memory import _tiny_model


@pytest.mark.parametrize("beacon_pos", ["append", "intersect"])
def test_layout_visibility_labels_and_cache(beacon_pos):
    beacon = BeaconConfig(beacon_window=4, beacon_stride=4, beacon_ratio=2, beacon_pos=beacon_pos)
    ids = torch.tensor([[10, 11, 20, 21, 22, 23, 24, 30, 40, 41, 31]])
    labels = ids.clone()
    regions = [(2, 7), (8, 10)]
    memory = _Qwen3BeaconMemory(_tiny_model(beacon_pos), beacon)
    memory.prepare(ids, labels, regions)

    expected_steps = (
        [[10, 11], [20, 21, 128, 22, 23, 128], [24, 128], [30], [40, 41, 128], [31]]
        if beacon_pos == "intersect" else
        [[10, 11], [20, 21, 22, 23, 128, 128], [24, 128], [30], [40, 41, 128], [31]]
    )
    retained = []
    original_pos = 0
    for expected in expected_steps:
        win_ids, win_labels, mask, positions, layer_past = memory.step()
        assert win_ids.tolist() == [expected]
        for layer_idx, (key, value, beacon_size, indices) in layer_past:
            if key is not None:
                assert key.flatten().tolist() == retained
        current_len = len(expected)
        assert mask.shape == (1, 1, current_len, len(retained) + current_len)
        assert (mask[..., :len(retained)] == 0).all()
        assert torch.equal(mask[0, 0, :, len(retained):] == 0, torch.ones(current_len, current_len).bool().tril())
        if beacon_pos == "intersect" and expected == [20, 21, 128, 22, 23, 128]:
            assert (mask[0, 0, 5, len(retained):len(retained) + 5] == 0).all()
            assert (mask[0, 0, 3, len(retained):len(retained) + 3] == 0).all()
            assert (mask[0, 0, 2, len(retained) + 3:] < 0).all()
            assert indices.tolist() == [0, 0, 1, 0, 0, 1]
        raw_count = sum(token != 128 for token in expected)
        raw_mask = win_ids[0] != 128
        shifted = torch.cat([labels[:, 1:], labels.new_full((1, 1), -100)], dim=1)
        assert win_labels[0, raw_mask].tolist() == shifted[0, original_pos:original_pos + raw_count].tolist()
        assert (win_labels[0, ~raw_mask] == -100).all()
        original_pos += raw_count
        fake_cache = win_ids.float().reshape(1, 1, current_len, 1)
        updates = [(layer_idx, (fake_cache, fake_cache, beacon_size, indices)) for layer_idx, _ in layer_past]
        memory.update_memory(updates)
        retained.extend([128] * beacon_size if beacon_size else expected)
    assert memory.finish
    assert retained == [10, 11, 128, 128, 128, 30, 128, 31]


def test_intersect_config_round_trip_and_invalid_position():
    config = BeaconConfig(beacon_pos="intersect")
    assert BeaconConfig.from_dict(config.to_dict()).beacon_pos == "intersect"
    with pytest.raises(AssertionError, match="beacon_pos"):
        BeaconConfig(beacon_pos="invalid")


@pytest.mark.parametrize("raw_length", [1, 2, 3, 4, 5, 8])
@pytest.mark.parametrize("ratio", [1, 2, 4])
def test_intersect_order_keeps_tokens_and_tail_beacon(raw_length, ratio):
    beacon_size = (raw_length + ratio - 1) // ratio
    appended = torch.cat([torch.arange(raw_length), torch.full((beacon_size,), -1)])
    order = beacon_intersect_order(raw_length, ratio, appended.device)
    expected = []
    for start in range(0, raw_length, ratio):
        expected.extend(range(start, min(start + ratio, raw_length)))
        expected.append(-1)
    assert appended[order].tolist() == expected
    assert sorted(order.tolist()) == list(range(raw_length + beacon_size))
