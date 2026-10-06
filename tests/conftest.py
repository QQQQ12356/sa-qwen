"""CPU 单元测试使用 Transformers 原生回退，避免调用仅支持 CUDA 的 FLA。"""
import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def cpu_qwen35_kernel_fallback():
    if os.environ.get('BEACON_TEST_DEVICE', 'cpu') != 'cpu':
        yield
        return
    from transformers.models.qwen3_5 import modeling_qwen3_5
    with patch.multiple(
        modeling_qwen3_5, chunk_gated_delta_rule=None,
        fused_recurrent_gated_delta_rule=None, causal_conv1d_fn=None,
        FusedRMSNormGated=None,
    ):
        yield
