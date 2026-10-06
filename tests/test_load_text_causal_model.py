"""``load_text_causal_model`` 对两种 checkpoint 布局的兼容性测试。

原生 SFT（``native_trainer``）保存的是纯文本 ``Qwen3_5ForCausalLM``，
其 config 是 ``model_type: qwen3_5_text``、**没有** ``text_config``；
而基础模型 ``Qwen/Qwen3.5-2B`` 是多模态 ``Qwen3_5ForConditionalGeneration``，
文本主干挂在 ``config.text_config`` 下。评测入口两种都要能加载。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from search_comp.milestones.qwen35_text import load_text_causal_model


@pytest.fixture(scope="module")
def tiny_text_checkpoint(tmp_path_factory):
    """按原生 SFT 的保存格式，产出一个 tiny 纯文本 checkpoint。"""
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    from test_beacon_qwen35_linear_memory import _tiny_model
    config = _tiny_model().config
    path = tmp_path_factory.mktemp("text_ckpt") / "final"
    Qwen3_5ForCausalLM(config).save_pretrained(str(path))
    return str(path)


def test_saved_checkpoint_is_text_only_layout(tiny_text_checkpoint):
    # Arrange / Act
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(tiny_text_checkpoint, trust_remote_code=True)

    # Assert：正是原生 SFT 保存出来的形态——没有 text_config
    assert config.model_type == "qwen3_5_text"
    assert not hasattr(config, "text_config")


def test_loads_native_text_checkpoint(tiny_text_checkpoint):
    # Act
    model = load_text_causal_model(tiny_text_checkpoint, device_map="cpu")

    # Assert：加载成功且处于推理态
    assert model.config.vocab_size == 128
    assert not model.training


def test_rejects_beacon_checkpoint(tiny_text_checkpoint, tmp_path):
    """Beacon checkpoint 也是纯文本布局，但不能走这条路径。

    否则 beacon 压缩参数会被当作「多余的键」静默丢弃，得到一个无压缩模型。
    """
    # Arrange：把 tiny 纯文本 checkpoint 伪装成 Beacon 保存的形态
    import json
    import shutil

    beacon_dir = tmp_path / "beacon_ckpt"
    shutil.copytree(tiny_text_checkpoint, beacon_dir)
    config_path = beacon_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = ["BeaconQwen3_5ForCausalLM"]
    config["beacon_window"] = 512
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # Act / Assert
    with pytest.raises(ValueError, match="Beacon"):
        load_text_causal_model(str(beacon_dir), device_map="cpu")
