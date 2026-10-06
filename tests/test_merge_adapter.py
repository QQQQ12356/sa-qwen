"""``merge_adapter`` 转换入口的单元测试。

不加载真实模型：把 ``subprocess`` 调用换成记录 argv 的桩，只验证默认路径推导、
参数拼装（含覆盖优先级）与各失败分支的报错信息。
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from search_comp.trainer import merge_adapter

_BEACON_CONFIG = {"enable_beacon": True, "beacon_window": 8192, "beacon_ratio": 32}


def _value_of(command, flag):
    """取出 ``--flag value`` 形式的参数值。"""
    return command[command.index(flag) + 1]


def _make_run(
    tmp_path,
    *,
    use_lora: bool = True,
    include_resolved: bool = True,
    adapter_files=("adapter_config.json", "adapter_model.safetensors", "beacon_state.pt"),
    base_model: str = "Qwen/Qwen3.5-2B",
):
    """搭一个最小实验目录，返回 ``(exp_dir, adapter_dir)``。"""
    exp_dir = tmp_path / "exp"
    adapter_dir = exp_dir / "final_adapter"
    adapter_dir.mkdir(parents=True)
    if include_resolved:
        (exp_dir / "resolved_config.json").write_text(
            json.dumps(
                {
                    "model_name_or_path": base_model,
                    "use_lora": use_lora,
                    "beacon": _BEACON_CONFIG,
                }
            ),
            encoding="utf-8",
        )
    for name in adapter_files:
        (adapter_dir / name).write_bytes(b"")
    return exp_dir, adapter_dir


@pytest.fixture
def recorded(monkeypatch):
    """替换掉真正拉起子进程的那一步，记录拼好的命令。"""
    calls = []
    monkeypatch.setattr(merge_adapter, "run_merge", lambda command: calls.append(command))
    return calls


def test_exp_dir_only_derives_paths_and_config(tmp_path, recorded):
    exp_dir, adapter_dir = _make_run(tmp_path)

    merge_adapter.main(["--exp_dir", str(exp_dir)])

    assert len(recorded) == 1
    command = recorded[0]
    assert command[:3] == [
        sys.executable,
        "-m",
        "search_comp.trainer.merge_beacon_lora",
    ]
    assert _value_of(command, "--base_model_path") == "Qwen/Qwen3.5-2B"
    assert _value_of(command, "--adapter_path") == str(adapter_dir)
    assert _value_of(command, "--output_path") == str(exp_dir / "final")
    # beacon 段原样透传（不需要人工拼 JSON）
    assert json.loads(_value_of(command, "--beacon_config_json")) == _BEACON_CONFIG
    assert _value_of(command, "--device-map") == "cpu"


def test_explicit_flags_override_run_config(tmp_path, recorded):
    exp_dir, _ = _make_run(tmp_path)
    custom = {"enable_beacon": True, "beacon_window": 4096}
    adapter_dir = tmp_path / "elsewhere" / "checkpoint-500"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_bytes(b"")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")
    (adapter_dir / "beacon_state.pt").write_bytes(b"")

    merge_adapter.main(
        [
            "--exp_dir",
            str(exp_dir),
            "--adapter_path",
            str(adapter_dir),
            "--output_path",
            str(tmp_path / "merged"),
            "--base_model_path",
            "Qwen/Qwen3.5-4B",
            "--beacon_config_json",
            json.dumps(custom),
            "--device-map",
            "auto",
        ]
    )

    command = recorded[0]
    assert _value_of(command, "--base_model_path") == "Qwen/Qwen3.5-4B"
    assert _value_of(command, "--adapter_path") == str(adapter_dir)
    assert _value_of(command, "--output_path") == str(tmp_path / "merged")
    assert json.loads(_value_of(command, "--beacon_config_json")) == custom
    assert _value_of(command, "--device-map") == "auto"


def test_adapter_path_alone_needs_no_exp_dir(tmp_path, recorded):
    adapter_dir = tmp_path / "final_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_bytes(b"")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"")
    (adapter_dir / "beacon_state.pt").write_bytes(b"")

    merge_adapter.main(
        [
            "--adapter_path",
            str(adapter_dir),
            "--base_model_path",
            "Qwen/Qwen3.5-2B",
            "--beacon_config_json",
            json.dumps(_BEACON_CONFIG),
        ]
    )

    command = recorded[0]
    # 没有 exp_dir 时输出目录缺省为 adapter 的同级 final
    assert _value_of(command, "--output_path") == str(tmp_path / "final")


def test_missing_resolved_config_lists_both_fixes(tmp_path, capsys):
    exp_dir, _ = _make_run(tmp_path, include_resolved=False)

    with pytest.raises(SystemExit) as excinfo:
        merge_adapter.main(["--exp_dir", str(exp_dir)])

    assert excinfo.value.code != 0
    stderr = capsys.readouterr().err
    assert "resolved_config.json" in stderr
    assert "--base_model_path" in stderr and "--beacon_config_json" in stderr


def test_missing_beacon_state_points_at_use_lora(tmp_path, capsys):
    # use_lora: false 的 run 只有全量权重，没有可合并的 adapter
    exp_dir, _ = _make_run(
        tmp_path, use_lora=False, adapter_files=("adapter_config.json",)
    )

    with pytest.raises(SystemExit) as excinfo:
        merge_adapter.main(["--exp_dir", str(exp_dir)])

    assert excinfo.value.code != 0
    stderr = capsys.readouterr().err
    assert "beacon_state.pt" in stderr
    assert "use_lora" in stderr


def test_missing_adapter_dir_is_reported(tmp_path, capsys):
    exp_dir, adapter_dir = _make_run(tmp_path)
    for child in adapter_dir.iterdir():
        child.unlink()
    adapter_dir.rmdir()

    with pytest.raises(SystemExit) as excinfo:
        merge_adapter.main(["--exp_dir", str(exp_dir)])

    assert excinfo.value.code != 0
    assert str(adapter_dir) in capsys.readouterr().err


def test_existing_output_requires_overwrite(tmp_path, recorded, capsys):
    exp_dir, _ = _make_run(tmp_path)
    final_dir = exp_dir / "final"
    final_dir.mkdir()
    (final_dir / "model.safetensors").write_bytes(b"")

    with pytest.raises(SystemExit) as excinfo:
        merge_adapter.main(["--exp_dir", str(exp_dir)])

    assert excinfo.value.code != 0
    assert "--overwrite" in capsys.readouterr().err
    assert recorded == []

    # 显式 --overwrite 后放行
    merge_adapter.main(["--exp_dir", str(exp_dir), "--overwrite"])
    assert len(recorded) == 1


def test_empty_output_dir_is_not_treated_as_existing(tmp_path, recorded):
    exp_dir, _ = _make_run(tmp_path)
    (exp_dir / "final").mkdir()

    merge_adapter.main(["--exp_dir", str(exp_dir)])

    assert len(recorded) == 1


def test_run_merge_propagates_subprocess_failure(monkeypatch):
    def _fake_run(command, check):
        assert check is True
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(merge_adapter.subprocess, "run", _fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        merge_adapter.run_merge(["python", "-m", "x"])
