"""训练与评测入口共享的配置、运行目录和 JSON 产物工具。"""

from __future__ import annotations

import json
import os
import platform
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


def load_yaml_config(config_path: str, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """加载 YAML，并应用 ``key=value`` 形式的点号路径覆盖。"""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件顶层必须是映射: {config_path}")
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"参数覆盖必须使用 key=value: {override}")
        key, raw_value = override.split("=", 1)
        cursor = config
        parts = key.split(".")
        if any(not part for part in parts):
            raise ValueError(f"无效配置路径: {key}")
        for part in parts[:-1]:
            value = cursor.setdefault(part, {})
            if not isinstance(value, dict):
                raise ValueError(f"配置路径不是映射，无法覆盖: {key}")
            cursor = value
        cursor[parts[-1]] = yaml.safe_load(raw_value)
    return config


def require_keys(config: dict[str, Any], keys: Iterable[str]) -> None:
    """检查入口运行所需的顶层配置。"""
    missing = [key for key in keys if config.get(key) in (None, "")]
    if missing:
        raise ValueError(f"配置缺少必填参数: {', '.join(missing)}")


def resolve_experiment_dir(config: dict[str, Any]) -> Path:
    """返回标准实验目录 ``output_dir/models/exp_name``。"""
    require_keys(config, ("output_dir", "exp_name"))
    return Path(config["output_dir"]) / "models" / str(config["exp_name"])


def write_json(path: str | Path, payload: Any) -> None:
    """以 UTF-8、可读格式写 JSON。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary, output)


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    """追加一条 JSONL，适合训练日志和可恢复评测结果。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        output_file.flush()


def prepare_run_artifacts(
    run_dir: str | Path,
    config: dict[str, Any],
    config_path: str,
    overrides: Iterable[str],
) -> Path:
    """创建运行目录并保存解析后的配置、环境和命令。"""
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "config_path": str(Path(config_path).resolve()),
        "overrides": list(overrides),
        "command": " ".join(shlex.quote(arg) for arg in sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        metadata["cuda_device_count"] = torch.cuda.device_count()
        metadata["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    try:
        import transformers

        metadata["transformers"] = transformers.__version__
    except ImportError:
        metadata["transformers"] = None
    write_json(output / "run_metadata.json", metadata)
    return output


def count_parameters(model) -> dict[str, int]:
    """统计总参数与可训练参数。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total_parameters": total, "trainable_parameters": trainable}
