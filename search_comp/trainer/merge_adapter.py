"""把 Beacon 训练的 ``final_adapter`` 转换为合并后的 ``final``。

训练收尾会自动做这件事（``beacon_trainer._merge_final_adapter_in_subprocess``），
但重跑实验、恢复被中断的 run、或者只想换个输出目录时，需要手动再跑一次。手动拼
``--beacon_config_json`` 很麻烦：``adapter_config.json`` 里的
``base_model_name_or_path`` 是空串，基座路径与 beacon 结构只能从同目录的
``resolved_config.json`` 里读。

本模块就是这层包装：给一个实验目录，自动补齐基座路径与 beacon 配置，再委托
:mod:`search_comp.trainer.merge_beacon_lora` 执行合并（该模块本身不改动）。

用法::

    python -m search_comp.trainer.merge_adapter --exp_dir outputs/models/beacon_xxx

    # 只指向 adapter（例如训练中断的 checkpoint-N），或换输出目录
    python -m search_comp.trainer.merge_adapter \
        --adapter_path outputs/models/beacon_xxx/final_adapter \
        --output_path outputs/models/beacon_xxx/final_merged

    # resolved_config.json 缺失/需要改结构时显式覆盖
    python -m search_comp.trainer.merge_adapter --exp_dir ... \
        --base_model_path Qwen/Qwen3.5-2B --beacon_config_json '{"enable_beacon": true}'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

#: 真正执行合并的模块（``python -m`` 调用，避免改动它本身）
MERGE_MODULE = "search_comp.trainer.merge_beacon_lora"
#: 合并基座所需的 LoRA adapter 配置
ADAPTER_CONFIG_FILE = "adapter_config.json"
#: Beacon 全量参数的 state dict，只有 LoRA 训练才会产出
BEACON_STATE_FILE = "beacon_state.pt"
#: 训练时保存的解析后配置（含 model_name_or_path 与 beacon 段）
RESOLVED_CONFIG_FILE = "resolved_config.json"


class MergeAdapterError(ValueError):
    """路径推导或输入校验失败；由 :func:`main` 转成 argparse 报错退出。"""


@dataclass(frozen=True)
class MergePlan:
    """校验通过的合并方案。"""

    base_model_path: str
    beacon_config: dict
    adapter_path: Path
    output_path: Path


def build_command(plan: MergePlan, device_map: str) -> list[str]:
    """拼出调用 :mod:`search_comp.trainer.merge_beacon_lora` 的 argv。"""
    return [
        sys.executable,
        "-m",
        MERGE_MODULE,
        "--base_model_path",
        plan.base_model_path,
        "--adapter_path",
        str(plan.adapter_path),
        "--output_path",
        str(plan.output_path),
        # beacon 段作为一个 argv 元素整体传递，无需 shell 转义
        "--beacon_config_json",
        json.dumps(plan.beacon_config, ensure_ascii=False),
        "--device-map",
        device_map,
    ]


def run_merge(command: Sequence[str]) -> None:
    """拉起合并子进程；失败时抛 ``CalledProcessError``，不吞错误。"""
    subprocess.run(list(command), check=True)


# ----------------------------------------------------------------------
# 输入解析与校验
# ----------------------------------------------------------------------
def _load_run_config(exp_dir: Path) -> Optional[dict]:
    config_path = exp_dir / RESOLVED_CONFIG_FILE
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MergeAdapterError(f"{config_path} 不是合法 JSON: {error}") from error
    if not isinstance(config, dict):
        raise MergeAdapterError(f"{config_path} 顶层不是 JSON 对象")
    return config


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.adapter_path is None and args.exp_dir is None:
        raise MergeAdapterError("必须提供 --exp_dir 或 --adapter_path。")
    if args.adapter_path is not None:
        adapter_path = Path(args.adapter_path)
    else:
        adapter_path = Path(args.exp_dir) / "final_adapter"

    if args.output_path is not None:
        output_path = Path(args.output_path)
    elif args.exp_dir is not None:
        output_path = Path(args.exp_dir) / "final"
    else:
        # 只给了 adapter：输出到同级目录的 final
        output_path = adapter_path.parent / "final"
    return adapter_path, output_path


def _resolve_base_model(args: argparse.Namespace, run_config: Optional[dict]) -> str:
    if args.base_model_path:
        return args.base_model_path
    value = (run_config or {}).get("model_name_or_path")
    if not value:
        raise MergeAdapterError(
            f"无法得到基座模型路径：resolved_config.json 里没有 model_name_or_path，"
            f"请用 --base_model_path 显式指定"
        )
    return str(value)


def _resolve_beacon_config(args: argparse.Namespace, run_config: Optional[dict]) -> dict:
    if args.beacon_config_json:
        try:
            parsed = json.loads(args.beacon_config_json)
        except json.JSONDecodeError as error:
            raise MergeAdapterError(f"--beacon_config_json 不是合法 JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise MergeAdapterError("--beacon_config_json 必须是 JSON 对象")
        return parsed

    beacon = (run_config or {}).get("beacon")
    if beacon is None:
        raise MergeAdapterError(
            "无法得到 beacon 配置：resolved_config.json 里没有 beacon 段，"
            "请用 --beacon_config_json 显式指定"
        )
    if not isinstance(beacon, dict):
        raise MergeAdapterError("resolved_config.json 的 beacon 段不是 JSON 对象")
    return beacon


def _check_adapter_dir(adapter_path: Path, run_config: Optional[dict]) -> None:
    if not adapter_path.is_dir():
        raise MergeAdapterError(f"adapter 目录不存在: {adapter_path}")

    missing = [
        name
        for name in (ADAPTER_CONFIG_FILE, BEACON_STATE_FILE)
        if not (adapter_path / name).is_file()
    ]
    if not any(adapter_path.glob("adapter_model.*")):
        missing.append("adapter_model.*")
    if not missing:
        return

    if BEACON_STATE_FILE in missing and (run_config or {}).get("use_lora") is False:
        raise MergeAdapterError(
            f"{adapter_path} 缺少 {BEACON_STATE_FILE}：该 run 的 use_lora=false，"
            f"训练已直接保存全量权重（final/），没有 LoRA adapter 需要合并"
        )
    raise MergeAdapterError(f"adapter 目录不完整，缺少 {', '.join(missing)}: {adapter_path}")


def _check_output_path(output_path: Path, overwrite: bool) -> None:
    if overwrite or not output_path.exists():
        return
    if output_path.is_dir() and not any(output_path.iterdir()):
        return  # 空目录可安全复用
    raise MergeAdapterError(
        f"输出路径已存在且非空: {output_path}；确认覆盖请加 --overwrite"
    )


def resolve_plan(args: argparse.Namespace) -> MergePlan:
    """把命令行参数解析成一个校验过的 :class:`MergePlan`。"""
    adapter_path, output_path = _resolve_paths(args)
    run_config = _load_run_config(Path(args.exp_dir)) if args.exp_dir is not None else None
    if run_config is None and args.exp_dir is not None:
        if not args.base_model_path or not args.beacon_config_json:
            config_path = Path(args.exp_dir) / RESOLVED_CONFIG_FILE
            raise MergeAdapterError(
                f"缺少 {config_path}：无法自动推导基座路径与 beacon 配置。"
                f"请用 --base_model_path 与 --beacon_config_json 显式提供"
            )

    plan = MergePlan(
        base_model_path=_resolve_base_model(args, run_config),
        beacon_config=_resolve_beacon_config(args, run_config),
        adapter_path=adapter_path,
        output_path=output_path,
    )
    _check_adapter_dir(plan.adapter_path, run_config)
    _check_output_path(plan.output_path, args.overwrite)
    return plan


# ----------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 Beacon 训练的 final_adapter 合并为可推理的 final checkpoint"
    )
    parser.add_argument("--exp_dir", type=str, default=None, help="实验目录（含 resolved_config.json）")
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="adapter 目录，默认 <exp_dir>/final_adapter；也可指向 checkpoint-N",
    )
    parser.add_argument(
        "--output_path", type=str, default=None, help="合并产物目录，默认 <exp_dir>/final"
    )
    parser.add_argument(
        "--base_model_path", type=str, default=None,
        help=f"基座模型或 HF ID，默认读 {RESOLVED_CONFIG_FILE} 的 model_name_or_path",
    )
    parser.add_argument(
        "--beacon_config_json", type=str, default=None,
        help=f"beacon 配置 JSON，默认读 {RESOLVED_CONFIG_FILE} 的 beacon 段",
    )
    parser.add_argument(
        "--device-map", choices=("cpu", "auto"), default="cpu",
        help="合并设备映射；默认 CPU，避免训练结束后再次占满 GPU 显存。",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="允许覆盖已存在且非空的输出目录",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        plan = resolve_plan(args)
    except MergeAdapterError as error:
        parser.error(str(error))

    print(
        f"[merge-adapter] 合并 {plan.adapter_path} -> {plan.output_path}"
        f"（base={plan.base_model_path}, device_map={args.device_map}）",
        flush=True,
    )
    run_merge(build_command(plan, args.device_map))


if __name__ == "__main__":
    main()
