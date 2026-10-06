"""将一个已保存的 Beacon LoRA adapter 合并为独立推理 checkpoint。"""

from __future__ import annotations

import argparse
import json
import os

import torch

from ..models.beacon_config import BeaconConfig
from ..models.model_loader import load_beacon_model


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 Beacon LoRA adapter")
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--beacon_config_json", required=True)
    parser.add_argument(
        "--device-map",
        choices=("cpu", "auto"),
        default="cpu",
        help="合并设备映射；默认 CPU，避免训练进程结束后再次占满 GPU 显存。",
    )
    args = parser.parse_args()

    from peft import PeftModel
    from ..milestones.qwen35_text import load_text_tokenizer

    beacon_config = BeaconConfig.from_dict(json.loads(args.beacon_config_json))
    model = load_beacon_model(
        args.base_model_path,
        beacon_config=beacon_config,
        device_map=args.device_map,
    )
    beacon_state_path = os.path.join(args.adapter_path, "beacon_state.pt")
    beacon_state = torch.load(beacon_state_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(beacon_state, strict=False)
    unexpected = [name for name in unexpected if "beacon" in name]
    if unexpected:
        raise RuntimeError(f"Beacon 参数载入出现未知键: {unexpected}")
    if not any("beacon" in name for name in beacon_state):
        raise RuntimeError(f"adapter 中缺少 Beacon 参数: {beacon_state_path}")

    merged = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    merged.save_pretrained(args.output_path)
    load_text_tokenizer(args.adapter_path).save_pretrained(args.output_path)
    print(f"[beacon] 已合并 final checkpoint -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
