"""把原生 SFT 的 LoRA adapter 合并为独立推理 checkpoint。

原生训练（``use_lora: true``）保存的 ``final/`` 里只有 adapter
（``adapter_model.safetensors`` + ``adapter_config.json``），而评测入口
``search_comp.evaluation.native_interactive_eval`` 经
:func:`search_comp.milestones.qwen35_text.load_text_causal_model` 加载完整模型，
不认识 adapter，因此评测前需先合并。

比 Beacon 版（:mod:`search_comp.trainer.merge_beacon_lora`）简单：原生模型没有
额外的 Beacon 压缩参数，单独一个 ``merge_and_unload`` 即可。

用法::

    python -m search_comp.trainer.merge_lora \
        --base_model_path Qwen/Qwen3.5-2B \
        --adapter_path outputs/models/native_qwen3_searchr1_v1/final \
        --output_path outputs/models/native_qwen3_searchr1_v1/final_merged
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="合并原生 SFT 的 LoRA adapter")
    parser.add_argument("--base_model_path", required=True, help="基础模型或 HF ID")
    parser.add_argument("--adapter_path", required=True, help="训练保存的 adapter 目录")
    parser.add_argument("--output_path", required=True, help="合并后 checkpoint 输出目录")
    parser.add_argument(
        "--device-map",
        choices=("cpu", "auto"),
        default="cpu",
        help="合并设备映射；默认 CPU，避免训练进程结束后再次占满 GPU 显存。",
    )
    args = parser.parse_args()

    from peft import PeftModel

    from ..milestones.qwen35_text import load_text_causal_model, load_text_tokenizer

    model = load_text_causal_model(args.base_model_path, device_map=args.device_map)
    merged = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    merged.save_pretrained(args.output_path)
    load_text_tokenizer(args.adapter_path).save_pretrained(args.output_path)
    print(f"[native] 已合并 LoRA adapter -> {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
