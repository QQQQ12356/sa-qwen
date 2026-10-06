"""标准 Transformers Trainer 的日志与产物回调。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from .runtime import append_jsonl, write_json


class JsonlMetricsCallback(TrainerCallback):
    """将 Trainer 日志实时追加到 ``trainer_metrics.jsonl``。"""

    def __init__(self, run_dir: str | Path, append: bool = False):
        self.run_dir = Path(run_dir)
        self.metrics_path = self.run_dir / "trainer_metrics.jsonl"
        self.started_at = time.time()
        self.append = append

    def on_train_begin(self, args, state, control, **kwargs):
        if self.metrics_path.exists() and not self.append:
            self.metrics_path.unlink()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record: dict[str, Any] = {
            "event": "log",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "elapsed_seconds": round(time.time() - self.started_at, 3),
        }
        record.update(logs)
        append_jsonl(self.metrics_path, record)

    def on_train_end(self, args, state, control, **kwargs):
        write_json(
            self.run_dir / "trainer_state_summary.json",
            {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
                "elapsed_seconds": round(time.time() - self.started_at, 3),
                "log_history": state.log_history,
            },
        )
