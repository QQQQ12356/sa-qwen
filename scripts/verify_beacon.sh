#!/usr/bin/env bash
# 验证 Qwen3.5 Beacon 移植的正确性：
# 1) 空压缩区（all-keep）的 beacon 前向应与原生前向等价；
# 2) 压缩只作用于检索内容：文档 K/V 被丢弃、只保留 beacon，损失不含文档。
set -euo pipefail
source "$(dirname "$0")/common.sh"
PYTHON_BIN=$(resolve_python)
LOG_PATH=${LOG_PATH:-$(new_log_path verify_beacon)}

run_logged "$LOG_PATH" env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONPATH="$PWD" \
  "$PYTHON_BIN" -u - <<'PY'
import torch
from search_comp.models.model_loader import load_beacon_model
from search_comp.models.beacon_config import BeaconConfig
from search_comp.milestones.qwen35_text import load_text_tokenizer

tok = load_text_tokenizer("Qwen/Qwen3.5-2B")
bc = BeaconConfig(enable_beacon=True, beacon_window=32, beacon_stride=32, beacon_ratio=16, beacon_param="q k v")
m = load_beacon_model("Qwen/Qwen3.5-2B", beacon_config=bc).eval()

ids = tok("The capital of France is Paris and it is a beautiful city.", add_special_tokens=False, return_tensors="pt")["input_ids"].to(m.device)
labels = torch.full_like(ids, -100); labels[:, -5:] = ids[:, -5:]

m.set_beacon_config(BeaconConfig(enable_beacon=False))
native = m(input_ids=ids, labels=labels).loss.item()
m.set_beacon_config(bc)
bloss, _ = m(input_ids=ids, labels=labels, compress_regions=[])
diff = abs(bloss.item() - native)
print(f"[verify] native={native:.4f} beacon(all-keep)={bloss.item():.4f} diff={diff:.5f}")
assert diff < 0.05, "all-keep 应与原生等价"

q = tok("Who founded Google?", add_special_tokens=False).input_ids
doc = tok("Google was founded in 1998 by Larry Page and Sergey Brin while PhD students at Stanford University.", add_special_tokens=False).input_ids
ans = tok("Larry Page and Sergey Brin", add_special_tokens=False).input_ids
prefix = torch.tensor([q + doc], device=m.device)
m.prefill_and_get_cache(prefix, regions=[(len(q), len(q) + len(doc))])
assert not hasattr(m._mem, "_linear_cache"), "压缩边界不得保留原始 DynamicCache"
assert not m._mem._linear_conv, "压缩边界不得保留 information 的卷积尾状态"
assert m._mem._linear_recurrent, "线性层应提交 Beacon writer recurrent state"

ids2 = torch.tensor([q + doc + ans], device=m.device)
lab2 = torch.full_like(ids2, -100); lab2[:, len(q)+len(doc):] = ids2[:, len(q)+len(doc):]
loss, _ = m(input_ids=ids2, labels=lab2, compress_start=len(q), compress_end=len(q)+len(doc))
cache = m._mem._cache[3][0].shape[2]
expect = len(q) + (len(doc) + bc.beacon_ratio - 1)//bc.beacon_ratio + len(ans)
print(f"[verify] raw={len(ids2[0])} -> cache={cache} (expect ~{expect}); loss={loss.item():.4f}")
assert cache < len(ids2[0]), "文档 K/V 应被压缩"
assert torch.isfinite(loss), "损失应为有限值"
print("[verify] Beacon 移植正确性验证通过 ✓")
PY
