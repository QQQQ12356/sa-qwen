#!/usr/bin/env bash
# 保留原入口名称；训练数据统一为 Search-R1 messages，参数透传给脚本 24。
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec bash "$SCRIPT_DIR/24_beacon_train_searchr1.sh" "$@"
