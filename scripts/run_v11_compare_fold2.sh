#!/usr/bin/env bash
# V11 controlled comparison on LOSO fold2.
# Runs sequentially: V10 heatmap+offset baseline, V11a BCIR, V11b SAR.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

echo "===== V11 baseline: heatmap+offset fold 2 ====="
bash scripts/run_v11_baseline_fold2.sh

echo "===== V11a BCIR fold 2 ====="
bash scripts/run_v11a_bcir_fold2.sh

echo "===== V11b SAR fold 2 ====="
bash scripts/run_v11b_sar_fold2.sh

echo "===== V11 fold 2 comparison complete ====="
