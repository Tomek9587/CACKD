#!/usr/bin/env bash
# Re-evaluate V11 SAR tuned LOSO checkpoints.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type sar --backbone resnet18 --sar_scale 16.0 \
  --batch_size 8 --num_workers 4 --amp \
  --adabn --adabn_passes 1 --eval_only"

for FOLD in 0 1 2; do
  echo "===== Eval V11 SAR tuned LOSO fold $FOLD ====="
  python -m surgical_tracking.src.dr_11 \
    --out_dir logs/v11_fold${FOLD}_sar_tuned \
    --fold $FOLD $COMMON
done
