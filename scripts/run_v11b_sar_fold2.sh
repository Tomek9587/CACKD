#!/usr/bin/env bash
# V11b SAR on LOSO fold2.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

python -m surgical_tracking.src.dr_11 \
  --out_dir logs/v11_fold2_sar \
  --protocol loso --fold 2 --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type sar \
  --backbone resnet18 \
  --sar_scale 16.0 \
  --batch_size 8 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 20 --seed 42
