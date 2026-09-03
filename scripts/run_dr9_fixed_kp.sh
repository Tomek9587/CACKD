#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --sigma 2.0 --pretrained --use_offset \
  --backbone resnet34 --decoder_stride 2 \
  --batch_size 12 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 20 --seed 42 \
  --adabn --aug_color 0.3 --aug_blur 0.2 --aug_noise 0.1 --aug_cutout 0.1 \
  --kp_weights 2 1 1 1"

for fold in 0 2; do
  echo "===== V9 Fold $fold: ResNet34 + ASPP + H/2 + fixed kp=2 + AdaBN + aug ====="
  python -m surgical_tracking.src.dr_9 \
    --out_dir logs/v9_loso_fold${fold}_fixedkp2 \
    $COMMON --fold $fold
done
