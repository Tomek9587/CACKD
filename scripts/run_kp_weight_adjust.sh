#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --sigma 2.0 --pretrained --use_offset \
  --batch_size 16 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 20 --seed 42 \
  --adabn --aug_color 0.3 --aug_blur 0.2 --aug_noise 0.1 --aug_cutout 0.1"

for fold in 0 2; do
  echo "===== Fold $fold: kp_weights=1.5 1 1 1 + AdaBN + aug (full LOSO val) ====="
  python -m surgical_tracking.src.dr_8 \
    --out_dir logs/v8_loso_fold${fold}_kp15_adabn_aug \
    $COMMON --fold $fold --kp_weights 1.5 1 1 1
done
