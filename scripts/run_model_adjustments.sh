#!/usr/bin/env bash
# Run model adjustment ablations for LOSO fold0 and fold2.
# Experiments:
# 1. baseline + AdaBN
# 2. kp reweighting + AdaBN + stronger augmentation
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --sigma 2.0 --pretrained --use_offset \
  --batch_size 16 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 20 --seed 42"

for fold in 0 2; do
  echo "===== Fold $fold: baseline + AdaBN ====="
  python -m surgical_tracking.src.dr_8 \
    --out_dir logs/v8_loso_fold${fold}_adabn \
    $COMMON --fold $fold --adabn

done

for fold in 0 2; do
  echo "===== Fold $fold: kp reweight + AdaBN + stronger aug ====="
  python -m surgical_tracking.src.dr_8 \
    --out_dir logs/v8_loso_fold${fold}_kp_adabn_aug \
    $COMMON --fold $fold --adabn \
    --kp_weights 2 1 1 1 \
    --aug_color 0.3 --aug_blur 0.2 --aug_noise 0.1 --aug_cutout 0.1
done
