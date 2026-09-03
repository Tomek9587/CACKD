#!/usr/bin/env bash
# V11 SAR + enhancements on full LOSO (folds 0,1,2).
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type sar --backbone resnet18 --sar_scale 16.0 \
  --batch_size 8 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 20 --seed 42 \
  --adabn --adabn_passes 1 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

for FOLD in 0 1 2; do
  echo "===== V11 SAR enhanced LOSO fold $FOLD ====="
  python -m surgical_tracking.src.dr_11 \
    --out_dir logs/v11_fold${FOLD}_sar_enhanced \
    --fold $FOLD $COMMON
done

echo "===== V11 SAR enhanced full LOSO complete ====="
