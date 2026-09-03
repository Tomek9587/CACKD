#!/usr/bin/env bash
# Strict controlled ablation on LOSO fold2.
# Only architecture changes; all training hyperparameters are fixed.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}

cd "$ROOT"

FOLD=2
# Common training protocol (identical for V8 and V9 ablations)
COMMON="--protocol loso --fold $FOLD --img_size 512 --sigma 2.0 --pretrained --use_offset \
  --batch_size 12 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 20 --seed 42 \
  --adabn --aug_color 0.3 --aug_blur 0.2 --aug_noise 0.1 --aug_cutout 0.1 \
  --kp_weights 2 1 1 1"

# 1. V8 baseline (ResNet18, H/4, no ASPP)
echo "===== [1/5] V8 baseline: ResNet18 + H/4 + no ASPP ====="
python -m surgical_tracking.src.dr_8 \
  --out_dir logs/abl_fold${FOLD}_v8_baseline \
  $COMMON

# 2. V9: only backbone upgrade (ResNet34, H/4, no ASPP)
echo "===== [2/5] V9 ablation: ResNet34 + H/4 + no ASPP ====="
python -m surgical_tracking.src.dr_9 \
  --out_dir logs/abl_fold${FOLD}_v9_r34_s4_noaspp \
  $COMMON --backbone resnet34 --decoder_stride 4 --no_aspp

# 3. V9: only ASPP (ResNet18, H/4, ASPP)
echo "===== [3/5] V9 ablation: ResNet18 + H/4 + ASPP ====="
python -m surgical_tracking.src.dr_9 \
  --out_dir logs/abl_fold${FOLD}_v9_r18_s4_aspp \
  $COMMON --backbone resnet18 --decoder_stride 4

# 4. V9: only H/2 decoder (ResNet18, H/2, no ASPP)
echo "===== [4/5] V9 ablation: ResNet18 + H/2 + no ASPP ====="
python -m surgical_tracking.src.dr_9 \
  --out_dir logs/abl_fold${FOLD}_v9_r18_s2_noaspp \
  $COMMON --backbone resnet18 --decoder_stride 2 --no_aspp

# 5. V9: full model (ResNet34, H/2, ASPP)
echo "===== [5/5] V9 full: ResNet34 + H/2 + ASPP ====="
python -m surgical_tracking.src.dr_9 \
  --out_dir logs/abl_fold${FOLD}_v9_r34_s2_aspp \
  $COMMON --backbone resnet34 --decoder_stride 2
