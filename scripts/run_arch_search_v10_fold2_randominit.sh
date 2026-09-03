#!/usr/bin/env bash
# V10 architecture search on LOSO fold2 with RANDOM initialization for backbones
# whose pretrained weights are not locally available (FastViT, RepViT, MobileNetV4).
# Clean baseline training: no AdaBN, no kp weighting, default light augmentation.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

FOLD=2
COMMON="--protocol loso --fold $FOLD --img_size 512 --sigma 2.0 --no_pretrained \
  --decoder_stride 4 --use_offset \
  --batch_size 8 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 20 --seed 42"

BACKBONES=(
  "fastvit_t8"
  "fastvit_s12"
  "repvit_m0_9"
  "mobilenetv4_conv_small"
)

for bb in "${BACKBONES[@]}"; do
  echo "===== V10 fold $FOLD backbone=$bb (random init, clean baseline) ====="
  python -m surgical_tracking.src.dr_10 \
    --out_dir logs/v10_archsearch_fold${FOLD}_${bb}_randominit \
    $COMMON --backbone $bb
done
