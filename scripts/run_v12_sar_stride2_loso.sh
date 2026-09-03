#!/usr/bin/env bash
# V12: SAR with decoder_stride=2 (256x256 grid) + relaxed kp_adaptive weights (no normalization).
# Changes vs V11 SAR tuned:
#   --decoder_stride 2         (was 4) -> SAR grid 256x256, precision floor 8px -> 4px
#   --kp_adapt_max_weight 8.0  (was 3.0) -> allow kp0 to get up to 8x weight
#   --kp_adapt_min_weight 0.3  (was 0.5) -> allow easy kps to be downweighted
#   (no --kp_adapt_normalize)  (was always normalized to sum=4) -> weights now reflect true difficulty
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 2 --decoder_type sar --backbone resnet18 --sar_scale 16.0 \
  --sar_decode_mode softmax --sar_topk 9 --sar_temperature 0.5 \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

for FOLD in 0 1 2; do
  echo "===== V12 SAR stride2 LOSO fold $FOLD ====="
  python -m surgical_tracking.src.dr_11 \
    --out_dir logs/v12_fold${FOLD}_sar_stride2 \
    --fold $FOLD $COMMON
done

echo "===== V12 SAR stride2 LOSO complete ====="
