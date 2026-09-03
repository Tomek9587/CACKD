#!/usr/bin/env bash
# V12: Ensemble (SAR + heatmap) + temporal conv. Full LOSO.
# vs V11 SAR tuned:
#   --decoder_type ensemble       (was sar)        -> SAR + heatmap heads, predict averaged xy
#   --use_temporal                (new)            -> 1D temporal conv kernel=3 on decoder feature
#   --seq_len 8                   (new)            -> 8-frame clips
#   --batch_size 4                (was 8)          -> 4 clips * 8 frames = 32 frames/batch (same memory)
#   --decoder_stride 4            (reverted from 2, stride=2 was worse)
#   kp_adapt_max_weight 8.0       (was 3.0)        -> allow kp0 up to 8x
#   no --kp_adapt_normalize       (was normalized) -> weights reflect true difficulty
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type ensemble --backbone resnet18 --sar_scale 16.0 \
  --sar_decode_mode softmax --sar_topk 9 --sar_temperature 0.5 \
  --use_temporal --temporal_kernel 3 --seq_len 8 \
  --ensemble_alpha 0.5 --sar_weight 1.0 \
  --batch_size 4 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

for FOLD in 0 1 2; do
  echo "===== V12 ensemble+temporal LOSO fold $FOLD ====="
  python -m surgical_tracking.src.dr_12 \
    --out_dir logs/v12_fold${FOLD}_ensemble_temporal \
    --fold $FOLD $COMMON
done

echo "===== V12 ensemble+temporal LOSO complete ====="
