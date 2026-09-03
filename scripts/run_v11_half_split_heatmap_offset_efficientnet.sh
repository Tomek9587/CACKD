#!/usr/bin/env bash
# Train EfficientNetV2-S heatmap+offset baseline on half-split for frame-level ensemble.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

EXP_DIR=logs/v11_half_heatmap_offset_efficientnet_v2_s
LOG="${EXP_DIR}/train.log"
mkdir -p "$EXP_DIR"

python -m surgical_tracking.src.dr_11 \
  --out_dir "$EXP_DIR" \
  --protocol half --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type heatmap_offset --use_offset \
  --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 3.0 --kp_adapt_min_weight 0.5 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1 \
  2>&1 | tee "$LOG"

python -m surgical_tracking.src.dr_11 \
  --eval_only --out_dir "$EXP_DIR" --checkpoint "$EXP_DIR/best.pt" \
  --protocol half --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type heatmap_offset --use_offset \
  --backbone efficientnet_v2_s \
  --batch_size 8 --num_workers 4 --amp \
  --adabn --adabn_passes 1 \
  2>&1 | tee -a "$LOG"

echo "===== All done =====" | tee -a "$LOG"
