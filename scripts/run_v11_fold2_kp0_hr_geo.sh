#!/usr/bin/env bash
# Fold2 (seq3) heatmap_offset + EfficientNetV2-S with kp0 high-res head + geometry constraints.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --pretrained \
  --decoder_stride 4 --decoder_type heatmap_offset --use_offset \
  --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 3.0 --kp_adapt_min_weight 0.5 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

# Baseline from sigma sweep: sigma=3.0 gave the lowest overall error on fold2.
SIGMA=3.0
EXP_DIR="logs/v11_fold2_heatmap_offset_efficientnet_v2_s_sigma${SIGMA}_kp0hr_rigid01_geo001"
LOG="${EXP_DIR}/train.log"
mkdir -p "$EXP_DIR"

echo "===== V11 heatmap_offset EfficientNetV2-S fold2 kp0_hr + rigid_length + geometry ====="
python -m surgical_tracking.src.dr_11 \
  --out_dir "$EXP_DIR" --fold 2 --sigma $SIGMA $COMMON \
  --use_kp0_hr \
  --rigid_length_loss_weight 0.1 --rigid_length_edges "0,2;2,1;2,3;1,3" \
  --geometry_loss_weight 0.01 --geometry_loss_num_std 3.0 \
  2>&1 | tee "$LOG"

echo "===== Eval kp0_hr + geometry ====="
python -m surgical_tracking.src.dr_11 \
  --eval_only --out_dir "$EXP_DIR" --checkpoint "$EXP_DIR/best.pt" \
  --fold 2 --sigma $SIGMA $COMMON \
  --use_kp0_hr \
  --rigid_length_loss_weight 0.1 --rigid_length_edges "0,2;2,1;2,3;1,3" \
  --geometry_loss_weight 0.01 --geometry_loss_num_std 3.0 \
  2>&1 | tee -a "$LOG"

echo "===== Fold2 kp0_hr + geometry complete ====="
