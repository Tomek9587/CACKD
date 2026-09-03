#!/usr/bin/env bash
# Fold2 (seq3) per-keypoint sigma ablation.
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

# (suffix, kp_sigmas)
confs=(
  "kp_sigmas_4333|4.0 3.0 3.0 3.0"
  "kp_sigmas_5333|5.0 3.0 3.0 3.0"
  "kp_sigmas_6333|6.0 3.0 3.0 3.0"
)

for item in "${confs[@]}"; do
  suffix="${item%%|*}"
  sigmas="${item#*|}"
  EXP_DIR="logs/v11_fold2_heatmap_offset_efficientnet_v2_s_${suffix}"
  LOG="${EXP_DIR}/train.log"
  mkdir -p "$EXP_DIR"

  echo "===== V11 fold2 ${suffix} ====="
  python -m surgical_tracking.src.dr_11 \
    --out_dir "$EXP_DIR" --fold 2 --sigma 3.0 $COMMON \
    --kp_sigmas $sigmas \
    2>&1 | tee "$LOG"

  echo "===== Eval ${suffix} ====="
  python -m surgical_tracking.src.dr_11 \
    --eval_only --out_dir "$EXP_DIR" --checkpoint "$EXP_DIR/best.pt" \
    --fold 2 --sigma 3.0 $COMMON \
    --kp_sigmas $sigmas \
    2>&1 | tee -a "$LOG"
done

echo "===== Fold2 per-keypoint sigma ablation complete ====="
