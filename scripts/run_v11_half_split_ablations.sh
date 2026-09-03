#!/usr/bin/env bash
# V11 SAR half-split ablation pipeline:
#   1. baseline eval (existing best checkpoint)
#   2. train/eval with larger sar_scale=24
#   3. train/eval with geometry loss (weight=0.1)
#   4. train/eval with sar_scale=24 + geometry loss
# All use the literature-standard first-half/second-half split.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

LOG="logs/v11_half_split_ablations.log"
ensure_dir() { mkdir -p "$(dirname "$1")"; }
ensure_dir "$LOG"
: > "$LOG"

COMMON="--protocol half --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type sar --backbone resnet18 \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 3.0 --kp_adapt_min_weight 0.5 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

SOFT_DECODE="--sar_decode_mode softmax --sar_topk 9 --sar_temperature 1.0"

eval_ckpt() {
    local out_dir=$1
    local name=$2
    echo "===== Evaluating ${name} =====" | tee -a "$LOG"
    python -m surgical_tracking.src.dr_11 \
        --eval_only --out_dir "$out_dir" --checkpoint "$out_dir/best.pt" \
        $COMMON $SOFT_DECODE 2>&1 | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# 1. Baseline (already trained)
# ---------------------------------------------------------------------------
echo "===== Baseline: V11 SAR half-split best checkpoint =====" | tee -a "$LOG"
eval_ckpt logs/v11_sar_half_split "baseline"

# ---------------------------------------------------------------------------
# 2. Larger SAR kernel scale
# ---------------------------------------------------------------------------
EXP_DIR=logs/v11_half_sar_scale24
echo "===== Training ${EXP_DIR} (sar_scale=24) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --out_dir "$EXP_DIR" $COMMON --sar_scale 24.0 2>&1 | tee -a "$LOG"
eval_ckpt "$EXP_DIR" "sar_scale=24"

# ---------------------------------------------------------------------------
# 3. Skeleton geometry loss
# ---------------------------------------------------------------------------
EXP_DIR=logs/v11_half_sar_geo
echo "===== Training ${EXP_DIR} (geometry_loss_weight=0.1) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --out_dir "$EXP_DIR" $COMMON --geometry_loss_weight 0.1 2>&1 | tee -a "$LOG"
eval_ckpt "$EXP_DIR" "geometry_loss"

# ---------------------------------------------------------------------------
# 4. Combined: larger scale + geometry loss
# ---------------------------------------------------------------------------
EXP_DIR=logs/v11_half_sar_scale24_geo
echo "===== Training ${EXP_DIR} (sar_scale=24 + geometry_loss) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --out_dir "$EXP_DIR" $COMMON --sar_scale 24.0 --geometry_loss_weight 0.1 2>&1 | tee -a "$LOG"
eval_ckpt "$EXP_DIR" "sar_scale=24 + geometry_loss"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "===== Summary of mean_error_px =====" | tee -a "$LOG"
grep -E '\[V11 sar\] epoch [0-9]+/[0-9]+ .*val_err=' "$LOG" | grep -E '(baseline|sar_scale=24|geometry_loss)' || true
grep -E '\[V11 eval\] half:' "$LOG" | tee -a "$LOG"
echo "===== All done =====" | tee -a "$LOG"
