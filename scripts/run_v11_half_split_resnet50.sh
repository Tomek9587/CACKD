#!/usr/bin/env bash
# V11 SAR + geometry loss + Assoc v7 on ResNet50 (half-split proof-of-concept).
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

LOG="logs/v11_half_split_resnet50.log"
ensure_dir() { mkdir -p "$(dirname "$1")"; }
ensure_dir "$LOG"
: > "$LOG"

COMMON="--protocol half --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type sar --backbone resnet50 --sar_scale 16.0 \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 3.0 --kp_adapt_min_weight 0.5 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

SOFT_DECODE="--sar_decode_mode softmax --sar_topk 9 --sar_temperature 1.0"

# ---------------------------------------------------------------------------
# 1. Train SAR + geometry loss baseline with ResNet50
# ---------------------------------------------------------------------------
GEO_DIR=logs/v11_half_sar_geo_resnet50
echo "===== Training ${GEO_DIR} (ResNet50 + SAR + geometry loss) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --out_dir "$GEO_DIR" $COMMON --geometry_loss_weight 0.1 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# 2. Freeze SAR pipeline and train AssocHead v7 from the ResNet50 geo baseline
# ---------------------------------------------------------------------------
ASSOC_DIR=logs/v11_half_sar_geo_assoc_v7_resnet50
echo "===== Training ${ASSOC_DIR} (freeze SAR, train AssocHead, warm-start from ${GEO_DIR}) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --out_dir "$ASSOC_DIR" \
    --protocol half --img_size 512 --sigma 2.0 --pretrained \
    --decoder_stride 4 --decoder_type sar --backbone resnet50 --sar_scale 16.0 \
    --batch_size 8 --epochs 30 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
    --warmup_epochs 1 --patience 10 --seed 42 \
    --adabn --adabn_passes 1 \
    --use_assoc --assoc_loss_weight 10.0 --assoc_sigma_px 32.0 --freeze_sar \
    --pt_checkpoint "$GEO_DIR/best.pt" 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# 3. Evaluate argmax and soft top9
# ---------------------------------------------------------------------------
echo "===== Evaluating ${ASSOC_DIR} (argmax) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --eval_only --out_dir "$ASSOC_DIR" --checkpoint "$ASSOC_DIR/best.pt" \
    --protocol half --img_size 512 --sigma 2.0 --pretrained \
    --decoder_stride 4 --decoder_type sar --backbone resnet50 --sar_scale 16.0 \
    --batch_size 8 --num_workers 4 \
    --adabn --adabn_passes 1 \
    --use_assoc --assoc_loss_weight 10.0 --assoc_sigma_px 32.0 --freeze_sar 2>&1 | tee -a "$LOG"

echo "===== Evaluating ${ASSOC_DIR} (soft top9) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --eval_only --out_dir "$ASSOC_DIR" --checkpoint "$ASSOC_DIR/best.pt" \
    --protocol half --img_size 512 --sigma 2.0 --pretrained \
    --decoder_stride 4 --decoder_type sar --backbone resnet50 --sar_scale 16.0 \
    --batch_size 8 --num_workers 4 \
    --adabn --adabn_passes 1 \
    --use_assoc --assoc_loss_weight 10.0 --assoc_sigma_px 32.0 --freeze_sar \
    $SOFT_DECODE 2>&1 | tee -a "$LOG"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "===== Summary =====" | tee -a "$LOG"
grep -E '\[V11 sar\] epoch [0-9]+/[0-9]+ .*val_err=' "$LOG" | tail -20 | tee -a "$LOG"
grep -E '\[V11 eval\] half:' "$LOG" | tee -a "$LOG"
echo "===== All done =====" | tee -a "$LOG"
