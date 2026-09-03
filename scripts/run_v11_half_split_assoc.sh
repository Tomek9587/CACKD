#!/usr/bin/env bash
# V11 SAR + pure-vector Association Field (PAF-style) half-split experiment.
# Freezes the SAR pipeline and only trains a larger association head on top of
# the frozen geometry-loss baseline.  Inference uses relative SAR-confidence
# gating (alpha=1.5 in assoc_refine).
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

LOG="logs/v11_half_split_assoc_v7.log"
ensure_dir() { mkdir -p "$(dirname "$1")"; }
ensure_dir "$LOG"
: > "$LOG"

COMMON="--protocol half --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type sar --backbone resnet18 \
  --batch_size 8 --epochs 30 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 1 --patience 10 --seed 42 \
  --adabn --adabn_passes 1"

SOFT_DECODE="--sar_decode_mode softmax --sar_topk 9 --sar_temperature 1.0"
GEO_CKPT="logs/v11_half_sar_geo/best.pt"

EXP_DIR=logs/v11_half_sar_geo_assoc_v7

echo "===== Training ${EXP_DIR} (freeze SAR, train AssocHead, warm-start from ${GEO_CKPT}) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --out_dir "$EXP_DIR" $COMMON \
    --use_assoc --assoc_loss_weight 10.0 --assoc_sigma_px 32.0 --freeze_sar \
    --pt_checkpoint "$GEO_CKPT" 2>&1 | tee -a "$LOG"

echo "===== Evaluating ${EXP_DIR} (argmax) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --eval_only --out_dir "$EXP_DIR" --checkpoint "$EXP_DIR/best.pt" \
    $COMMON --use_assoc --assoc_loss_weight 10.0 --assoc_sigma_px 32.0 --freeze_sar 2>&1 | tee -a "$LOG"

echo "===== Evaluating ${EXP_DIR} (soft top9) =====" | tee -a "$LOG"
python -m surgical_tracking.src.dr_11 \
    --eval_only --out_dir "$EXP_DIR" --checkpoint "$EXP_DIR/best.pt" \
    $COMMON $SOFT_DECODE --use_assoc --assoc_loss_weight 10.0 --assoc_sigma_px 32.0 --freeze_sar 2>&1 | tee -a "$LOG"

echo "===== Summary =====" | tee -a "$LOG"
grep -E '\[V11 sar\] epoch [0-9]+/[0-9]+ .*val_err=' "$LOG" | tail -20 | tee -a "$LOG"
grep -E '\[V11 eval\] half:' "$LOG" | tee -a "$LOG"
echo "===== All done =====" | tee -a "$LOG"
