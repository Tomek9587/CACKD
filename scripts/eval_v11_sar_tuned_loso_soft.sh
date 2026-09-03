#!/usr/bin/env bash
# Evaluate the SAR tuned checkpoints (fold 0/1/2) with soft top-k decoding.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type sar --backbone resnet18 --sar_scale 16.0 \
  --sar_decode_mode softmax --sar_topk 9 --sar_temperature 0.5 \
  --batch_size 8 --num_workers 4 --amp \
  --adabn --adabn_passes 1 --kp_adaptive --eval_only"

for FOLD in 0 1 2; do
  echo "===== V11 SAR tuned fold $FOLD (soft top9 t0.5) ====="
  python -m surgical_tracking.src.dr_11 \
    --out_dir logs/v11_fold${FOLD}_sar_tuned_soft \
    $COMMON --fold $FOLD \
    --checkpoint logs/v11_fold${FOLD}_sar_tuned/best.pt
done

echo "===== V11 SAR tuned soft decode evaluation complete ====="
