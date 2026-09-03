#!/usr/bin/env bash
# Evaluate V11 SAR half-split best checkpoint with soft top-k decode.
set -euo pipefail
ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
cd "$ROOT"
python -m surgical_tracking.src.dr_11 \
  --eval_only --out_dir logs/v11_sar_half_split \
  --checkpoint logs/v11_sar_half_split/best.pt \
  --protocol half --img_size 512 --sigma 2.0 \
  --decoder_stride 4 --decoder_type sar --backbone resnet18 --sar_scale 16.0 \
  --batch_size 8 --num_workers 4 \
  --adabn --adabn_passes 1 \
  --sar_decode_mode softmax --sar_topk 9 --sar_temperature 1.0
