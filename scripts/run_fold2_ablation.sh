#!/usr/bin/env bash
# Fold2 ablation: isolate contribution of ensemble vs temporal.
# All three configs use dr_12.py sequence sampling (fair comparison).
# Run serially to avoid OOM.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1
cd "$ROOT"

BASE="--protocol loso --fold 2 --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --backbone resnet18 --sar_scale 16.0 \
  --sar_decode_mode softmax --sar_topk 9 --sar_temperature 0.5 \
  --seq_len 8 \
  --batch_size 4 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

echo "########## Config 1: ensemble-only (no temporal) ##########"
python -m surgical_tracking.src.dr_12 \
  --out_dir logs/v12_fold2_ensemble_only \
  --decoder_type ensemble --ensemble_alpha 0.5 --sar_weight 1.0 \
  $BASE 2>&1 | grep -E "^\[V12 ensemble|Best val|Early stopping" | tee logs/v12_fold2_ensemble_only.log

echo ""
echo "########## Config 2: SAR + temporal (no ensemble) ##########"
python -m surgical_tracking.src.dr_12 \
  --out_dir logs/v12_fold2_sar_temporal \
  --decoder_type sar --use_temporal --temporal_kernel 3 \
  $BASE 2>&1 | grep -E "^\[V12 sar|Best val|Early stopping" | tee logs/v12_fold2_sar_temporal.log

echo ""
echo "########## Config 3: SAR-only baseline (seq sampling, no temporal, no ensemble) ##########"
python -m surgical_tracking.src.dr_12 \
  --out_dir logs/v12_fold2_sar_only_seq \
  --decoder_type sar \
  $BASE 2>&1 | grep -E "^\[V12 sar|Best val|Early stopping" | tee logs/v12_fold2_sar_only_seq.log

echo ""
echo "########## Fold2 ablation complete ##########"
echo "Summary:"
for cfg in ensemble_only sar_temporal sar_only_seq; do
  f="logs/v12_fold2_${cfg}/best.pt"
  if [ -f "$f" ]; then
    python -c "import torch; c=torch.load('$f',map_location='cpu'); print(f'  {cfg}: epoch={c[\"epoch\"]} err={c[\"metric\"]:.2f}px')"
  fi
done
