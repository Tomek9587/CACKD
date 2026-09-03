#!/bin/bash
# Run V14 geometry-constrained method on RMIT 3-fold LOSO
# Fixed parameters - NO tuning on validation set
cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

COMMON="--img_size 512 --sigma 2.0 --pretrained --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --use_krm \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

for fold in 0 1 2; do
  out="logs/v14_geo_rmit_f${fold}"
  if [ -f "$out/best.pt" ]; then
    echo "[SKIP] fold $fold"
    continue
  fi
  echo "[RUN] RMIT fold $fold"
  python -m surgical_tracking.src.dr_geo --out_dir "$out" \
    --manifest data/manifests/rmit_manifest.json --protocol loso --fold $fold $COMMON \
    > "logs/v14_geo_rmit_f${fold}.out" 2>&1
  echo "[DONE] RMIT fold $fold"
done

echo "=== RMIT 3-fold complete ==="
