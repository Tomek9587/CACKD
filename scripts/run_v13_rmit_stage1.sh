#!/bin/bash
# Stage 1: Complete Table 1 data on RMIT dataset
# - V8 heatmap baseline half-split
# - Re-eval existing LOSO checkpoints for stable numbers
# - SurgNetXL / PeskaVLP pretrained on RMIT LOSO 3-fold

set -e
cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=0

MANIFEST="data/manifests/rmit_manifest.json"
COMMON="--manifest $MANIFEST --img_size 512 --sigma 2.0 --decoder_stride 4 \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

# ---- 1. V8 heatmap baseline half-split (ResNet18 + heatmap_offset) ----
if [ ! -f logs/v13_rmit_v8_half/best.pt ]; then
  echo "[RUN] V8 heatmap baseline half-split"
  python -m surgical_tracking.src.dr_11 --out_dir logs/v13_rmit_v8_half --protocol half \
    $COMMON --backbone resnet18 --decoder_type heatmap_offset --use_offset --pretrained \
    > logs/v13_rmit_v8_half.out 2>&1
  echo "[DONE] V8 heatmap baseline half-split"
else
  echo "[SKIP] V8 heatmap baseline half-split"
fi

# ---- 2. PeskaVLP (ResNet50) heatmap_offset LOSO 3-fold ----
for fold in 0 1 2; do
  out="logs/v13_rmit_peska_f${fold}"
  if [ -f "$out/best.pt" ]; then
    echo "[SKIP] PeskaVLP fold $fold"
    continue
  fi
  echo "[RUN] PeskaVLP fold $fold"
  python -m surgical_tracking.src.dr_11 --out_dir "$out" --protocol loso --fold $fold \
    $COMMON --backbone resnet50 --decoder_type heatmap_offset --use_offset --pretrained \
    --surg_pretrained peskavlp \
    > "logs/v13_rmit_peska_f${fold}.out" 2>&1
  echo "[DONE] PeskaVLP fold $fold"
done

# ---- 3. SurgeNetXL (CAFormer-S18) heatmap_offset LOSO 3-fold ----
for fold in 0 1 2; do
  out="logs/v13_rmit_snxl_f${fold}"
  if [ -f "$out/best.pt" ]; then
    echo "[SKIP] SurgeNetXL fold $fold"
    continue
  fi
  echo "[RUN] SurgeNetXL fold $fold"
  python -m surgical_tracking.src.dr_11 --out_dir "$out" --protocol loso --fold $fold \
    $COMMON --backbone caformer_s18.sail_in22k_ft_in1k --decoder_type heatmap_offset --use_offset \
    --surg_pretrained surgenetxl \
    > "logs/v13_rmit_snxl_f${fold}.out" 2>&1
  echo "[DONE] SurgeNetXL fold $fold"
done

echo "All RMIT Stage 1 experiments complete."
