#!/bin/bash
# Ablation study: Baseline → +LGM → +LGM+KRM
# Run all three on RMIT fold0, then cataract fold0
# Same parameters everywhere, NO tuning

cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

COMMON="--img_size 512 --sigma 2.0 --pretrained --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

run() {
  local out=$1; shift
  if [ -f "logs/$out/best.pt" ]; then echo "[SKIP] $out"; return; fi
  echo "[RUN] $out"
  python -m surgical_tracking.src.dr_geo --out_dir "logs/$out" "$@" > "logs/$out.out" 2>&1
  echo "[DONE] $out"
}

# === RMIT fold0 ===
echo "===== RMIT fold0 ablation ====="
# (a) Baseline: no geometry
run v14_abl_rmit_base --manifest data/manifests/rmit_manifest.json --protocol loso --fold 0 \
  $COMMON --no_lgm

# (b) +LGM: learnable geometry
run v14_abl_rmit_lgm --manifest data/manifests/rmit_manifest.json --protocol loso --fold 0 \
  $COMMON

# (c) +LGM +KRM: learnable geometry + keypoint refinement
run v14_abl_rmit_lgm_krm --manifest data/manifests/rmit_manifest.json --protocol loso --fold 0 \
  $COMMON --use_krm

# === cataract fold0 ===
echo "===== cataract fold0 ablation ====="
run v14_abl_cat_base --manifest data/manifests/cataract_manifest.json --protocol loso --fold 0 \
  $COMMON --no_lgm

run v14_abl_cat_lgm --manifest data/manifests/cataract_manifest.json --protocol loso --fold 0 \
  $COMMON

run v14_abl_cat_lgm_krm --manifest data/manifests/cataract_manifest.json --protocol loso --fold 0 \
  $COMMON --use_krm

echo "===== Ablation complete ====="
