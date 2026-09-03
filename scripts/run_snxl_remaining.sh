#!/bin/bash
# Run all remaining SurgeNetXL experiments
set -e
cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

COMMON="--img_size 512 --sigma 2.0 --decoder_stride 4 --decoder_type heatmap_offset --use_offset \
  --backbone caformer_s18.sail_in22k_ft_in1k --no_pretrained --surg_pretrained surgenetxl \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

run_one() {
  local out=$1; local manifest=$2; local fold=$3
  if [ -f "logs/$out/best.pt" ]; then echo "[SKIP] $out"; return; fi
  echo "[RUN] $out fold $fold"
  python -m surgical_tracking.src.dr_11 --out_dir "logs/$out" --manifest "$manifest" \
    --protocol loso --fold $fold $COMMON > "logs/$out.out" 2>&1
  echo "[DONE] $out"
}

# RMIT fold1, fold2 (fold0 already running)
run_one v13_rmit_snxl_f1 data/manifests/rmit_manifest.json 1
run_one v13_rmit_snxl_f2 data/manifests/rmit_manifest.json 2

# cataract fold0, 10, 20
run_one v13_cat_snxl_hm_f0 data/manifests/cataract_manifest.json 0
run_one v13_cat_snxl_hm_f10 data/manifests/cataract_manifest.json 10
run_one v13_cat_snxl_hm_f20 data/manifests/cataract_manifest.json 20

echo "All SurgeNetXL experiments complete."
