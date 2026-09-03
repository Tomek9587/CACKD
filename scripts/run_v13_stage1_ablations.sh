#!/bin/bash
# Run Stage 1 ablations on cataract_keypoints dataset (3-fold LOSO)
# Each config runs on fold 0, 10, 20 (representative folds)

set -e

cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=0

MANIFEST="data/manifests/cataract_manifest.json"
FOLDS="0 10 20"

COMMON="--manifest $MANIFEST --protocol loso --img_size 512 --sigma 2.0 --decoder_stride 4 \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

run() {
  local name=$1
  shift
  for fold in $FOLDS; do
    local out="logs/v13_cat_${name}_f${fold}"
    if [ -f "$out/best.pt" ]; then
      echo "[SKIP] $name fold $fold (already done)"
      continue
    fi
    echo "[RUN] $name fold $fold"
    python -m surgical_tracking.src.dr_11 --out_dir "$out" --fold $fold $COMMON "$@" \
      > "logs/v13_cat_${name}_f${fold}.out" 2>&1
    echo "[DONE] $name fold $fold"
  done
}

# 1. ResNet18 + heatmap_offset (baseline)
run "r18_hm" --backbone resnet18 --decoder_type heatmap_offset --use_offset --pretrained

# 2. ResNet18 + SAR
run "r18_sar" --backbone resnet18 --decoder_type sar --sar_scale 16.0 \
  --sar_decode_mode softmax --sar_topk 9 --sar_temperature 0.5 --pretrained

# 3. ResNet18 + heatmap only (no offset) -- NOT SUPPORTED (use_offset defaults True)
# run "r18_hmonly" --backbone resnet18 --decoder_type heatmap --pretrained

# 4. ConvNeXt-Tiny + heatmap_offset
run "cnt_hm" --backbone convnext_tiny --decoder_type heatmap_offset --use_offset --pretrained

# 5. PeskaVLP (ResNet50) + heatmap_offset
run "peska_hm" --backbone resnet50 --decoder_type heatmap_offset --use_offset \
  --pretrained --surg_pretrained peskavlp

# 6. SurgeNetXL (CAFormer-S18) + heatmap_offset
run "snxl_hm" --backbone caformer_s18.sail_in22k_ft_in1k --decoder_type heatmap_offset --use_offset \
  --surg_pretrained surgenetxl

echo "All Stage 1 ablations complete."
