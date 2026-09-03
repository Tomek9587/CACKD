#!/bin/bash
# Cross-Dataset Transfer with Confidence-Aware Filtering
# Uses V11 baseline (RMIT-only) to compute confidence on cataract,
# then trains on RMIT + cataract (confidence-weighted)

cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

BASELINE="logs/v11_fold0_heatmap_offset_efficientnet_v2_s/best.pt"
COMBINED="data/manifests/rmit_cataract_manifest.json"
AUX="data/manifests/cataract_manifest.json"
CONF_CACHE="logs/cdt_confidence_f0.json"

COMMON="--img_size 512 --sigma 2.0 --pretrained --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

OUT="logs/cdt_rmit_f0"
if [ -f "$OUT/best.pt" ]; then
  echo "[SKIP] $OUT (already done)"
  exit 0
fi

echo "[RUN] CDT RMIT fold0"
python -m surgical_tracking.src.dr_xtransfer \
  --manifest "$COMBINED" --aux_manifest "$AUX" \
  --baseline_ckpt "$BASELINE" --conf_cache "$CONF_CACHE" \
  --protocol loso --fold 0 --out_dir "$OUT" \
  $COMMON > "logs/cdt_rmit_f0.out" 2>&1
echo "[DONE] CDT RMIT fold0"
