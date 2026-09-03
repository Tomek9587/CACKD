#!/bin/bash
# CDT Half-split on RMIT (compare with literature: Du 4.87px, Kayhan 4.82px)
# Baseline V11 half-split: 3.76px
# Uses V11 half-split checkpoint for confidence computation on cataract

cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

BASELINE="logs/v11_half_heatmap_offset_efficientnet_v2_s/best.pt"
COMBINED="data/manifests/rmit_cataract_manifest.json"
AUX="data/manifests/cataract_manifest.json"
CONF="logs/cdt_confidence_half.json"

COMMON="--img_size 512 --sigma 2.0 --pretrained --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

OUT="logs/cdt_rmit_half"
if [ -f "$OUT/best.pt" ]; then
  echo "[SKIP] CDT half-split (already done)"
  exit 0
fi

echo "[RUN] CDT RMIT half-split"
python -m surgical_tracking.src.dr_xtransfer \
  --manifest "$COMBINED" --aux_manifest "$AUX" \
  --baseline_ckpt "$BASELINE" --conf_cache "$CONF" \
  --protocol half --fold 0 --out_dir "$OUT" \
  $COMMON > "logs/cdt_rmit_half.out" 2>&1
echo "[DONE] CDT RMIT half-split"
