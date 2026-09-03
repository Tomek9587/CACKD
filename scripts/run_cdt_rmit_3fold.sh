#!/bin/bash
# CDT 3-fold LOSO on RMIT
# Each fold: V11 baseline (RMIT-only) → confidence on cataract → train RMIT+cataract

cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

COMBINED="data/manifests/rmit_cataract_manifest.json"
AUX="data/manifests/cataract_manifest.json"

COMMON="--img_size 512 --sigma 2.0 --pretrained --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

for fold in 0 1 2; do
  BASELINE="logs/v11_fold${fold}_heatmap_offset_efficientnet_v2_s/best.pt"
  CONF="logs/cdt_confidence_f${fold}.json"
  OUT="logs/cdt_rmit_f${fold}"

  if [ -f "$OUT/best.pt" ]; then
    echo "[SKIP] fold $fold (already done)"
    continue
  fi

  echo "[RUN] CDT RMIT fold $fold"
  python -m surgical_tracking.src.dr_xtransfer \
    --manifest "$COMBINED" --aux_manifest "$AUX" \
    --baseline_ckpt "$BASELINE" --conf_cache "$CONF" \
    --protocol loso --fold $fold --out_dir "$OUT" \
    $COMMON > "logs/cdt_rmit_f${fold}.out" 2>&1
  echo "[DONE] fold $fold"
done

echo "=== CDT RMIT 3-fold complete ==="
