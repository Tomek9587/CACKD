#!/bin/bash
# Master script: Run all remaining experiments (Exp #1-4)
# #1: Cataract baseline EffV2-S (full 30-fold LOSO, current dr_11.py)
# #2: Naive aux fold1+fold2 (dr_xtransfer.py --naive_aux)
# #3: CDT scale ablation (1x, 2x, 3x=default, 5x) on fold0
# #4: CDT + LGM (v14_geo_rmit) fold2 补完

cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

echo "========================================"
echo "  Starting all remaining experiments"
echo "  Time: $(date)"
echo "========================================"

# ================================================================
# #1: Cataract baseline EffV2-S (dr_11.py, current code)
# ================================================================
echo ""
echo "===== #1: Cataract EffV2-S baseline (30-fold LOSO) ====="

CAT_MANIFEST="data/manifests/cataract_manifest.json"
CAT_COMMON="--manifest $CAT_MANIFEST --protocol loso --img_size 512 --sigma 2.0 --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

for fold in $(seq 0 29); do
  OUT="logs/v15_cat_effv2_f${fold}"
  if [ -f "$OUT/best.pt" ]; then
    echo "[SKIP] cataract EffV2 fold $fold (already done)"
    continue
  fi
  echo "[RUN] cataract EffV2 fold $fold - $(date)"
  python -m surgical_tracking.src.dr_11 --out_dir "$OUT" --fold $fold $CAT_COMMON --pretrained \
    > "logs/v15_cat_effv2_f${fold}.out" 2>&1
  echo "[DONE] cataract EffV2 fold $fold - $(date)"
done

echo "===== #1 complete ====="

# ================================================================
# #2: Naive aux fold1 + fold2 (no confidence filtering)
# ================================================================
echo ""
echo "===== #2: Naive aux fold1 + fold2 ====="

COMBINED="data/manifests/rmit_cataract_manifest.json"
AUX="data/manifests/cataract_manifest.json"

NAIVE_COMMON="--manifest $COMBINED --aux_manifest $AUX --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

for fold in 1 2; do
  BASELINE="logs/v11_fold${fold}_heatmap_offset_efficientnet_v2_s/best.pt"
  OUT="logs/cdt_naive_rmit_f${fold}"
  if [ -f "$OUT/best.pt" ]; then
    echo "[SKIP] naive fold $fold (already done)"
    continue
  fi
  echo "[RUN] Naive aux RMIT fold $fold - $(date)"
  python -m surgical_tracking.src.dr_xtransfer \
    --baseline_ckpt "$BASELINE" --naive_aux \
    --protocol loso --fold $fold --out_dir "$OUT" \
    $NAIVE_COMMON > "logs/cdt_naive_rmit_f${fold}.out" 2>&1
  echo "[DONE] naive fold $fold - $(date)"
done

echo "===== #2 complete ====="

# ================================================================
# #3: CDT scale ablation (1x, 2x, 3x, 5x on fold 0)
# ================================================================
echo ""
echo "===== #3: CDT confidence scale ablation ====="

CDT_COMMON="--manifest $COMBINED --aux_manifest $AUX --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

BASELINE_F0="logs/v11_fold0_heatmap_offset_efficientnet_v2_s/best.pt"

for SCALE in 1.0 2.0 3.0 5.0; do
  OUT="logs/cdt_scale${SCALE}_rmit_f0"
  CONF="logs/cdt_confidence_f0_s${SCALE}.json"
  if [ -f "$OUT/best.pt" ]; then
    echo "[SKIP] CDT scale=$SCALE fold0 (already done)"
    continue
  fi
  echo "[RUN] CDT scale=$SCALE fold0 - $(date)"
  python -m surgical_tracking.src.dr_xtransfer \
    --baseline_ckpt "$BASELINE_F0" --conf_cache "$CONF" \
    --conf_scale_multiplier $SCALE \
    --protocol loso --fold 0 --out_dir "$OUT" \
    $CDT_COMMON > "logs/cdt_scale${SCALE}_rmit_f0.out" 2>&1
  echo "[DONE] CDT scale=$SCALE fold0 - $(date)"
done

# Also run 3-fold for scale=3.0 (default CDT) if not already done
# (should already exist as cdt_rmit_f0/f1/f2, skip if present)

echo "===== #3 complete ====="

# ================================================================
# #4: v14_geo_rmit fold2 (LGM+KRM on RMIT)
# ================================================================
echo ""
echo "===== #4: CDT + LGM (v14_geo) RMIT fold2 ====="

GEO_COMMON="--img_size 512 --sigma 2.0 --pretrained --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --use_krm \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

OUT="logs/v14_geo_rmit_f2"
if [ -f "$OUT/best.pt" ]; then
  echo "[SKIP] v14_geo_rmit_f2 (already done)"
else
  echo "[RUN] v14_geo RMIT fold2 - $(date)"
  python -m surgical_tracking.src.dr_geo --out_dir "$OUT" \
    --manifest data/manifests/rmit_manifest.json --protocol loso --fold 2 $GEO_COMMON \
    > "logs/v14_geo_rmit_f2.out" 2>&1
  echo "[DONE] v14_geo RMIT fold2 - $(date)"
fi

echo "===== #4 complete ====="

echo ""
echo "========================================"
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Time: $(date)"
echo "========================================"
