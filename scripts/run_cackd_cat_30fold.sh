#!/bin/bash
# CAC-KD 30-fold LOSO on cataract (RMIT -> cataract direction)
# Reliable parallelism: 3 independent workers, each runs 10 folds serially.
# Worker A: 0,3,6,9,12,15,18,21,24,27
# Worker B: 1,4,7,10,13,16,19,22,25,28
# Worker C: 2,5,8,11,14,17,20,23,26,29
set -uo pipefail

cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

CAT_MANIFEST="data/manifests/cataract_manifest.json"
RMIT_AUX="data/manifests/rmit_manifest.json"
MASTER_LOG="logs/run_cackd_cat_30fold_master.out"

COMMON="--manifest $CAT_MANIFEST --aux_manifest $RMIT_AUX \
  --protocol loso --img_size 512 --sigma 2.0 --pretrained \
  --decoder_stride 4 --decoder_type heatmap_offset --use_offset \
  --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 \
  --amp --grad_clip 1.0 --warmup_epochs 5 --patience 40 --seed 42 \
  --geo_weight 0.5 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

run_fold() {
  local fold=$1
  local OUT="logs/cackd_cat_f${fold}"
  if [ -f "$OUT/best.pt" ]; then
    echo "[SKIP] fold $fold (done) - $(date)" >> "$MASTER_LOG"
    return 0
  fi
  local BL="logs/v15_cat_effv2_f${fold}/best.pt"
  local CONF="logs/cackd_cat_conf_f${fold}.json"
  if [ ! -f "$BL" ]; then
    echo "[ERROR] baseline missing fold $fold: $BL" >> "$MASTER_LOG"
    return 1
  fi
  echo "[RUN] fold $fold - $(date)" >> "$MASTER_LOG"
  python -m surgical_tracking.src.dr_cackd \
    --baseline_ckpt "$BL" \
    --conf_cache "$CONF" \
    --fold "$fold" \
    --out_dir "$OUT" \
    $COMMON \
    > "logs/cackd_cat_f${fold}.out" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && [ -f "$OUT/best.pt" ]; then
    echo "[DONE] fold $fold (rc=0) - $(date)" >> "$MASTER_LOG"
  else
    echo "[FAIL] fold $fold (rc=$rc) - $(date)" >> "$MASTER_LOG"
  fi
}

echo "========================================" > "$MASTER_LOG"
echo "  CAC-KD Cataract 30-fold LOSO"        >> "$MASTER_LOG"
echo "  3 independent workers (10 folds each)" >> "$MASTER_LOG"
echo "  Time: $(date)"                         >> "$MASTER_LOG"
echo "========================================" >> "$MASTER_LOG"

# Launch 3 workers in parallel, each serially processes its folds
(
  for f in 0 3 6 9 12 15 18 21 24 27; do run_fold $f; done
  echo "[WORKER A done] - $(date)" >> "$MASTER_LOG"
) &
WA=$!

(
  for f in 1 4 7 10 13 16 19 22 25 28; do run_fold $f; done
  echo "[WORKER B done] - $(date)" >> "$MASTER_LOG"
) &
WB=$!

(
  for f in 2 5 8 11 14 17 20 23 26 29; do run_fold $f; done
  echo "[WORKER C done] - $(date)" >> "$MASTER_LOG"
) &
WC=$!

echo "  Worker PIDs: A=$WA B=$WB C=$WC" >> "$MASTER_LOG"

# Wait for all workers
wait $WA; echo "[A exited]" >> "$MASTER_LOG"
wait $WB; echo "[B exited]" >> "$MASTER_LOG"
wait $WC; echo "[C exited]" >> "$MASTER_LOG"

echo "" >> "$MASTER_LOG"
echo "========================================"  >> "$MASTER_LOG"
echo "  ALL CAC-KD CATARACT FOLDS COMPLETE"      >> "$MASTER_LOG"
echo "  Time: $(date)"                            >> "$MASTER_LOG"
echo "========================================"  >> "$MASTER_LOG"

# Summary: collect best errors
echo "" >> "$MASTER_LOG"
echo "=== Results summary ===" >> "$MASTER_LOG"
python3 - <<'PYEOF'
import re, os, statistics
results = []
for f in range(30):
    out = f"logs/cackd_cat_f{f}.out"
    best = None
    if os.path.exists(out):
        txt = open(out, errors="ignore").read()
        m = re.findall(r"Best:\s*([\d.]+)\s*px", txt)
        if m:
            best = float(m[-1])
    results.append((f, best))
    status = f"{best:.2f}px" if best else "N/A"
    print(f"  fold {f:2d}: {status}")
valid = [b for _, b in results if b is not None]
if valid:
    print(f"\n  Mean: {statistics.mean(valid):.2f}px  ({len(valid)}/{len(results)} folds)")
PYEOF
