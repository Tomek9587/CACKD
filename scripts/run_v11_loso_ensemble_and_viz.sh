#!/usr/bin/env bash
# Full LOSO ensemble: train missing fold0/fold1 sigma3/sigma5, evaluate, aggregate, and visualize fold2.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

COMMON="--protocol loso --img_size 512 --pretrained \
  --decoder_stride 4 --decoder_type heatmap_offset --use_offset \
  --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 1e-4 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 \
  --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 3.0 --kp_adapt_min_weight 0.5 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

# 1) Train / eval missing single-sigma checkpoints for all folds.
for FOLD in 0 1 2; do
  for SIGMA in 3.0 5.0; do
    EXP_DIR="logs/v11_fold${FOLD}_heatmap_offset_efficientnet_v2_s_sigma${SIGMA}"
    LOG="${EXP_DIR}/train.log"
    mkdir -p "$EXP_DIR"

    if [ -f "$EXP_DIR/best.pt" ]; then
      echo "===== Fold ${FOLD} sigma=${SIGMA} checkpoint exists, eval only ====="
      python -m surgical_tracking.src.dr_11 \
        --eval_only --out_dir "$EXP_DIR" --checkpoint "$EXP_DIR/best.pt" \
        --fold $FOLD --sigma $SIGMA $COMMON \
        2>&1 | tee "$LOG"
    else
      echo "===== Train fold ${FOLD} sigma=${SIGMA} ====="
      python -m surgical_tracking.src.dr_11 \
        --out_dir "$EXP_DIR" --fold $FOLD --sigma $SIGMA $COMMON \
        2>&1 | tee "$LOG"

      echo "===== Eval fold ${FOLD} sigma=${SIGMA} ====="
      python -m surgical_tracking.src.dr_11 \
        --eval_only --out_dir "$EXP_DIR" --checkpoint "$EXP_DIR/best.pt" \
        --fold $FOLD --sigma $SIGMA $COMMON \
        2>&1 | tee -a "$LOG"
    fi
  done
done

# 2) Ensemble evaluation per fold.
ENSEMBLE_DIR="logs/v11_loso_ensemble"
mkdir -p "$ENSEMBLE_DIR"
for FOLD in 0 1 2; do
  echo "===== Ensemble eval fold ${FOLD} ====="
  python scripts/eval_v11_loso_ensemble.py \
    --fold $FOLD \
    --sigma3_ckpt "logs/v11_fold${FOLD}_heatmap_offset_efficientnet_v2_s_sigma3.0/best.pt" \
    --sigma5_ckpt "logs/v11_fold${FOLD}_heatmap_offset_efficientnet_v2_s_sigma5.0/best.pt" \
    --w3 0.7 --w5 0.3 \
    --out_dir "$ENSEMBLE_DIR"
done

# 3) Aggregate overall LOSO metrics.
echo "===== Aggregate LOSO ensemble metrics ====="
python - <<'PY'
import json, glob, os
files = sorted(glob.glob("logs/v11_loso_ensemble/fold*_ensemble_metrics.json"))
all_err, all_pck15, all_pck20 = [], [], []
kp_err = {k: [] for k in range(4)}
kp_pck15 = {k: [] for k in range(4)}
for f in files:
    d = json.load(open(f))
    m = d["ensemble"]
    n = len(files)
    all_err.append(m["mean_error_px"])
    all_pck15.append(m["pck@15"])
    all_pck20.append(m["pck@20"])
    for k in range(4):
        kp_err[k].append(m[f"kp{k}_mean_error_px"])
        kp_pck15[k].append(m[f"kp{k}_pck@15"])

print(f"LOSO ensemble (w3=0.7 w5=0.3): avg over {n} folds")
print(f"  mean_error_px: {sum(all_err)/n:.2f}")
print(f"  PCK@15       : {sum(all_pck15)/n:.3f}")
print(f"  PCK@20       : {sum(all_pck20)/n:.3f}")
for k in range(4):
    print(f"  kp{k} mean_error_px: {sum(kp_err[k])/n:.2f}  PCK@15: {sum(kp_pck15[k])/n:.3f}")

# Also print individual sigma averages for comparison
print("\nSingle-model averages:")
for key in ["sigma3", "sigma5"]:
    err = [json.load(open(f))[key]["mean_error_px"] for f in files]
    p15 = [json.load(open(f))[key]["pck@15"] for f in files]
    p20 = [json.load(open(f))[key]["pck@20"] for f in files]
    print(f"  {key}: mean_error={sum(err)/n:.2f} PCK@15={sum(p15)/n:.3f} PCK@20={sum(p20)/n:.3f}")
PY

# 4) Visualize fold2 (seq3) ensemble.
echo "===== Visualize fold2 ensemble ====="
python scripts/viz_v11_loso_ensemble.py \
  --fold 2 \
  --sigma3_ckpt "logs/v11_fold2_heatmap_offset_efficientnet_v2_s_sigma3.0/best.pt" \
  --sigma5_ckpt "logs/v11_fold2_heatmap_offset_efficientnet_v2_s_sigma5.0/best.pt" \
  --w3 0.7 --w5 0.3 \
  --out_dir "logs/v11_loso_ensemble/viz_seq3" \
  --num_samples 20

echo "===== LOSO ensemble + viz complete ====="
