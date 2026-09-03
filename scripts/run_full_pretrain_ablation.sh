#!/bin/bash
# run_full_pretrain_ablation.sh
# 通宵批量跑：V8 baseline / +FOVEA / +DR 在 within-seq 与 LOSO 上的完整对比，
# 外加 fold0 的 freeze-backbone 探针。

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PARENT_ROOT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PARENT_ROOT:${PYTHONPATH:-}"

mkdir -p logs

DR_CKPT="logs/pretrain_dr/best.pt"
FOVEA_CKPT="logs/pretrain_fovea/best.pt"
MANIFEST="data/manifests/rmit_manifest.json"
COMMON_ARGS=(
  --manifest "$MANIFEST"
  --img_size 512
  --sigma 2.0
  --use_offset
  --batch_size 12
  --epochs 100
  --lr 1e-4
  --wd 1e-4
  --num_workers 4
  --amp
  --grad_clip 1.0
  --warmup_epochs 5
  --patience 20
  --seed 42
)

LOG="logs/run_full_pretrain_ablation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

run_cmd() {
  echo ""
  echo "[$SECONDS] ============================================================"
  echo "[$SECONDS] RUNNING: $*"
  timeout 1800 "$@" || echo "[$SECONDS] FAILED or TIMEOUT: $?"
  echo "[$SECONDS] DONE: $*"
}

echo "[$SECONDS] Starting full pretraining ablation..."

# ------------------ Within-sequence ------------------
for SEQ in seq1 seq2 seq3; do
  run_cmd python -m surgical_tracking.src.dr_8 \
    --out_dir "logs/v8_within_${SEQ}_baseline" \
    --protocol within_seq \
    --seq "$SEQ" \
    "${COMMON_ARGS[@]}" \
    --pretrained
done

for SEQ in seq1 seq2 seq3; do
  run_cmd python -m surgical_tracking.src.dr_8 \
    --out_dir "logs/v8_within_${SEQ}_fovea" \
    --protocol within_seq \
    --seq "$SEQ" \
    "${COMMON_ARGS[@]}" \
    --pretrained \
    --pt_checkpoint "$FOVEA_CKPT"
done

for SEQ in seq1 seq2 seq3; do
  run_cmd python -m surgical_tracking.src.dr_8 \
    --out_dir "logs/v8_within_${SEQ}_dr" \
    --protocol within_seq \
    --seq "$SEQ" \
    "${COMMON_ARGS[@]}" \
    --no_pretrained \
    --dr_checkpoint "$DR_CKPT"
done

# ------------------ LOSO ------------------
for FOLD in 0 1 2; do
  run_cmd python -m surgical_tracking.src.dr_8 \
    --out_dir "logs/v8_loso_fold${FOLD}_baseline" \
    --protocol loso \
    --fold "$FOLD" \
    "${COMMON_ARGS[@]}" \
    --pretrained
done

for FOLD in 0 1 2; do
  run_cmd python -m surgical_tracking.src.dr_8 \
    --out_dir "logs/v8_loso_fold${FOLD}_fovea" \
    --protocol loso \
    --fold "$FOLD" \
    "${COMMON_ARGS[@]}" \
    --pretrained \
    --pt_checkpoint "$FOVEA_CKPT"
done

for FOLD in 0 1 2; do
  run_cmd python -m surgical_tracking.src.dr_8 \
    --out_dir "logs/v8_loso_fold${FOLD}_dr" \
    --protocol loso \
    --fold "$FOLD" \
    "${COMMON_ARGS[@]}" \
    --no_pretrained \
    --dr_checkpoint "$DR_CKPT"
done

# ------------------ Freeze-backbone probes (fold0 only) ------------------
run_cmd python -m surgical_tracking.src.dr_8 \
  --out_dir "logs/v8_loso_fold0_fovea_freeze" \
  --protocol loso \
  --fold 0 \
  "${COMMON_ARGS[@]}" \
  --pretrained \
  --pt_checkpoint "$FOVEA_CKPT" \
  --freeze_backbone

run_cmd python -m surgical_tracking.src.dr_8 \
  --out_dir "logs/v8_loso_fold0_dr_freeze" \
  --protocol loso \
  --fold 0 \
  "${COMMON_ARGS[@]}" \
  --no_pretrained \
  --dr_checkpoint "$DR_CKPT" \
  --freeze_backbone

echo ""
echo "[$SECONDS] ALL RUNS COMPLETE. Log: $LOG"
