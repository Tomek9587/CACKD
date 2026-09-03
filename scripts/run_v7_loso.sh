#!/bin/bash
# run_v7_loso.sh — V7 Rigid Instrument Tracker, LOSO 3-fold

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

FOLD="${1:-0}"
OUT_DIR="logs/v7_loso_fold${FOLD}"
MANIFEST="data/manifests/rmit_manifest.json"

CMD="python -m surgical_tracking.src.dr_7 \
    --out_dir $OUT_DIR \
    --manifest $MANIFEST \
    --img_size 512 \
    --backbone resnet18 \
    --pretrained \
    --hidden_dim 256 \
    --dropout 0.3 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --wd 1e-4 \
    --num_workers 4 \
    --amp \
    --grad_clip 1.0 \
    --fold $FOLD \
    --warmup_epochs 5 \
    --patience 20 \
    --seed 42"

echo "=== V7 LOSO Fold $FOLD ==="
echo "Command: $CMD"
eval $CMD
