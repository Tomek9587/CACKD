#!/bin/bash
# run_v9_loso.sh — V9 geometry-aware heatmap tracker, leave-one-sequence-out

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

FOLD="${1:-0}"
OUT_DIR="logs/v9_loso_fold${FOLD}"
MANIFEST="data/manifests/rmit_manifest.json"

python -m surgical_tracking.src.dr_9 \
    --out_dir $OUT_DIR \
    --manifest $MANIFEST \
    --protocol loso \
    --fold $FOLD \
    --img_size 512 \
    --sigma 2.0 \
    --pretrained \
    --use_offset \
    --use_geometry \
    --w_geo 1.0 \
    --w_align 0.5 \
    --geo_warmup_epochs 10 \
    --batch_size 12 \
    --epochs 100 \
    --lr 1e-4 \
    --wd 1e-4 \
    --num_workers 4 \
    --amp \
    --grad_clip 1.0 \
    --warmup_epochs 5 \
    --patience 20 \
    --seed 42
