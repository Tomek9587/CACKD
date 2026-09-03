#!/bin/bash
# run_v7_overfit.sh — V7 overfit test on seq1 only

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

OUT_DIR="logs/v7_overfit"
MANIFEST="data/manifests/rmit_manifest.json"

python -m surgical_tracking.src.dr_7 \
    --out_dir $OUT_DIR \
    --manifest $MANIFEST \
    --img_size 512 \
    --backbone resnet18 \
    --no_pretrained \
    --hidden_dim 256 \
    --dropout 0.1 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-3 \
    --wd 0.0 \
    --num_workers 0 \
    --grad_clip 1.0 \
    --fold 0 \
    --warmup_epochs 0 \
    --patience 100 \
    --seed 42
