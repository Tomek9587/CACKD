#!/bin/bash
# run_v7_within_seq.sh — V7 within-sequence training
# Usage: bash scripts/run_v7_within_seq.sh seq1

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

SEQ="${1:-seq1}"
OUT_DIR="logs/v7_within_${SEQ}"
MANIFEST="data/manifests/rmit_manifest.json"

python -m surgical_tracking.src.dr_7 \
    --out_dir $OUT_DIR \
    --manifest $MANIFEST \
    --protocol within_seq \
    --seq $SEQ \
    --within_seq_val_ratio 0.2 \
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
    --warmup_epochs 5 \
    --patience 20 \
    --seed 42
