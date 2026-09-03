#!/bin/bash
# run_v6_loso.sh — 标准 LOSO 3 折交叉验证
# 用法: bash scripts/run_v6_loso.sh [INIT_CKPT] [FOLD]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

INIT_CKPT="${1:-}"
FOLD="${2:-0}"  # 0, 1, 2

OUT_DIR="logs/v6_loso_fold${FOLD}"
MANIFEST="data/manifests/rmit_manifest.json"

CMD="python -m surgical_tracking.src.dr_6 \
    --stage rmit \
    --out_dir $OUT_DIR \
    --rmit_manifest_json $MANIFEST \
    --img_size 512 \
    --patch_size 16 \
    --embed_dim 384 \
    --depth 8 \
    --heads 6 \
    --feature_layers 2,5,7 \
    --fpn_out_dim 256 \
    --batch_size 8 \
    --epochs 50 \
    --rmit_lr 3e-5 \
    --wd 0.01 \
    --freeze_backbone \
    --lora_r 4 \
    --lora_rank_strategy late_heavy \
    --use_dora \
    --use_ema \
    --ema_decay 0.999 \
    --amp \
    --grad_accum 2 \
    --max_frame_gap 3 \
    --split_mode loso \
    --fold $FOLD \
    --lambda_hm 1.0 \
    --lambda_xy 1.0 \
    --lambda_geo 1.0 \
    --warmup_epochs 3 \
    --patience 15 \
    --offset_dropout 0.2 \
    --num_vis_per_seq 5 \
    --seed 42 \
    --skip_stage2_backbone"

if [ -n "$INIT_CKPT" ]; then
    CMD="$CMD --init_ckpt $INIT_CKPT"
fi

echo "=== V6 LOSO Fold $FOLD ==="
echo "Command: $CMD"
eval $CMD
