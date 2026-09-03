#!/bin/bash
# run_v6_overfit.sh — 单 batch 过拟合测试
# 目标: 用 1 个 batch 的 4 关键点数据，验证模型能否过拟合 (loss → 0)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$SCRIPT_DIR/.."

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

OUT_DIR="logs/v6_overfit"
MANIFEST="data/manifests/rmit_manifest.json"
INIT_CKPT="${1:-}"  # 可选: Stage1 或 Stage2 的 ckpt

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
    --batch_size 4 \
    --epochs 100 \
    --rmit_lr 1e-4 \
    --wd 0.0 \
    --freeze_backbone \
    --lora_r 4 \
    --lora_rank_strategy late_heavy \
    --use_dora \
    --max_frame_gap 1 \
    --max_pairs_per_seq 8 \
    --split_mode custom \
    --exclude_seqs seq2 seq3 \
    --lambda_hm 1.0 \
    --lambda_xy 1.0 \
    --lambda_geo 1.0 \
    --warmup_epochs 0 \
    --patience 200 \
    --offset_dropout 0.1 \
    --num_vis_per_seq 2 \
    --seed 42"

if [ -n "$INIT_CKPT" ]; then
    CMD="$CMD --init_ckpt $INIT_CKPT"
fi

echo "=== V6 Overfit Test ==="
echo "Command: $CMD"
eval $CMD
