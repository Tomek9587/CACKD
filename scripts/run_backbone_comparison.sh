#!/bin/bash
# Rerun backbone/pretrain comparison on cataract (fold 0/10/20) with V11 code
# Same protocol as 4.1 baseline (AdaBN + kp_adaptive + strong aug)
# 3 folds run in parallel per config

set -e
cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CAT_MANIFEST="data/manifests/cataract_manifest.json"
COMMON="--manifest $CAT_MANIFEST --protocol loso --img_size 512 --sigma 2.0 --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 \
  --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 \
  --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

FOLDS="0 10 20"

run_config() {
    local name=$1
    local backbone=$2
    local extra=$3

    echo ""
    echo "===== Running $name (backbone=$backbone) ====="
    echo "  Time: $(date)"

    pids=()
    for fold in $FOLDS; do
        out="logs/v15_cat_${name}_f${fold}"
        if [ -f "$out/best.pt" ]; then
            echo "  [SKIP] fold $fold (already exists)"
            continue
        fi
        echo "  [RUN] fold $fold"
        python src/dr_11.py --out_dir "$out" --fold $fold \
            $COMMON --backbone "$backbone" $extra \
            > "logs/v15_cat_${name}_f${fold}.out" 2>&1 &
        pids+=($!)
        echo "    PID=$!"
    done

    fail=0
    for pid in "${pids[@]}"; do
        if ! wait $pid; then
            fail=1
            echo "  [FAIL] PID $pid failed"
        fi
    done

    if [ $fail -eq 0 ]; then
        echo "  [DONE] $name completed at $(date)"
    else
        echo "  [WARN] $name had failures, check logs"
    fi
}

# 1. ResNet18 + ImageNet
run_config "r18" "resnet18" ""

# 2. ConvNeXt-Tiny + ImageNet
run_config "cnt" "convnext_tiny" ""

# 3. ResNet50 + PeskaVLP
run_config "peska" "resnet50" "--surg_pretrained peskavlp"

# 4. CAFormer-S18 + SurgeNetXL
run_config "snxl" "caformer_s18.sail_in22k_ft_in1k" "--surg_pretrained surgenetxl"

echo ""
echo "========================================"
echo "  All backbone comparison experiments done"
echo "  Time: $(date)"
echo "========================================"

# Print results
echo ""
echo "=== Results Summary ==="
for name in r18 cnt peska snxl; do
    for fold in $FOLDS; do
        logfile="logs/v15_cat_${name}_f${fold}.out"
        if [ -f "$logfile" ]; then
            err=$(grep -i "Best:" "$logfile" 2>/dev/null | tail -1)
            if [ -z "$err" ]; then
                err=$(grep -i "best.*err" "$logfile" 2>/dev/null | tail -1)
            fi
            echo "  ${name} f${fold}: ${err:-N/A}"
        fi
    done
done
