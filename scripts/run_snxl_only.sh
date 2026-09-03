#!/bin/bash
# Run SurgeNetXL (CAFormer-S18) on cataract fold 0/10/20
# batch_size=4 to fit 3 parallel on 24GB GPU
cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

COMMON="--manifest data/manifests/cataract_manifest.json --protocol loso --img_size 512 --sigma 2.0 --decoder_stride 4 --decoder_type heatmap_offset --use_offset --batch_size 4 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 --amp --grad_clip 1.0 --warmup_epochs 5 --patience 40 --seed 42 --adabn --adabn_passes 1 --kp_adaptive --kp_adapt_momentum 0.9 --kp_adapt_max_weight 8.0 --kp_adapt_min_weight 0.3 --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1"

for fold in 0 10 20; do
  rm -rf "logs/v15_cat_snxl_f${fold}"
  echo "[RUN] snxl fold $fold"
  python src/dr_11.py --out_dir "logs/v15_cat_snxl_f${fold}" --fold $fold \
    $COMMON --backbone caformer_s18.sail_in22k_ft_in1k --surg_pretrained surgenetxl \
    > "logs/v15_cat_snxl_f${fold}.out" 2>&1 &
done

echo "3 folds started. Waiting..."
wait
echo "=== ALL DONE ==="
for f in 0 10 20; do
  echo -n "f${f}: "
  grep "Best val error" "logs/v15_cat_snxl_f${f}.out" 2>/dev/null | tail -1
done
