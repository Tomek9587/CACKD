#!/bin/bash
cd /root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

python -m surgical_tracking.src.dr_cackd \
  --manifest data/manifests/rmit_cataract_manifest.json \
  --aux_manifest data/manifests/cataract_manifest.json \
  --baseline_ckpt logs/v11_fold0_heatmap_offset_efficientnet_v2_s/best.pt \
  --conf_cache logs/cdt_confidence_f0.json \
  --protocol loso --fold 0 \
  --out_dir logs/cackd_rmit_f0 \
  --img_size 512 --sigma 2.0 --pretrained --decoder_stride 4 \
  --decoder_type heatmap_offset --use_offset --backbone efficientnet_v2_s \
  --batch_size 8 --epochs 100 --lr 5e-5 --wd 1e-4 --num_workers 4 \
  --amp --grad_clip 1.0 --warmup_epochs 5 --patience 40 --seed 42 \
  --geo_weight 0.5 \
  --aug_color 0.3 --aug_rotate 10 --aug_blur 0.3 --aug_noise 0.05 --aug_cutout 0.1 \
  > logs/cackd_rmit_f0.out 2>&1
echo "[DONE] CAC-KD fold0"
