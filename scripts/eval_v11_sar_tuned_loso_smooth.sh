#!/usr/bin/env bash
# P0-1: Evaluate SAR tuned checkpoints with test-time trajectory smoothing.
# Tests One-Euro and Savitzky-Golay on all 3 LOSO folds.
set -euo pipefail

ROOT=/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking
export PYTHONPATH=/root/autodl-tmp/root/autodl-tmp/longfei:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1

cd "$ROOT"

MODEL_ARGS="--decoder_type sar --decoder_stride 4 --backbone resnet18 \
  --sar_decode_mode softmax --sar_topk 9 --sar_temperature 0.5 \
  --img_size 512 --sigma 2.0 --batch_size 8 --num_workers 4 \
  --adabn --adabn_passes 1"

echo "########## One-Euro Filter ##########"
for FOLD in 0 1 2; do
  echo "===== fold $FOLD (One-Euro) ====="
  python -m surgical_tracking.src.eval_with_smoothing \
    --manifest data/manifests/rmit_manifest.json \
    --checkpoint logs/v11_fold${FOLD}_sar_tuned/best.pt \
    --protocol loso --fold $FOLD \
    $MODEL_ARGS \
    --smooth_method one_euro \
    --out_json results/smooth_fold${FOLD}_one_euro.json
done

echo ""
echo "########## Savitzky-Golay (w=7) ##########"
for FOLD in 0 1 2; do
  echo "===== fold $FOLD (SG) ====="
  python -m surgical_tracking.src.eval_with_smoothing \
    --manifest data/manifests/rmit_manifest.json \
    --checkpoint logs/v11_fold${FOLD}_sar_tuned/best.pt \
    --protocol loso --fold $FOLD \
    $MODEL_ARGS \
    --smooth_method sg --sg_window 7 \
    --out_json results/smooth_fold${FOLD}_sg.json
done

echo ""
echo "########## Summary ##########"
python -c "
import json, glob, os
for method in ['one_euro', 'sg']:
    files = sorted(glob.glob(f'results/smooth_fold*_{method}.json'))
    if not files:
        continue
    raw_avg, sm_avg, pck20_avg = 0, 0, 0
    print(f'\n--- {method} ---')
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        r, s = d['raw'], d['smooth']
        raw_avg += r['mean_error_px']
        sm_avg += s['mean_error_px']
        pck20_avg += s['pck@20']
        print(f\"  {d['seq']}: raw={r['mean_error_px']:.2f} -> smooth={s['mean_error_px']:.2f}  (delta={d['delta_px']:+.2f})  pck@20={s['pck@20']:.3f}\")
    n = len(files)
    print(f'  AVG: raw={raw_avg/n:.2f} -> smooth={sm_avg/n:.2f}  pck@20={pck20_avg/n:.3f}')
"

echo "===== Smoothing evaluation complete ====="
