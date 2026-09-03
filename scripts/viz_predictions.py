#!/usr/bin/env python3
"""
Prediction overlay visualization: GT vs Baseline vs CAC-KD on cataract dataset.

For each selected fold:
  1. Load baseline + CAC-KD checkpoints
  2. Run inference on test frames
  3. Select worst-error + random frames
  4. Draw overlay: GT (green) vs Baseline (cyan) vs CAC-KD (red)

Output: analysis/figs/predictions/
"""

import sys
import os
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dr_11 import (
    HeatmapKeypointNet, RMITHeatmapDataset, resolve_loso_split,
    load_pretrained_weights, adapt_bn, letterbox_resize,
)
from dr_cackd import CACKeypointNet

# ============================================================
# Config
# ============================================================
IMG_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 4
MANIFEST = os.path.join(ROOT, "data/manifests/cataract_manifest.json")
OUT_DIR = os.path.join(ROOT, "analysis/figs/predictions")
os.makedirs(OUT_DIR, exist_ok=True)

# Representative folds: big improvement, best accuracy, etc.
FOLDS = [4, 3, 6, 28]
N_WORST = 2
N_RANDOM = 2

# Colors
C_GT = '#00FF00'       # green
C_BASE = '#00BFFF'     # deep sky blue
C_CACKD = '#FF3030'    # red

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'figure.dpi': 300,
})


# ============================================================
# Model builders
# ============================================================
def build_baseline():
    return HeatmapKeypointNet(
        img_size=IMG_SIZE, pretrained=False,
        backbone="efficientnet_v2_s",
        decoder_channels=[256, 128, 64],
        decoder_stride=4, use_aspp=False, use_offset=True,
        decoder_type="heatmap_offset", bcir_beta=10.0,
        sar_decode_mode="argmax", sar_topk=0, sar_temperature=1.0,
    )


def build_cackd():
    return CACKeypointNet(
        num_keypoints=4,
        img_size=IMG_SIZE, pretrained=False,
        backbone="efficientnet_v2_s",
        decoder_channels=[256, 128, 64],
        decoder_stride=4, use_aspp=False, use_offset=True,
        decoder_type="heatmap_offset", bcir_beta=10.0,
        sar_decode_mode="argmax", sar_topk=0, sar_temperature=1.0,
    )


# ============================================================
# Inference
# ============================================================
def run_inference(model, loader, device):
    """Run inference and return per-frame records."""
    records = []
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device)
            out = model(img)
            xy = out["xy"].cpu().numpy() * (IMG_SIZE - 1)
            gt = batch["gt_xy"].numpy() * (IMG_SIZE - 1)
            paths = batch["path"]
            for i in range(len(paths)):
                err = np.linalg.norm(xy[i] - gt[i], axis=-1).mean()
                records.append({
                    "path": paths[i],
                    "pred": xy[i],
                    "gt": gt[i],
                    "err": float(err),
                })
    return records


# ============================================================
# Drawing
# ============================================================
def crop_bounds(gt, pred_b, pred_c, img_size, margin=80):
    """Compute crop bounds around all keypoints."""
    all_pts = np.vstack([gt, pred_b, pred_c])
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    cx1 = max(0, int(x_min - margin))
    cy1 = max(0, int(y_min - margin))
    cx2 = min(img_size, int(x_max + margin))
    cy2 = min(img_size, int(y_max + margin))
    return cx1, cy1, cx2, cy2


def draw_overlay(ax, img_np, gt, pred_b, pred_c, title, crop=None):
    """Draw keypoint overlay on image."""
    if crop is not None:
        cx1, cy1, cx2, cy2 = crop
        img_np = img_np[cy1:cy2, cx1:cx2]
        off = np.array([cx1, cy1])
        gt = gt - off
        pred_b = pred_b - off
        pred_c = pred_c - off

    ax.imshow(img_np)
    ax.set_xticks([])
    ax.set_yticks([])

    for k in range(4):
        # GT→Baseline error line
        ax.plot([gt[k, 0], pred_b[k, 0]], [gt[k, 1], pred_b[k, 1]],
                color=C_BASE, linewidth=1.2, alpha=0.6, zorder=3)
        # GT→CAC-KD error line
        ax.plot([gt[k, 0], pred_c[k, 0]], [gt[k, 1], pred_c[k, 1]],
                color=C_CACKD, linewidth=1.2, alpha=0.6, zorder=3)

        # GT (green circle)
        ax.scatter(gt[k, 0], gt[k, 1], c=C_GT, s=90, marker='o',
                   edgecolors='black', linewidths=0.8, zorder=5)
        # Baseline (cyan x)
        ax.scatter(pred_b[k, 0], pred_b[k, 1], c=C_BASE, s=70, marker='x',
                   linewidths=2, zorder=4)
        # CAC-KD (red +)
        ax.scatter(pred_c[k, 0], pred_c[k, 1], c=C_CACKD, s=70, marker='P',
                   edgecolors='darkred', linewidths=0.5, zorder=4)

    ax.set_title(title, fontsize=8, pad=4)


# ============================================================
# Main
# ============================================================
def process_fold(fold, device):
    """Process a single fold: load models, infer, draw."""
    print(f"\n{'='*60}")
    print(f"Processing fold {fold}")
    print(f"{'='*60}")

    train_names, val_names, test_name = resolve_loso_split(MANIFEST, fold)
    print(f"Test case: {test_name}")

    # Dataset
    ds = RMITHeatmapDataset(
        MANIFEST, IMG_SIZE, val_names, False,
        sigma=2.0, augment=False, within_seq_val_ratio=0.0,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test frames: {len(ds)}")

    # --- Baseline ---
    base_ckpt = os.path.join(ROOT, f"logs/v15_cat_effv2_f{fold}/best.pt")
    model_b = build_baseline().to(device)
    load_pretrained_weights(model_b, base_ckpt)
    adapt_bn(model_b, loader, device, passes=1)
    model_b.eval()
    recs_b = run_inference(model_b, loader, device)
    del model_b
    torch.cuda.empty_cache()

    # --- CAC-KD ---
    cackd_ckpt = os.path.join(ROOT, f"logs/cackd_cat_f{fold}/best.pt")
    model_c = build_cackd().to(device)
    load_pretrained_weights(model_c, cackd_ckpt)
    adapt_bn(model_c, loader, device, passes=1)
    model_c.eval()
    recs_c = run_inference(model_c, loader, device)
    del model_c
    torch.cuda.empty_cache()

    # Merge records
    merged = []
    for rb, rc in zip(recs_b, recs_c):
        merged.append({
            "path": rb["path"],
            "gt": rb["gt"],
            "pred_b": rb["pred"],
            "pred_c": rc["pred"],
            "err_b": rb["err"],
            "err_c": rc["err"],
        })

    # Sort by baseline error (worst first)
    merged.sort(key=lambda r: r["err_b"], reverse=True)
    worst = merged[:N_WORST]

    rng = np.random.default_rng(42)
    random_idx = rng.choice(len(merged), size=min(N_RANDOM, len(merged)),
                            replace=False)
    random_recs = [merged[i] for i in random_idx]

    selected = [(f"worst{i+1}", r) for i, r in enumerate(worst)] + \
               [(f"rand{i+1}", r) for i, r in enumerate(random_recs)]

    fold_imgs = []
    for tag, rec in selected:
        img = Image.open(rec["path"]).convert("RGB")
        img_canvas, _, _, _, _ = letterbox_resize(img, IMG_SIZE)
        img_np = np.array(img_canvas)

        gt = rec["gt"]
        pred_b = rec["pred_b"]
        pred_c = rec["pred_c"]

        crop = crop_bounds(gt, pred_b, pred_c, IMG_SIZE, margin=80)

        title = (f"f{fold} {tag} | "
                 f"Base={rec['err_b']:.1f}px  CAC-KD={rec['err_c']:.1f}px")

        # Individual high-res figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        draw_overlay(ax, img_np, gt, pred_b, pred_c, title, crop=crop)
        plt.tight_layout()
        safe_tag = tag.replace(' ', '_')
        out_path = os.path.join(OUT_DIR, f"cat_f{fold}_{safe_tag}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {out_path}")

        fold_imgs.append((tag, img_np, gt, pred_b, pred_c, title, crop))

    return fold_imgs


def make_grid(all_fold_imgs):
    """Create a combined grid figure."""
    n_folds = len(all_fold_imgs)
    n_cols = N_WORST + N_RANDOM

    fig, axes = plt.subplots(n_folds, n_cols,
                            figsize=(4.2 * n_cols, 3.8 * n_folds + 1))
    if n_folds == 1:
        axes = axes[np.newaxis, :]

    for row, (fold, imgs) in enumerate(all_fold_imgs):
        for col, (tag, img_np, gt, pred_b, pred_c, title, crop) in enumerate(imgs):
            ax = axes[row, col]
            draw_overlay(ax, img_np, gt, pred_b, pred_c, title, crop=crop)

            if row == 0 and col == 0:
                # Legend
                from matplotlib.lines import Line2D
                legend = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GT,
                           markersize=8, markeredgecolor='black', label='Ground Truth'),
                    Line2D([0], [0], marker='x', color=C_BASE, markersize=8,
                           linewidth=2, label='Baseline'),
                    Line2D([0], [0], marker='P', color=C_CACKD, markersize=8,
                           markeredgecolor='darkred', label='CAC-KD'),
                ]
                ax.legend(handles=legend, fontsize=7, loc='lower right',
                          framealpha=0.9)

    fig.suptitle('Prediction Overlay: GT vs Baseline vs CAC-KD (Cataract LOSO)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(OUT_DIR, "prediction_grid.png")
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nGrid saved: {out_path}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_fold_imgs = []
    for fold in FOLDS:
        imgs = process_fold(fold, device)
        all_fold_imgs.append((fold, imgs))

    make_grid(all_fold_imgs)
    print("\nAll prediction overlays generated.")
