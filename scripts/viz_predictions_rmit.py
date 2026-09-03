#!/usr/bin/env python3
"""
RMIT 3-fold prediction overlay: GT vs Baseline vs CDT vs CAC-KD.

For each LOSO fold (0, 1, 2):
  1. Load baseline + CDT + CAC-KD checkpoints
  2. Run inference on test frames
  3. Select worst-error + random frames
  4. Draw overlay: GT (green) vs Baseline (blue) vs CDT (orange) vs CAC-KD (red)

Output: analysis/figs/predictions/rmit_*.png
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
from dr_cackd import CACKeypointNet, UncertaintyModulatedGeometryRefinement

# ============================================================
# Config
# ============================================================
IMG_SIZE = 512
BATCH_SIZE = 8
NUM_WORKERS = 4
MANIFEST = os.path.join(ROOT, "data/manifests/rmit_manifest.json")
OUT_DIR = os.path.join(ROOT, "analysis/figs/predictions")
os.makedirs(OUT_DIR, exist_ok=True)

FOLDS = [0, 1, 2]
N_WORST = 2
N_RANDOM = 2

# Colors
C_GT = '#00FF00'
C_BASE = '#00BFFF'
C_CDT = '#FFA500'
C_CACKD = '#FF3030'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'figure.dpi': 300,
})


def build_model():
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


def run_inference(model, loader, device):
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


def crop_bounds(pts_list, img_size, margin=80):
    all_pts = np.vstack(pts_list)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    cx1 = max(0, int(x_min - margin))
    cy1 = max(0, int(y_min - margin))
    cx2 = min(img_size, int(x_max + margin))
    cy2 = min(img_size, int(y_max + margin))
    return cx1, cy1, cx2, cy2


def draw_overlay(ax, img_np, gt, preds, title, crop=None):
    """preds: dict of {name: (4,2) array}"""
    if crop is not None:
        cx1, cy1, cx2, cy2 = crop
        img_np = img_np[cy1:cy2, cx1:cx2]
        off = np.array([cx1, cy1])
        gt = gt - off
        preds = {k: v - off for k, v in preds.items()}

    ax.imshow(img_np)
    ax.set_xticks([])
    ax.set_yticks([])

    styles = {
        'base':  (C_BASE,  'x', 70, 2),
        'cdt':   (C_CDT,  'D', 55, 0),
        'cackd': (C_CACKD, 'P', 70, 0),
    }

    # Error lines first (behind markers)
    for name, (color, _, _, _) in styles.items():
        pred = preds[name]
        for k in range(4):
            ax.plot([gt[k, 0], pred[k, 0]], [gt[k, 1], pred[k, 1]],
                    color=color, linewidth=1, alpha=0.5, zorder=3)

    # GT (on top)
    for k in range(4):
        ax.scatter(gt[k, 0], gt[k, 1], c=C_GT, s=90, marker='o',
                   edgecolors='black', linewidths=0.8, zorder=5)

    # Predictions
    for name, (color, marker, size, lw) in styles.items():
        pred = preds[name]
        edge = 'darkred' if name == 'cackd' else ('darkorange' if name == 'cdt' else 'navy')
        ax.scatter(pred[:, 0], pred[:, 1], c=color, s=size, marker=marker,
                   edgecolors=edge, linewidths=lw, zorder=4)

    ax.set_title(title, fontsize=8, pad=4)


def load_and_infer(ckpt_path, loader, device, is_cackd=False):
    """Load checkpoint, adapt BN, run inference, free model."""
    if is_cackd:
        model = build_cackd().to(device)
    else:
        model = build_model().to(device)
    load_pretrained_weights(model, ckpt_path)
    adapt_bn(model, loader, device, passes=1)
    model.eval()
    recs = run_inference(model, loader, device)
    del model
    torch.cuda.empty_cache()
    return recs


def process_fold(fold, device):
    print(f"\n{'='*60}\nProcessing RMIT fold {fold}\n{'='*60}")

    train_names, val_names, test_name = resolve_loso_split(MANIFEST, fold)
    print(f"Test sequence: {test_name}")

    ds = RMITHeatmapDataset(
        MANIFEST, IMG_SIZE, val_names, False,
        sigma=2.0, augment=False, within_seq_val_ratio=0.0,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test frames: {len(ds)}")

    # Load 3 models
    base_ckpt = os.path.join(ROOT, f"logs/v11_fold{fold}_heatmap_offset_efficientnet_v2_s/best.pt")
    cdt_ckpt = os.path.join(ROOT, f"logs/cdt_rmit_f{fold}/best.pt")
    cackd_ckpt = os.path.join(ROOT, f"logs/cac_s2_f{fold}/best.pt")

    recs_b = load_and_infer(base_ckpt, loader, device, is_cackd=False)
    recs_cdt = load_and_infer(cdt_ckpt, loader, device, is_cackd=False)
    recs_c = load_and_infer(cackd_ckpt, loader, device, is_cackd=True)

    # Merge
    merged = []
    for rb, rcdt, rc in zip(recs_b, recs_cdt, recs_c):
        merged.append({
            "path": rb["path"],
            "gt": rb["gt"],
            "pred_b": rb["pred"],
            "pred_cdt": rcdt["pred"],
            "pred_c": rc["pred"],
            "err_b": rb["err"],
            "err_cdt": rcdt["err"],
            "err_c": rc["err"],
        })

    # Sort by baseline error
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
        preds = {
            'base': rec["pred_b"],
            'cdt': rec["pred_cdt"],
            'cackd': rec["pred_c"],
        }
        crop = crop_bounds([gt, rec["pred_b"], rec["pred_cdt"], rec["pred_c"]],
                           IMG_SIZE, margin=80)

        title = (f"f{fold}({test_name}) {tag} | "
                 f"Base={rec['err_b']:.1f}  CDT={rec['err_cdt']:.1f}  "
                 f"CAC-KD={rec['err_c']:.1f}px")

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        draw_overlay(ax, img_np, gt, preds, title, crop=crop)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"rmit_f{fold}_{tag}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {out_path}")

        fold_imgs.append((tag, img_np, gt, preds, title, crop))

    return fold_imgs


def make_grid(all_fold_imgs):
    n_folds = len(all_fold_imgs)
    n_cols = N_WORST + N_RANDOM

    fig, axes = plt.subplots(n_folds, n_cols,
                            figsize=(4.5 * n_cols, 4.0 * n_folds + 1))
    if n_folds == 1:
        axes = axes[np.newaxis, :]

    for row, (fold, imgs) in enumerate(all_fold_imgs):
        for col, (tag, img_np, gt, preds, title, crop) in enumerate(imgs):
            ax = axes[row, col]
            draw_overlay(ax, img_np, gt, preds, title, crop=crop)

            if row == 0 and col == 0:
                from matplotlib.lines import Line2D
                legend = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GT,
                           markersize=8, markeredgecolor='black', label='GT'),
                    Line2D([0], [0], marker='x', color=C_BASE, markersize=8,
                           linewidth=2, label='Baseline'),
                    Line2D([0], [0], marker='D', color=C_CDT, markersize=7,
                           label='CDT'),
                    Line2D([0], [0], marker='P', color=C_CACKD, markersize=8,
                           markeredgecolor='darkred', label='CAC-KD'),
                ]
                ax.legend(handles=legend, fontsize=7, loc='lower right',
                          framealpha=0.9)

    fig.suptitle('RMIT 3-Fold LOSO: GT vs Baseline vs CDT vs CAC-KD',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = os.path.join(OUT_DIR, "rmit_prediction_grid.png")
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
    print("\nAll RMIT prediction overlays generated.")
