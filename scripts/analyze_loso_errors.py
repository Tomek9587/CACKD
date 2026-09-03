#!/usr/bin/env python
"""
analyze_loso_errors.py — LOSO failure analysis for V8 baseline.
For each fold, load best checkpoint, run inference on the held-out test sequence,
compute per-keypoint / per-frame error, save CSV, and visualize worst/best cases.
"""
import argparse
import csv
import os
from typing import Dict, List

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PROJECT_ROOT))
from surgical_tracking.src.dr_8 import (
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    load_rmit_manifest_sequences,
    resolve_loso_split,
    seed_all,
)


def draw_keypoints(img: Image.Image, pts_pred: np.ndarray, pts_gt: np.ndarray, radius: int = 4):
    """Draw green=GT, red=pred."""
    draw = ImageDraw.Draw(img)
    for k in range(pts_gt.shape[0]):
        xg, yg = pts_gt[k]
        xp, yp = pts_pred[k]
        draw.ellipse((xg - radius, yg - radius, xg + radius, yg + radius), fill=(0, 255, 0))
        draw.ellipse((xp - radius, yp - radius, xp + radius, yp + radius), fill=(255, 0, 0))
        draw.line((xg, yg, xp, yp), fill=(255, 255, 0), width=2)
    return img


@torch.no_grad()
def analyze_fold(fold: int, cfg, device: torch.device) -> Dict:
    manifest = cfg.manifest
    img_size = cfg.img_size
    train_names, val_names, test_name = resolve_loso_split(manifest, fold)
    print(f"[fold {fold}] train={sorted(train_names)}, test={test_name}")

    model = HeatmapKeypointNet(img_size=img_size, pretrained=False, use_offset=True).to(device)
    ckpt_path = os.path.join(cfg.log_dir, f"v8_loso_fold{fold}_baseline", "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = RMITHeatmapDataset(
        manifest, img_size, {test_name}, is_training=False,
        sigma=cfg.sigma, augment=False, within_seq_val_ratio=0.0,
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print(f"[fold {fold}] test samples: {len(ds)}")

    all_records: List[dict] = []
    kps = ["kp0", "kp1", "kp2", "kp3"]
    for batch in loader:
        img = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        paths = batch["path"]
        pred = model(img)["xy"]
        pred_px = pred * (img_size - 1)
        gt_px = gt_xy * (img_size - 1)
        dist = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1))  # [B, 4]
        for b in range(img.size(0)):
            rec = {
                "path": paths[b],
                "seq": test_name,
                "mean_error_px": float(dist[b].mean().item()),
                "max_error_px": float(dist[b].max().item()),
            }
            for k, name in enumerate(kps):
                rec[name] = float(dist[b, k].item())
            all_records.append(rec)

    all_records.sort(key=lambda r: r["mean_error_px"], reverse=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    csv_path = os.path.join(cfg.out_dir, f"fold{fold}_{test_name}_errors.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "seq", "mean_error_px", "max_error_px"] + kps)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"[fold {fold}] saved CSV: {csv_path}")

    # Per-keypoint mean
    per_kp = {name: float(np.mean([r[name] for r in all_records])) for name in kps}
    overall = float(np.mean([r["mean_error_px"] for r in all_records]))
    print(f"[fold {fold}] overall mean error: {overall:.2f}px")
    for name, val in per_kp.items():
        print(f"[fold {fold}] {name}: {val:.2f}px")

    # Visualize worst/best N
    vis_dir = os.path.join(cfg.out_dir, f"fold{fold}_{test_name}_viz")
    os.makedirs(vis_dir, exist_ok=True)
    for label, recs in [("worst", all_records[: cfg.top_n]), ("best", all_records[-cfg.top_n :])]:
        subdir = os.path.join(vis_dir, label)
        os.makedirs(subdir, exist_ok=True)
        for rank, rec in enumerate(recs):
            img = Image.open(rec["path"]).convert("RGB")
            # re-run single image to get coords
            # Instead, just recompute on the fly
            # (simpler: load via dataset not needed; just open and resize?)
            # We'll draw using stored errors only; for exact points, recompute below.
        # More robust: recompute for visualization by iterating dataset index
    # Actually recompute exact coords for worst/best to draw
    worst_paths = {r["path"] for r in all_records[: cfg.top_n]}
    best_paths = {r["path"] for r in all_records[-cfg.top_n :]}
    target_paths = worst_paths | best_paths
    for batch in loader:
        paths = batch["path"]
        selected = [i for i, p in enumerate(paths) if p in target_paths]
        if not selected:
            continue
        img = batch["img"][selected].to(device)
        gt_xy = batch["gt_xy"][selected].to(device)
        pred = model(img)["xy"]
        pred_px = pred * (img_size - 1)
        gt_px = gt_xy * (img_size - 1)
        for idx_in_batch, sel in enumerate(selected):
            path = paths[sel]
            img_pil = Image.open(path).convert("RGB")
            # Resize to img_size for overlay (letterbox already padded; drawing on 512x512)
            img_pil = img_pil.resize((img_size, img_size), Image.BILINEAR)
            draw_keypoints(img_pil, pred_px[idx_in_batch].cpu().numpy(), gt_px[idx_in_batch].cpu().numpy())
            mean_err = next(r["mean_error_px"] for r in all_records if r["path"] == path)
            label = "worst" if path in worst_paths else "best"
            fname = os.path.basename(path)
            out_path = os.path.join(vis_dir, label, f"{label}_{mean_err:.2f}px_{fname}")
            img_pil.save(out_path)
    print(f"[fold {fold}] visualized top/bottom {cfg.top_n} in {vis_dir}")
    return {"fold": fold, "test": test_name, "overall": overall, "per_kp": per_kp, "csv": csv_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifests/rmit_manifest.json")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--out_dir", default="logs/loso_failure_analysis")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary = []
    for fold in range(3):
        summary.append(analyze_fold(fold, args, device))

    print("\n========== Summary ==========")
    for s in summary:
        print(f"fold{s['fold']} (test {s['test']}): {s['overall']:.2f}px")
        for name, val in s["per_kp"].items():
            print(f"  {name}: {val:.2f}px")


if __name__ == "__main__":
    main()
