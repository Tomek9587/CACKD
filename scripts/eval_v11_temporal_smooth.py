#!/usr/bin/env python
"""Temporal smoothing post-processing for V11 LOSO predictions."""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d

ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dr_11 import (
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    resolve_loso_split,
    load_pretrained_weights,
    adapt_bn,
)


def build_model(cfg):
    return HeatmapKeypointNet(
        img_size=cfg.img_size,
        pretrained=cfg.pretrained,
        backbone=cfg.backbone,
        decoder_channels=cfg.decoder_channels,
        decoder_stride=cfg.decoder_stride,
        use_aspp=getattr(cfg, "use_aspp", False),
        use_offset=cfg.use_offset,
        decoder_type=cfg.decoder_type,
        bcir_beta=cfg.bcir_beta,
        sar_decode_mode=getattr(cfg, "sar_decode_mode", "argmax"),
        sar_topk=getattr(cfg, "sar_topk", 0),
        sar_temperature=getattr(cfg, "sar_temperature", 1.0),
        use_kp0_hr=getattr(cfg, "use_kp0_hr", False),
    )


def smooth_gaussian(pred, sigma):
    """pred: [T,4,2] numpy. Apply 1D Gaussian filter per keypoint/coord."""
    if sigma <= 0:
        return pred
    out = np.zeros_like(pred)
    for k in range(pred.shape[1]):
        for d in range(2):
            out[:, k, d] = gaussian_filter1d(pred[:, k, d], sigma=sigma, mode="nearest")
    return out


def smooth_ema(pred, alpha):
    """Bidirectional exponential moving average (forward then backward)."""
    T = pred.shape[0]
    out = pred.copy()
    # forward
    for t in range(1, T):
        out[t] = alpha * out[t - 1] + (1 - alpha) * pred[t]
    # backward
    for t in range(T - 2, -1, -1):
        out[t] = alpha * out[t + 1] + (1 - alpha) * out[t]
    return out


def smooth_moving_avg(pred, window):
    """Simple causal moving average (past window)."""
    T = pred.shape[0]
    out = np.zeros_like(pred)
    for t in range(T):
        start = max(0, t - window + 1)
        out[t] = pred[start : t + 1].mean(axis=0)
    return out


def unnormalize_to_orig(pred_canvas, gt_canvas, scales, pads_top, pads_left):
    # pred_canvas, gt_canvas: [T,4,2] in canvas coords
    # scales, pads: 1-D arrays of length T
    scales = np.array(scales).reshape(-1, 1, 1)
    pads_left = np.array(pads_left).reshape(-1, 1, 1)
    pred = (pred_canvas - pads_left) / scales
    gt = (gt_canvas - pads_left) / scales
    return pred, gt


def compute_metrics(pred, gt):
    # pred, gt: [N,4,2] numpy arrays, original pixel coords
    dist = np.linalg.norm(pred - gt, axis=-1)
    metrics = {
        "mean_error_px": float(dist.mean()),
        "rmse_px": float(np.sqrt((dist ** 2).mean())),
        "max_error_px": float(dist.max()),
        "pck@15": float((dist < 15).mean()),
        "pck@20": float((dist < 20).mean()),
        "pck@50": float((dist < 50).mean()),
        "pck@5": float((dist < 5).mean()),
        "pck@10": float((dist < 10).mean()),
    }
    for thr in [15, 20]:
        mask = dist < thr
        if mask.any():
            metrics[f"rmse@{thr}_success_px"] = float(np.sqrt((dist[mask] ** 2).mean()))
        else:
            metrics[f"rmse@{thr}_success_px"] = float("nan")
    for k in range(4):
        d = dist[:, k]
        metrics[f"kp{k}_mean_error_px"] = float(d.mean())
        metrics[f"kp{k}_rmse_px"] = float(np.sqrt((d ** 2).mean()))
        metrics[f"kp{k}_pck@15"] = float((d < 15).mean())
        metrics[f"kp{k}_pck@20"] = float((d < 20).mean())
        metrics[f"kp{k}_pck@50"] = float((d < 50).mean())
    return metrics


def evaluate_smoothing(checkpoint, fold, device, cfg, methods):
    manifest = os.path.join(ROOT, "data/manifests/rmit_manifest.json")
    _, val_names, test_name = resolve_loso_split(manifest, fold)
    val_ds = RMITHeatmapDataset(
        manifest, cfg.img_size, val_names, False,
        sigma=cfg.sigma, kp_sigmas=getattr(cfg, "kp_sigmas", None), augment=False, within_seq_val_ratio=0.0,
    )
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(cfg).to(device)
    load_pretrained_weights(model, checkpoint)
    if cfg.adabn:
        adapt_bn(model, val_ld, device, passes=cfg.adabn_passes)
    model.eval()

    all_pred_canvas, all_gt_canvas = [], []
    all_paths = []
    with torch.no_grad():
        for batch in val_ld:
            img = batch["img"].to(device)
            pred = model(img)["xy"].cpu() * (cfg.img_size - 1)
            gt = batch["gt_xy"] * (cfg.img_size - 1)
            all_pred_canvas.append(pred)
            all_gt_canvas.append(gt)
            all_paths.extend(batch["path"])

    pred_canvas = torch.cat(all_pred_canvas, dim=0).numpy()
    gt_canvas = torch.cat(all_gt_canvas, dim=0).numpy()

    # compute letterbox scale/pad from original image sizes
    scales, pads_top, pads_left = [], [], []
    for p in all_paths:
        w, h = Image.open(p).size
        scale = min(cfg.img_size / w, cfg.img_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_left = (cfg.img_size - new_w) // 2
        pad_top = (cfg.img_size - new_h) // 2
        scales.append(scale)
        pads_top.append(pad_top)
        pads_left.append(pad_left)
    pred_orig, gt_orig = unnormalize_to_orig(pred_canvas, gt_canvas, scales, pads_top, pads_left)

    # baseline
    baseline = compute_metrics(pred_orig, gt_orig)
    results = {"none": baseline}

    for name, func in methods:
        smoothed = func(pred_orig)
        m = compute_metrics(smoothed, gt_orig)
        results[name] = m

    return results, test_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--kp_sigmas", type=float, nargs=4, default=None, help="must match training checkpoint")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s")
    parser.add_argument("--decoder_channels", type=int, nargs=3, default=[256, 128, 64])
    parser.add_argument("--decoder_stride", type=int, default=4)
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--decoder_type", type=str, default="heatmap_offset")
    parser.add_argument("--bcir_beta", type=float, default=10.0)
    parser.add_argument("--use_aspp", action="store_true", default=False)
    parser.add_argument("--use_kp0_hr", action="store_true", help="must match training checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adabn", action="store_true", default=True)
    parser.add_argument("--adabn_passes", type=int, default=1)
    parser.add_argument("--gaussian_sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    parser.add_argument("--ema_alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--ma_windows", type=int, nargs="+", default=[3, 5, 7])
    args = parser.parse_args()
    if args.no_pretrained:
        args.pretrained = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = args

    methods = [(f"gauss_s{s}", lambda p, s=s: smooth_gaussian(p, s)) for s in args.gaussian_sigmas]
    methods += [(f"ema_a{a}", lambda p, a=a: smooth_ema(p, a)) for a in args.ema_alphas]
    methods += [(f"ma_w{w}", lambda p, w=w: smooth_moving_avg(p, w)) for w in args.ma_windows]

    results, test_name = evaluate_smoothing(
        args.checkpoint, args.fold, device, cfg, methods
    )

    print(f"\nTemporal smoothing on {test_name} (fold {args.fold})")
    print(f"{'method':>12} | {'err(px)':>8} | {'PCK@15':>7} | {'PCK@20':>7} | {'PCK@50':>7}")
    print("-" * 55)
    names = ["none"] + [m[0] for m in methods]
    for name in names:
        m = results[name]
        print(f"{name:>12} | {m['mean_error_px']:8.2f} | {m['pck@15']:7.3f} | {m['pck@20']:7.3f} | {m['pck@50']:7.3f}")

    out_dir = os.path.dirname(args.checkpoint)
    out_path = os.path.join(out_dir, "temporal_smooth_results.json")
    with open(out_path, "w") as f:
        json.dump({"seq": test_name, "fold": args.fold, "results": {k: v for k, v in results.items()}}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
