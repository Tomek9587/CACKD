#!/usr/bin/env python
"""
eval_with_smoothing.py - Evaluate SAR/heatmap checkpoints with test-time trajectory smoothing.

Loads a trained checkpoint, runs frame-by-frame inference in temporal order,
applies One-Euro or Savitzky-Golay smoothing to each keypoint trajectory,
and reports metrics (mean error, PCK@20, PCK@50) before and after smoothing.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src_dir)

from dr_11 import (
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    compute_metrics,
    load_pretrained_weights,
    resolve_loso_split,
    resolve_within_seq_split,
    adapt_bn,
)


# ---- One-Euro Filter ----
class OneEuroFilter1D:
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=30.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(dt, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dt = 1.0
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def smooth_one_euro(traj, min_cutoff=1.0, beta=0.007, d_cutoff=30.0):
    f = OneEuroFilter1D(min_cutoff, beta, d_cutoff)
    return np.array([f(x) for x in traj])


def smooth_sg(traj, window=7, polyorder=2):
    try:
        from scipy.signal import savgol_filter
        w = min(window, len(traj))
        if w % 2 == 0:
            w -= 1
        if w < polyorder + 2:
            return traj.copy()
        return savgol_filter(traj, w, polyorder)
    except ImportError:
        # moving average fallback
        w = min(window, len(traj))
        kernel = np.ones(w) / w
        pad = w // 2
        padded = np.pad(traj, pad, mode="edge")
        return np.convolve(padded, kernel, mode="valid")[:len(traj)]


@torch.no_grad()
def evaluate_with_smoothing(model, loader, device, img_size, smooth_method="one_euro",
                            sg_window=7, oe_min_cutoff=1.0, oe_beta=0.007):
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        img = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        pred = model(img)
        all_pred.append(pred["xy"].cpu())
        all_gt.append(gt_xy.cpu())

    all_pred = torch.cat(all_pred, dim=0) * (img_size - 1)  # [N, 4, 2]
    all_gt = torch.cat(all_gt, dim=0) * (img_size - 1)

    raw_metrics = compute_metrics(all_pred, all_gt, img_size)

    if smooth_method == "none":
        return raw_metrics, raw_metrics

    pred_np = all_pred.numpy()
    smoothed = pred_np.copy()
    N, K, _ = pred_np.shape
    for k in range(K):
        for c in range(2):
            traj = pred_np[:, k, c]
            if smooth_method == "one_euro":
                smoothed[:, k, c] = smooth_one_euro(traj, oe_min_cutoff, oe_beta)
            elif smooth_method == "sg":
                smoothed[:, k, c] = smooth_sg(traj, sg_window)
            elif smooth_method == "both":
                s1 = smooth_one_euro(traj, oe_min_cutoff, oe_beta)
                s2 = smooth_sg(traj, sg_window)
                smoothed[:, k, c] = 0.5 * (s1 + s2)

    smooth_metrics = compute_metrics(torch.from_numpy(smoothed).float(), all_gt, img_size)
    return raw_metrics, smooth_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--protocol", type=str, default="loso", choices=["loso", "within_seq"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seq", type=str, default="seq1")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # model config (must match training)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--decoder_channels", type=int, nargs=3, default=[256, 128, 64])
    parser.add_argument("--decoder_stride", type=int, default=4, choices=[2, 4])
    parser.add_argument("--decoder_type", type=str, default="sar")
    parser.add_argument("--use_aspp", action="store_true")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--bcir_beta", type=float, default=10.0)
    parser.add_argument("--sar_decode_mode", type=str, default="softmax")
    parser.add_argument("--sar_topk", type=int, default=9)
    parser.add_argument("--sar_temperature", type=float, default=0.5)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")

    # adabn
    parser.add_argument("--adabn", action="store_true")
    parser.add_argument("--adabn_passes", type=int, default=1)

    # smoothing
    parser.add_argument("--smooth_method", type=str, default="one_euro",
                        choices=["one_euro", "sg", "both", "none"])
    parser.add_argument("--sg_window", type=int, default=7)
    parser.add_argument("--oe_min_cutoff", type=float, default=1.0)
    parser.add_argument("--oe_beta", type=float, default=0.007)

    parser.add_argument("--out_json", type=str, default=None)

    args = parser.parse_args()
    if args.no_pretrained:
        args.pretrained = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HeatmapKeypointNet(
        img_size=args.img_size,
        pretrained=args.pretrained,
        backbone=args.backbone,
        decoder_channels=args.decoder_channels,
        decoder_stride=args.decoder_stride,
        use_aspp=args.use_aspp,
        use_offset=args.use_offset,
        decoder_type=args.decoder_type,
        bcir_beta=args.bcir_beta,
        sar_decode_mode=args.sar_decode_mode,
        sar_topk=args.sar_topk,
        sar_temperature=args.sar_temperature,
    ).to(device)
    load_pretrained_weights(model, args.checkpoint)

    if args.protocol == "loso":
        train_names, val_names, test_name = resolve_loso_split(args.manifest, args.fold)
    else:
        train_names, val_names, test_name = resolve_within_seq_split(args.manifest, args.seq)
    print(f"[eval] protocol={args.protocol} fold={args.fold} seq={test_name}")

    val_ratio = 0.0 if args.protocol == "loso" else 0.2
    val_ds = RMITHeatmapDataset(
        args.manifest, args.img_size, val_names, False,
        sigma=args.sigma, augment=False, within_seq_val_ratio=val_ratio,
    )
    val_ld = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if args.adabn:
        adapt_bn(model, val_ld, device, passes=args.adabn_passes)

    raw, smooth = evaluate_with_smoothing(
        model, val_ld, device, args.img_size,
        smooth_method=args.smooth_method,
        sg_window=args.sg_window,
        oe_min_cutoff=args.oe_min_cutoff,
        oe_beta=args.oe_beta,
    )

    print(f"\n=== {test_name} (smooth={args.smooth_method}) ===")
    print(f"RAW    : err={raw['mean_error_px']:.2f}px  pck@20={raw['pck@20']:.3f}  pck@50={raw['pck@50']:.3f}")
    for k in range(4):
        print(f"         kp{k}: err={raw[f'kp{k}_mean_error_px']:.2f}px  pck@20={raw[f'kp{k}_pck@20']:.3f}")
    print(f"SMOOTH : err={smooth['mean_error_px']:.2f}px  pck@20={smooth['pck@20']:.3f}  pck@50={smooth['pck@50']:.3f}")
    for k in range(4):
        print(f"         kp{k}: err={smooth[f'kp{k}_mean_error_px']:.2f}px  pck@20={smooth[f'kp{k}_pck@20']:.3f}")

    delta = raw['mean_error_px'] - smooth['mean_error_px']
    print(f"DELTA  : {delta:+.2f}px")

    if args.out_json:
        result = {
            "seq": test_name,
            "fold": args.fold,
            "smooth_method": args.smooth_method,
            "raw": raw,
            "smooth": smooth,
            "delta_px": delta,
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] saved to {args.out_json}")


if __name__ == "__main__":
    main()
