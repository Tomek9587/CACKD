#!/usr/bin/env python
"""Visualize sigma3 / sigma5 / ensemble predictions for a LOSO fold."""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dr_11 import (
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    resolve_loso_split,
    load_pretrained_weights,
    adapt_bn,
    letterbox_resize,
)


def build_model(img_size):
    return HeatmapKeypointNet(
        img_size=img_size,
        pretrained=True,
        backbone="efficientnet_v2_s",
        decoder_channels=[256, 128, 64],
        decoder_stride=4,
        use_aspp=False,
        use_offset=True,
        decoder_type="heatmap_offset",
        bcir_beta=10.0,
        sar_decode_mode="argmax",
        sar_topk=0,
        sar_temperature=1.0,
    )


def denormalize(img_tensor):
    """Reverse normalize_img: tensor [C,H,W] -> numpy [H,W,C] uint8."""
    # normalize_img in dr_11 likely does img / 255.0 then ImageNet normalization.
    # We'll just rescale to [0,1] using min/max for visualization.
    x = img_tensor.cpu().numpy().transpose(1, 2, 0)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 255).astype(np.uint8)


def draw_frame(ax, img_np, pred, gt, title):
    ax.imshow(img_np)
    colors = {"gt": "lime", "sigma3": "cyan", "sigma5": "red", "ensemble": "yellow"}
    for k in range(4):
        ax.scatter(gt[k, 0], gt[k, 1], c=colors["gt"], s=80, marker="o", edgecolors="black", linewidths=0.5, zorder=4)
        ax.scatter(pred["sigma3"][k, 0], pred["sigma3"][k, 1], c=colors["sigma3"], s=60, marker="x", linewidths=1.5, zorder=3)
        ax.scatter(pred["sigma5"][k, 0], pred["sigma5"][k, 1], c=colors["sigma5"], s=60, marker="+", linewidths=1.5, zorder=3)
        ax.scatter(pred["ensemble"][k, 0], pred["ensemble"][k, 1], c=colors["ensemble"], s=40, marker="*", zorder=5)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sigma3_ckpt", type=str, required=True)
    parser.add_argument("--sigma5_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--w3", type=float, default=0.7)
    parser.add_argument("--w5", type=float, default=0.3)
    parser.add_argument("--num_samples", type=int, default=20, help="number of worst-error frames to visualize")
    parser.add_argument("--adabn", action="store_true", default=True)
    parser.add_argument("--adabn_passes", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = os.path.join(ROOT, "data/manifests/rmit_manifest.json")
    train_names, val_names, test_name = resolve_loso_split(manifest, args.fold)

    def make_loader(sigma):
        ds = RMITHeatmapDataset(
            manifest, args.img_size, val_names, False,
            sigma=sigma, augment=False, within_seq_val_ratio=0.0,
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    ld3 = make_loader(3.0)
    ld5 = make_loader(5.0)

    models = {}
    for name, ckpt, ld in [("sigma3", args.sigma3_ckpt, ld3),
                           ("sigma5", args.sigma5_ckpt, ld5)]:
        m = build_model(args.img_size).to(device)
        load_pretrained_weights(m, ckpt)
        if args.adabn:
            adapt_bn(m, ld, device, passes=args.adabn_passes)
        m.eval()
        models[name] = m

    # collect predictions
    records = []
    with torch.no_grad():
        for b3, b5 in zip(ld3, ld5):
            img = b3["img"].to(device)
            xy3 = models["sigma3"](img)["xy"].cpu() * (args.img_size - 1)
            xy5 = models["sigma5"](img)["xy"].cpu() * (args.img_size - 1)
            ens = args.w3 * xy3 + args.w5 * xy5
            gt = b3["gt_xy"] * (args.img_size - 1)
            paths = b3["path"]
            for i in range(img.size(0)):
                err = torch.norm(ens[i] - gt[i], dim=-1).mean().item()
                records.append({
                    "path": paths[i],
                    "err": err,
                    "gt": gt[i].numpy(),
                    "sigma3": xy3[i].numpy(),
                    "sigma5": xy5[i].numpy(),
                    "ensemble": ens[i].numpy(),
                })

    os.makedirs(args.out_dir, exist_ok=True)

    # worst ensemble-error frames
    records.sort(key=lambda r: r["err"], reverse=True)
    worst = records[:args.num_samples]

    for rank, rec in enumerate(worst):
        img = Image.open(rec["path"]).convert("RGB")
        img_canvas, _, _, _, _ = letterbox_resize(img, args.img_size)
        img_np = np.array(img_canvas)
        pred = {"sigma3": rec["sigma3"], "sigma5": rec["sigma5"], "ensemble": rec["ensemble"]}
        title = f"{test_name} rank={rank+1} mean_err={rec['err']:.1f}"
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        draw_frame(ax, img_np, pred, rec["gt"], title)
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"{test_name}_worst_{rank+1:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # random sample frames
    rng = np.random.default_rng(42)
    random_recs = rng.choice(records, size=min(args.num_samples, len(records)), replace=False)
    for idx, rec in enumerate(random_recs):
        img = Image.open(rec["path"]).convert("RGB")
        img_canvas, _, _, _, _ = letterbox_resize(img, args.img_size)
        img_np = np.array(img_canvas)
        pred = {"sigma3": rec["sigma3"], "sigma5": rec["sigma5"], "ensemble": rec["ensemble"]}
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        draw_frame(ax, img_np, pred, rec["gt"], f"{test_name} random_{idx+1}")
        fig.tight_layout()
        out_path = os.path.join(args.out_dir, f"{test_name}_random_{idx+1:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {args.num_samples} worst + {len(random_recs)} random visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
