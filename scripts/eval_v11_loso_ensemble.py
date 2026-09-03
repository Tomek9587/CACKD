#!/usr/bin/env python
"""Evaluate single-fold ensemble of sigma=3 and sigma=5 heatmap_offset models."""
import argparse
import json
import os
import sys

import torch
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
    compute_metrics,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--sigma3_ckpt", type=str, required=True)
    parser.add_argument("--sigma5_ckpt", type=str, required=True)
    parser.add_argument("--w3", type=float, default=0.7, help="weight for sigma=3 predictions")
    parser.add_argument("--w5", type=float, default=0.3, help="weight for sigma=5 predictions")
    parser.add_argument("--out_dir", type=str, required=True)
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
        if not os.path.exists(ckpt):
            raise FileNotFoundError(ckpt)
        m = build_model(args.img_size).to(device)
        load_pretrained_weights(m, ckpt)
        if args.adabn:
            adapt_bn(m, ld, device, passes=args.adabn_passes)
        m.eval()
        models[name] = m

    all_xy = {"sigma3": [], "sigma5": [], "ensemble": [], "gt": []}
    with torch.no_grad():
        for b3, b5 in zip(ld3, ld5):
            img = b3["img"].to(device)
            xy3 = models["sigma3"](img)["xy"].cpu()
            xy5 = models["sigma5"](img)["xy"].cpu()
            ens = args.w3 * xy3 + args.w5 * xy5
            all_xy["sigma3"].append(xy3)
            all_xy["sigma5"].append(xy5)
            all_xy["ensemble"].append(ens)
            all_xy["gt"].append(b3["gt_xy"])

    for k in all_xy:
        all_xy[k] = torch.cat(all_xy[k], dim=0) * (args.img_size - 1)

    results = {
        "fold": args.fold,
        "seq": test_name,
        "w3": args.w3,
        "w5": args.w5,
    }
    for name in ["sigma3", "sigma5", "ensemble"]:
        results[name] = compute_metrics(all_xy[name], all_xy["gt"], args.img_size)

    os.makedirs(args.out_dir, exist_ok=True)
    out_json = os.path.join(args.out_dir, f"fold{args.fold}_ensemble_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"=== Fold {args.fold} ({test_name}) ===")
    for name in ["sigma3", "sigma5", "ensemble"]:
        m = results[name]
        print(f"{name:10s}: err={m['mean_error_px']:.2f} pck@15={m['pck@15']:.3f} "
              f"pck@20={m['pck@20']:.3f} kp0={m['kp0_pck@15']:.3f} kp1={m['kp1_pck@15']:.3f}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
