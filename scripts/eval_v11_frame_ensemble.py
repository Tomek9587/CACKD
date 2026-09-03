#!/usr/bin/env python
"""Frame-level ensemble of two V11 checkpoints (e.g. SAR + heatmap_offset)."""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dr_11 import (
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    resolve_loso_split,
    resolve_within_seq_split,
    resolve_half_split,
    load_pretrained_weights,
    adapt_bn,
    compute_metrics,
)


def build_model(cfg):
    return HeatmapKeypointNet(
        img_size=cfg.img_size,
        pretrained=cfg.pretrained,
        backbone=cfg.backbone,
        decoder_channels=cfg.decoder_channels,
        decoder_stride=cfg.decoder_stride,
        use_aspp=cfg.use_aspp,
        use_offset=cfg.use_offset,
        decoder_type=cfg.decoder_type,
        bcir_beta=cfg.bcir_beta,
        sar_decode_mode=cfg.sar_decode_mode,
        sar_topk=cfg.sar_topk,
        sar_temperature=cfg.sar_temperature,
    )


@torch.no_grad()
def evaluate_ensemble(model_a, model_b, loader, device, img_size, alphas, adabn=False, adabn_passes=1):
    if adabn:
        adapt_bn(model_a, loader, device, passes=adabn_passes)
        adapt_bn(model_b, loader, device, passes=adabn_passes)
    model_a.eval()
    model_b.eval()

    all_a, all_b, all_gt = [], [], []
    for batch in loader:
        img = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        pred_a = model_a(img)["xy"]
        pred_b = model_b(img)["xy"]
        all_a.append(pred_a.cpu())
        all_b.append(pred_b.cpu())
        all_gt.append(gt_xy.cpu())

    pred_a = torch.cat(all_a, dim=0)  # [N,4,2] in [0,1]
    pred_b = torch.cat(all_b, dim=0)
    gt = torch.cat(all_gt, dim=0)

    results = {}
    for alpha in alphas:
        fused = (1.0 - alpha) * pred_a + alpha * pred_b
        metrics = compute_metrics(fused * (img_size - 1), gt * (img_size - 1), img_size)
        results[float(alpha)] = metrics
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str,
                        default=os.path.join(ROOT, "data/manifests/rmit_manifest.json"))
    parser.add_argument("--protocol", type=str, default="loso", choices=["loso", "within_seq", "half"])
    parser.add_argument("--seq", type=str, default="seq1", choices=["seq1", "seq2", "seq3"])
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--decoder_channels", type=int, nargs=3, default=[256, 128, 64])
    parser.add_argument("--decoder_stride", type=int, default=4)
    parser.add_argument("--use_aspp", action="store_true")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--bcir_beta", type=float, default=10.0)

    # model A (usually SAR)
    parser.add_argument("--decoder_type_a", type=str, default="sar", choices=["sar", "heatmap_offset"])
    parser.add_argument("--sar_decode_mode_a", type=str, default="softmax", choices=["argmax", "softmax"])
    parser.add_argument("--sar_topk_a", type=int, default=9)
    parser.add_argument("--sar_temperature_a", type=float, default=0.5)
    parser.add_argument("--checkpoint_a", type=str, required=True)

    # model B (usually heatmap_offset)
    parser.add_argument("--decoder_type_b", type=str, default="heatmap_offset", choices=["sar", "heatmap_offset"])
    parser.add_argument("--sar_decode_mode_b", type=str, default="softmax")
    parser.add_argument("--sar_topk_b", type=int, default=9)
    parser.add_argument("--sar_temperature_b", type=float, default=0.5)
    parser.add_argument("--checkpoint_b", type=str, required=True)

    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adabn", action="store_true")
    parser.add_argument("--adabn_passes", type=int, default=1)
    args = parser.parse_args()
    if args.no_pretrained:
        args.pretrained = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.protocol == "loso":
        _, val_names, test_name = resolve_loso_split(args.manifest, args.fold)
        val_half = None
    elif args.protocol == "within_seq":
        _, val_names, test_name = resolve_within_seq_split(args.manifest, args.seq)
        val_half = None
    else:  # half
        _, val_names, test_name = resolve_half_split(args.manifest)
        val_half = "second"

    val_ds = RMITHeatmapDataset(
        args.manifest, args.img_size, val_names, False,
        sigma=args.sigma, augment=False, within_seq_val_ratio=0.0,
        half_split=val_half,
    )
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # build/load A
    cfg_a = argparse.Namespace(**vars(args))
    cfg_a.decoder_type = args.decoder_type_a
    cfg_a.sar_decode_mode = args.sar_decode_mode_a
    cfg_a.sar_topk = args.sar_topk_a
    cfg_a.sar_temperature = args.sar_temperature_a
    model_a = build_model(cfg_a).to(device)
    load_pretrained_weights(model_a, args.checkpoint_a)

    # build/load B
    cfg_b = argparse.Namespace(**vars(args))
    cfg_b.decoder_type = args.decoder_type_b
    cfg_b.sar_decode_mode = args.sar_decode_mode_b
    cfg_b.sar_topk = args.sar_topk_b
    cfg_b.sar_temperature = args.sar_temperature_b
    model_b = build_model(cfg_b).to(device)
    load_pretrained_weights(model_b, args.checkpoint_b)

    results = evaluate_ensemble(
        model_a, model_b, val_ld, device, args.img_size,
        alphas=args.alphas, adabn=args.adabn, adabn_passes=args.adabn_passes,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "frame_ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump({"seq": test_name, "alphas": {str(k): v for k, v in results.items()}}, f, indent=2)

    print(f"\nFrame-level ensemble on {test_name}")
    print(f"{'alpha':>6} | {'err(px)':>8} | {'PCK@20':>7} | {'PCK@50':>7}")
    print("-" * 40)
    for alpha in args.alphas:
        m = results[float(alpha)]
        print(f"{alpha:6.2f} | {m['mean_error_px']:8.2f} | {m['pck@20']:7.3f} | {m['pck@50']:7.3f}")


if __name__ == "__main__":
    main()
