#!/usr/bin/env python
"""
eval_loso_checkpoint.py — evaluate a V8/V9 checkpoint on the full held-out LOSO sequence.
Prints overall and per-keypoint error; optionally applies AdaBN before eval.
"""
import argparse
import importlib
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_PROJECT_ROOT))


def resolve_loso_split(manifest_json, fold, module):
    sequences = module.load_rmit_manifest_sequences(manifest_json)
    names = sorted([s["name"] for s in sequences])
    test_name = names[fold % len(names)]
    return test_name


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="data/manifests/rmit_manifest.json")
    parser.add_argument("--fold", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--adabn", action="store_true")
    parser.add_argument("--adabn_passes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--src_module", type=str, default="dr_8", choices=["dr_8", "dr_9"])
    # V9 specific
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--decoder_stride", type=int, default=None, choices=[2, 4])
    parser.add_argument("--use_aspp", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    module = importlib.import_module(f"surgical_tracking.src.{args.src_module}")

    module.seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_name = resolve_loso_split(args.manifest, args.fold, module)
    net_cls = getattr(module, "HeatmapKeypointNet")
    net_kwargs = dict(img_size=args.img_size, pretrained=False, use_offset=True)
    if args.src_module == "dr_9":
        if args.backbone is not None:
            net_kwargs["backbone"] = args.backbone
        if args.decoder_stride is not None:
            net_kwargs["decoder_stride"] = args.decoder_stride
        if args.use_aspp is not None:
            net_kwargs["use_aspp"] = args.use_aspp
    model = net_cls(**net_kwargs).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Use within_seq_val_ratio=0.0 to evaluate the ENTIRE held-out sequence
    val_ds = module.RMITHeatmapDataset(
        args.manifest, args.img_size, {test_name}, is_training=False,
        sigma=args.sigma, augment=False, within_seq_val_ratio=0.0,
    )
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    if args.adabn:
        module.adapt_bn(model, val_ld, device, passes=args.adabn_passes)

    all_pred, all_gt = [], []
    for batch in val_ld:
        img = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        pred_xy = model(img)["xy"]
        all_pred.append(pred_xy.cpu())
        all_gt.append(gt_xy.cpu())

    all_pred = torch.cat(all_pred) * (args.img_size - 1)
    all_gt = torch.cat(all_gt) * (args.img_size - 1)
    dist = torch.sqrt(((all_pred - all_gt) ** 2).sum(dim=-1))  # [N, 4]

    overall = float(dist.mean().item())
    median = float(dist.median().item())
    print(f"checkpoint: {args.checkpoint}")
    print(f"fold {args.fold} test={test_name}  n={len(val_ds)}")
    print(f"overall mean error: {overall:.2f}px  median: {median:.2f}px")
    for k in range(4):
        print(f"  kp{k}: mean={dist[:, k].mean():.2f}px  median={dist[:, k].median():.2f}px  "
              f"pck5={float((dist[:, k] < 5).float().mean()):.3f}  "
              f"pck10={float((dist[:, k] < 10).float().mean()):.3f}  "
              f"pck20={float((dist[:, k] < 20).float().mean()):.3f}")
    print(f"  pck@5  = {float((dist < 5).float().mean()):.3f}")
    print(f"  pck@10 = {float((dist < 10).float().mean()):.3f}")
    print(f"  pck@20 = {float((dist < 20).float().mean()):.3f}")


if __name__ == "__main__":
    main()
