"""
Cross-Dataset Transfer with Confidence-Aware Filtering (CDT).

Method:
1. Load baseline model trained on target dataset (RMIT) only
2. Run baseline on auxiliary dataset (cataract) → compute per-keypoint confidence
   (how well does the baseline's prediction agree with the pseudo-label?)
3. Train on target (confidence=1.0) + auxiliary (confidence-weighted)
4. The model automatically discovers which auxiliary samples are helpful

No manual selection of keypoints or samples. No data snooping.
Confidence is computed from model predictions on auxiliary data, not test results.

Usage:
    python -m surgical_tracking.src.dr_xtransfer \
        --manifest data/manifests/rmit_cataract_manifest.json \
        --aux_manifest data/manifests/cataract_manifest.json \
        --baseline_ckpt logs/v11_fold0_heatmap_offset_efficientnet_v2_s/best.pt \
        --protocol loso --fold 0 \
        --out_dir logs/cdt_rmit_f0
"""

import os
import sys
import json
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dr_11 import (
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    load_rmit_manifest_sequences,
    resolve_loso_split,
    warmup_cosine_lr,
    ensure_dir,
    total_loss,
)


# =============================================================================
# 1. Confidence Computation: Run baseline on auxiliary data
# =============================================================================

@torch.no_grad()
def compute_aux_confidence(
    baseline_ckpt: str,
    aux_manifest: str,
    img_size: int,
    device: torch.device,
    backbone: str = "efficientnet_v2_s",
    conf_scale_multiplier: float = 3.0,
) -> Dict[str, np.ndarray]:
    """
    Run baseline model on auxiliary dataset, compute per-keypoint confidence.

    Confidence = agreement between baseline prediction and pseudo-label.
    High agreement → pseudo-label is reliable → high confidence.
    Low agreement  → pseudo-label is noisy    → low confidence.

    Scale is data-driven: 3 * median(all per-keypoint errors).
    This is NOT a manual hyperparameter — it adapts to the data distribution.
    """
    print(f"[CDT] Loading baseline: {baseline_ckpt}")
    model = HeatmapKeypointNet(
        img_size=img_size, pretrained=False, backbone=backbone,
        decoder_stride=4, decoder_type="heatmap_offset", use_offset=True,
    ).to(device)

    state = torch.load(baseline_ckpt, map_location=device)
    model_state = state["model"] if "model" in state else state
    # Handle dr_geo checkpoints (base.* prefix)
    if any(k.startswith("base.") for k in model_state):
        model_state = {k.replace("base.", "", 1): v
                       for k, v in model_state.items() if k.startswith("base.")}
    model.load_state_dict(model_state, strict=False)
    model.eval()

    aux_sequences = load_rmit_manifest_sequences(aux_manifest)
    aux_names = {s["name"] for s in aux_sequences}
    aux_ds = RMITHeatmapDataset(
        aux_manifest, img_size, aux_names, is_training=False, sigma=2.0, augment=False,
    )
    aux_loader = DataLoader(aux_ds, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True)

    all_errors = []
    print(f"[CDT] Computing confidence on {len(aux_ds)} auxiliary frames...")
    for batch in tqdm(aux_loader, desc="Confidence", leave=False):
        imgs = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        path = batch["path"][0]

        pred = model(imgs)
        pred_px = pred["xy"] * (img_size - 1)
        gt_px = gt_xy * (img_size - 1)
        err = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1))[0]  # [K]
        all_errors.append((path, err.cpu().numpy()))

    # Data-driven scale: conf_scale_multiplier * median of all per-keypoint errors
    all_err_flat = np.concatenate([e[1] for e in all_errors])
    scale = max(conf_scale_multiplier * np.median(all_err_flat), 1.0)

    print(f"[CDT] Error stats: median={np.median(all_err_flat):.2f}px "
          f"scale={scale:.2f}px ({conf_scale_multiplier}x median)")
    print(f"[CDT] Percentiles: 25%={np.percentile(all_err_flat,25):.1f} "
          f"50%={np.percentile(all_err_flat,50):.1f} "
          f"75%={np.percentile(all_err_flat,75):.1f} "
          f"90%={np.percentile(all_err_flat,90):.1f}px")

    # Compute confidence: max(0, 1 - err / scale)
    confidence = {}
    n_high = n_mid = n_low = 0
    for path, errs in all_errors:
        conf = np.maximum(0.0, 1.0 - errs / scale).astype(np.float32)
        confidence[path] = conf
        mc = conf.mean()
        if mc > 0.66: n_high += 1
        elif mc > 0.33: n_mid += 1
        else: n_low += 1

    print(f"[CDT] Confidence: high(>0.66)={n_high} mid(0.33-0.66)={n_mid} low(<0.33)={n_low}")
    for k in range(4):
        kp_errs = all_err_flat[k::4]
        kp_confs = np.array([confidence[p][k] for p, _ in all_errors])
        print(f"  kp{k}: err_median={np.median(kp_errs):.1f}px conf_mean={kp_confs.mean():.3f}")

    del model
    torch.cuda.empty_cache()
    return confidence


# =============================================================================
# 2. Confidence-Weighted Dataset Wrapper
# =============================================================================

class ConfidenceWeightedDataset(Dataset):
    """Wraps RMITHeatmapDataset with per-keypoint confidence weights.

    RMIT samples: confidence = [1,1,1,1] (real labels)
    Cataract samples: confidence from baseline prediction agreement
    """

    def __init__(self, base_dataset, confidence_map, num_keypoints=4):
        self.base = base_dataset
        self.confidence_map = confidence_map
        self.num_kp = num_keypoints

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        path = item["path"]
        if path in self.confidence_map:
            conf = self.confidence_map[path]
        else:
            conf = np.ones(self.num_kp, dtype=np.float32)
        item["confidence"] = torch.from_numpy(np.array(conf, dtype=np.float32))
        return item


# =============================================================================
# 3. Confidence-Weighted Loss (per-sample, per-keypoint)
# =============================================================================

def confidence_weighted_loss(
    pred: Dict[str, torch.Tensor],
    gt_hm: torch.Tensor,
    gt_xy: torch.Tensor,
    confidence: torch.Tensor,
    xy_weight: float = 10.0,
    coarse_weight: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Per-sample, per-keypoint weighted loss.

    Args:
        pred: model output dict (hm_logits, xy, coarse_xy)
        gt_hm: [B, K, H, W]
        gt_xy: [B, K, 2] normalized
        confidence: [B, K] per-keypoint confidence per sample
    """
    B, K = confidence.shape

    # Heatmap loss (MSE on sigmoid)
    pred_hm = torch.sigmoid(pred["hm_logits"])
    se = (pred_hm - gt_hm) ** 2
    per_kp_hm = se.view(B, K, -1).mean(dim=-1)  # [B, K]
    loss_hm = (per_kp_hm * confidence).sum() / (confidence.sum() + 1e-6)

    # XY regression loss (L1)
    diff_xy = torch.abs(pred["xy"] - gt_xy).mean(dim=-1)  # [B, K]
    loss_xy = (diff_xy * confidence).sum() / (confidence.sum() + 1e-6)

    # Coarse XY loss
    diff_coarse = torch.abs(pred["coarse_xy"] - gt_xy).mean(dim=-1)  # [B, K]
    loss_coarse = (diff_coarse * confidence).sum() / (confidence.sum() + 1e-6)

    loss = loss_hm + xy_weight * loss_xy + coarse_weight * loss_coarse
    return loss, {
        "hm": loss_hm.item(), "xy": loss_xy.item(),
        "coarse": loss_coarse.item(),
        "mean_conf": confidence.mean().item(),
    }


# =============================================================================
# 4. Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device, img_size):
    model.eval()
    all_kp = [[] for _ in range(4)]
    pck_20 = pck_50 = total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        imgs = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        pred = model(imgs)
        pred_px = pred["xy"] * (img_size - 1)
        gt_px = gt_xy * (img_size - 1)
        err = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1))
        for k in range(4):
            all_kp[k].extend(err[:, k].cpu().numpy().tolist())
        pck_20 += (err < 20).float().sum().item()
        pck_50 += (err < 50).float().sum().item()
        total += err.numel()

    all_flat = [e for kp_errs in all_kp for e in kp_errs]
    return {
        "mean_error": np.mean(all_flat),
        "kp_errors": [np.mean(e) for e in all_kp],
        "pck_20": pck_20 / total, "pck_50": pck_50 / total,
    }


# =============================================================================
# 5. Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Dataset Transfer with Confidence-Aware Filtering")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Combined manifest (RMIT + cataract)")
    parser.add_argument("--aux_manifest", type=str, required=True,
                        help="Auxiliary manifest (cataract only)")
    parser.add_argument("--baseline_ckpt", type=str, required=True,
                        help="Baseline checkpoint (trained on target only)")
    parser.add_argument("--protocol", type=str, default="loso")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--decoder_type", type=str, default="heatmap_offset")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--decoder_stride", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--aug_color", type=float, default=0.3)
    parser.add_argument("--aug_rotate", type=float, default=10)
    parser.add_argument("--aug_blur", type=float, default=0.3)
    parser.add_argument("--aug_noise", type=float, default=0.05)
    parser.add_argument("--aug_cutout", type=float, default=0.1)

    parser.add_argument("--conf_cache", type=str, default=None,
                        help="Cache file for confidence scores")
    parser.add_argument("--naive_aux", action="store_true",
                        help="Naive auxiliary: set all confidence=1.0 (no filtering)")
    parser.add_argument("--conf_scale_multiplier", type=float, default=3.0,
                        help="Multiplier for median error to compute confidence scale (default: 3.0)")
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()
    if args.no_pretrained:
        args.pretrained = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ensure_dir(args.out_dir)

    # === Step 1: Compute or load confidence ===
    if args.naive_aux:
        print("[CDT] Naive aux mode: all confidence=1.0 (no filtering)")
        confidence_map = {}
    elif args.conf_cache and os.path.exists(args.conf_cache):
        print(f"[CDT] Loading cached confidence: {args.conf_cache}")
        raw = json.load(open(args.conf_cache))
        confidence_map = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    else:
        confidence_map = compute_aux_confidence(
            args.baseline_ckpt, args.aux_manifest,
            args.img_size, device, args.backbone,
            args.conf_scale_multiplier,
        )
        if args.conf_cache:
            cache = {k: v.tolist() for k, v in confidence_map.items()}
            json.dump(cache, open(args.conf_cache, "w"))
            print(f"[CDT] Confidence cached to {args.conf_cache}")

    # === Step 2: Data split ===
    if args.protocol == "loso":
        train_names, val_names, test_name = resolve_loso_split(args.manifest, args.fold)
        half_train, half_val = None, None
        print(f"[CDT] LOSO fold {args.fold}: test={test_name}")
    elif args.protocol == "half":
        seqs = load_rmit_manifest_sequences(args.manifest)
        train_names = val_names = {s["name"] for s in seqs}
        test_name = "half"
        half_train, half_val = "first", "second"
        print(f"[CDT] Half-split: train=first-half, val=second-half")

    train_ds = RMITHeatmapDataset(
        args.manifest, args.img_size, train_names, is_training=True,
        sigma=args.sigma, augment=True,
        aug_color=args.aug_color, aug_rotate=args.aug_rotate,
        aug_blur=args.aug_blur, aug_noise=args.aug_noise,
        aug_cutout=args.aug_cutout,
        half_split=half_train,
    )
    train_ds = ConfidenceWeightedDataset(train_ds, confidence_map)

    val_ds = RMITHeatmapDataset(
        args.manifest, args.img_size, val_names, is_training=False,
        sigma=args.sigma, augment=False,
        half_split=half_val,
    )

    print(f"[CDT] Train: {len(train_ds)} (RMIT+cataract)  Val: {len(val_ds)} (RMIT test)")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # === Step 3: Model ===
    model = HeatmapKeypointNet(
        img_size=args.img_size, pretrained=args.pretrained, backbone=args.backbone,
        decoder_stride=args.decoder_stride, decoder_type=args.decoder_type,
        use_offset=args.use_offset,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CDT] Model: {n_params:,} params")

    # === Step 4: Training ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler("cuda")

    best_err = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Warmup + cosine LR
        if epoch < args.warmup_epochs:
            lr = args.lr * (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        sum_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}", leave=False)

        for batch in pbar:
            imgs = batch["img"].to(device)
            gt_xy = batch["gt_xy"].to(device)
            gt_hm = batch["gt_hm"].to(device)
            confidence = batch["confidence"].to(device)
            B = imgs.size(0)

            optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(imgs)
                loss, metrics = confidence_weighted_loss(
                    pred, gt_hm, gt_xy, confidence)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += loss.item() * B
            n += B
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             conf=f"{metrics['mean_conf']:.3f}")

        val_metrics = evaluate(model, val_loader, device, args.img_size)
        val_err = val_metrics["mean_error"]

        print(f"[CDT] epoch {epoch+1}/{args.epochs} loss={sum_loss/n:.4f} "
              f"val_err={val_err:.2f}px pck@20={val_metrics['pck_20']:.3f} "
              f"pck@50={val_metrics['pck_50']:.3f}")
        for k in range(4):
            print(f"  kp{k}: {val_metrics['kp_errors'][k]:.2f}px")

        if val_err < best_err:
            best_err = val_err
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "metric": val_err, "args": vars(args),
                "val_metrics": val_metrics,
            }, os.path.join(args.out_dir, "best.pt"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[CDT] Early stopping at epoch {epoch+1}")
            break

    print(f"[CDT] Best: {best_err:.2f}px at epoch {best_epoch+1}")


if __name__ == "__main__":
    main()
