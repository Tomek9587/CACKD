#!/usr/bin/env python
"""SOTA comparison: HRNet-W32 backbone + SimCC head.

Architecture:
  - Backbone: HRNet-W32 (CVPR 2019) via timm, ImageNet pretrained
  - Head: SimCC (ECCV 2022) 1D coordinate classification

Trained on identical data, splits (cataract-only 30-fold LOSO), augmentations,
optimizer, and evaluation as the CAC-KD baseline (dr_11.py).  The only differences
from baseline are backbone (EffV2-S -> HRNet-W32) and head (heatmap+offset -> SimCC).

Usage:
  python -m surgical_tracking.src.sota_hrnet_simcc \
    --manifest data/manifests/cataract_manifest.json \
    --protocol loso --fold 0 --out_dir logs/sota_hrnet_simcc_f0
"""
import argparse
import collections
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Reuse baseline components
sys.path.insert(0, os.path.dirname(__file__))
from dr_11 import (
    RMITHeatmapDataset,
    resolve_loso_split,
    compute_metrics,
    adapt_bn,
    warmup_cosine_lr,
    ensure_dir,
    TimmEncoder,
    UpBlock,
    ConvBlock,
    ASPP,
)


@torch.no_grad()
def adapt_bn_safe(model: nn.Module, loader: DataLoader, device: torch.device, passes: int = 1):
    """AdaBN that skips batch_size=1 batches to avoid BatchNorm errors in ASPP global pooling."""
    was_training = model.training
    model.train()
    skipped = 0
    for _ in range(passes):
        for batch in loader:
            img = batch["img"].to(device, non_blocking=True)
            if img.size(0) == 1:
                skipped += 1
                continue
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                _ = model(img)
    if not was_training:
        model.eval()
    if skipped:
        print(f"[AdaBN] skipped {skipped} batch(es) with size 1")


@torch.no_grad()
def evaluate_simcc(model: nn.Module, loader: DataLoader, device: torch.device,
                   img_size: int, adabn: bool = False, adabn_passes: int = 1) -> Dict[str, float]:
    """Evaluate with safe AdaBN (skips batch_size=1)."""
    if adabn:
        adapt_bn_safe(model, loader, device, passes=adabn_passes)
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        img = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        pred = model(img)
        all_pred.append(pred["xy"].cpu())
        all_gt.append(gt_xy.cpu())
    all_pred = torch.cat(all_pred, dim=0) * (img_size - 1)
    all_gt = torch.cat(all_gt, dim=0) * (img_size - 1)
    return compute_metrics(all_pred, all_gt, img_size)

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

NUM_KEYPOINTS = 4


# =============================================================================
# SimCC Head
# =============================================================================
class SimCCHead(nn.Module):
    """SimCC 1D coordinate classification head.

    Predicts separate 1D distributions for x and y axes by pooling the
    feature map along the perpendicular axis, then classifying along the
    target axis.
    """

    def __init__(self, in_channels: int, num_keypoints: int = 4):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_keypoints, 1),
        )

    def forward(self, feat: torch.Tensor, img_size: int) -> Dict[str, torch.Tensor]:
        """
        Args:
            feat: (B, C, Hf, Wf) decoder feature at stride 4
            img_size: target resolution for 1D distributions
        Returns:
            dict with:
              xy: (B, K, 2) normalized [0,1] coordinates
              x_logits: (B, K, W)
              y_logits: (B, K, H)
        """
        k_feat = self.conv(feat)  # (B, K, Hf, Wf)

        # x-axis: pool over H -> (B, K, Wf) -> upsample to W
        x_logits = k_feat.mean(dim=2)  # (B, K, Wf)
        x_logits = F.interpolate(
            x_logits, size=img_size, mode="linear", align_corners=False
        )  # (B, K, W)

        # y-axis: pool over W -> (B, K, Hf) -> upsample to H
        y_logits = k_feat.mean(dim=3)  # (B, K, Hf)
        y_logits = F.interpolate(
            y_logits, size=img_size, mode="linear", align_corners=False
        )  # (B, K, H)

        # Decode: soft-argmax (expected value) for sub-pixel accuracy
        x_prob = F.softmax(x_logits.float(), dim=-1)
        y_prob = F.softmax(y_logits.float(), dim=-1)

        coords = torch.arange(img_size, device=feat.device, dtype=torch.float32)
        x_coord = (x_prob * coords).sum(dim=-1) / (img_size - 1)  # (B, K)
        y_coord = (y_prob * coords).sum(dim=-1) / (img_size - 1)  # (B, K)

        xy = torch.stack([x_coord, y_coord], dim=-1)  # (B, K, 2)
        return {"xy": xy, "x_logits": x_logits, "y_logits": y_logits}


# =============================================================================
# Model
# =============================================================================
class HRNetSimCC(nn.Module):
    """HRNet-W32 encoder + U-Net decoder + SimCC head."""

    def __init__(
        self,
        img_size: int = 512,
        pretrained: bool = True,
        backbone: str = "hrnet_w32",
        decoder_channels: List[int] = [256, 128, 64],
    ):
        super().__init__()
        self.img_size = img_size
        self.backbone_name = backbone

        self.encoder = TimmEncoder(
            backbone,
            pretrained=pretrained,
            target_reductions=[4, 8, 16],
            target_channels=[64, 128, 256],
        )
        enc_c = [64, 128, 256]
        self.aspp = nn.Identity()  # no ASPP — match baseline (baseline uses Identity when use_aspp=False)
        self.up1 = UpBlock(enc_c[2], enc_c[1], decoder_channels[0])
        self.up2 = UpBlock(decoder_channels[0], enc_c[0], decoder_channels[1])
        self.bottleneck = ConvBlock(decoder_channels[1], decoder_channels[2])
        self.simcc_head = SimCCHead(decoder_channels[2], NUM_KEYPOINTS)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        e2, e3, e4 = self.encoder(x)
        e4 = self.aspp(e4)
        d1 = self.up1(e4, e3)
        d2 = self.up2(d1, e2)
        d3 = self.bottleneck(d2)
        return self.simcc_head(d3, self.img_size)


# =============================================================================
# Loss
# =============================================================================
def make_1d_gaussian_targets(
    size: int, centers: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Generate 1D Gaussian targets.

    Args:
        size: distribution length (img_size)
        centers: (B, K) center positions in [0, size-1]
        sigma: Gaussian std dev
    Returns:
        (B, K, size) normalized to sum=1
    """
    coords = torch.arange(size, device=centers.device, dtype=torch.float32)
    coords = coords.view(1, 1, size)  # (1, 1, size)
    diff = (coords - centers.unsqueeze(-1)) / sigma
    target = torch.exp(-0.5 * diff ** 2)
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return target


def simcc_loss(
    pred: Dict[str, torch.Tensor],
    gt_xy: torch.Tensor,
    img_size: int,
    sigma: float = 6.0,
    kp_weights: Optional[torch.Tensor] = None,
    xy_weight: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SimCC loss: KL divergence on 1D distributions + optional L1 on coords.

    Args:
        pred: model output with x_logits, y_logits, xy
        gt_xy: (B, K, 2) normalized [0,1]
        img_size: distribution size
        sigma: Gaussian target sigma
        kp_weights: (K,) per-keypoint weights
        xy_weight: weight for auxiliary L1 coordinate loss
    """
    B, K, _ = gt_xy.shape
    device = gt_xy.device

    # GT positions in pixel space
    gt_px = gt_xy * (img_size - 1)  # (B, K, 2)
    x_centers = gt_px[..., 0]  # (B, K)
    y_centers = gt_px[..., 1]  # (B, K)

    # 1D Gaussian targets
    x_target = make_1d_gaussian_targets(img_size, x_centers, sigma)  # (B, K, W)
    y_target = make_1d_gaussian_targets(img_size, y_centers, sigma)  # (B, K, H)

    # Cross-entropy (KL) loss
    x_logp = F.log_softmax(pred["x_logits"].float(), dim=-1)
    y_logp = F.log_softmax(pred["y_logits"].float(), dim=-1)
    loss_x = -(x_target * x_logp).sum(dim=-1)  # (B, K)
    loss_y = -(y_target * y_logp).sum(dim=-1)  # (B, K)

    if kp_weights is not None:
        loss_x = loss_x * kp_weights.view(1, K)
        loss_y = loss_y * kp_weights.view(1, K)

    loss_cls = (loss_x + loss_y).mean()

    # Auxiliary L1 on decoded coordinates
    loss_reg = F.l1_loss(pred["xy"], gt_xy)

    loss = loss_cls + xy_weight * loss_reg
    return loss, {"cls": loss_cls.item(), "reg": loss_reg.item()}


# =============================================================================
# Training
# =============================================================================
def train_one_fold(cfg, device: torch.device) -> None:
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    model = HRNetSimCC(
        img_size=cfg.img_size,
        pretrained=cfg.pretrained,
        backbone=cfg.backbone,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[SimCC] Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.wd
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))

    # LOSO split
    train_names, val_names, test_name = resolve_loso_split(cfg.manifest, cfg.fold)
    print(f"[SimCC] LOSO fold {cfg.fold}: test={test_name}")

    train_ds = RMITHeatmapDataset(
        cfg.manifest, cfg.img_size, train_names, True,
        sigma=cfg.sigma, augment=True,
        aug_color=cfg.aug_color, aug_rotate=cfg.aug_rotate,
        aug_blur=cfg.aug_blur, aug_noise=cfg.aug_noise, aug_cutout=cfg.aug_cutout,
    )
    val_ds = RMITHeatmapDataset(
        cfg.manifest, cfg.img_size, val_names, False,
        sigma=cfg.sigma, augment=False,
    )

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    print(f"[SimCC] Train: {len(train_ds)}, Val: {len(val_ds)}")

    # kp_adaptive weights
    kp_weights = None
    if cfg.kp_adaptive:
        kp_weights = torch.ones(NUM_KEYPOINTS, dtype=torch.float32, device=device)
        print(f"[SimCC] Adaptive kp weights init: {kp_weights.tolist()}")

    total_steps = max(1, len(train_ld)) * cfg.epochs
    warmup_steps = max(1, len(train_ld)) * cfg.warmup_epochs
    global_step = 0
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_terms = collections.defaultdict(float)

        iterator = enumerate(train_ld)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_ld),
                            desc=f"SimCC {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iterator:
            img = batch["img"].to(device, non_blocking=True)
            gt_xy = batch["gt_xy"].to(device, non_blocking=True)

            lr_now = warmup_cosine_lr(cfg.lr, global_step, total_steps, warmup_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast("cuda", enabled=(cfg.amp and device.type == "cuda")):
                pred = model(img)
                loss, loss_dict = simcc_loss(
                    pred, gt_xy, cfg.img_size,
                    sigma=cfg.simcc_sigma,
                    kp_weights=kp_weights,
                    xy_weight=cfg.xy_weight,
                )

            if not torch.isfinite(loss):
                print(f"[SimCC] skip non-finite loss at step {global_step}")
                optim.zero_grad(set_to_none=True)
                continue

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()
            for lk, lv in loss_dict.items():
                running_terms[lk] += lv
            global_step += 1

            if global_step % 20 == 0:
                writer.add_scalar("simcc/train_loss", loss.item(), global_step)
                for lk, lv in loss_dict.items():
                    writer.add_scalar(f"simcc/train_{lk}", lv, global_step)
                writer.add_scalar("simcc/lr", lr_now, global_step)

        # Validation
        metrics = evaluate_simcc(model, val_ld, device, cfg.img_size,
                                  adabn=cfg.adabn, adabn_passes=cfg.adabn_passes)
        for k, v in metrics.items():
            writer.add_scalar(f"simcc/val_{k}", v, epoch)

        # kp_adaptive update
        if cfg.kp_adaptive:
            per_kp = torch.tensor(
                [metrics[f"kp{k}_mean_error_px"] for k in range(NUM_KEYPOINTS)],
                dtype=torch.float32, device=device,
            )
            mean_err = per_kp.mean().clamp_min(1e-6)
            rel = per_kp / mean_err
            mom = cfg.kp_adapt_momentum
            kp_weights = mom * kp_weights + (1.0 - mom) * rel
            kp_weights = torch.clamp(kp_weights, cfg.kp_adapt_min, cfg.kp_adapt_max)
            kp_weights = kp_weights / kp_weights.sum() * NUM_KEYPOINTS
            print(f"[SimCC] adaptive kp_weights: {kp_weights.tolist()}")

        avg_loss = running_loss / max(len(train_ld), 1)
        avg_terms = {k: v / max(len(train_ld), 1) for k, v in running_terms.items()}
        terms_str = " ".join(f"{k}={v:.4f}" for k, v in avg_terms.items())
        print(
            f"[SimCC] epoch {epoch+1}/{cfg.epochs} loss={avg_loss:.4f} {terms_str} "
            f"val_err={metrics['mean_error_px']:.2f}px "
            f"pck@20={metrics['pck@20']:.3f} pck@50={metrics['pck@50']:.3f}"
        )

        metric = metrics["mean_error_px"]
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metric": metric},
                os.path.join(cfg.out_dir, "best.pt"),
            )
        else:
            patience_counter += 1

        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "metric": metric},
            os.path.join(cfg.out_dir, "last.pt"),
        )

        if patience_counter >= cfg.patience:
            print(f"[SimCC] Early stopping at epoch {epoch+1}")
            break

    writer.close()
    print(f"Best: {best_metric:.2f}px")
    print(f"[SimCC] Best val error: {best_metric:.2f}px at fold {cfg.fold}")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="SOTA: HRNet-W32 + SimCC")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--manifest", type=str,
                   default="data/manifests/cataract_manifest.json")
    p.add_argument("--protocol", type=str, default="loso", choices=["loso"])
    p.add_argument("--fold", type=int, default=0)

    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--simcc_sigma", type=float, default=6.0,
                   help="Gaussian sigma for 1D SimCC targets")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--backbone", type=str, default="hrnet_w32")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--adabn", action="store_true", default=True)
    p.add_argument("--adabn_passes", type=int, default=1)
    p.add_argument("--kp_adaptive", action="store_true", default=True)
    p.add_argument("--kp_adapt_momentum", type=float, default=0.9)
    p.add_argument("--kp_adapt_min", type=float, default=0.3)
    p.add_argument("--kp_adapt_max", type=float, default=8.0)
    p.add_argument("--xy_weight", type=float, default=2.0,
                   help="weight for auxiliary L1 coordinate loss")

    p.add_argument("--aug_color", type=float, default=0.3)
    p.add_argument("--aug_rotate", type=float, default=10.0)
    p.add_argument("--aug_blur", type=float, default=0.3)
    p.add_argument("--aug_noise", type=float, default=0.05)
    p.add_argument("--aug_cutout", type=float, default=0.1)
    return p.parse_args()


def main():
    cfg = parse_args()
    if cfg.no_pretrained:
        cfg.pretrained = False

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SimCC] Device: {device}")
    print(f"[SimCC] Config: {vars(cfg)}")

    train_one_fold(cfg, device)


if __name__ == "__main__":
    main()
