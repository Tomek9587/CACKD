#!/usr/bin/env python
"""
dr_8_dann.py — V8 heatmap tracker with Domain-Adversarial Neural Network (DANN).

Adds a domain classifier on the encoder output with Gradient Reversal Layer.
In LOSO, training sequences are labeled source (0) and the held-out test sequence
is labeled target (1). The domain classifier is trained to distinguish them while
the encoder learns to fool it, encouraging domain-invariant features.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

# Assume project-root parent is on PYTHONPATH so 'surgical_tracking.src.dr_8' resolves.
from surgical_tracking.src.dr_8 import (
    ConvBlock,
    UpBlock,
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    load_rmit_manifest_sequences,
    resolve_loso_split,
    seed_all,
    ensure_dir,
    load_image_rgb,
    normalize_img,
    pil_to_tensor,
    letterbox_resize,
    make_gaussian,
    soft_argmax_2d,
    mse_heatmap_loss,
    total_loss,
    compute_metrics,
    evaluate,
    warmup_cosine_lr,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# 0. Gradient Reversal Layer
# =============================================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return GradientReversalFunction.apply(x, alpha)


# =============================================================================
# 1. Domain-aware dataset
# =============================================================================
class DomainRMITHeatmapDataset(RMITHeatmapDataset):
    """RMIT dataset that additionally returns a binary domain label.

    source (train) sequences -> domain 0
    target (test) sequence   -> domain 1
    """
    def __init__(
        self,
        manifest_json: str,
        img_size: int,
        split_names: set,
        is_training: bool,
        train_names: set,
        test_name: str,
        sigma: float = 2.0,
        augment: bool = True,
        within_seq_val_ratio: float = 0.2,
    ):
        self.train_names = set(train_names)
        self.test_name = test_name
        super().__init__(
            manifest_json, img_size, split_names, is_training,
            sigma=sigma, augment=augment, within_seq_val_ratio=within_seq_val_ratio,
        )

    def _load_sequence(self, frames_dir: str, ann_csv: str) -> List[dict]:
        records = []
        seq_name = os.path.basename(frames_dir)
        with open(ann_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pts = []
                for k in range(4):
                    xk = row.get(f"x{k}")
                    yk = row.get(f"y{k}")
                    if xk is None or yk is None:
                        break
                    pts.append([float(xk), float(yk)])
                if len(pts) != 4:
                    continue
                frame_name = row.get("frame")
                path = frame_name if os.path.isabs(frame_name) else os.path.join(frames_dir, frame_name)
                if not os.path.exists(path):
                    continue
                records.append({"path": path, "xy": np.array(pts, dtype=np.float32), "seq": seq_name})
        return records

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        rec = self.records[idx]
        domain = 0 if rec["seq"] in self.train_names else 1
        item["domain"] = torch.tensor(domain, dtype=torch.long)
        return item


# =============================================================================
# 2. Domain-adversarial model
# =============================================================================
class DomainHeatmapKeypointNet(nn.Module):
    def __init__(
        self,
        img_size: int,
        pretrained: bool = True,
        decoder_channels: List[int] = [256, 128, 64],
        use_offset: bool = True,
        domain_hidden: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.use_offset = use_offset

        resnet = models.resnet18(weights=("IMAGENET1K_V1" if pretrained else None))
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # /4
        self.enc3 = resnet.layer2                                          # /8
        self.enc4 = resnet.layer3                                          # /16

        # Decoder
        self.up1 = UpBlock(256, 128, decoder_channels[0])
        self.up2 = UpBlock(decoder_channels[0], 64, decoder_channels[1])
        self.bottleneck = ConvBlock(decoder_channels[1], decoder_channels[2])

        # Heatmap head
        self.hm_head = nn.Sequential(
            nn.Conv2d(decoder_channels[2], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4, 1),
        )

        if use_offset:
            self.offset_head = nn.Sequential(
                nn.Conv2d(decoder_channels[2], 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 8, 1),
            )

        # Domain discriminator on enc4 features
        self.grl = GradientReversalLayer()
        self.domain_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, domain_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(domain_hidden, 1),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Dict[str, torch.Tensor]:
        B = x.size(0)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d1 = self.up1(e4, e3)
        d2 = self.up2(d1, e2)
        d3 = self.bottleneck(d2)

        hm_logits = self.hm_head(d3)
        hm_logits = F.interpolate(hm_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        coarse_xy = soft_argmax_2d(hm_logits, temperature=1.0)
        coarse_xy_norm = coarse_xy / float(self.img_size - 1)

        if self.use_offset:
            offset_logits = self.offset_head(d3)
            offset_logits = F.interpolate(offset_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            offset = torch.tanh(offset_logits).view(B, 4, 2, self.img_size, self.img_size)
            grid = coarse_xy_norm * 2 - 1
            grid = grid.unsqueeze(2).unsqueeze(3)
            sampled_offsets = []
            for k in range(4):
                o_k = offset[:, k]
                sampled = F.grid_sample(o_k, grid[:, k], mode="bilinear", padding_mode="border", align_corners=True)
                sampled_offsets.append(sampled.squeeze(-1).squeeze(-1))
            offset_xy = torch.stack(sampled_offsets, dim=1) * 0.05
            xy_norm = (coarse_xy_norm + offset_xy).clamp(0, 1)
        else:
            xy_norm = coarse_xy_norm

        # Domain logits
        rev = self.grl(e4, alpha)
        domain_logits = self.domain_head(rev).squeeze(-1)  # [B]

        return {
            "hm_logits": hm_logits,
            "coarse_xy": coarse_xy_norm,
            "xy": xy_norm,
            "domain_logits": domain_logits,
        }


# =============================================================================
# 3. Training with DANN
# =============================================================================
def dann_total_loss(pred: Dict[str, torch.Tensor], gt_hm: torch.Tensor, gt_xy: torch.Tensor, domain_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Keypoint loss only on source (domain_mask==1) samples. Domain loss on all."""
    loss_hm = mse_heatmap_loss(pred["hm_logits"], gt_hm)
    loss_xy = F.l1_loss(pred["xy"], gt_xy, reduction="none").mean(dim=(1, 2))  # [B]
    loss_coarse = F.l1_loss(pred["coarse_xy"], gt_xy, reduction="none").mean(dim=(1, 2))  # [B]

    # Average keypoint loss only over source samples
    src_count = domain_mask.sum().clamp_min(1.0)
    loss_xy_src = (loss_xy * domain_mask).sum() / src_count
    loss_coarse_src = (loss_coarse * domain_mask).sum() / src_count
    loss_hm_src = (loss_hm * domain_mask).sum() / src_count  # loss_hm is scalar per-sample? Actually mse_heatmap returns mean over batch.
    # Recalculate per-sample hm loss for masking:
    # Easier: compute per-sample heatmap loss
    pred_sig = torch.sigmoid(pred["hm_logits"])
    per_sample_hm = F.mse_loss(pred_sig, gt_hm, reduction="none").mean(dim=(1, 2, 3))  # [B]
    loss_hm_src = (per_sample_hm * domain_mask).sum() / src_count

    loss = loss_hm_src + 10.0 * loss_xy_src + 5.0 * loss_coarse_src
    return loss, {
        "hm": loss_hm_src.item(),
        "xy": loss_xy_src.item(),
        "coarse": loss_coarse_src.item(),
    }


def compute_lambda(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """Domain adversarial factor: starts near 0, smoothly grows to 1."""
    p = float(epoch) / max(total_epochs, 1)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


@torch.no_grad()
def evaluate_dann(model: nn.Module, loader: DataLoader, device: torch.device, img_size: int) -> Dict[str, float]:
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        img = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        pred = model(img, alpha=0.0)["xy"]
        all_pred.append(pred.cpu())
        all_gt.append(gt_xy.cpu())
    all_pred = torch.cat(all_pred, dim=0) * (img_size - 1)
    all_gt = torch.cat(all_gt, dim=0) * (img_size - 1)
    return compute_metrics(all_pred, all_gt, img_size)


def train_dann(cfg, device: torch.device) -> None:
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    model = DomainHeatmapKeypointNet(
        img_size=cfg.img_size,
        pretrained=cfg.pretrained,
        use_offset=cfg.use_offset,
        domain_hidden=cfg.domain_hidden,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[DANN] Trainable: {trainable:,} / {total:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))

    if cfg.protocol != "loso":
        raise ValueError("DANN mode only supports --protocol loso")

    train_names, val_names, test_name = resolve_loso_split(cfg.manifest, cfg.fold)
    print(f"[DANN] LOSO fold {cfg.fold}: source={sorted(train_names)}, target={test_name}")

    # Train set includes both source and target; keypoint labels used only for source.
    all_names = train_names | val_names
    train_ds = DomainRMITHeatmapDataset(
        cfg.manifest, cfg.img_size, all_names, True,
        train_names=train_names, test_name=test_name,
        sigma=cfg.sigma, augment=True, within_seq_val_ratio=0.0,
    )
    val_ds = DomainRMITHeatmapDataset(
        cfg.manifest, cfg.img_size, val_names, False,
        train_names=train_names, test_name=test_name,
        sigma=cfg.sigma, augment=False, within_seq_val_ratio=0.0,
    )

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print(f"[DANN] Train: {len(train_ds)}, Val: {len(val_ds)}")

    total_steps = max(1, len(train_ld)) * cfg.epochs
    warmup_steps = max(1, len(train_ld)) * cfg.warmup_epochs
    global_step = 0
    best_metric = float("inf")
    patience_counter = 0

    domain_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_hm = 0.0
        running_xy = 0.0
        running_dom = 0.0
        running_dom_acc = 0.0

        lambda_p = compute_lambda(epoch, cfg.epochs, cfg.lambda_gamma) * cfg.lambda_max

        iterator = enumerate(train_ld)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_ld), desc=f"DANN {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iterator:
            img = batch["img"].to(device, non_blocking=True)
            gt_xy = batch["gt_xy"].to(device, non_blocking=True)
            gt_hm = batch["gt_hm"].to(device, non_blocking=True)
            domain = batch["domain"].to(device, non_blocking=True).float()  # 0=source, 1=target
            src_mask = (domain == 0).float()

            lr_now = warmup_cosine_lr(cfg.lr, global_step, total_steps, warmup_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
                pred = model(img, alpha=lambda_p)
                kp_loss, loss_dict = dann_total_loss(pred, gt_hm, gt_xy, src_mask)
                dom_loss = domain_criterion(pred["domain_logits"], domain)
                loss = kp_loss + dom_loss

            if not torch.isfinite(loss):
                print(f"[DANN] skip non-finite loss")
                optim.zero_grad(set_to_none=True)
                continue

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()
            running_hm += loss_dict["hm"]
            running_xy += loss_dict["xy"]
            running_dom += dom_loss.item()

            with torch.no_grad():
                dom_pred = (torch.sigmoid(pred["domain_logits"]) > 0.5).float()
                running_dom_acc += (dom_pred == domain).float().mean().item()

            global_step += 1
            if global_step % 20 == 0:
                writer.add_scalar("dann/train_loss", loss.item(), global_step)
                writer.add_scalar("dann/train_kp_loss", kp_loss.item(), global_step)
                writer.add_scalar("dann/train_dom_loss", dom_loss.item(), global_step)
                writer.add_scalar("dann/lambda", lambda_p, global_step)
                writer.add_scalar("dann/lr", lr_now, global_step)

        metrics = evaluate_dann(model, val_ld, device, cfg.img_size)
        for k, v in metrics.items():
            writer.add_scalar(f"dann/val_{k}", v, epoch)

        avg_loss = running_loss / max(len(train_ld), 1)
        avg_hm = running_hm / max(len(train_ld), 1)
        avg_xy = running_xy / max(len(train_ld), 1)
        avg_dom = running_dom / max(len(train_ld), 1)
        avg_dom_acc = running_dom_acc / max(len(train_ld), 1)
        msg = (
            f"[DANN] epoch {epoch+1}/{cfg.epochs} "
            f"loss={avg_loss:.4f} hm={avg_hm:.4f} xy={avg_xy:.4f} dom={avg_dom:.4f} dom_acc={avg_dom_acc:.3f} "
            f"lambda={lambda_p:.3f} val_err={metrics['mean_error_px']:.2f}px "
            f"pck@5={metrics['pck@5']:.3f} pck@10={metrics['pck@10']:.3f} pck@20={metrics['pck@20']:.3f}"
        )
        print(msg)

        metric = metrics["mean_error_px"]
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch, "metric": metric},
                       os.path.join(cfg.out_dir, "best.pt"))
        else:
            patience_counter += 1

        torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "epoch": epoch, "metric": metric},
                   os.path.join(cfg.out_dir, "last.pt"))

        if patience_counter >= cfg.patience:
            print(f"[DANN] Early stopping at epoch {epoch+1}")
            break

    writer.close()
    print(f"[DANN] Best val error: {best_metric:.2f}px")


# =============================================================================
# 4. Args
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DR-SurgBridgeNet V8 + DANN")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="data/manifests/rmit_manifest.json")
    parser.add_argument("--protocol", type=str, default="loso", choices=["loso"])
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--domain_hidden", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--lambda_gamma", type=float, default=10.0)
    parser.add_argument("--lambda_max", type=float, default=1.0)

    args = parser.parse_args()
    if args.no_pretrained:
        args.pretrained = False
    return args


def main():
    cfg = parse_args()
    seed_all(cfg.seed)
    ensure_dir(cfg.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_dann(cfg, device)


if __name__ == "__main__":
    main()
