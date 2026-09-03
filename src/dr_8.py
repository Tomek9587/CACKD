#!/usr/bin/env python
"""
dr_8.py — Heatmap-based Keypoint Tracker with U-Net decoder

V8 核心改变：
1. 放弃 rigid 5 参数，改用 4 通道 heatmap + soft-argmax
2. ResNet18 encoder + U-Net decoder（带 skip connection）
3. 输出分辨率 H/4 x W/4，再插值到原图尺寸
4. Offset head 做亚像素精修
5. Heatmap sigma=2，focal loss + L1 offset loss
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

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# 0. 工具函数
# =============================================================================
def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize_img(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)
    arr = arr.transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def letterbox_resize(
    img: Image.Image,
    target_size: int,
    gt_xy: Optional[np.ndarray] = None,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, Optional[np.ndarray], float, int, int]:
    orig_w, orig_h = img.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    new_img = Image.new("RGB", (target_size, target_size), fill)
    new_img.paste(img, (pad_left, pad_top))

    if gt_xy is not None:
        gt_xy = gt_xy.copy()
        gt_xy[:, 0] = gt_xy[:, 0] * scale + pad_left
        gt_xy[:, 1] = gt_xy[:, 1] * scale + pad_top
        return new_img, gt_xy, scale, pad_top, pad_left
    return new_img, None, scale, pad_top, pad_left


def make_gaussian(size: int, x: float, y: float, sigma: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    heatmap = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmap


def soft_argmax_2d(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Args:
        heatmap: [B, K, H, W]
    Returns:
        xy: [B, K, 2] 坐标 (x, y)
    """
    bsz, K, H, W = heatmap.shape
    probs = F.softmax((heatmap / temperature).view(bsz, K, -1), dim=-1).view(bsz, K, H, W)
    ys = torch.linspace(0, H - 1, H, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, H, 1)
    xs = torch.linspace(0, W - 1, W, device=heatmap.device, dtype=heatmap.dtype).view(1, 1, 1, W)
    y = (probs * ys).sum(dim=(2, 3))
    x = (probs * xs).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1)


# =============================================================================
# 1. 数据集
# =============================================================================
def load_rmit_manifest_sequences(manifest_json: str) -> List[dict]:
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["sequences"] if isinstance(manifest, dict) and "sequences" in manifest else manifest


class RMITHeatmapDataset(Dataset):
    def __init__(
        self,
        manifest_json: str,
        img_size: int,
        split_names: set,
        is_training: bool,
        sigma: float = 2.0,
        augment: bool = True,
        within_seq_val_ratio: float = 0.2,
        aug_color: float = 0.15,
        aug_rotate: float = 5.0,
        aug_blur: float = 0.0,
        aug_noise: float = 0.0,
        aug_cutout: float = 0.0,
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.sigma = sigma
        self.augment = augment
        self.aug_color = aug_color
        self.aug_rotate = aug_rotate
        self.aug_blur = aug_blur
        self.aug_noise = aug_noise
        self.aug_cutout = aug_cutout

        sequences = load_rmit_manifest_sequences(manifest_json)
        selected = [s for s in sequences if s["name"] in split_names]

        self.records: List[dict] = []
        for seq in selected:
            self.records.extend(self._load_sequence(seq["frames_dir"], seq["ann_csv"]))
        self.records.sort(key=lambda r: r["path"])

        # 同序列时序划分
        if len(split_names) == 1 and within_seq_val_ratio > 0:
            n = len(self.records)
            n_val = max(1, int(round(n * within_seq_val_ratio)))
            if is_training:
                self.records = self.records[: n - n_val]
            else:
                self.records = self.records[n - n_val:]

    def _load_sequence(self, frames_dir: str, ann_csv: str) -> List[dict]:
        records = []
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
                records.append({"path": path, "xy": np.array(pts, dtype=np.float32)})
        return records

    def _augment(self, img: Image.Image, xy: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        if self.aug_color > 0:
            if random.random() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-self.aug_color, self.aug_color))
            if random.random() < 0.5:
                img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-self.aug_color, self.aug_color))
            if random.random() < 0.3:
                img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-self.aug_color, self.aug_color))

        if self.aug_blur > 0 and random.random() < self.aug_blur:
            # mild blur or sharpen
            factor = random.uniform(0.4, 1.8)
            img = ImageEnhance.Sharpness(img).enhance(factor)

        if self.aug_noise > 0 and random.random() < self.aug_noise:
            arr = np.asarray(img, dtype=np.float32)
            std = self.aug_noise * 255.0
            noise = np.random.normal(0.0, std, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
            img = Image.fromarray(arr)

        if self.aug_cutout > 0 and random.random() < self.aug_cutout:
            img_w, img_h = img.size
            cut_w = max(8, random.randint(img_w // 10, img_w // 5))
            cut_h = max(8, random.randint(img_h // 10, img_h // 5))
            x0 = random.randint(0, max(1, img_w - cut_w))
            y0 = random.randint(0, max(1, img_h - cut_h))
            img.paste((0, 0, 0), (x0, y0, x0 + cut_w, y0 + cut_h))

        if self.aug_rotate > 0 and random.random() < 0.3:
            angle = random.uniform(-self.aug_rotate, self.aug_rotate)
            cx = cy = self.img_size / 2.0
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
            new_xy = xy.copy()
            for k in range(4):
                x, y = xy[k, 0] - cx, xy[k, 1] - cy
                new_xy[k, 0] = x * cos_a - y * sin_a + cx
                new_xy[k, 1] = x * sin_a + y * cos_a + cy
            xy = new_xy

        return img, xy

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        img = load_image_rgb(rec["path"])
        xy = rec["xy"].copy()

        img, xy, scale, pad_top, pad_left = letterbox_resize(img, self.img_size, xy)

        if self.is_training and self.augment:
            img, xy = self._augment(img, xy)

        xy = np.clip(xy, 0, self.img_size - 1)
        xy_norm = xy / float(self.img_size - 1)

        # 生成 heatmap
        gt_hm = torch.zeros(4, self.img_size, self.img_size, dtype=torch.float32)
        for k in range(4):
            gt_hm[k] = make_gaussian(self.img_size, xy[k, 0], xy[k, 1], self.sigma)

        return {
            "img": normalize_img(pil_to_tensor(img)),
            "gt_xy": torch.from_numpy(xy_norm).float(),
            "gt_hm": gt_hm,
            "path": rec["path"],
        }


# =============================================================================
# 2. 网络：ResNet18 + U-Net decoder
# =============================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HeatmapKeypointNet(nn.Module):
    def __init__(
        self,
        img_size: int,
        pretrained: bool = True,
        decoder_channels: List[int] = [256, 128, 64],
        use_offset: bool = True,
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
        self.up1 = UpBlock(256, 128, decoder_channels[0])   # /16 -> /8
        self.up2 = UpBlock(decoder_channels[0], 64, decoder_channels[1])  # /8 -> /4

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
                nn.Conv2d(64, 8, 1),  # 4 points * 2 channels
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)

        e1 = self.enc1(x)    # [B, 64, H/2, W/2]
        e2 = self.enc2(e1)   # [B, 64, H/4, W/4]
        e3 = self.enc3(e2)   # [B, 128, H/8, W/8]
        e4 = self.enc4(e3)   # [B, 256, H/16, W/16]

        d1 = self.up1(e4, e3)  # [B, 256, H/8, W/8]
        d2 = self.up2(d1, e2)  # [B, 128, H/4, W/4]
        d3 = self.bottleneck(d2)  # [B, 64, H/4, W/4]

        hm_logits = self.hm_head(d3)  # [B, 4, H/4, W/4]
        hm_logits = F.interpolate(hm_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        coarse_xy = soft_argmax_2d(hm_logits, temperature=1.0)  # [B, 4, 2]
        coarse_xy_norm = coarse_xy / float(self.img_size - 1)

        if self.use_offset:
            offset_logits = self.offset_head(d3)  # [B, 8, H/4, W/4]
            offset_logits = F.interpolate(offset_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            offset = torch.tanh(offset_logits).view(B, 4, 2, self.img_size, self.img_size)

            # sample offset at coarse location
            grid = coarse_xy_norm * 2 - 1  # [B, 4, 2]
            grid = grid.unsqueeze(2).unsqueeze(3)  # [B, 4, 1, 1, 2]
            sampled_offsets = []
            for k in range(4):
                o_k = offset[:, k]  # [B, 2, H, W]
                sampled = F.grid_sample(o_k, grid[:, k], mode="bilinear", padding_mode="border", align_corners=True)
                sampled_offsets.append(sampled.squeeze(-1).squeeze(-1))  # [B, 2]
            offset_xy = torch.stack(sampled_offsets, dim=1) * 0.05  # 缩放 offset 范围

            xy_norm = (coarse_xy_norm + offset_xy).clamp(0, 1)
        else:
            xy_norm = coarse_xy_norm

        return {
            "hm_logits": hm_logits,
            "coarse_xy": coarse_xy_norm,
            "xy": xy_norm,
        }


# =============================================================================
# 3. 损失函数
# =============================================================================
def mse_heatmap_loss(pred: torch.Tensor, target: torch.Tensor, kp_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE on sigmoid heatmap, optional per-keypoint weighting.
    pred/target: [B, K, H, W]
    """
    pred = torch.sigmoid(pred)
    bsz, K, H, W = pred.shape
    se = (pred - target) ** 2  # [B, K, H, W]
    per_kp = se.view(bsz, K, -1).mean(dim=-1)  # [B, K]
    if kp_weights is not None:
        w = kp_weights.to(pred.device).view(K)
        per_kp = per_kp * w
        loss = per_kp.sum() / (w.sum() * bsz)
    else:
        loss = per_kp.mean()
    return loss


def xy_regression_loss(pred: torch.Tensor, target: torch.Tensor,
                       loss_type: str = "l1",
                       wing_w: float = 10.0, wing_epsilon: float = 2.0,
                       kp_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Per-keypoint regression loss with optional weighting.
    pred/target: [B, K, 2]
    """
    diff = pred - target  # [B, K, 2]
    if loss_type == "smooth_l1":
        err = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none").mean(dim=-1)  # [B, K]
    elif loss_type == "wing":
        abs_diff = torch.abs(diff)
        # wing loss per coordinate
        wing = torch.where(
            abs_diff < wing_w,
            wing_w * torch.log(1.0 + abs_diff / wing_epsilon),
            abs_diff - wing_w + wing_w * torch.log(1.0 + wing_w / wing_epsilon),
        )
        err = wing.mean(dim=-1)  # [B, K]
    else:  # l1
        err = torch.abs(diff).mean(dim=-1)  # [B, K]

    if kp_weights is not None:
        w = kp_weights.to(pred.device).view(-1)
        err = err * w
        loss = err.sum() / w.sum()
    else:
        loss = err.mean()
    return loss


def total_loss(
    pred: Dict[str, torch.Tensor],
    gt_hm: torch.Tensor,
    gt_xy: torch.Tensor,
    kp_weights: Optional[torch.Tensor] = None,
    xy_loss_type: str = "l1",
    wing_w: float = 10.0,
    wing_epsilon: float = 2.0,
    xy_weight: float = 10.0,
    coarse_weight: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_hm = mse_heatmap_loss(pred["hm_logits"], gt_hm, kp_weights=kp_weights)
    loss_xy = xy_regression_loss(pred["xy"], gt_xy,
                                 loss_type=xy_loss_type, wing_w=wing_w, wing_epsilon=wing_epsilon,
                                 kp_weights=kp_weights)
    loss_coarse = xy_regression_loss(pred["coarse_xy"], gt_xy,
                                     loss_type=xy_loss_type, wing_w=wing_w, wing_epsilon=wing_epsilon,
                                     kp_weights=kp_weights)

    loss = loss_hm + xy_weight * loss_xy + coarse_weight * loss_coarse
    return loss, {
        "hm": loss_hm.item(),
        "xy": loss_xy.item(),
        "coarse": loss_coarse.item(),
    }


# =============================================================================
# 4. 评估
# =============================================================================
def compute_metrics(pred_xy_px: torch.Tensor, gt_xy_px: torch.Tensor, img_size: int = None) -> Dict[str, float]:
    dist = torch.sqrt(((pred_xy_px - gt_xy_px) ** 2).sum(dim=-1))  # [B, 4]
    metrics = {
        "mean_error_px": float(dist.mean().item()),
        "max_error_px": float(dist.max().item()),
        "pck@20": float((dist < 20).float().mean().item()),
        "pck@50": float((dist < 50).float().mean().item()),
        # 保留内部阈值，仅用于调试，不作为汇报重点
        "pck@5": float((dist < 5).float().mean().item()),
        "pck@10": float((dist < 10).float().mean().item()),
    }
    for k in range(4):
        metrics[f"kp{k}_mean_error_px"] = float(dist[:, k].mean().item())
        metrics[f"kp{k}_pck@20"] = float((dist[:, k] < 20).float().mean().item())
        metrics[f"kp{k}_pck@50"] = float((dist[:, k] < 50).float().mean().item())
    return metrics


@torch.no_grad()
def adapt_bn(model: nn.Module, loader: DataLoader, device: torch.device, passes: int = 1) -> None:
    """AdaBN: update BatchNorm running statistics using target-domain data.
    Keeps affine params frozen; only running mean/var are updated.
    """
    was_training = model.training
    model.train()
    for _ in range(passes):
        for batch in loader:
            img = batch["img"].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                _ = model(img)
    if not was_training:
        model.eval()
    print(f"[AdaBN] updated BN stats with {passes} pass(es) over {len(loader.dataset)} target samples")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, img_size: int, adabn: bool = False, adabn_passes: int = 1) -> Dict[str, float]:
    if adabn:
        adapt_bn(model, loader, device, passes=adabn_passes)
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        img = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        pred = model(img)
        pred_xy = pred["xy"]
        all_pred.append(pred_xy.cpu())
        all_gt.append(gt_xy.cpu())
    all_pred = torch.cat(all_pred, dim=0) * (img_size - 1)
    all_gt = torch.cat(all_gt, dim=0) * (img_size - 1)
    return compute_metrics(all_pred, all_gt, img_size)


# =============================================================================
# 5. 划分
# =============================================================================
def resolve_loso_split(manifest_json: str, fold: int) -> Tuple[set, set, str]:
    sequences = load_rmit_manifest_sequences(manifest_json)
    names = sorted([s["name"] for s in sequences])
    fold = fold % len(names)
    test_name = names[fold]
    train_names = {n for n in names if n != test_name}
    return train_names, {test_name}, test_name


def resolve_within_seq_split(manifest_json: str, seq_name: str) -> Tuple[set, set, str]:
    return {seq_name}, {seq_name}, seq_name


# =============================================================================
# 6. 训练
# =============================================================================
def warmup_cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def load_pretrained_weights(model: nn.Module, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[V8] Loaded pretrained checkpoint from {checkpoint_path}")
    if missing:
        print(f"[V8] Missing keys: {missing}")
    if unexpected:
        print(f"[V8] Unexpected keys: {unexpected}")


def load_encoder_from_dr(model: HeatmapKeypointNet, dr_checkpoint_path: str):
    """Load ResNet18 encoder weights from a DRClassifier checkpoint.

    The DR checkpoint stores the full feature extractor as a Sequential with keys
    0.* (conv1), 1.* (bn1), 3.* (maxpool), 4.* (layer1), 5.* (layer2),
    6.* (layer3), 7.* (layer4).
    """
    ckpt = torch.load(dr_checkpoint_path, map_location="cpu")
    enc_state = ckpt.get("encoder")
    if enc_state is None:
        raise ValueError(f"DR checkpoint {dr_checkpoint_path} has no 'encoder' key")

    resnet = models.resnet18(weights=None)
    temp_enc = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu,
        resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
    )
    temp_enc.load_state_dict(enc_state, strict=True)

    model.enc1[0].load_state_dict(resnet.conv1.state_dict())
    model.enc1[1].load_state_dict(resnet.bn1.state_dict())
    model.enc2[0].load_state_dict(resnet.maxpool.state_dict())
    model.enc2[1].load_state_dict(resnet.layer1.state_dict())
    model.enc3.load_state_dict(resnet.layer2.state_dict())
    model.enc4.load_state_dict(resnet.layer3.state_dict())
    print(f"[V8] Loaded DR encoder from {dr_checkpoint_path}")


def train_rmit(cfg, device: torch.device) -> None:
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    model = HeatmapKeypointNet(
        img_size=cfg.img_size,
        pretrained=cfg.pretrained,
        use_offset=cfg.use_offset,
    ).to(device)

    if cfg.dr_checkpoint and os.path.exists(cfg.dr_checkpoint):
        load_encoder_from_dr(model, cfg.dr_checkpoint)

    if cfg.pt_checkpoint and os.path.exists(cfg.pt_checkpoint):
        load_pretrained_weights(model, cfg.pt_checkpoint)

    if cfg.freeze_backbone:
        for p in model.enc1.parameters():
            p.requires_grad = False
        for p in model.enc2.parameters():
            p.requires_grad = False
        for p in model.enc3.parameters():
            p.requires_grad = False
        for p in model.enc4.parameters():
            p.requires_grad = False
        print("[V8] Freeze backbone encoder")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[V8] Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    kp_weights = None
    if cfg.kp_adaptive:
        init_w = cfg.kp_weights if cfg.kp_weights is not None else [1.0, 1.0, 1.0, 1.0]
        kp_weights = torch.tensor(init_w, dtype=torch.float32)
        print(f"[V8] Adaptive kp weights initialized: {kp_weights.tolist()}")
    elif cfg.kp_weights is not None:
        kp_weights = torch.tensor(cfg.kp_weights, dtype=torch.float32)
        if kp_weights.numel() != 4:
            raise ValueError("--kp_weights must have exactly 4 values")
        print(f"[V8] Static kp_weights: {kp_weights.tolist()}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.wd,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))

    if cfg.protocol == "loso":
        train_names, val_names, test_name = resolve_loso_split(cfg.manifest, cfg.fold)
        print(f"[V8] LOSO fold {cfg.fold}: train={sorted(train_names)}, val={test_name}")
    else:
        train_names, val_names, test_name = resolve_within_seq_split(cfg.manifest, cfg.seq)
        print(f"[V8] Within-seq {cfg.seq}")

    train_ds = RMITHeatmapDataset(
        cfg.manifest, cfg.img_size, train_names, True,
        sigma=cfg.sigma, augment=True, within_seq_val_ratio=cfg.within_seq_val_ratio,
        aug_color=cfg.aug_color, aug_rotate=cfg.aug_rotate,
        aug_blur=cfg.aug_blur, aug_noise=cfg.aug_noise, aug_cutout=cfg.aug_cutout,
    )
    # LOSO: validate on the entire held-out sequence, not a temporal split
    val_ratio = 0.0 if cfg.protocol == "loso" else cfg.within_seq_val_ratio
    val_ds = RMITHeatmapDataset(
        cfg.manifest, cfg.img_size, val_names, False,
        sigma=cfg.sigma, augment=False, within_seq_val_ratio=val_ratio,
    )

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print(f"[V8] Train: {len(train_ds)}, Val: {len(val_ds)}")

    total_steps = max(1, len(train_ld)) * cfg.epochs
    warmup_steps = max(1, len(train_ld)) * cfg.warmup_epochs
    global_step = 0
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_hm = 0.0
        running_xy = 0.0

        iterator = enumerate(train_ld)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_ld), desc=f"V8 {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iterator:
            img = batch["img"].to(device, non_blocking=True)
            gt_xy = batch["gt_xy"].to(device, non_blocking=True)
            gt_hm = batch["gt_hm"].to(device, non_blocking=True)

            lr_now = warmup_cosine_lr(cfg.lr, global_step, total_steps, warmup_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
                pred = model(img)
                loss, loss_dict = total_loss(
                    pred, gt_hm, gt_xy,
                    kp_weights=kp_weights,
                    xy_loss_type=cfg.xy_loss_type,
                    wing_w=cfg.wing_w,
                    wing_epsilon=cfg.wing_epsilon,
                    xy_weight=cfg.xy_weight,
                    coarse_weight=cfg.coarse_weight,
                )

            if not torch.isfinite(loss):
                print(f"[V8] skip non-finite loss")
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
            global_step += 1

            if global_step % 20 == 0:
                writer.add_scalar("v8/train_loss", loss.item(), global_step)
                writer.add_scalar("v8/train_hm", loss_dict["hm"], global_step)
                writer.add_scalar("v8/train_xy", loss_dict["xy"], global_step)
                writer.add_scalar("v8/lr", lr_now, global_step)

        metrics = evaluate(model, val_ld, device, cfg.img_size, adabn=cfg.adabn, adabn_passes=cfg.adabn_passes)
        for k, v in metrics.items():
            writer.add_scalar(f"v8/val_{k}", v, epoch)

        avg_loss = running_loss / max(len(train_ld), 1)
        avg_hm = running_hm / max(len(train_ld), 1)
        avg_xy = running_xy / max(len(train_ld), 1)
        # adaptive keypoint reweighting based on validation per-kp error
        if cfg.kp_adaptive:
            per_kp_err = torch.tensor([metrics[f"kp{k}_mean_error_px"] for k in range(4)], dtype=torch.float32)
            mean_err = per_kp_err.mean().clamp_min(1e-6)
            rel = per_kp_err / mean_err  # >1 for hard keypoints
            mom = cfg.kp_adapt_momentum
            kp_weights = mom * kp_weights + (1.0 - mom) * rel
            kp_weights = torch.clamp(kp_weights, cfg.kp_adapt_min_weight, cfg.kp_adapt_max_weight)
            # normalize to keep sum roughly constant (4)
            kp_weights = kp_weights / kp_weights.sum() * 4.0
            print(f"[V8] adaptive kp_weights: {kp_weights.tolist()}")
            for k in range(4):
                writer.add_scalar(f"v8/kp_weight_{k}", kp_weights[k].item(), epoch)

        msg = (
            f"[V8] epoch {epoch+1}/{cfg.epochs} "
            f"loss={avg_loss:.4f} hm={avg_hm:.4f} xy={avg_xy:.4f} "
            f"val_err={metrics['mean_error_px']:.2f}px "
            f"pck@20={metrics['pck@20']:.3f} pck@50={metrics['pck@50']:.3f}"
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
            print(f"[V8] Early stopping at epoch {epoch+1}")
            break

    writer.close()
    print(f"[V8] Best val error: {best_metric:.2f}px")


# =============================================================================
# 7. 命令行
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DR-SurgBridgeNet V8: Heatmap Keypoint Tracker")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking/data/manifests/rmit_manifest.json")
    parser.add_argument("--protocol", type=str, default="within_seq", choices=["loso", "within_seq"])
    parser.add_argument("--seq", type=str, default="seq1", choices=["seq1", "seq2", "seq3"])
    parser.add_argument("--within_seq_val_ratio", type=float, default=0.2)

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--pt_checkpoint", type=str, default=None, help="pretrained encoder+decoder checkpoint from FOVEA")
    parser.add_argument("--dr_checkpoint", type=str, default=None, help="DR classification encoder checkpoint")
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)

    # loss / keypoint reweighting
    parser.add_argument("--kp_weights", type=float, nargs=4, default=None, help="per-keypoint loss weights, e.g. 2 1 1 1")
    parser.add_argument("--kp_adaptive", action="store_true", help="adapt kp weights each epoch based on val per-kp error")
    parser.add_argument("--kp_adapt_momentum", type=float, default=0.9)
    parser.add_argument("--kp_adapt_max_weight", type=float, default=3.0)
    parser.add_argument("--kp_adapt_min_weight", type=float, default=0.5)
    parser.add_argument("--xy_loss_type", type=str, default="l1", choices=["l1", "smooth_l1", "wing"])
    parser.add_argument("--wing_w", type=float, default=10.0)
    parser.add_argument("--wing_epsilon", type=float, default=2.0)
    parser.add_argument("--xy_weight", type=float, default=10.0)
    parser.add_argument("--coarse_weight", type=float, default=5.0)

    # AdaBN
    parser.add_argument("--adabn", action="store_true", help="adapt BN stats on validation set before evaluating")
    parser.add_argument("--adabn_passes", type=int, default=1)

    # stronger augmentation
    parser.add_argument("--aug_color", type=float, default=0.15, help="color jitter range")
    parser.add_argument("--aug_rotate", type=float, default=5.0, help="max rotation degrees")
    parser.add_argument("--aug_blur", type=float, default=0.0, help="prob of sharpness/blur augmentation")
    parser.add_argument("--aug_noise", type=float, default=0.0, help="prob of additive Gaussian noise (std as fraction of 255)")
    parser.add_argument("--aug_cutout", type=float, default=0.0, help="prob of random cutout")

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
    train_rmit(cfg, device)


if __name__ == "__main__":
    main()
