#!/usr/bin/env python
"""
dr_11.py — V11 advanced decoding heads (BCIR + SAR) on top of V10 backbone/decoder.

Adds two recent (2023-2024) keypoint localization modules:
- BCIR: Bias-Compensated Integral Regression (TPAMI 2023) soft-argmax decoding
        with Gaussian-prior heatmap supervision.
- SAR:  Spatial-Aware Regression (CVPR 2024) per-grid coordinate+confidence head
        trained with a unified Laplacian-kernel objective.

Baseline `heatmap_offset` retains the original V10 behaviour.
"""
import argparse
import collections
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

import timm

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


def bcir_decode(hm_logits: torch.Tensor, beta: float = 10.0, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bias-Compensated Integral Regression (BCIR) decoding.

    Args:
        hm_logits: [B, K, H, W] raw heatmap logits.  Enforced non-negative via
                   softplus so that background pixels contribute exp(0)=1,
                   matching the BCIR derivation.
        beta:      softmax temperature scaling.

    Returns:
        biased_xy: [B, K, 2] standard soft-argmax expectation (x, y) in pixel coords.
        bcir_xy:   [B, K, 2] bias-compensated coordinates (x, y) in pixel coords.
    """
    # Use a bounded non-negative heatmap (sigmoid) for BCIR: it matches the
    # detection loss supervision and prevents exp(beta*H) from overflowing.
    orig_dtype = hm_logits.dtype
    H = torch.sigmoid(hm_logits).float()
    B, K, h, w = H.shape
    flat = H.view(B, K, -1)
    C = torch.sum(torch.exp(beta * H), dim=(2, 3))  # [B, K]
    probs = F.softmax(beta * flat, dim=-1).view(B, K, h, w)
    ys = torch.linspace(0, h - 1, h, device=H.device, dtype=torch.float32).view(1, 1, h, 1)
    xs = torch.linspace(0, w - 1, w, device=H.device, dtype=torch.float32).view(1, 1, 1, w)
    y_re = (probs * ys).sum(dim=(2, 3))
    x_re = (probs * xs).sum(dim=(2, 3))
    biased_xy = torch.stack([x_re, y_re], dim=-1)

    denom = (C - h * w).clamp_min(eps)
    scale = C / denom
    # BCIR closed-form correction, applicable to all quadrants (BCIR paper, Eq.18 / Appendix B).
    # Map: our x_re is horizontal (paper's y_r), y_re is vertical (paper's x_r).
    x_ro = scale * x_re - (h * h * w) / (2.0 * denom)
    y_ro = scale * y_re - (h * w * w) / (2.0 * denom)
    bcir_xy = torch.stack([x_ro, y_ro], dim=-1)
    return biased_xy.to(orig_dtype), bcir_xy.to(orig_dtype)


class SARHead(nn.Module):
    """Spatial-Aware Regression head (CVPR 2024)."""

    def __init__(self, in_channels: int, num_keypoints: int = 4):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.logit_conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.offset_conv = nn.Conv2d(in_channels, num_keypoints * 2, kernel_size=1)
        # Low-confidence init for logits, near-zero init for offsets.
        nn.init.normal_(self.logit_conv.weight, std=0.001)
        nn.init.constant_(self.logit_conv.bias, -math.log((1.0 - 0.01) / 0.01))
        nn.init.normal_(self.offset_conv.weight, std=0.001)
        nn.init.constant_(self.offset_conv.bias, 0.0)

    def _locations(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        device = x.device
        shifts_x = torch.arange(w, dtype=torch.float32, device=device)
        shifts_y = torch.arange(h, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        # Normalise by (w-1, h-1) to match gt_xy_norm used elsewhere.
        shift_x = shift_x / max(w - 1, 1)
        shift_y = shift_y / max(h - 1, 1)
        loc = torch.stack([shift_x, shift_y], dim=-1)  # [H, W, 2]
        return loc.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 2, H, W]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        logit = self.logit_conv(x).sigmoid()  # [B, K, H, W]
        offset = self.offset_conv(x).view(B, self.num_keypoints, 2, x.size(2), x.size(3))
        location = self._locations(offset)  # [1, 1, 2, H, W]
        keypoint = (location - offset).clamp(0.0, 1.0)  # [B, K, 2, H, W]
        return {"logit": logit, "keypoint": keypoint}


class AssocHead(nn.Module):
    """Association field head: predict edge-wise displacement vectors.

    For each directed skeletal edge (src -> dst) we output 2 channels:
      - normalized displacement vector from src to dst.
    Inference refines the dst endpoint by pulling it toward
    ``src + predicted_vector``, gated by the relative SAR confidence of the
    two endpoints.  Removing the learned edge confidence avoids the instability
    observed when the confidence branch dominated training.
    """

    def __init__(self, in_channels: int, num_edges: int):
        super().__init__()
        self.num_edges = num_edges
        hidden = max(in_channels, 256)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_edges * 2, 1),
        )
        nn.init.normal_(self.conv[-1].weight, std=0.001)
        nn.init.constant_(self.conv[-1].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, E*2, H, W] -> [B, E, 2, H, W]
        B = x.size(0)
        out = self.conv(x).view(B, self.num_edges, 2, x.size(2), x.size(3))
        return out


def sar_decode(
    logit: torch.Tensor,
    keypoint: torch.Tensor,
    mode: str = "argmax",
    topk: int = 0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Decode SAR predictions.
    - argmax: coordinate of the grid with max confidence.
    - softmax: weighted average of all grid coordinates (or top-k) by softmax(logit).
    logit:    [B, K, H, W]
    keypoint: [B, K, 2, H, W]
    Returns xy_norm: [B, K, 2]
    """
    if mode not in ("argmax", "softmax"):
        raise ValueError(f"Unknown sar_decode mode: {mode}")
    B, K, H, W = logit.shape
    logit_flat = logit.view(B * K, -1)
    kp_flat = keypoint.view(B * K, 2, -1).permute(0, 2, 1)

    if mode == "argmax":
        _, maxinds = logit_flat.max(dim=1)
        coords = kp_flat[torch.arange(B * K, device=keypoint.device), maxinds]
    else:
        if topk > 0 and topk < logit_flat.size(1):
            vals, inds = torch.topk(logit_flat, topk, dim=1)
            weights = torch.full_like(logit_flat, float("-inf"))
            weights.scatter_(1, inds, vals)
        else:
            weights = logit_flat
        weights = F.softmax(weights / temperature, dim=1)
        coords = (weights.unsqueeze(-1) * kp_flat).sum(dim=1)

    return coords.view(B, K, 2).clamp(0.0, 1.0)


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
        kp_sigmas: Optional[List[float]] = None,
        augment: bool = True,
        within_seq_val_ratio: float = 0.2,
        half_split: Optional[str] = None,
        aug_color: float = 0.15,
        aug_rotate: float = 5.0,
        aug_blur: float = 0.0,
        aug_noise: float = 0.0,
        aug_cutout: float = 0.0,
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.sigma = sigma
        self.kp_sigmas = kp_sigmas  # per-keypoint sigma, length 4
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
            recs = self._load_sequence(seq["frames_dir"], seq["ann_csv"])
            recs.sort(key=lambda r: r["path"])
            if half_split is not None:
                mid = len(recs) // 2
                if half_split == "first":
                    recs = recs[:mid]
                elif half_split == "second":
                    recs = recs[mid:]
                else:
                    raise ValueError(f"half_split must be 'first' or 'second', got {half_split}")
            self.records.extend(recs)

        # 同序列时序划分（仅单序列且未使用 half_split 时）
        if half_split is None and len(split_names) == 1 and within_seq_val_ratio > 0:
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
            s = self.kp_sigmas[k] if self.kp_sigmas is not None else self.sigma
            gt_hm[k] = make_gaussian(self.img_size, xy[k, 0], xy[k, 1], s)

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


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (simplified)."""

    def __init__(self, in_ch: int = 256, out_ch: int = 256, rates: List[int] = [6, 12, 18]):
        super().__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.atrous_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            for r in rates
        ])
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (2 + len(rates)), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        feats = [self.conv_1x1(x)]
        for conv in self.atrous_convs:
            feats.append(conv(x))
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=(h, w), mode="bilinear", align_corners=False)
        feats.append(global_feat)
        x = torch.cat(feats, dim=1)
        return self.project(x)


class Conv1x1BNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _get_timm_feature_indices(backbone: str, target_reductions: List[int] = [4, 8, 16]) -> Tuple[Tuple[int, ...], List[int]]:
    """Find timm feature indices whose spatial reduction matches target strides."""
    temp = timm.create_model(backbone, features_only=True, pretrained=False)
    info = temp.feature_info.info
    indices: List[int] = []
    channels: List[int] = []
    for tr in target_reductions:
        matches = [i for i, d in enumerate(info) if d["reduction"] == tr]
        if not matches:
            raise ValueError(f"Backbone {backbone} has no feature map with reduction={tr}; available: {info}")
        idx = matches[0]
        indices.append(idx)
        channels.append(info[idx]["num_chs"])
    return tuple(indices), channels


class TimmEncoder(nn.Module):
    """Generic timm backbone encoder that returns multi-scale features at fixed channels."""

    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        target_reductions: List[int] = [4, 8, 16],
        target_channels: List[int] = [64, 128, 256],
    ):
        super().__init__()
        self.backbone_name = backbone
        out_indices, in_channels = _get_timm_feature_indices(backbone, target_reductions)
        self.out_indices = out_indices
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            out_indices=out_indices,
            pretrained=pretrained,
        )
        actual_reductions = self.backbone.feature_info.reduction()
        if actual_reductions != target_reductions:
            raise ValueError(f"{backbone} reductions {actual_reductions} != {target_reductions}")
        self.projs = nn.ModuleList([
            Conv1x1BNReLU(c, t) for c, t in zip(in_channels, target_channels)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)  # list of 3 feature maps
        return [proj(f) for proj, f in zip(self.projs, feats)]


# torchvision backbones supported natively (no timm offline weights needed)
TORCHVISION_BACKBONES = (
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "convnext_tiny",
    "efficientnet_v2_s",
    "regnet_y_400mf",
    "swin_t",
)


class TorchvisionEncoder(nn.Module):
    """Generic torchvision encoder that returns multi-scale features at /4, /8, /16."""

    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        target_channels: List[int] = [64, 128, 256],
    ):
        super().__init__()
        self.backbone_name = backbone
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone in ("resnet18", "resnet34"):
            resnet = getattr(models, backbone)(weights=weights)
            self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # /4
            self.enc3 = resnet.layer2  # /8
            self.enc4 = resnet.layer3  # /16
            base_channels = [64, 128, 256]
            self._mode = "resnet"
        elif backbone in ("resnet50", "resnet101", "resnet152"):
            resnet = getattr(models, backbone)(weights=weights)
            self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)  # /4
            self.enc3 = resnet.layer2  # /8
            self.enc4 = resnet.layer3  # /16
            base_channels = [256, 512, 1024]
            self._mode = "resnet"
        elif backbone == "convnext_tiny":
            net = models.convnext_tiny(weights=weights)
            self.body = net.features
            # features[0]=stem(/4,96); [1]=stage0(/4,96); [2]=down(/8,192); [3]=stage1(/8,192); [4]=down(/16,384); [5]=stage2(/16,384)
            self.return_indices = (1, 3, 5)
            base_channels = [96, 192, 384]
            self._mode = "sequential"
        elif backbone == "efficientnet_v2_s":
            net = models.efficientnet_v2_s(weights=weights)
            self.body = net.features
            # features[2](/4,48); [3](/8,64); [5](/16,160)
            self.return_indices = (2, 3, 5)
            base_channels = [48, 64, 160]
            self._mode = "sequential"
        elif backbone == "regnet_y_400mf":
            net = models.regnet_y_400mf(weights=weights)
            self.body = nn.Sequential(net.stem, *net.trunk_output)
            # stem(/2,32); trunk[0](/4,48); trunk[1](/8,104); trunk[2](/16,208); trunk[3](/32,440)
            self.return_indices = (1, 2, 3)
            base_channels = [48, 104, 208]
            self._mode = "sequential"
        elif backbone == "swin_t":
            net = models.swin_t(weights=weights)
            self.body = net.features
            # features[1](/4,96); features[3](/8,192); features[5](/16,384)
            self.return_indices = (1, 3, 5)
            base_channels = [96, 192, 384]
            self._mode = "swin"
        else:
            raise ValueError(f"Unsupported TorchvisionEncoder backbone: {backbone}")

        self.projs = nn.ModuleList([
            Conv1x1BNReLU(c, t) for c, t in zip(base_channels, target_channels)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self._mode == "resnet":
            e1 = self.enc1(x)
            feats = [
                self.enc2(e1),  # /4
                self.enc3(self.enc2(e1)),  # /8
                self.enc4(self.enc3(self.enc2(e1))),  # /16
            ]
        elif self._mode == "sequential":
            feats = []
            x = self.body[0](x)
            for i, layer in enumerate(self.body[1:], start=1):
                x = layer(x)
                if i in self.return_indices:
                    feats.append(x.contiguous())
        elif self._mode == "swin":
            feats = []
            x = self.body[0](x)
            for i, layer in enumerate(self.body[1:], start=1):
                x = layer(x)
                if i in self.return_indices:
                    # Swin features are channels-last [B,H,W,C]
                    feats.append(x.permute(0, 3, 1, 2).contiguous())
        else:
            raise ValueError(f"Unknown encoder mode: {self._mode}")

        return [proj(f) for proj, f in zip(self.projs, feats)]


class HeatmapKeypointNet(nn.Module):
    def __init__(
        self,
        img_size: int,
        pretrained: bool = True,
        backbone: str = "resnet18",
        decoder_channels: List[int] = [256, 128, 64],
        decoder_stride: int = 4,
        use_aspp: bool = False,
        use_offset: bool = True,
        decoder_type: str = "heatmap_offset",
        bcir_beta: float = 10.0,
        sar_decode_mode: str = "argmax",
        sar_topk: int = 0,
        sar_temperature: float = 1.0,
        use_assoc: bool = False,
        use_kp0_hr: bool = False,
    ):
        super().__init__()
        self.img_size = img_size
        self.backbone = backbone
        self.decoder_stride = decoder_stride
        self.decoder_type = decoder_type
        self.bcir_beta = bcir_beta
        self.use_offset = use_offset and (decoder_type == "heatmap_offset")
        self.sar_decode_mode = sar_decode_mode
        self.sar_topk = sar_topk
        self.sar_temperature = sar_temperature
        self.use_assoc = use_assoc
        self.use_kp0_hr = use_kp0_hr and (decoder_type == "heatmap_offset")

        if decoder_type not in ("heatmap_offset", "bcir", "sar"):
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

        if decoder_stride not in (2, 4):
            raise ValueError("decoder_stride must be 2 or 4")

        if backbone in TORCHVISION_BACKBONES:
            self.encoder = TorchvisionEncoder(backbone, pretrained=pretrained)
        else:
            self.encoder = TimmEncoder(backbone, pretrained=pretrained)
        enc_channels = [64, 128, 256]  # projected channels

        self.aspp = ASPP(enc_channels[2], enc_channels[2]) if use_aspp else nn.Identity()

        # Decoder: /16 -> /8 -> /4
        self.up1 = UpBlock(enc_channels[2], enc_channels[1], decoder_channels[0])
        self.up2 = UpBlock(decoder_channels[0], enc_channels[0], decoder_channels[1])
        self.bottleneck = ConvBlock(decoder_channels[1], decoder_channels[2])

        # Optional high-resolution stage for stride=2 (SAR grid 256x256 at img_size=512)
        if decoder_stride == 2:
            self.up3 = nn.Sequential(
                nn.ConvTranspose2d(decoder_channels[2], decoder_channels[2], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(decoder_channels[2]),
                nn.ReLU(inplace=True),
            )
        else:
            self.up3 = None

        if decoder_type in ("heatmap_offset", "bcir"):
            # Heatmap head
            self.hm_head = nn.Sequential(
                nn.Conv2d(decoder_channels[2], 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 4, 1),
            )

        if self.use_offset:
            self.offset_head = nn.Sequential(
                nn.Conv2d(decoder_channels[2], 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 8, 1),  # 4 points * 2 channels
            )

        if self.use_kp0_hr:
            self.kp0_hr_up = nn.Sequential(
                nn.ConvTranspose2d(decoder_channels[2], decoder_channels[2], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(decoder_channels[2]),
                nn.ReLU(inplace=True),
            )
            self.kp0_hm_head = nn.Sequential(
                nn.Conv2d(decoder_channels[2], 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
            self.kp0_offset_head = nn.Sequential(
                nn.Conv2d(decoder_channels[2], 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 2, 1),
            )

        if decoder_type == "sar":
            self.sar_head = SARHead(decoder_channels[2], num_keypoints=4)
            if self.use_assoc:
                self.assoc_head = AssocHead(decoder_channels[2], num_edges=len(ASSOC_EDGES))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)

        e2, e3, e4 = self.encoder(x)  # /4, /8, /16
        e4 = self.aspp(e4)

        d1 = self.up1(e4, e3)  # [B, dec[0], H/8, W/8]
        d2 = self.up2(d1, e2)  # [B, dec[1], H/4, W/4]
        d3 = self.bottleneck(d2)  # [B, dec[2], H/4, W/4]

        if self.decoder_type == "sar":
            feat = self.up3(d3) if self.up3 is not None else d3
            sar_out = self.sar_head(feat)
            xy_norm = sar_decode(
                sar_out["logit"], sar_out["keypoint"],
                mode=self.sar_decode_mode,
                topk=self.sar_topk,
                temperature=self.sar_temperature,
            )
            out = {
                "xy": xy_norm,
                "logit": sar_out["logit"],
                "keypoint": sar_out["keypoint"],
            }
            if self.use_assoc:
                assoc = self.assoc_head(feat)
                out["assoc"] = assoc
                if not self.training:
                    out["xy"] = assoc_refine(xy_norm, assoc, sar_out["logit"], self.img_size)
            return out

        hm_logits = self.hm_head(d3)  # [B, 4, H/4, W/4]
        hm_logits = F.interpolate(hm_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        if self.use_kp0_hr:
            d3_hr = self.kp0_hr_up(d3)
            kp0_hm_logits = self.kp0_hm_head(d3_hr)
            kp0_hm_logits = F.interpolate(kp0_hm_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            hm_logits[:, 0:1] = kp0_hm_logits

        if self.decoder_type == "bcir":
            biased_xy, bcir_xy = bcir_decode(hm_logits, beta=self.bcir_beta)
            return {
                "hm_logits": hm_logits,
                "coarse_xy": biased_xy / float(self.img_size - 1),
                "xy": bcir_xy / float(self.img_size - 1),
            }

        # decoder_type == "heatmap_offset"
        coarse_xy = soft_argmax_2d(hm_logits, temperature=1.0)  # [B, 4, 2]
        coarse_xy_norm = coarse_xy / float(self.img_size - 1)

        if self.use_offset:
            offset_logits = self.offset_head(d3)  # [B, 8, H/4, W/4]
            offset_logits = F.interpolate(offset_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
            if self.use_kp0_hr:
                kp0_offset_logits = self.kp0_offset_head(d3_hr)
                kp0_offset_logits = F.interpolate(kp0_offset_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
                offset_logits[:, 0:2] = kp0_offset_logits
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


def bcir_loss(
    pred: Dict[str, torch.Tensor],
    gt_hm: torch.Tensor,
    gt_xy: torch.Tensor,
    lambda_de: float = 1.0,
    kp_weights: Optional[torch.Tensor] = None,
    xy_loss_type: str = "l1",
    wing_w: float = 10.0,
    wing_epsilon: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BCIR training loss: L_re on bias-compensated coordinates + lambda_de * L_de."""
    loss_de = mse_heatmap_loss(pred["hm_logits"], gt_hm, kp_weights=kp_weights)
    loss_re = xy_regression_loss(pred["xy"], gt_xy,
                                 loss_type=xy_loss_type, wing_w=wing_w, wing_epsilon=wing_epsilon,
                                 kp_weights=kp_weights)
    loss = loss_re + lambda_de * loss_de
    return loss, {
        "re": loss_re.item(),
        "de": loss_de.item(),
    }


def sar_loss(
    pred: Dict[str, torch.Tensor],
    gt_xy: torch.Tensor,
    img_size: int,
    sar_scale: float = 16.0,
    kp_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SAR unified Laplacian-kernel objective with optional per-keypoint weighting."""
    logit = pred["logit"]          # [B, K, H, W]
    keypoint = pred["keypoint"]    # [B, K, 2, H, W]
    B, K, H, W = logit.shape
    logit_flat = logit.view(B * K, -1)
    norm_cls = logit_flat / (logit_flat.sum(dim=1, keepdim=True) + 1e-6)
    kp_flat = keypoint.view(B * K, 2, -1).permute(0, 2, 1)  # [B*K, HW, 2]
    gt = gt_xy.view(B * K, 1, 2)
    # Convert normalised coordinate difference to pixel space, scaled by sar_scale.
    dist = torch.abs(kp_flat - gt) * (img_size - 1) / sar_scale
    reg_score = torch.exp(-dist.sum(dim=2))  # [B*K, HW]
    loss_per_kp = -torch.log((norm_cls * reg_score).sum(dim=1) + 1e-6).view(B, K)
    if kp_weights is not None:
        w = kp_weights.to(logit.device).view(K)
        loss = (loss_per_kp * w).sum() / (w.sum() * B)
    else:
        loss = loss_per_kp.mean()
    return loss, {"sar": loss.item()}


def compute_geometry_refs(manifest_json: str, split_names: set, img_size: int, half_split: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std of pairwise keypoint distances (in pixel space) from training data.

    Returns:
        mean: [P] tensor, std: [P] tensor where P = number of unique pairs.
    """
    ds = RMITHeatmapDataset(
        manifest_json, img_size, split_names, is_training=True,
        sigma=2.0, augment=False, within_seq_val_ratio=0.0, half_split=half_split,
    )
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    dists = [[] for _ in pairs]
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=False)
    for batch in loader:
        xy = batch["gt_xy"] * (img_size - 1)  # [B, 4, 2]
        for pidx, (i, j) in enumerate(pairs):
            d = torch.sqrt(((xy[:, i] - xy[:, j]) ** 2).sum(dim=-1))
            dists[pidx].append(d)
    mean = torch.stack([torch.cat(d).mean() for d in dists])
    std = torch.stack([torch.cat(d).std() for d in dists])
    return mean, std


def geometry_loss(pred_xy_px: torch.Tensor, ref_mean: torch.Tensor, ref_std: torch.Tensor, num_std: float = 3.0) -> torch.Tensor:
    """Soft skeleton constraint based on relative deviation from training-set pairwise distances.

    Only penalizes deviations beyond `num_std * ref_std / ref_mean` to allow normal articulation.
    Using relative error makes the loss scale-invariant and prevents it from dominating early training.
    """
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    losses = []
    for pidx, (i, j) in enumerate(pairs):
        d = torch.sqrt(((pred_xy_px[:, i] - pred_xy_px[:, j]) ** 2).sum(dim=-1))
        rel_tol = num_std * (ref_std[pidx] / ref_mean[pidx].clamp_min(1e-6))
        rel_err = torch.abs(d / ref_mean[pidx].clamp_min(1e-6) - 1.0) - rel_tol
        losses.append(F.relu(rel_err) ** 2)
    return torch.stack(losses).mean()


def _parse_edges(edge_str: str):
    edges = []
    for part in edge_str.split(";"):
        part = part.strip()
        if not part:
            continue
        i, j = part.split(",")
        edges.append((int(i), int(j)))
    return edges


def rigid_length_loss(pred_xy_px: torch.Tensor, gt_xy_px: torch.Tensor,
                      edges: List[Tuple[int, int]] = None,
                      loss_type: str = "smooth_l1") -> torch.Tensor:
    """Per-sample edge-length matching loss.

    Penalises the difference between predicted pairwise distances and the
    ground-truth distances for the current frame. This is much stronger than
    the dataset-level geometry_loss because it uses the true lengths.
    """
    if edges is None:
        edges = [(0, 2), (2, 1), (2, 3), (1, 3)]
    losses = []
    for i, j in edges:
        pred_len = torch.sqrt(((pred_xy_px[:, i] - pred_xy_px[:, j]) ** 2).sum(dim=-1))
        gt_len = torch.sqrt(((gt_xy_px[:, i] - gt_xy_px[:, j]) ** 2).sum(dim=-1))
        diff = pred_len - gt_len
        if loss_type == "l1":
            losses.append(torch.abs(diff))
        else:  # smooth_l1
            losses.append(F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none"))
    return torch.stack(losses).mean()


# RMIT instrument skeleton edges: shaft (2) as root, tips (1,3) and end (0) as leaves.
ASSOC_EDGES = [(2, 0), (2, 1), (2, 3), (1, 3)]


def association_loss(pred_assoc: torch.Tensor, gt_xy: torch.Tensor, img_size: int, sigma_px: float = 16.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PAF-style association field loss (pure vector, no edge confidence).

    Each directed edge predicts the normalized displacement from src to dst.
    The field is supervised only around the src endpoint, where the vector is
    meaningful; this prevents conflicting gradients near the dst endpoint.

    Args:
        pred_assoc: [B, E, 2, H, W] displacement vectors (src -> dst)
        gt_xy: [B, 4, 2] normalized to [0, 1]
        img_size: input image size
        sigma_px: spatial support of the association field (px)
    Returns:
        loss, loss_dict
    """
    B, E, _, H, W = pred_assoc.shape
    device = pred_assoc.device
    gt_px = gt_xy * (img_size - 1)  # [B, 4, 2]

    # Feature grid coordinates (normalized image coords)
    stride = img_size / W
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=device),
        torch.arange(W, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1) * stride / (img_size - 1)  # [H, W, 2] in [0,1]

    vec_losses = []
    for e, (src, dst) in enumerate(ASSOC_EDGES):
        src_pos = gt_xy[:, src].view(B, 1, 1, 2)  # [B, 1, 1, 2] normalized
        dst_pos = gt_xy[:, dst].view(B, 1, 1, 2)
        target = dst_pos - src_pos  # normalized displacement

        # Gaussian mask around the source endpoint only
        dist2_src = ((grid.unsqueeze(0) * (img_size - 1) - gt_px[:, src].view(B, 1, 1, 2)) ** 2).sum(dim=-1)
        mask = torch.exp(-dist2_src / (2.0 * sigma_px ** 2))  # [B, H, W]

        # Vector loss
        pred_vec = pred_assoc[:, e].permute(0, 2, 3, 1)  # [B, H, W, 2]
        diff = pred_vec - target
        vec_loss_e = (mask.unsqueeze(-1) * (diff ** 2)).sum() / (mask.sum() * 2.0 + 1e-6)
        vec_losses.append(vec_loss_e)

    vec_loss = torch.stack(vec_losses).mean()
    per_edge = {f"assoc_vec_e{e}": vec_losses[e].item() for e in range(E)}
    return vec_loss, per_edge


def assoc_refine(
    pred_xy: torch.Tensor,
    pred_assoc: torch.Tensor,
    logit: torch.Tensor,
    img_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Refine keypoint predictions using PAF-style vector fields.

    For each directed skeletal edge (i -> j) we sample the predicted
    displacement at endpoint i.  If endpoint i is more confident than j,
    we pull j toward ``i + predicted_vector``.  Because raw SAR logits are
    highly saturated, the gating weight is the *relative* confidence margin
    ``conf_i - conf_j`` rather than an absolute threshold.  This version does
    not rely on a learned edge confidence branch, which was found to
    destabilize training.

    Args:
        pred_xy: [B, 4, 2] direct SAR predictions (normalized)
        pred_assoc: [B, E, 2, H, W] displacement vectors (src -> dst)
        logit: [B, 4, H, W] SAR confidence logits
        img_size: input image size (kept for interface compatibility)
    Returns:
        refined_xy: [B, 4, 2] normalized
    """
    B, K, _ = pred_xy.shape

    # Normalized src coords -> [-1, 1] feature grid
    grid_xy = pred_xy * 2.0 - 1.0  # [B, 4, 2]

    # Per-keypoint SAR confidences
    conf = torch.zeros(B, K, device=pred_xy.device)
    for k in range(K):
        logit_k = logit[:, k].unsqueeze(1)  # [B, 1, H, W]
        sampled = F.grid_sample(
            logit_k, grid_xy[:, k:k+1].unsqueeze(1),
            mode="bilinear", padding_mode="border", align_corners=True,
        ).view(B)
        conf[:, k] = torch.sigmoid(sampled / temperature)

    delta = torch.zeros_like(pred_xy)
    alpha = 1.5  # global damping of association corrections
    for e, (i, j) in enumerate(ASSOC_EDGES):
        # Sample predicted displacement at the source endpoint i
        vec_i = F.grid_sample(
            pred_assoc[:, e], grid_xy[:, i].unsqueeze(1).unsqueeze(1),
            mode="bilinear", padding_mode="border", align_corners=True,
        ).squeeze(-1).permute(0, 2, 1).squeeze(1)  # [B, 2]

        # Proposal for where endpoint j should be
        cand_j = pred_xy[:, i] + vec_i  # [B, 2]

        # Pull target j toward the source proposal when source i is more confident.
        # The raw SAR confidences are saturated, so we use the relative margin
        # between source and target as a soft gating weight.
        w = (conf[:, i] - conf[:, j]).clamp(min=0.0)  # [B]

        delta[:, j] = delta[:, j] + alpha * w.unsqueeze(-1) * (cand_j - pred_xy[:, j])

    refined = pred_xy + delta
    return refined.clamp(0.0, 1.0)


# =============================================================================
# 4. 评估
# =============================================================================
def compute_metrics(pred_xy_px: torch.Tensor, gt_xy_px: torch.Tensor, img_size: int = None) -> Dict[str, float]:
    dist = torch.sqrt(((pred_xy_px - gt_xy_px) ** 2).sum(dim=-1))  # [B, 4]
    metrics = {
        "mean_error_px": float(dist.mean().item()),
        "rmse_px": float(torch.sqrt((dist ** 2).mean()).item()),
        "max_error_px": float(dist.max().item()),
        "pck@15": float((dist < 15).float().mean().item()),
        "pck@20": float((dist < 20).float().mean().item()),
        "pck@50": float((dist < 50).float().mean().item()),
        # 保留内部阈值，仅用于调试，不作为汇报重点
        "pck@5": float((dist < 5).float().mean().item()),
        "pck@10": float((dist < 10).float().mean().item()),
    }
    # 与 Du et al. 对齐：只统计检测成功（在阈值内）的点的 RMSE
    for thr in [15, 20]:
        mask = dist < thr
        if mask.any():
            metrics[f"rmse@{thr}_success_px"] = float(torch.sqrt((dist[mask] ** 2).mean()).item())
        else:
            metrics[f"rmse@{thr}_success_px"] = float("nan")
    for k in range(4):
        d = dist[:, k]
        metrics[f"kp{k}_mean_error_px"] = float(d.mean().item())
        metrics[f"kp{k}_rmse_px"] = float(torch.sqrt((d ** 2).mean()).item())
        metrics[f"kp{k}_pck@15"] = float((d < 15).float().mean().item())
        metrics[f"kp{k}_pck@20"] = float((d < 20).float().mean().item())
        metrics[f"kp{k}_pck@50"] = float((d < 50).float().mean().item())
        for thr in [15, 20]:
            m = d < thr
            if m.any():
                metrics[f"kp{k}_rmse@{thr}_success_px"] = float(torch.sqrt((d[m] ** 2).mean()).item())
            else:
                metrics[f"kp{k}_rmse@{thr}_success_px"] = float("nan")
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
    names = [s["name"] for s in sequences]  # preserve manifest order so auxiliary datasets can be appended after RMIT
    fold = fold % len(names)
    test_name = names[fold]
    train_names = {n for n in names if n != test_name}
    return train_names, {test_name}, test_name


def resolve_within_seq_split(manifest_json: str, seq_name: str) -> Tuple[set, set, str]:
    return {seq_name}, {seq_name}, seq_name


def resolve_half_split(manifest_json: str) -> Tuple[set, set, str]:
    sequences = load_rmit_manifest_sequences(manifest_json)
    names = {s["name"] for s in sequences}
    return names, names, "half"


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
    print(f"[V10] Loaded pretrained checkpoint from {checkpoint_path}")
    if missing:
        print(f"[V10] Missing keys: {missing}")
    if unexpected:
        print(f"[V10] Unexpected keys: {unexpected}")


def load_encoder_from_dr(model: HeatmapKeypointNet, dr_checkpoint_path: str):
    """DR checkpoint loading is not supported for generic timm backbones in V10."""
    print(f"[V10] DR checkpoint loading not supported for backbone {model.backbone}; skipping")
    return


SURG_PRETRAINED_PATHS = {
    "surgenetxl": "/root/autodl-tmp/glaucoma_surgery_segmentation/pretrained_model/SurgeNetXL_checkpoint_epoch0050_teacher.pth",
    "peskavlp": "/root/autodl-tmp/glaucoma_surgery_segmentation/pretrained_model/PeskaVLP.pth",
    "surgenet_convnextv2": "/root/autodl-tmp/glaucoma_surgery_segmentation/SurgeNet_ConvNextv2_checkpoint_epoch0050_teacher.pth",
}


def load_surgical_pretrained(model: HeatmapKeypointNet, kind: str):
    """Load surgical-domain pretrained backbone weights into model.encoder.

    Supports:
      - surgenetxl: CAFormer-S18 DINO pretrained, mapped to timm caformer_s18
      - peskavlp: ResNet-based video-language pretrained, mapped to resnet18/34/50
      - surgenet_convnextv2: ConvNeXt-V2 DINO pretrained (legacy)
    """
    ckpt_path = SURG_PRETRAINED_PATHS.get(kind)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[V10] Surgical pretrained '{kind}' not found at {ckpt_path}; skipping")
        return

    print(f"[V10] Loading surgical pretrained '{kind}' from {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Strip common prefixes
    sd = {k: v for k, v in sd.items() if "num_batches_tracked" not in k}

    enc = model.encoder
    if kind == "surgenetxl":
        # SurgeNetXL: metaformer keys -> timm caformer_s18 keys via surgenetxl_model mapping
        try:
            import sys as _sys
            _sys.path.insert(0, "/root/autodl-tmp/glaucoma_surgery_segmentation/glaucoma_surgery_segmentation/code")
            from surgenetxl_model import build_surgenetxl_encoder
            # build a fresh encoder and copy its weights into model.encoder.backbone
            snxl = build_surgenetxl_encoder(ckpt_path, "cpu", activation="starrelu")
            # snxl is nn.Sequential(backbone, pool); extract backbone
            snxl_backbone = snxl[0] if isinstance(snxl, nn.Sequential) else snxl
            # Try to load into enc.backbone (timm features_only caformer_s18)
            # The features_only version shares the same backbone weights
            missing, unexpected = enc.backbone.load_state_dict(snxl_backbone.state_dict(), strict=False)
            print(f"[V10] SurgeNetXL loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[V10] SurgeNetXL loading failed: {e}")
    elif kind == "peskavlp":
        # PeskaVLP: keys are backbone_img.model.* (ResNet50 with bottleneck); map to torchvision resnet
        peska_sd = {}
        for k, v in sd.items():
            if k.startswith("backbone_img.model."):
                peska_sd[k[len("backbone_img.model."):]] = v
        # enc is TorchvisionEncoder for resnet; load state dict into its sub-modules
        try:
            resnet = getattr(models, enc.backbone_name)(weights=None)
            missing, unexpected = resnet.load_state_dict(peska_sd, strict=False)
            n_loaded = len(peska_sd) - len(missing)
            print(f"[V10] PeskaVLP resnet load: loaded {n_loaded}/{len(peska_sd)} (missing={len(missing)}, unexpected={len(unexpected)})")
            if n_loaded == 0:
                print("[V10] PeskaVLP keys do not match resnet; skipping")
                return
            # Copy the resnet weights into enc's enc1-enc4 modules
            if enc.backbone_name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
                enc.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
                enc.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)
                enc.enc3 = resnet.layer2
                enc.enc4 = resnet.layer3
                # Move to the same device as the rest of the model
                try:
                    device = next(model.parameters()).device
                    enc.to(device)
                except Exception:
                    pass
                print("[V10] PeskaVLP weights copied into encoder")
        except Exception as e:
            print(f"[V10] PeskaVLP loading failed: {e}")
    elif kind == "surgenet_convnextv2":
        # Legacy SurgeNet ConvNeXt-V2: keys start with backbone.
        try:
            c2_sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
            missing, unexpected = enc.backbone.load_state_dict(c2_sd, strict=False)
            print(f"[V10] SurgeNet ConvNeXt-V2 loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            print(f"[V10] SurgeNet ConvNeXt-V2 loading failed: {e}")


def train_rmit(cfg, device: torch.device) -> None:
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    model = HeatmapKeypointNet(
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
        use_assoc=cfg.use_assoc,
        use_kp0_hr=cfg.use_kp0_hr,
    ).to(device)

    if cfg.dr_checkpoint and os.path.exists(cfg.dr_checkpoint):
        load_encoder_from_dr(model, cfg.dr_checkpoint)

    if cfg.pt_checkpoint and os.path.exists(cfg.pt_checkpoint):
        load_pretrained_weights(model, cfg.pt_checkpoint)

    if cfg.surg_pretrained:
        load_surgical_pretrained(model, cfg.surg_pretrained)
        # Ensure model is on the correct device after potentially replacing encoder layers
        model.to(device)

    if cfg.freeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("[V10] Freeze backbone encoder")

    if getattr(cfg, "freeze_sar", False):
        for m in (model.encoder, model.aspp, model.up1, model.up2, model.bottleneck, model.up3, model.sar_head):
            if m is None:
                continue
            for p in m.parameters():
                p.requires_grad = False
        print("[V10] Freeze SAR pipeline (encoder / decoder / SAR head), train association head only")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[V10] Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    if cfg.eval_only:
        ckpt_path = cfg.checkpoint if cfg.checkpoint else os.path.join(cfg.out_dir, "best.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        load_pretrained_weights(model, ckpt_path)
        if cfg.protocol == "loso":
            train_names, val_names, test_name = resolve_loso_split(cfg.manifest, cfg.fold)
        elif cfg.protocol == "half":
            train_names, val_names, test_name = resolve_half_split(cfg.manifest)
        else:
            train_names, val_names, test_name = resolve_within_seq_split(cfg.manifest, cfg.seq)
        print(f"[V11 eval] protocol={cfg.protocol} fold={cfg.fold} seq={test_name}")
        val_ratio = 0.0 if cfg.protocol in ("loso", "half") else cfg.within_seq_val_ratio
        half = "second" if cfg.protocol == "half" else None
        val_ds = RMITHeatmapDataset(
            cfg.manifest, cfg.img_size, val_names, False,
            sigma=cfg.sigma, kp_sigmas=cfg.kp_sigmas, augment=False, within_seq_val_ratio=val_ratio,
            half_split=half,
        )
        val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        metrics = evaluate(model, val_ld, device, cfg.img_size, adabn=cfg.adabn, adabn_passes=cfg.adabn_passes)
        print(f"[V11 eval] {test_name}: err={metrics['mean_error_px']:.2f}px pck@20={metrics['pck@20']:.3f} pck@50={metrics['pck@50']:.3f}")
        print(json.dumps(metrics, indent=2))
        return

    kp_weights = None
    if cfg.kp_adaptive:
        init_w = cfg.kp_weights if cfg.kp_weights is not None else [1.0, 1.0, 1.0, 1.0]
        kp_weights = torch.tensor(init_w, dtype=torch.float32)
        print(f"[V10] Adaptive kp weights initialized: {kp_weights.tolist()}")
    elif cfg.kp_weights is not None:
        kp_weights = torch.tensor(cfg.kp_weights, dtype=torch.float32)
        if kp_weights.numel() != 4:
            raise ValueError("--kp_weights must have exactly 4 values")
        print(f"[V10] Static kp_weights: {kp_weights.tolist()}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.wd,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))

    if cfg.protocol == "loso":
        train_names, val_names, test_name = resolve_loso_split(cfg.manifest, cfg.fold)
        print(f"[V10] LOSO fold {cfg.fold}: train={sorted(train_names)}, val={test_name}")
    elif cfg.protocol == "half":
        train_names, val_names, test_name = resolve_half_split(cfg.manifest)
        print(f"[V10] Half split: train=first halves, val=second halves, seqs={sorted(train_names)}")
    else:
        train_names, val_names, test_name = resolve_within_seq_split(cfg.manifest, cfg.seq)
        print(f"[V10] Within-seq {cfg.seq}")

    train_half = "first" if cfg.protocol == "half" else None
    val_half = "second" if cfg.protocol == "half" else None
    train_ds = RMITHeatmapDataset(
        cfg.manifest, cfg.img_size, train_names, True,
        sigma=cfg.sigma, kp_sigmas=cfg.kp_sigmas, augment=True, within_seq_val_ratio=cfg.within_seq_val_ratio,
        half_split=train_half,
        aug_color=cfg.aug_color, aug_rotate=cfg.aug_rotate,
        aug_blur=cfg.aug_blur, aug_noise=cfg.aug_noise, aug_cutout=cfg.aug_cutout,
    )
    # LOSO / half: validate on the held-out portion, not a within-seq ratio split
    val_ratio = 0.0 if cfg.protocol in ("loso", "half") else cfg.within_seq_val_ratio
    val_ds = RMITHeatmapDataset(
        cfg.manifest, cfg.img_size, val_names, False,
        sigma=cfg.sigma, kp_sigmas=cfg.kp_sigmas, augment=False, within_seq_val_ratio=val_ratio,
        half_split=val_half,
    )

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print(f"[V10] Train: {len(train_ds)}, Val: {len(val_ds)}")

    geo_mean, geo_std = None, None
    if cfg.geometry_loss_weight > 0.0:
        print(f"[V10] Computing geometry refs for split {sorted(train_names)}...")
        geo_mean, geo_std = compute_geometry_refs(cfg.manifest, train_names, cfg.img_size, half_split=train_half)
        geo_mean, geo_std = geo_mean.to(device), geo_std.to(device)
        print(f"[V10] Geometry refs mean: {geo_mean.cpu().tolist()}")

    total_steps = max(1, len(train_ld)) * cfg.epochs
    warmup_steps = max(1, len(train_ld)) * cfg.warmup_epochs
    global_step = 0
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_terms: Dict[str, float] = collections.defaultdict(float)

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
                if cfg.decoder_type == "heatmap_offset":
                    loss, loss_dict = total_loss(
                        pred, gt_hm, gt_xy,
                        kp_weights=kp_weights,
                        xy_loss_type=cfg.xy_loss_type,
                        wing_w=cfg.wing_w,
                        wing_epsilon=cfg.wing_epsilon,
                        xy_weight=cfg.xy_weight,
                        coarse_weight=cfg.coarse_weight,
                    )
                elif cfg.decoder_type == "bcir":
                    lambda_de = 1.0 if epoch < cfg.bcir_gaussian_epochs else 0.0
                    loss, loss_dict = bcir_loss(
                        pred, gt_hm, gt_xy,
                        lambda_de=lambda_de,
                        kp_weights=kp_weights,
                        xy_loss_type=cfg.xy_loss_type,
                        wing_w=cfg.wing_w,
                        wing_epsilon=cfg.wing_epsilon,
                    )
                elif cfg.decoder_type == "sar":
                    loss, loss_dict = sar_loss(
                        pred, gt_xy, cfg.img_size, sar_scale=cfg.sar_scale,
                        kp_weights=kp_weights,
                    )
                else:
                    raise ValueError(f"Unknown decoder_type: {cfg.decoder_type}")

                if cfg.geometry_loss_weight > 0.0 and geo_mean is not None:
                    pred_xy_px = pred["xy"] * (cfg.img_size - 1)
                    geo = geometry_loss(pred_xy_px, geo_mean, geo_std, cfg.geometry_loss_num_std)
                    loss = loss + cfg.geometry_loss_weight * geo
                    loss_dict["geo"] = geo.item()

                if cfg.rigid_length_loss_weight > 0.0:
                    pred_xy_px = pred["xy"] * (cfg.img_size - 1)
                    gt_xy_px = gt_xy * (cfg.img_size - 1)
                    rl = rigid_length_loss(
                        pred_xy_px, gt_xy_px,
                        edges=_parse_edges(cfg.rigid_length_edges),
                        loss_type=cfg.rigid_length_loss_type,
                    )
                    loss = loss + cfg.rigid_length_loss_weight * rl
                    loss_dict["rigid_len"] = rl.item()

                if cfg.use_assoc and "assoc" in pred:
                    assoc_loss_val, assoc_dict = association_loss(
                        pred["assoc"], gt_xy, cfg.img_size, sigma_px=cfg.assoc_sigma_px,
                    )
                    loss = loss + cfg.assoc_loss_weight * assoc_loss_val
                    loss_dict["assoc"] = assoc_loss_val.item()
                    loss_dict.update(assoc_dict)

            if not torch.isfinite(loss):
                print(f"[V11] skip non-finite loss")
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
                writer.add_scalar("v11/train_loss", loss.item(), global_step)
                for lk, lv in loss_dict.items():
                    writer.add_scalar(f"v11/train_{lk}", lv, global_step)
                writer.add_scalar("v11/lr", lr_now, global_step)

        metrics = evaluate(model, val_ld, device, cfg.img_size, adabn=cfg.adabn, adabn_passes=cfg.adabn_passes)
        for k, v in metrics.items():
            writer.add_scalar(f"v11/val_{k}", v, epoch)

        avg_loss = running_loss / max(len(train_ld), 1)
        avg_terms = {k: v / max(len(train_ld), 1) for k, v in running_terms.items()}
        # adaptive keypoint reweighting based on validation per-kp error
        if cfg.kp_adaptive:
            per_kp_err = torch.tensor([metrics[f"kp{k}_mean_error_px"] for k in range(4)], dtype=torch.float32)
            mean_err = per_kp_err.mean().clamp_min(1e-6)
            rel = per_kp_err / mean_err  # >1 for hard keypoints
            mom = cfg.kp_adapt_momentum
            kp_weights = mom * kp_weights + (1.0 - mom) * rel
            kp_weights = torch.clamp(kp_weights, cfg.kp_adapt_min_weight, cfg.kp_adapt_max_weight)
            if cfg.kp_adapt_normalize:
                kp_weights = kp_weights / kp_weights.sum() * 4.0
            print(f"[V10] adaptive kp_weights: {kp_weights.tolist()}")
            for k in range(4):
                writer.add_scalar(f"v11/kp_weight_{k}", kp_weights[k].item(), epoch)

        loss_terms = " ".join([f"{k}={v:.4f}" for k, v in avg_terms.items()])
        msg = (
            f"[V11 {cfg.decoder_type}] epoch {epoch+1}/{cfg.epochs} "
            f"loss={avg_loss:.4f} {loss_terms} "
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
            print(f"[V10] Early stopping at epoch {epoch+1}")
            break

    writer.close()
    print(f"[V10] Best val error: {best_metric:.2f}px")


# =============================================================================
# 7. 命令行
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DR-SurgBridgeNet V11: BCIR / SAR / heatmap+offset keypoint tracker")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking/data/manifests/rmit_manifest.json")
    parser.add_argument("--protocol", type=str, default="within_seq", choices=["loso", "within_seq", "half"])
    parser.add_argument("--seq", type=str, default="seq1", choices=["seq1", "seq2", "seq3"])
    parser.add_argument("--within_seq_val_ratio", type=float, default=0.2)

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--kp_sigmas", type=float, nargs=4, default=None, help="per-keypoint Gaussian sigma for heatmap, e.g. 5 3 3 3")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument(
        "--backbone", type=str, default="resnet18",
        help="backbone name. torchvision: resnet18/34/50/101/152, convnext_tiny, efficientnet_v2_s, regnet_y_400mf, swin_t. "
             "timm (offline weights may fail): convnext_tiny.fb_in1k, fastvit_t8, efficientnet_b0, etc.",
    )
    parser.add_argument("--decoder_channels", type=int, nargs=3, default=[256, 128, 64])
    parser.add_argument("--decoder_stride", type=int, default=4, choices=[2, 4])
    parser.add_argument("--use_aspp", action="store_true")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--decoder_type", type=str, default="heatmap_offset", choices=["heatmap_offset", "bcir", "sar"])
    parser.add_argument("--bcir_beta", type=float, default=10.0, help="BCIR softmax temperature scaling")
    parser.add_argument("--bcir_gaussian_epochs", type=int, default=50, help="epochs with Gaussian detection loss for BCIR")
    parser.add_argument("--sar_scale", type=float, default=16.0, help="SAR Laplacian kernel spatial scale")
    parser.add_argument("--sar_decode_mode", type=str, default="argmax", choices=["argmax", "softmax"], help="SAR decode mode")
    parser.add_argument("--sar_topk", type=int, default=0, help="SAR softmax decode top-k (0=all)")
    parser.add_argument("--sar_temperature", type=float, default=1.0, help="SAR softmax temperature")
    parser.add_argument("--geometry_loss_weight", type=float, default=0.0, help="weight for soft skeleton geometry loss (0=disabled)")
    parser.add_argument("--geometry_loss_num_std", type=float, default=3.0, help="geometry loss tolerance in multiples of training std")
    parser.add_argument("--use_kp0_hr", action="store_true", help="add high-resolution dedicated head for keypoint 0 (heatmap_offset only)")
    parser.add_argument("--rigid_length_loss_weight", type=float, default=0.0, help="weight for per-sample rigid edge-length loss (0=disabled)")
    parser.add_argument("--rigid_length_edges", type=str, default="0,2;2,1;2,3;1,3", help="edges for rigid length loss as 'i,j;i,j'")
    parser.add_argument("--rigid_length_loss_type", type=str, default="smooth_l1", choices=["l1", "smooth_l1"], help="rigid length loss type")
    parser.add_argument("--use_assoc", action="store_true", help="add association field branch (PAF-style skeleton constraint)")
    parser.add_argument("--assoc_loss_weight", type=float, default=1.0, help="weight for association field loss")
    parser.add_argument("--assoc_sigma_px", type=float, default=16.0, help="spatial support of association field supervision (px)")
    parser.add_argument("--pt_checkpoint", type=str, default=None, help="pretrained encoder+decoder checkpoint from FOVEA")
    parser.add_argument("--dr_checkpoint", type=str, default=None, help="DR classification encoder checkpoint")
    parser.add_argument("--surg_pretrained", type=str, default=None,
                        choices=["surgenetxl", "peskavlp", "surgenet_convnextv2"],
                        help="load surgical-domain pretrained backbone weights")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to model checkpoint for eval_only mode")
    parser.add_argument("--eval_only", action="store_true", help="only evaluate the checkpoint and exit")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--freeze_sar", action="store_true", help="freeze everything except the association head (for PAF-only fine-tuning)")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)

    # loss / keypoint reweighting
    parser.add_argument("--kp_weights", type=float, nargs=4, default=None, help="per-keypoint loss weights, e.g. 2 1 1 1")
    parser.add_argument("--kp_adaptive", action="store_true", help="adapt kp weights each epoch based on val per-kp error")
    parser.add_argument("--kp_adapt_momentum", type=float, default=0.9)
    parser.add_argument("--kp_adapt_max_weight", type=float, default=8.0)
    parser.add_argument("--kp_adapt_min_weight", type=float, default=0.3)
    parser.add_argument("--kp_adapt_normalize", action="store_true", help="normalize adaptive kp weights to sum=4 (legacy behaviour)")
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
