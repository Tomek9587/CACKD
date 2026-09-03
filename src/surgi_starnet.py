#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SurgiStarNet for RMIT Surgical Instrument Tip Tracking (Task #12, #15)

Based on StarNet (CVPR 2024): Element-wise multiplication (Star Operation) as attention replacement.
- Backbone: StarNet (~1.2M parameters)
- Head: U-Net style decoder with skip connections
- Preprocessing: Circular FOV detection and cropping

Three-Stage Training:
- Stage 1: DR classification (pretrain backbone)
- Stage 2: FOVEA localization (heatmap head)
- Stage 3: RMIT tracking (fine-tune on surgical data)
"""

import os
import sys
import json
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF


# =============================================================================
# Logger Setup
# =============================================================================

def setup_logger(out_dir: str, name: str = 'train') -> logging.Logger:
    """
    设置同时输出到控制台和文件的 logger。
    
    Args:
        out_dir: 输出目录
        name: logger 名称
        
    Returns:
        logger: 配置好的 logger 对象
    """
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f'{name}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有 handlers
    
    # 文件 handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 控制台 handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# =============================================================================
# Soft Argmax Implementation
# =============================================================================

def soft_argmax_2d(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute soft-argmax for 2D heatmap.
    
    Args:
        heatmap: (B, 1, H, W) or (B, H, W)
        temperature: Softmax temperature (lower = sharper)
    
    Returns:
        coords: (B, 2) with (x, y) in normalized coordinates [0, 1]
    """
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(1)
    B, C, H, W = heatmap.shape
    
    # Flatten spatial dimensions and apply softmax
    heatmap_flat = heatmap.view(B, C, -1)  # (B, C, H*W)
    weights = F.softmax(heatmap_flat / temperature, dim=-1)  # (B, C, H*W)
    weights = weights.view(B, C, H, W)
    
    # Create coordinate grids
    y_coords = torch.linspace(0, 1, H, device=heatmap.device).view(1, 1, H, 1)
    x_coords = torch.linspace(0, 1, W, device=heatmap.device).view(1, 1, 1, W)
    
    # Compute expected coordinates
    x = (weights * x_coords).sum(dim=(2, 3))  # (B, C)
    y = (weights * y_coords).sum(dim=(2, 3))  # (B, C)
    
    coords = torch.stack([x.squeeze(-1), y.squeeze(-1)], dim=-1)  # (B, 2)
    return coords


def generate_gaussian_heatmap(
    coord: np.ndarray,
    heatmap_size: Tuple[int, int],
    sigma: float = 2.0
) -> np.ndarray:
    """
    Generate Gaussian heatmap centered at coord.
    
    Args:
        coord: (x, y) in pixel coordinates
        heatmap_size: (H, W)
        sigma: Gaussian sigma in pixels
    
    Returns:
        heatmap: (H, W) float32
    """
    H, W = heatmap_size
    x, y = coord
    
    # Create coordinate grids
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    
    # Gaussian
    heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    heatmap = heatmap.astype(np.float32)
    
    # Normalize to [0, 1]
    heatmap = heatmap / (heatmap.max() + 1e-8)
    
    return heatmap


# =============================================================================
# Circular FOV Detection and Cropping
# =============================================================================

def detect_circular_fov(image: np.ndarray, expand_ratio: float = 0.1) -> Tuple[int, int, int, int]:
    """
    Detect the circular field of view and return bounding box.
    
    Args:
        image: RGB image (H, W, 3)
        expand_ratio: Expand bbox by this ratio
    
    Returns:
        bbox: (x1, y1, x2, y2) for cropping
    """
    # Convert to grayscale
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Threshold to find non-black region
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to center crop
        H, W = gray.shape
        size = min(H, W)
        x1 = (W - size) // 2
        y1 = (H - size) // 2
        return (x1, y1, x1 + size, y1 + size)
    
    # Find largest contour (the circular FOV)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Expand the bbox
    expand_w = int(w * expand_ratio)
    expand_h = int(h * expand_ratio)
    x1 = max(0, x - expand_w)
    y1 = max(0, y - expand_h)
    x2 = min(image.shape[1], x + w + expand_w)
    y2 = min(image.shape[0], y + h + expand_h)
    
    return (x1, y1, x2, y2)


def crop_and_transform(
    image: np.ndarray,
    gt_xy: np.ndarray,
    target_size: int,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    no_crop: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Crop image to FOV and transform coordinates.
    
    Args:
        image: RGB image (H, W, 3)
        gt_xy: Ground truth (x, y) in original coordinates
        target_size: Output size (square)
        bbox: Optional pre-computed bbox (x1, y1, x2, y2)
        no_crop: If True, skip crop and only resize
    
    Returns:
        cropped_image: (target_size, target_size, 3)
        new_gt_xy: (x, y) in cropped coordinates
        transform_info: Dict for coordinate restoration
    """
    if no_crop:
        # No crop mode: directly resize to target size
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # Transform coordinates (only scale, no offset)
        scale_x = target_size / orig_w
        scale_y = target_size / orig_h
        
        new_x = gt_xy[0] * scale_x
        new_y = gt_xy[1] * scale_y
        
        new_gt_xy = np.array([new_x, new_y], dtype=np.float32)
        
        # Transform info for restoration (no crop, only resize)
        transform_info = {
            'original_shape': (orig_h, orig_w),
            'target_size': target_size,
            'scale': (scale_x, scale_y),
            'offset': (0, 0),  # No offset in no_crop mode
            'no_crop': True
        }
        
        return resized, new_gt_xy, transform_info
    
    # Original crop mode
    if bbox is None:
        bbox = detect_circular_fov(image)
    
    x1, y1, x2, y2 = bbox
    crop_h = y2 - y1
    crop_w = x2 - x1
    
    # Crop
    cropped = image[y1:y2, x1:x2].copy()
    
    # Resize
    cropped = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Transform coordinates
    scale_x = target_size / crop_w
    scale_y = target_size / crop_h
    
    new_x = (gt_xy[0] - x1) * scale_x
    new_y = (gt_xy[1] - y1) * scale_y
    
    new_gt_xy = np.array([new_x, new_y], dtype=np.float32)
    
    # Transform info for restoration
    transform_info = {
        'original_bbox': bbox,
        'original_shape': image.shape[:2],
        'target_size': target_size,
        'scale': (scale_x, scale_y),
        'offset': (x1, y1),
        'no_crop': False
    }
    
    return cropped, new_gt_xy, transform_info


def restore_coordinates(
    pred_xy: np.ndarray,
    transform_info: Dict[str, Any]
) -> np.ndarray:
    """
    Restore predicted coordinates to original image coordinates.
    
    Args:
        pred_xy: (x, y) in cropped coordinates
        transform_info: Transform info from crop_and_transform
    
    Returns:
        original_xy: (x, y) in original image coordinates
    """
    x1, y1 = transform_info['offset']
    scale_x, scale_y = transform_info['scale']
    
    original_x = pred_xy[0] / scale_x + x1
    original_y = pred_xy[1] / scale_y + y1
    
    return np.array([original_x, original_y], dtype=np.float32)


# =============================================================================
# StarNet Architecture (CVPR 2024)
# =============================================================================

class StarOperation(nn.Module):
    """StarNet核心: element-wise乘法替代注意力, CVPR 2024"""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim, 1)  # 映射分支1
        self.fc2 = nn.Conv2d(dim, dim, 1)  # 映射分支2
        self.act = nn.ReLU6()
    
    def forward(self, x):
        return self.act(self.fc1(x)) * self.fc2(x)  # Star Operation: 逐元素相乘


class StarBlock(nn.Module):
    """StarNet基本块"""
    def __init__(self, dim, mlp_ratio=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)  # 深度可分离卷积
        self.norm = nn.BatchNorm2d(dim)
        self.star = StarOperation(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_dim, 1),
            nn.ReLU6(),
            nn.Conv2d(mlp_dim, dim, 1)
        )
        self.norm2 = nn.BatchNorm2d(dim)
    
    def forward(self, x):
        residual = x
        x = self.norm(self.dwconv(x))
        x = self.star(x)
        x = residual + x
        residual = x
        x = self.norm2(self.mlp(x))
        return residual + x


class SurgiStarNetHeatmap(nn.Module):
    """
    StarNet backbone with U-Net style decoder for tip detection.
    ~1.2M parameters, trained from scratch.
    """
    
    def __init__(
        self,
        heatmap_size: int = 64,
        dims: Tuple[int, int, int, int] = (24, 48, 96, 192),
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        mlp_ratio: float = 2.0
    ):
        super().__init__()
        
        self.heatmap_size = heatmap_size
        self.dims = dims
        
        # Stem: Conv2d(3, 32, 3, 2, 1) + BN + ReLU → 128x128x32
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: 下采样Conv + 2x StarBlock, dim=32 → 64x64x32
        self.downsample1 = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0])
        )
        self.stage1 = nn.Sequential(*[StarBlock(dims[0], mlp_ratio) for _ in range(depths[0])])
        
        # Stage 2: 下采样Conv + 2x StarBlock, dim=64 → 32x32x64
        self.downsample2 = nn.Sequential(
            nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[1])
        )
        self.stage2 = nn.Sequential(*[StarBlock(dims[1], mlp_ratio) for _ in range(depths[1])])
        
        # Stage 3: 下采样Conv + 3x StarBlock, dim=128 → 16x16x128
        self.downsample3 = nn.Sequential(
            nn.Conv2d(dims[1], dims[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[2])
        )
        self.stage3 = nn.Sequential(*[StarBlock(dims[2], mlp_ratio) for _ in range(depths[2])])
        
        # Stage 4: 下采样Conv + 2x StarBlock, dim=256 → 8x8x256
        self.downsample4 = nn.Sequential(
            nn.Conv2d(dims[2], dims[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[3])
        )
        self.stage4 = nn.Sequential(*[StarBlock(dims[3], mlp_ratio) for _ in range(depths[3])])
        
        # Decoder with skip connections (U-Net style)
        # Up1: 8x8x256 → 16x16x128 (+ Stage3 skip)
        self.up1 = nn.ConvTranspose2d(dims[3], dims[2], kernel_size=4, stride=2, padding=1)
        self.up1_bn = nn.BatchNorm2d(dims[2])
        self.up1_conv = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.ReLU(inplace=True)
        )
        
        # Up2: 16x16x128 → 32x32x64 (+ Stage2 skip)
        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], kernel_size=4, stride=2, padding=1)
        self.up2_bn = nn.BatchNorm2d(dims[1])
        self.up2_conv = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.ReLU(inplace=True)
        )
        
        # Up3: 32x32x64 → 64x64x32 (+ Stage1 skip)
        self.up3 = nn.ConvTranspose2d(dims[1], dims[0], kernel_size=4, stride=2, padding=1)
        self.up3_bn = nn.BatchNorm2d(dims[0])
        self.up3_conv = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Final heatmap head: 64x64x32 → 64x64x1
        self.heatmap_head = nn.Conv2d(dims[0], 1, kernel_size=1)
        
        # Initialize weights
        self._init_weights()
        
        # Count parameters
        self._count_parameters()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SurgiStarNet] Total parameters: {total:,}")
        print(f"[SurgiStarNet] Trainable parameters: {trainable:,}")
        print(f"[SurgiStarNet] Model size: {total * 4 / 1024 / 1024:.2f} MB")
        
        # Estimate VRAM
        vram_estimate = total * 4 * 2 / 1024 / 1024 + 100
        print(f"[SurgiStarNet] Estimated VRAM (per sample): ~{vram_estimate:.0f} MB")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            dict with keys:
                - heatmap: (B, 1, heatmap_size, heatmap_size)
                - coords: (B, 2) decoded coordinates in [0, 1]
        """
        # Stem: 256x256 → 128x128
        x = self.stem(x)
        
        # Stage 1: 128x128 → 64x64
        x = self.downsample1(x)
        x = self.stage1(x)
        skip1 = x  # 64x64x32
        
        # Stage 2: 64x64 → 32x32
        x = self.downsample2(x)
        x = self.stage2(x)
        skip2 = x  # 32x32x64
        
        # Stage 3: 32x32 → 16x16
        x = self.downsample3(x)
        x = self.stage3(x)
        skip3 = x  # 16x16x128
        
        # Stage 4: 16x16 → 8x8
        x = self.downsample4(x)
        x = self.stage4(x)  # 8x8x256
        
        # Decoder with skip connections
        # Up1: 8x8 → 16x16
        x = F.relu(self.up1_bn(self.up1(x)))
        x = torch.cat([x, skip3], dim=1)
        x = self.up1_conv(x)
        
        # Up2: 16x16 → 32x32
        x = F.relu(self.up2_bn(self.up2(x)))
        x = torch.cat([x, skip2], dim=1)
        x = self.up2_conv(x)
        
        # Up3: 32x32 → 64x64
        x = F.relu(self.up3_bn(self.up3(x)))
        x = torch.cat([x, skip1], dim=1)
        x = self.up3_conv(x)
        
        # Heatmap head
        heatmap = self.heatmap_head(x)  # (B, 1, 64, 64)
        
        # Decode coordinates
        coords = soft_argmax_2d(heatmap, temperature=0.1)  # (B, 2)
        
        return {
            'heatmap': heatmap,
            'coords': coords
        }
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取 backbone 特征（用于 Stage 1 分类）。
        返回 Stage 4 的输出 (8x8x256)。
        """
        # Stem: 256x256 → 128x128
        x = self.stem(x)
        
        # Stage 1: 128x128 → 64x64
        x = self.downsample1(x)
        x = self.stage1(x)
        
        # Stage 2: 64x64 → 32x32
        x = self.downsample2(x)
        x = self.stage2(x)
        
        # Stage 3: 32x32 → 16x16
        x = self.downsample3(x)
        x = self.stage3(x)
        
        # Stage 4: 16x16 → 8x8
        x = self.downsample4(x)
        x = self.stage4(x)  # 8x8x256
        
        return x
    
    def extract_multi_scale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多尺度特征（用于 FOVEA Stage 2 跨域对比学习）。
        返回 stage2, stage3, stage4 的输出特征图（3 个尺度）。
        
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            List of [f2, f3, f4] feature maps
            - f2: [B, dims[1], H/8, W/8] (e.g., 48 channels)
            - f3: [B, dims[2], H/16, W/16] (e.g., 96 channels)
            - f4: [B, dims[3], H/32, W/32] (e.g., 192 channels)
        """
        # Stem: 256x256 → 128x128
        x = self.stem(x)
        
        # Stage 1: 128x128 → 64x64
        x = self.downsample1(x)
        x = self.stage1(x)
        
        # Stage 2: 64x64 → 32x32
        x = self.downsample2(x)
        f2 = self.stage2(x)  # [B, 48, H/8, W/8]
        
        # Stage 3: 32x32 → 16x16
        x = self.downsample3(f2)
        f3 = self.stage3(x)  # [B, 96, H/16, W/16]
        
        # Stage 4: 16x16 → 8x8
        x = self.downsample4(f3)
        f4 = self.stage4(x)  # [B, 192, H/32, W/32]
        
        return [f2, f3, f4]


class ClassificationHead(nn.Module):
    """
    临时分类头，用于 Stage 1 DR 分类。
    结构: GlobalAvgPool → FC → num_classes
    """
    
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        
        # Initialize
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) backbone features
            
        Returns:
            logits: (B, num_classes)
        """
        x = self.pool(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        logits = self.fc(x)  # (B, num_classes)
        return logits


class SurgiStarNetWithClassifier(nn.Module):
    """
    带分类头的 SurgiStarNet，用于 Stage 1 DR 分类训练。
    训练完成后可丢弃分类头，只保留 backbone。
    """
    
    def __init__(
        self,
        backbone: SurgiStarNetHeatmap,
        num_classes: int = 2,
        freeze_backbone: bool = False
    ):
        super().__init__()
        self.backbone = backbone
        self.classifier = ClassificationHead(
            in_channels=backbone.dims[-1],  # 最后一个 stage 的 dim (192)
            num_classes=num_classes
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) input image
            
        Returns:
            logits: (B, num_classes)
        """
        features = self.backbone.get_backbone_features(x)
        logits = self.classifier(features)
        return logits


# =============================================================================
# Dataset
# =============================================================================

class DRImageFolderDataset(Dataset):
    """
    DR 分类数据集，支持 ImageFolder 格式（目录结构：DR/和No_DR/子目录）。
    """
    
    def __init__(
        self,
        root_dir: str,
        img_size: int = 256,
        indices: Optional[List[int]] = None,
        augment: bool = True,
        label_map: Optional[Dict[str, int]] = None,
        logger: Optional[logging.Logger] = None
    ):
        from torchvision.datasets import ImageFolder
        
        self.base = ImageFolder(root=root_dir)
        self.img_size = img_size
        self.augment = augment
        self.indices = list(indices) if indices is not None else list(range(len(self.base.samples)))
        self.label_map = dict(label_map) if label_map is not None else {
            name: idx for name, idx in self.base.class_to_idx.items()
        }
        self.logger = logger
        
        self.items = []
        for sample_idx in self.indices:
            path, orig_label = self.base.samples[sample_idx]
            class_name = self.base.classes[orig_label]
            self.items.append((path, self.label_map[class_name]))
        
        # Data augmentation transforms
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        
        msg = f"[SurgiStarNet] DRImageFolderDataset: {root_dir}, {len(self.items)} samples, augment={self.augment}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[idx]
        
        # Load image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Data augmentation
        if self.augment:
            # Random horizontal flip
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
            # Color jitter
            image_tensor = self.color_jitter(image_tensor)
        
        # Normalize to [-1, 1] range
        image_tensor = image_tensor * 2 - 1
        
        return image_tensor, torch.tensor(label, dtype=torch.long)


class FOVEAPairedDataset(Dataset):
    """
    FOVEA 术前-术中图像配对数据集（跨域对比学习）。
    加载 pre_img 和 intra_img 进行配对训练。
    """
    
    def __init__(
        self,
        pairs: List[Dict],
        img_size: int = 256,
        augment: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.logger = logger
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        msg = f"[SurgiStarNet] FOVEAPairedDataset: {len(self.pairs)} pairs, augment={self.augment}"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        p = self.pairs[idx]
        
        # Load images
        pre = Image.open(p["pre_img"]).convert("RGB")
        intra = Image.open(p["intra_img"]).convert("RGB")
        
        # Resize
        pre = pre.resize((self.img_size, self.img_size))
        intra = intra.resize((self.img_size, self.img_size))
        
        # 同步增强（对两张图做相同的随机水平翻转）
        if self.augment and random.random() > 0.5:
            pre = TF.hflip(pre)
            intra = TF.hflip(intra)
        
        # Convert to tensor
        pre = TF.to_tensor(pre)
        intra = TF.to_tensor(intra)
        
        # Normalize using ImageNet statistics
        pre = self.normalize(pre)
        intra = self.normalize(intra)
        
        return pre, intra, p["patient_id"]


class ContrastiveProjector(nn.Module):
    """
    多尺度对比投影头（用于 FOVEA 跨域对比学习）。
    将多尺度特征投影到统一的嵌入空间进行对比。
    """
    
    def __init__(self, in_channels_list: List[int], embed_dim: int = 128):
        super().__init__()
        self.projectors = nn.ModuleList()
        for c in in_channels_list:
            self.projectors.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, c),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Linear(c, embed_dim),
            ))
    
    def forward(self, features_list: List[torch.Tensor]) -> List[torch.Tensor]:
        embeddings = []
        for proj, feat in zip(self.projectors, features_list):
            emb = proj(feat)
            emb = F.normalize(emb, dim=1)  # L2 normalize
            embeddings.append(emb)
        return embeddings


def multi_scale_info_nce(
    pre_embs: List[torch.Tensor],
    intra_embs: List[torch.Tensor],
    temperature: float = 0.07
) -> torch.Tensor:
    """
    多尺度 InfoNCE 对比损失。
    
    Args:
        pre_embs: list of [B, D] tensors from pre-operative images
        intra_embs: list of [B, D] tensors from intra-operative images
        temperature: softmax temperature (lower = sharper)

    同一 batch 位置 i 是正样本对（同一病人的 pre 和 intra）
    不同位置是负样本（不同病人）
    """
    total_loss = 0.0
    for pre_e, intra_e in zip(pre_embs, intra_embs):
        # [B, B] similarity matrix
        logits = torch.mm(pre_e, intra_e.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        # 双向 loss
        loss_p2i = F.cross_entropy(logits, labels)
        loss_i2p = F.cross_entropy(logits.t(), labels)
        total_loss += (loss_p2i + loss_i2p) / 2
    return total_loss / len(pre_embs)


def compute_retrieval_accuracy(
    pre_embs: List[torch.Tensor],
    intra_embs: List[torch.Tensor]
) -> float:
    """
    计算跨域检索准确率（Cross-Domain Retrieval Accuracy）。
    
    对于 batch 内的每个 pre embedding，在所有 intra embeddings 中找最相似的，
    如果找到的是对应的那个（同一病人），就算正确。
    
    Args:
        pre_embs: list of [B, D] tensors from pre-operative images
        intra_embs: list of [B, D] tensors from intra-operative images
    
    Returns:
        平均检索准确率 (0.0 - 1.0)
    """
    # 拼接所有尺度的 embedding
    all_pre = torch.cat(pre_embs, dim=1)   # [B, D*num_scales]
    all_intra = torch.cat(intra_embs, dim=1)
    all_pre = F.normalize(all_pre, dim=1)
    all_intra = F.normalize(all_intra, dim=1)
    
    # 计算相似度矩阵
    sim = torch.mm(all_pre, all_intra.t())  # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    
    # pre→intra accuracy
    p2i_acc = (sim.argmax(dim=1) == labels).float().mean().item()
    # intra→pre accuracy  
    i2p_acc = (sim.argmax(dim=0) == labels).float().mean().item()
    
    return (p2i_acc + i2p_acc) / 2


class RMITTipDataset(Dataset):
    """
    Dataset for RMIT surgical instrument tip tracking.
    """
    
    def __init__(
        self,
        manifest_path: str,
        sequences: List[str],
        img_size: int = 256,
        heatmap_size: int = 64,
        is_train: bool = True,
        augment: bool = True,
        no_crop: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        self.manifest_path = manifest_path
        self.sequences = sequences
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.is_train = is_train
        self.augment = augment and is_train
        self.no_crop = no_crop
        self.logger = logger
        
        # Load all samples
        self.samples = []
        self._load_manifest()
        
        # Data augmentation transforms
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
        
        msg = f"[SurgiStarNet] RMITTipDataset: {len(self.samples)} samples from {len(sequences)} sequences"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
    
    def _load_manifest(self):
        """Load manifest and build sample list."""
        with open(self.manifest_path, 'r') as f:
            manifest = json.load(f)
        
        for seq_info in manifest['sequences']:
            seq_name = seq_info['name']
            if seq_name not in self.sequences:
                continue
            
            frames_dir = seq_info['frames_dir']
            ann_csv = seq_info['ann_csv']
            
            # Load annotations
            with open(ann_csv, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
            
            for line in lines:
                parts = line.strip().split(',')
                frame_name = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                
                frame_path = os.path.join(frames_dir, frame_name)
                
                self.samples.append({
                    'frame_path': frame_path,
                    'gt_xy': np.array([x, y], dtype=np.float32),
                    'seq_name': seq_name
                })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['frame_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        gt_xy = sample['gt_xy'].copy()
        
        # Crop to FOV (or just resize if no_crop)
        cropped, gt_xy_cropped, transform_info = crop_and_transform(
            image, gt_xy, self.img_size, no_crop=self.no_crop
        )
        
        # Data augmentation (flip)
        flip = False
        if self.augment and np.random.rand() < 0.5:
            cropped = np.fliplr(cropped).copy()
            gt_xy_cropped[0] = self.img_size - gt_xy_cropped[0]
            flip = True
        
        # Convert to tensor
        image_tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
        
        # Color jitter (on tensor)
        if self.augment:
            image_tensor = self.color_jitter(image_tensor)
        
        # Generate heatmap
        heatmap = generate_gaussian_heatmap(
            gt_xy_cropped / self.img_size * self.heatmap_size,
            (self.heatmap_size, self.heatmap_size),
            sigma=1.5
        )
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)
        
        # Normalize coordinates to [0, 1]
        coords_normalized = gt_xy_cropped / self.img_size
        
        return {
            'image': image_tensor,
            'heatmap': heatmap_tensor,
            'gt_xy': torch.from_numpy(gt_xy_cropped),
            'gt_xy_normalized': torch.from_numpy(coords_normalized),
            'transform_info': transform_info,
            'frame_path': sample['frame_path'],
            'seq_name': sample['seq_name'],
            'original_gt_xy': torch.from_numpy(gt_xy),
            'flip': flip
        }


# =============================================================================
# Training and Evaluation
# =============================================================================

def build_dr_datasets(
    root_dir: str,
    img_size: int,
    val_ratio: float = 0.2,
    logger: Optional[logging.Logger] = None
) -> Tuple[DRImageFolderDataset, DRImageFolderDataset]:
    """
    构建 DR 分类数据集（支持 ImageFolder 格式）。
    
    Args:
        root_dir: ImageFolder 根目录（包含 DR/ 和 No_DR/ 子目录）
        img_size: 图像大小
        val_ratio: 验证集比例
        logger: 日志记录器
        
    Returns:
        train_ds, val_ds
    """
    from torchvision.datasets import ImageFolder
    
    base_ds = ImageFolder(root=root_dir)
    
    # 自动推断 label_map
    if set(base_ds.classes) == {"No_DR", "DR"}:
        label_map = {"No_DR": 0, "DR": 1}
    else:
        label_map = {name: idx for idx, name in enumerate(sorted(base_ds.classes))}
    
    # 按类别分层划分
    by_label: Dict[int, List[int]] = {}
    for i, (_path, label) in enumerate(base_ds.samples):
        mapped = label_map[base_ds.classes[label]]
        by_label.setdefault(mapped, []).append(i)
    
    train_indices, val_indices = [], []
    for _, indices in sorted(by_label.items()):
        indices = indices.copy()
        random.shuffle(indices)
        if len(indices) == 1:
            train_split, val_split = indices, indices
        else:
            n_val = max(1, int(round(len(indices) * val_ratio)))
            n_val = min(n_val, len(indices) - 1)
            val_split = indices[:n_val]
            train_split = indices[n_val:]
        train_indices.extend(train_split)
        val_indices.extend(val_split)
    
    return (
        DRImageFolderDataset(root_dir, img_size, train_indices, augment=True, label_map=label_map, logger=logger),
        DRImageFolderDataset(root_dir, img_size, val_indices, augment=False, label_map=label_map, logger=logger),
    )


def train_stage_dr(args, device: torch.device, logger: logging.Logger) -> str:
    """
    Stage 1: DR 分类训练。
    
    Returns:
        backbone_weights_path: 保存的 backbone 权重路径
    """
    logger.info("[SurgiStarNet] ========== Stage 1: DR Classification ==========")
    
    # Build datasets
    train_ds, val_ds = build_dr_datasets(
        args.dr_imagefolder_dir,
        args.img_size,
        args.dr_val_ratio,
        logger
    )
    
    # Infer num_classes
    train_labels = [label for _, label in train_ds.items]
    num_classes = max(train_labels) + 1
    logger.info(f"[SurgiStarNet] DR num_classes: {num_classes}")
    
    # Compute class weights for imbalanced data
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    logger.info(f"[SurgiStarNet] Class counts: {counts.tolist()}, weights: {weights.tolist()}")
    
    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    # Build model
    backbone = SurgiStarNetHeatmap(heatmap_size=args.heatmap_size).to(device)
    model = SurgiStarNetWithClassifier(
        backbone, num_classes=num_classes, freeze_backbone=False
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.dr_lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.dr_epochs, eta_min=1e-6)
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.dr_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
        
        train_acc = correct / total
        scheduler.step()
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.numel()
        
        val_acc = val_correct / val_total
        
        logger.info(f"[SurgiStarNet] DR epoch {epoch}/{args.dr_epochs} "
                    f"train_loss={train_loss/len(train_loader):.4f} "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            backbone_path = os.path.join(args.out_dir, 'dr_backbone.pth')
            torch.save(model.backbone.state_dict(), backbone_path)
            logger.info(f"[SurgiStarNet] Saved best backbone (val_acc={val_acc:.4f})")
    
    logger.info(f"[SurgiStarNet] Stage 1 DR completed. Best val_acc={best_acc:.4f} at epoch {best_epoch}")
    
    return os.path.join(args.out_dir, 'dr_backbone.pth')


def train_stage_fovea(args, device: torch.device, logger: logging.Logger) -> str:
    """
    Stage 2: FOVEA 跨域对比学习训练（Multi-Granularity Cross-Domain Contrastive Bridging）。
    
    核心思想：共享 backbone 分别编码术前和术中图像，在多个特征层级应用 InfoNCE 对比损失，
    迫使 backbone 学会跨域特征对齐。
    
    Returns:
        backbone_weights_path: 保存的 backbone 权重路径 (fovea_backbone.pth)
    """
    logger.info("[SurgiStarNet] ========== Stage 2: FOVEA Cross-Domain Contrastive Learning ==========")
    
    # Load FOVEA pairs JSON
    fovea_pairs_json = args.fovea_pairs_json if hasattr(args, 'fovea_pairs_json') and args.fovea_pairs_json else args.fovea_manifest_json
    logger.info(f"[SurgiStarNet] Loading FOVEA pairs from: {fovea_pairs_json}")
    
    with open(fovea_pairs_json, 'r') as f:
        all_pairs = json.load(f)
    
    logger.info(f"[SurgiStarNet] Total FOVEA pairs: {len(all_pairs)}")
    
    # Split train/val by patient_id (80/20)
    random.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * args.fovea_val_ratio))
    val_pairs = all_pairs[:n_val]
    train_pairs = all_pairs[n_val:]
    
    logger.info(f"[SurgiStarNet] Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    
    # Get image size for FOVEA (use fovea_img_size if available, else img_size)
    fovea_img_size = getattr(args, 'fovea_img_size', args.img_size)
    fovea_batch_size = getattr(args, 'fovea_batch_size', args.batch_size)
    fovea_embed_dim = getattr(args, 'fovea_embed_dim', 128)
    fovea_temperature = getattr(args, 'fovea_temperature', 0.07)
    
    # Build datasets
    train_ds = FOVEAPairedDataset(
        train_pairs,
        img_size=fovea_img_size,
        augment=True,
        logger=logger
    )
    val_ds = FOVEAPairedDataset(
        val_pairs,
        img_size=fovea_img_size,
        augment=False,
        logger=logger
    )
    
    # Dataloaders (shuffle=True, drop_last=True to ensure batch has different patients as negatives)
    train_loader = DataLoader(
        train_ds, batch_size=fovea_batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=fovea_batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    # Build backbone model
    backbone = SurgiStarNetHeatmap(heatmap_size=args.heatmap_size).to(device)
    
    # Load pretrained backbone from Stage 1
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        logger.info(f"[SurgiStarNet] Loading pretrained backbone from: {args.pretrained_weights}")
        backbone_state = torch.load(args.pretrained_weights, map_location=device)
        backbone.load_state_dict(backbone_state, strict=False)
        logger.info("[SurgiStarNet] Pretrained backbone loaded successfully")
    else:
        logger.info("[SurgiStarNet] Training from scratch (no pretrained weights)")
    
    # Create ContrastiveProjector (in_channels_list corresponds to backbone stage channels)
    # Default dims=(24, 48, 96, 192), we use stage2, stage3, stage4 -> [48, 96, 192]
    in_channels_list = [backbone.dims[1], backbone.dims[2], backbone.dims[3]]
    projector = ContrastiveProjector(in_channels_list, embed_dim=fovea_embed_dim).to(device)
    
    logger.info(f"[SurgiStarNet] ContrastiveProjector channels: {in_channels_list}, embed_dim={fovea_embed_dim}")
    
    # Optimizer: Adam, optimize backbone + projector
    fovea_lr = getattr(args, 'fovea_lr', 1e-4)
    optimizer = torch.optim.Adam(
        list(backbone.parameters()) + list(projector.parameters()),
        lr=fovea_lr
    )
    
    logger.info(f"[SurgiStarNet] Optimizer: Adam, lr={fovea_lr}")
    logger.info(f"[SurgiStarNet] Temperature: {fovea_temperature}")
    
    # Training loop
    best_val_loss = float('inf')
    best_val_retrieval_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.fovea_epochs + 1):
        # Train
        backbone.train()
        projector.train()
        train_loss = 0.0
        num_batches = 0
        
        for pre_img, intra_img, patient_ids in train_loader:
            pre_img = pre_img.to(device)
            intra_img = intra_img.to(device)
            
            optimizer.zero_grad()
            
            # Extract multi-scale features
            pre_feats = backbone.extract_multi_scale(pre_img)
            intra_feats = backbone.extract_multi_scale(intra_img)
            
            # Project to embedding space
            pre_embs = projector(pre_feats)
            intra_embs = projector(intra_feats)
            
            # Compute multi-scale InfoNCE loss
            loss = multi_scale_info_nce(pre_embs, intra_embs, temperature=fovea_temperature)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(projector.parameters()), 
                max_norm=1.0
            )
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / max(num_batches, 1)
        
        # Validate
        backbone.eval()
        projector.eval()
        val_loss = 0.0
        val_batches = 0
        val_retrieval_acc_sum = 0.0
        val_retrieval_batches = 0
        
        with torch.no_grad():
            for pre_img, intra_img, patient_ids in val_loader:
                pre_img = pre_img.to(device)
                intra_img = intra_img.to(device)
                
                # Extract multi-scale features
                pre_feats = backbone.extract_multi_scale(pre_img)
                intra_feats = backbone.extract_multi_scale(intra_img)
                
                # Project to embedding space
                pre_embs = projector(pre_feats)
                intra_embs = projector(intra_feats)
                
                # Compute multi-scale InfoNCE loss
                loss = multi_scale_info_nce(pre_embs, intra_embs, temperature=fovea_temperature)
                
                val_loss += loss.item()
                val_batches += 1
                
                # Compute retrieval accuracy
                retrieval_acc = compute_retrieval_accuracy(pre_embs, intra_embs)
                val_retrieval_acc_sum += retrieval_acc
                val_retrieval_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)
        avg_val_retrieval_acc = val_retrieval_acc_sum / max(val_retrieval_batches, 1)
        
        logger.info(f"[SurgiStarNet] FOVEA epoch {epoch}/{args.fovea_epochs} "
                    f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
                    f"val_retrieval_acc={avg_val_retrieval_acc:.1%}")
        
        # Save best model (only backbone, projector is not needed for Stage 3)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_retrieval_acc = avg_val_retrieval_acc
            best_epoch = epoch
            patience_counter = 0
            
            backbone_path = os.path.join(args.out_dir, 'fovea_backbone.pth')
            torch.save(backbone.state_dict(), backbone_path)
            logger.info(f"[SurgiStarNet] Saved best backbone (val_loss={avg_val_loss:.4f}, val_retrieval_acc={avg_val_retrieval_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"[SurgiStarNet] Early stopping at epoch {epoch}")
                break
    
    logger.info(f"[SurgiStarNet] Stage 2 FOVEA completed. Best val_loss={best_val_loss:.4f}, val_retrieval_acc={best_val_retrieval_acc:.1%} at epoch {best_epoch}")
    
    # Save final results
    results = {
        'best_val_loss': float(best_val_loss),
        'best_val_retrieval_acc': float(best_val_retrieval_acc),
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'num_train_pairs': len(train_pairs),
        'num_val_pairs': len(val_pairs),
        'embed_dim': fovea_embed_dim,
        'temperature': fovea_temperature,
        'learning_rate': fovea_lr,
    }
    with open(os.path.join(args.out_dir, 'fovea_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return os.path.join(args.out_dir, 'fovea_backbone.pth')


def compute_pck(errors: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
    """Compute PCK@threshold."""
    pck = {}
    for thresh in thresholds:
        pck[f'pck@{thresh}'] = (errors < thresh).mean()
    return pck


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        gt_heatmaps = batch['heatmap'].to(device)
        gt_coords = batch['gt_xy_normalized'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        pred_heatmaps = outputs['heatmap']
        pred_coords = outputs['coords']
        
        # Heatmap loss (MSE)
        heatmap_loss = F.mse_loss(pred_heatmaps, gt_heatmaps)
        
        # Coordinate loss (L1)
        coord_loss = F.l1_loss(pred_coords, gt_coords)
        
        # Total loss
        loss = heatmap_loss + 0.5 * coord_loss
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    img_size: int,
    num_vis_samples: int = 5,
    vis_dir: Optional[str] = None,
    epoch: int = 0
) -> Tuple[float, Dict[str, float], List[Dict]]:
    """
    Validate model and compute metrics.
    
    Returns:
        avg_loss: Average validation loss
        metrics: Dict of PCK metrics (both original and input space)
        vis_data: List of data for visualization
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_errors_orig = []  # Original image space errors
    all_errors_input = []  # Model input space errors
    vis_data = []
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        gt_heatmaps = batch['heatmap'].to(device)
        gt_coords = batch['gt_xy_normalized'].to(device)
        
        outputs = model(images)
        pred_heatmaps = outputs['heatmap']
        pred_coords = outputs['coords']
        
        # Compute loss
        heatmap_loss = F.mse_loss(pred_heatmaps, gt_heatmaps)
        coord_loss = F.l1_loss(pred_coords, gt_coords)
        loss = heatmap_loss + 0.5 * coord_loss
        
        total_loss += loss.item()
        num_batches += 1
        
        # Compute pixel errors in both spaces
        for i in range(images.shape[0]):
            # Predicted coordinates in cropped/input space (normalized 0-1)
            pred_xy_normalized = pred_coords[i].cpu().numpy()
            pred_xy_cropped = pred_xy_normalized * img_size
            
            # GT coordinates in input space (already in pixels, from dataset)
            gt_xy_cropped = batch['gt_xy'][i].numpy()
            
            # Compute error in model input space (cropped/resized 256x256 space)
            error_input = np.linalg.norm(pred_xy_cropped - gt_xy_cropped)
            all_errors_input.append(error_input)
            
            # Original GT coordinates
            original_gt_xy = batch['original_gt_xy'][i].numpy()
            
            # Restore predicted coordinates to original space
            transform_info = {
                'offset': (
                    batch['transform_info']['offset'][0][i].item(),
                    batch['transform_info']['offset'][1][i].item()
                ),
                'scale': (
                    batch['transform_info']['scale'][0][i].item(),
                    batch['transform_info']['scale'][1][i].item()
                )
            }
            
            pred_xy_original = restore_coordinates(pred_xy_cropped, transform_info)
            
            # Compute error in original coordinates
            error_orig = np.linalg.norm(pred_xy_original - original_gt_xy)
            all_errors_orig.append(error_orig)
            
            # Collect visualization data (use original space for visualization)
            if len(vis_data) < num_vis_samples:
                vis_data.append({
                    'frame_path': batch['frame_path'][i],
                    'pred_xy': pred_xy_original,
                    'gt_xy': original_gt_xy,
                    'error': error_orig,
                    'seq_name': batch['seq_name'][i]
                })
    
    avg_loss = total_loss / num_batches
    
    # Compute metrics for original space
    errors_orig = np.array(all_errors_orig)
    avg_error_orig = errors_orig.mean()
    pck_orig = compute_pck(errors_orig, [10, 20, 30, 50])
    
    # Compute metrics for input space
    errors_input = np.array(all_errors_input)
    avg_error_input = errors_input.mean()
    pck_input = compute_pck(errors_input, [10, 20, 30, 50])
    
    # Combine metrics with prefixes
    metrics = {
        # Original space metrics (for backward compatibility)
        'avg_err_px': avg_error_orig,
        'pck@10': pck_orig['pck@10'],
        'pck@20': pck_orig['pck@20'],
        'pck@30': pck_orig['pck@30'],
        'pck@50': pck_orig['pck@50'],
        # Original space with explicit prefix
        'orig_avg_err_px': avg_error_orig,
        'orig_pck@10': pck_orig['pck@10'],
        'orig_pck@20': pck_orig['pck@20'],
        'orig_pck@30': pck_orig['pck@30'],
        'orig_pck@50': pck_orig['pck@50'],
        # Input space metrics
        'input_avg_err_px': avg_error_input,
        'input_pck@10': pck_input['pck@10'],
        'input_pck@20': pck_input['pck@20'],
        'input_pck@30': pck_input['pck@30'],
        'input_pck@50': pck_input['pck@50'],
    }
    
    # Generate visualizations
    if vis_dir is not None and len(vis_data) > 0:
        epoch_vis_dir = os.path.join(vis_dir, f'epoch_{epoch:03d}')
        os.makedirs(epoch_vis_dir, exist_ok=True)
        
        for i, data in enumerate(vis_data):
            visualize_prediction(
                frame_path=data['frame_path'],
                pred_xy=data['pred_xy'],
                gt_xy=data['gt_xy'],
                error=data['error'],
                save_path=os.path.join(epoch_vis_dir, f'{i:02d}_err{data["error"]:.1f}px.png')
            )
    
    return avg_loss, metrics, vis_data


def visualize_prediction(
    frame_path: str,
    pred_xy: np.ndarray,
    gt_xy: np.ndarray,
    error: float,
    save_path: str
):
    """Visualize prediction on original image."""
    # Load original image
    image = cv2.imread(frame_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    
    # Draw GT (green)
    circle_gt = Circle((gt_xy[0], gt_xy[1]), radius=5, color='green', fill=False, linewidth=2)
    ax.add_patch(circle_gt)
    ax.plot(gt_xy[0], gt_xy[1], 'go', markersize=3, label='GT')
    
    # Draw Pred (red)
    circle_pred = Circle((pred_xy[0], pred_xy[1]), radius=5, color='red', fill=False, linewidth=2)
    ax.add_patch(circle_pred)
    ax.plot(pred_xy[0], pred_xy[1], 'ro', markersize=3, label='Pred')
    
    # Draw line connecting them (blue)
    ax.plot([gt_xy[0], pred_xy[0]], [gt_xy[1], pred_xy[1]], 'b-', linewidth=1)
    
    ax.set_title(f'Error: {error:.1f} px')
    ax.legend(loc='upper right')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def train_stage_rmit(args, device: torch.device, logger: logging.Logger) -> float:
    """
    Stage 3: RMIT surgical instrument tip tracking training.
    
    Returns:
        best_val_error: Best validation error (pixels)
    """
    logger.info("[SurgiStarNet] ========== Stage 3: RMIT Tracking ==========")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    
    # LOSO split
    all_seqs = ['seq1', 'seq2', 'seq3']
    if args.split_mode == 'loso':
        test_seq = all_seqs[args.fold]
        train_seqs = [s for s in all_seqs if s != test_seq]
    else:
        raise ValueError(f"Unknown split_mode: {args.split_mode}")
    
    logger.info(f"[SurgiStarNet] Split mode: {args.split_mode}, fold: {args.fold}")
    logger.info(f"[SurgiStarNet] Train sequences: {train_seqs}")
    logger.info(f"[SurgiStarNet] Test sequence: {test_seq}")
    
    # Datasets
    train_dataset = RMITTipDataset(
        manifest_path=args.rmit_manifest_json,
        sequences=train_seqs,
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        is_train=True,
        augment=True,
        no_crop=args.no_crop,
        logger=logger
    )
    
    val_dataset = RMITTipDataset(
        manifest_path=args.rmit_manifest_json,
        sequences=[test_seq],
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        is_train=False,
        augment=False,
        no_crop=args.no_crop,
        logger=logger
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = SurgiStarNetHeatmap(
        heatmap_size=args.heatmap_size
    ).to(device)
    
    # Load pretrained weights from Stage 2
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        logger.info(f"[SurgiStarNet] Loading pretrained weights from: {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("[SurgiStarNet] Pretrained weights loaded successfully")
    else:
        logger.info("[SurgiStarNet] Training from scratch (no pretrained weights)")
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training loop
    best_val_error = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss, metrics, vis_data = validate(
            model, val_loader, device, args.img_size,
            num_vis_samples=5, vis_dir=vis_dir, epoch=epoch
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log with both original and input space metrics
        logger.info(f"[SurgiStarNet] epoch {epoch}/{args.epochs} "
                    f"train_loss={train_loss:.4f} | "
                    f"orig: val_err={metrics['orig_avg_err_px']:.1f} "
                    f"pck@10={metrics['orig_pck@10']:.3f} "
                    f"pck@20={metrics['orig_pck@20']:.3f} "
                    f"pck@50={metrics['orig_pck@50']:.3f} | "
                    f"input: val_err={metrics['input_avg_err_px']:.1f} "
                    f"pck@10={metrics['input_pck@10']:.3f} "
                    f"pck@20={metrics['input_pck@20']:.3f} "
                    f"pck@50={metrics['input_pck@50']:.3f}")
        
        # Save best model
        if metrics['avg_err_px'] < best_val_error:
            best_val_error = metrics['avg_err_px']
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': best_val_error,
                'metrics': metrics
            }, checkpoint_path)
            logger.info(f"[SurgiStarNet] Saved best model (error: {best_val_error:.1f} px)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"[SurgiStarNet] Early stopping at epoch {epoch}")
                break
        
        # Save latest checkpoint
        latest_path = os.path.join(args.out_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_path)
    
    logger.info(f"[SurgiStarNet] Stage 3 RMIT completed. Best error: {best_val_error:.1f} px at epoch {best_epoch}")
    
    # Save final results with both original and input space metrics
    results = {
        'best_val_error_px': float(best_val_error),
        'fold': args.fold,
        'test_seq': test_seq,
        'train_seqs': train_seqs,
        'img_size': args.img_size,
        'epochs_trained': epoch,
        'best_epoch': best_epoch,
        # Original space metrics
        'orig_best_val_error_px': float(best_val_error),
    }
    
    # Try to load best model metrics if available
    best_checkpoint_path = os.path.join(args.out_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
        if 'metrics' in checkpoint:
            best_metrics = checkpoint['metrics']
            results['orig_pck@10'] = float(best_metrics.get('orig_pck@10', 0))
            results['orig_pck@20'] = float(best_metrics.get('orig_pck@20', 0))
            results['orig_pck@30'] = float(best_metrics.get('orig_pck@30', 0))
            results['orig_pck@50'] = float(best_metrics.get('orig_pck@50', 0))
            results['input_best_val_error_px'] = float(best_metrics.get('input_avg_err_px', 0))
            results['input_pck@10'] = float(best_metrics.get('input_pck@10', 0))
            results['input_pck@20'] = float(best_metrics.get('input_pck@20', 0))
            results['input_pck@30'] = float(best_metrics.get('input_pck@30', 0))
            results['input_pck@50'] = float(best_metrics.get('input_pck@50', 0))
    
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return best_val_error


def main():
    parser = argparse.ArgumentParser(description='SurgiStarNet for RMIT Tip Tracking (Three-Stage Training)')
    
    # Stage selection
    parser.add_argument('--stage', type=int, default=3, choices=[1, 2, 3],
                        help='Training stage: 1=DR, 2=FOVEA, 3=RMIT (default: 3)')
    
    # Output directory
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    
    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Pretrained weights path (from previous stage)')
    
    # Model configuration
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size after cropping')
    parser.add_argument('--heatmap_size', type=int, default=64,
                        help='Output heatmap size')
    
    # Training configuration (common)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_crop', action='store_true',
                        help='Skip FOV crop, directly resize to img_size')
    
    # Stage 1: DR classification
    parser.add_argument('--dr_imagefolder_dir', type=str, default=None,
                        help='DR dataset path (ImageFolder format with DR/ and No_DR/ subdirs)')
    parser.add_argument('--dr_num_classes', type=int, default=2,
                        help='DR classification num_classes (default: 2, auto-inferred from data)')
    parser.add_argument('--dr_epochs', type=int, default=30,
                        help='DR stage training epochs')
    parser.add_argument('--dr_lr', type=float, default=1e-3,
                        help='DR stage learning rate')
    parser.add_argument('--dr_val_ratio', type=float, default=0.2,
                        help='DR validation split ratio')
    
    # Stage 2: FOVEA cross-domain contrastive learning
    parser.add_argument('--fovea_manifest_json', type=str, default=None,
                        help='FOVEA manifest JSON path (legacy, use --fovea_pairs_json)')
    parser.add_argument('--fovea_pairs_json', type=str, default=None,
                        help='FOVEA pairs JSON path (pre_img, intra_img, patient_id)')
    parser.add_argument('--fovea_img_size', type=int, default=256,
                        help='FOVEA image size (default: 256)')
    parser.add_argument('--fovea_epochs', type=int, default=30,
                        help='FOVEA stage training epochs (default: 30)')
    parser.add_argument('--fovea_lr', type=float, default=1e-4,
                        help='FOVEA stage learning rate (default: 1e-4)')
    parser.add_argument('--fovea_batch_size', type=int, default=8,
                        help='FOVEA batch size (default: 8)')
    parser.add_argument('--fovea_embed_dim', type=int, default=128,
                        help='FOVEA contrastive embedding dimension (default: 128)')
    parser.add_argument('--fovea_temperature', type=float, default=0.07,
                        help='FOVEA InfoNCE temperature (default: 0.07)')
    parser.add_argument('--fovea_val_ratio', type=float, default=0.2,
                        help='FOVEA validation split ratio (default: 0.2)')
    
    # Stage 3: RMIT tracking
    parser.add_argument('--rmit_manifest_json', type=str, default=None,
                        help='RMIT manifest JSON path')
    parser.add_argument('--split_mode', type=str, default='loso',
                        choices=['loso'], help='Split mode')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2],
                        help='Fold index for LOSO')
    parser.add_argument('--epochs', type=int, default=100,
                        help='RMIT stage training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='RMIT stage learning rate')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logger
    os.makedirs(args.out_dir, exist_ok=True)
    logger = setup_logger(args.out_dir, 'train')
    
    # Log configuration
    logger.info("[SurgiStarNet] Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info(f"  device: {device}")
    
    # Run training based on stage
    if args.stage == 1:
        if not args.dr_imagefolder_dir:
            logger.error("Stage 1 requires --dr_imagefolder_dir")
            raise ValueError("Stage 1 requires --dr_imagefolder_dir")
        train_stage_dr(args, device, logger)
        
    elif args.stage == 2:
        # Check for either fovea_pairs_json or fovea_manifest_json
        fovea_json = args.fovea_pairs_json or args.fovea_manifest_json
        if not fovea_json:
            logger.error("Stage 2 requires --fovea_pairs_json or --fovea_manifest_json")
            raise ValueError("Stage 2 requires --fovea_pairs_json or --fovea_manifest_json")
        train_stage_fovea(args, device, logger)
        
    elif args.stage == 3:
        if not args.rmit_manifest_json:
            logger.error("Stage 3 requires --rmit_manifest_json")
            raise ValueError("Stage 3 requires --rmit_manifest_json")
        train_stage_rmit(args, device, logger)
    
    logger.info("[SurgiStarNet] Training completed!")


if __name__ == '__main__':
    main()
