#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SurgiRepFormer for RMIT Surgical Instrument Tip Tracking (Task #13)

Based on RepViT (CVPR 2024) structural re-parameterization:
- Training: Multi-branch for enhanced expressiveness
- Inference: Fused single-path for speed

Architecture: ~2.8M parameters
- Stem: RepConvBN downsampling
- 4 Stages: RepFormerBlocks with SE attention
- U-Net style skip connections for precise localization

Three-Stage Training (Task #16):
- Stage 1: DR classification (backbone pretraining)
- Stage 2: FOVEA localization (keypoint detection)
- Stage 3: RMIT tip tracking (fine-tuning)
"""

import os
import json
import argparse
import logging
import random
import csv
import math
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
# Logging System
# =============================================================================

def setup_logger(out_dir: str, name: str = 'train') -> logging.Logger:
    """
    Setup logger with file and console output.
    
    Args:
        out_dir: Output directory for log file
        name: Logger name
        
    Returns:
        logger: Configured logger instance
    """
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f'{name}.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# Global logger (will be set by main())
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = logging.getLogger('SurgiRepFormer')
        _logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        _logger.addHandler(ch)
    return _logger


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
# RepViT Core Components
# =============================================================================

class RepConvBN(nn.Module):
    """
    Re-parameterizable Conv+BN block.
    Training: multi-branch (kxk + 1x1 + optional identity)
    Inference: fused single conv
    """
    
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.pad = kernel_size // 2
        
        # Main kxk branch
        self.conv_kxk = nn.Conv2d(in_ch, out_ch, kernel_size, stride, self.pad, groups=groups, bias=False)
        self.bn_kxk = nn.BatchNorm2d(out_ch)
        
        # 1x1 branch (only when kernel_size > 1)
        if kernel_size > 1:
            self.conv_1x1 = nn.Conv2d(in_ch, out_ch, 1, stride, 0, groups=groups, bias=False)
            self.bn_1x1 = nn.BatchNorm2d(out_ch)
        else:
            self.conv_1x1 = None
            self.bn_1x1 = None
        
        # Identity branch (only when in_ch == out_ch and stride == 1)
        self.has_identity = (in_ch == out_ch and stride == 1)
        if self.has_identity:
            self.bn_identity = nn.BatchNorm2d(out_ch)
        
        # Fused conv for inference (initialized as None)
        self.fused_conv: Optional[nn.Conv2d] = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use fused conv if available
        if self.fused_conv is not None:
            return self.fused_conv(x)
        
        # Multi-branch during training
        out = self.bn_kxk(self.conv_kxk(x))
        
        if self.conv_1x1 is not None:
            out = out + self.bn_1x1(self.conv_1x1(x))
        
        if self.has_identity:
            out = out + self.bn_identity(x)
        
        return out
    
    def _fuse_conv_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse Conv + BN into single Conv weights and bias."""
        # BN parameters
        gamma = bn.weight
        beta = bn.bias
        mean = bn.running_mean
        var = bn.running_var
        eps = bn.eps
        
        # Compute fused weights
        std = torch.sqrt(var + eps)
        scale = gamma / std
        
        # Conv weights: (out_ch, in_ch/groups, kH, kW)
        w_conv = conv.weight
        w_fused = w_conv * scale.view(-1, 1, 1, 1)
        
        # Bias
        b_fused = beta - mean * scale
        
        return w_fused, b_fused
    
    def _get_identity_kernel(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create identity kernel for identity branch."""
        # Identity kernel: (out_ch, in_ch/groups, kH, kW)
        # For grouped conv, each group is independent
        in_ch_per_group = self.in_ch // self.groups
        
        kernel = torch.zeros(self.out_ch, in_ch_per_group, self.kernel_size, self.kernel_size,
                            device=self.bn_identity.weight.device,
                            dtype=self.bn_identity.weight.dtype)
        
        # Set center of kernel to 1 for identity
        center = self.kernel_size // 2
        for i in range(self.out_ch):
            group_idx = i // (self.out_ch // self.groups)
            in_idx = i % in_ch_per_group
            kernel[i, in_idx, center, center] = 1.0
        
        # Apply BN to identity
        gamma = self.bn_identity.weight
        beta = self.bn_identity.bias
        mean = self.bn_identity.running_mean
        var = self.bn_identity.running_var
        eps = self.bn_identity.eps
        
        std = torch.sqrt(var + eps)
        scale = gamma / std
        
        w_fused = kernel * scale.view(-1, 1, 1, 1)
        b_fused = beta - mean * scale
        
        return w_fused, b_fused
    
    def _pad_1x1_to_kxk(self, w: torch.Tensor) -> torch.Tensor:
        """Pad 1x1 kernel to kxk size."""
        if self.kernel_size == 1:
            return w
        pad_size = self.kernel_size // 2
        return F.pad(w, [pad_size, pad_size, pad_size, pad_size])
    
    def fuse(self):
        """Fuse multi-branch into single conv for inference."""
        if self.fused_conv is not None:
            return  # Already fused
        
        # Fuse kxk branch
        w_kxk, b_kxk = self._fuse_conv_bn(self.conv_kxk, self.bn_kxk)
        w_total = w_kxk
        b_total = b_kxk
        
        # Fuse 1x1 branch
        if self.conv_1x1 is not None:
            w_1x1, b_1x1 = self._fuse_conv_bn(self.conv_1x1, self.bn_1x1)
            w_1x1_padded = self._pad_1x1_to_kxk(w_1x1)
            w_total = w_total + w_1x1_padded
            b_total = b_total + b_1x1
        
        # Fuse identity branch
        if self.has_identity:
            w_id, b_id = self._get_identity_kernel()
            w_total = w_total + w_id
            b_total = b_total + b_id
        
        # Create fused conv
        self.fused_conv = nn.Conv2d(
            self.in_ch, self.out_ch, self.kernel_size, 
            self.stride, self.pad, groups=self.groups, bias=True
        )
        self.fused_conv.weight.data = w_total
        self.fused_conv.bias.data = b_total
        
        # Move to same device
        self.fused_conv = self.fused_conv.to(self.conv_kxk.weight.device)
        
        # Delete training branches to save memory
        del self.conv_kxk, self.bn_kxk
        if self.conv_1x1 is not None:
            del self.conv_1x1, self.bn_1x1
        if self.has_identity:
            del self.bn_identity


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class RepFormerBlock(nn.Module):
    """
    RepViT-style Transformer block.
    - Token mixer: Depthwise RepConvBN
    - Channel mixer: FFN with RepConvBN
    - SE attention for channel selection
    """
    
    def __init__(self, dim: int, mlp_ratio: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        # Token mixer: depthwise conv (groups=dim)
        self.token_mixer = RepConvBN(dim, dim, kernel_size=3, stride=1, groups=dim)
        self.act1 = nn.GELU()
        
        # SE attention
        self.se = SqueezeExcite(dim, reduction=4)
        
        # Channel mixer: FFN with 1x1 convs
        self.channel_mixer = nn.Sequential(
            RepConvBN(dim, hidden_dim, kernel_size=1, stride=1),
            nn.GELU(),
            RepConvBN(hidden_dim, dim, kernel_size=1, stride=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing with residual
        x = x + self.act1(self.token_mixer(x))
        # SE attention
        x = self.se(x)
        # Channel mixing with residual
        x = x + self.channel_mixer(x)
        return x
    
    def fuse(self):
        """Fuse all RepConvBN in this block."""
        self.token_mixer.fuse()
        for module in self.channel_mixer:
            if isinstance(module, RepConvBN):
                module.fuse()


class Downsample(nn.Module):
    """Downsampling with RepConvBN."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = RepConvBN(in_ch, out_ch, kernel_size=3, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
    
    def fuse(self):
        self.conv.fuse()


# =============================================================================
# SurgiRepFormer Model
# =============================================================================

class SurgiRepFormerHeatmap(nn.Module):
    """
    SurgiRepFormer: RepViT-based model for surgical tip detection.
    
    Architecture (~2.8M params):
    - Stem: RepConvBN downsampling
    - 4 Stages with RepFormerBlocks
    - U-Net decoder with skip connections
    - Heatmap output with soft-argmax
    """
    
    def __init__(self, heatmap_size: int = 64):
        super().__init__()
        self.heatmap_size = heatmap_size
        
        # Reduced channel dimensions for ~2.8M params
        # Stem: 256x256x3 -> 64x64x32
        self.stem = nn.Sequential(
            RepConvBN(3, 16, kernel_size=3, stride=2),   # -> 128x128x16
            nn.GELU(),
            RepConvBN(16, 32, kernel_size=3, stride=2),  # -> 64x64x32
            nn.GELU(),
        )
        
        # Stage 1: 64x64x32
        self.stage1 = nn.Sequential(
            RepFormerBlock(32, mlp_ratio=2.0),
            RepFormerBlock(32, mlp_ratio=2.0),
        )
        
        # Stage 2: 32x32x64
        self.down1 = Downsample(32, 64)
        self.stage2 = nn.Sequential(
            RepFormerBlock(64, mlp_ratio=2.0),
            RepFormerBlock(64, mlp_ratio=2.0),
            RepFormerBlock(64, mlp_ratio=2.0),
        )
        
        # Stage 3: 16x16x128
        self.down2 = Downsample(64, 128)
        self.stage3 = nn.Sequential(
            RepFormerBlock(128, mlp_ratio=2.0),
            RepFormerBlock(128, mlp_ratio=2.0),
            RepFormerBlock(128, mlp_ratio=2.0),
        )
        
        # Stage 4: 8x8x256
        self.down3 = Downsample(128, 256)
        self.stage4 = nn.Sequential(
            RepFormerBlock(256, mlp_ratio=2.0),
            RepFormerBlock(256, mlp_ratio=2.0),
        )
        
        # Decoder with skip connections
        # 8x8x256 -> 16x16x128
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.dec_conv1 = nn.Sequential(
            RepConvBN(256, 128, kernel_size=3, stride=1),  # 128+128=256 -> 128
            nn.GELU(),
        )
        
        # 16x16x128 -> 32x32x64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.dec_conv2 = nn.Sequential(
            RepConvBN(128, 64, kernel_size=3, stride=1),  # 64+64=128 -> 64
            nn.GELU(),
        )
        
        # 32x32x64 -> 64x64x32
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec_conv3 = nn.Sequential(
            RepConvBN(64, 32, kernel_size=3, stride=1),  # 32+32=64 -> 32
            nn.GELU(),
        )
        
        # Heatmap head: 64x64x32 -> 64x64x1
        self.heatmap_head = nn.Conv2d(32, 1, kernel_size=1)
        
        # Count parameters
        self._count_parameters()
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SurgiRepFormer] Total parameters: {total:,}")
        print(f"[SurgiRepFormer] Trainable parameters: {trainable:,}")
        print(f"[SurgiRepFormer] Model size: {total * 4 / 1024 / 1024:.2f} MB")
        
        # Estimate VRAM
        vram_estimate = total * 4 * 2 / 1024 / 1024 + 100
        print(f"[SurgiRepFormer] Estimated VRAM (per sample): ~{vram_estimate:.0f} MB")
    
    def extract_multi_scale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from encoder for contrastive learning.
        
        Args:
            x: (B, 3, H, W) input image
            
        Returns:
            List of 3 feature maps from stage2, stage3, stage4:
            - stage2: (B, 64, H/8, W/8)
            - stage3: (B, 128, H/16, W/16)
            - stage4: (B, 256, H/32, W/32)
        """
        # Encoder forward
        x = self.stem(x)           # 64x64x32
        x = self.stage1(x)         # 64x64x32
        
        x = self.down1(x)          # 32x32x64
        s2 = self.stage2(x)        # 32x32x64
        
        x = self.down2(s2)         # 16x16x128
        s3 = self.stage3(x)        # 16x16x128
        
        x = self.down3(s3)         # 8x8x256
        s4 = self.stage4(x)        # 8x8x256
        
        return [s2, s3, s4]  # channels: [64, 128, 256]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 3, 256, 256) input image
        
        Returns:
            dict with keys:
                - heatmap: (B, 1, 64, 64)
                - coords: (B, 2) in [0, 1]
        """
        # Encoder
        x = self.stem(x)           # 64x64x48
        s1 = self.stage1(x)        # 64x64x48
        
        x = self.down1(s1)         # 32x32x96
        s2 = self.stage2(x)        # 32x32x96
        
        x = self.down2(s2)         # 16x16x192
        s3 = self.stage3(x)        # 16x16x192
        
        x = self.down3(s3)         # 8x8x384
        x = self.stage4(x)         # 8x8x384
        
        # Decoder with skip connections
        x = self.up1(x)            # 16x16x192
        x = torch.cat([x, s3], dim=1)  # 16x16x384
        x = self.dec_conv1(x)      # 16x16x192
        
        x = self.up2(x)            # 32x32x96
        x = torch.cat([x, s2], dim=1)  # 32x32x192
        x = self.dec_conv2(x)      # 32x32x96
        
        x = self.up3(x)            # 64x64x48
        x = torch.cat([x, s1], dim=1)  # 64x64x96
        x = self.dec_conv3(x)      # 64x64x48
        
        # Heatmap
        heatmap = self.heatmap_head(x)  # 64x64x1
        
        # Decode coordinates
        coords = soft_argmax_2d(heatmap, temperature=0.1)
        
        return {
            'heatmap': heatmap,
            'coords': coords
        }
    
    def fuse(self):
        """Fuse all RepConvBN modules for inference optimization."""
        print("[SurgiRepFormer] Fusing multi-branch to single-path...")
        
        # Fuse stem
        for module in self.stem:
            if isinstance(module, RepConvBN):
                module.fuse()
        
        # Fuse stages
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            for block in stage:
                if isinstance(block, RepFormerBlock):
                    block.fuse()
        
        # Fuse downsamples
        self.down1.fuse()
        self.down2.fuse()
        self.down3.fuse()
        
        # Fuse decoder convs
        for dec in [self.dec_conv1, self.dec_conv2, self.dec_conv3]:
            for module in dec:
                if isinstance(module, RepConvBN):
                    module.fuse()
        
        # Count fused parameters
        total = sum(p.numel() for p in self.parameters())
        print(f"[SurgiRepFormer] Fused model parameters: {total:,}")
        print(f"[SurgiRepFormer] Fused model size: {total * 4 / 1024 / 1024:.2f} MB")
    
    def get_backbone_state_dict(self) -> Dict[str, Any]:
        """Get backbone state dict (excluding decoder and head) for Stage 1 -> Stage 2 transfer."""
        backbone_keys = ['stem', 'stage1', 'down1', 'stage2', 'down2', 'stage3', 'down3', 'stage4']
        state_dict = {}
        for key, val in self.state_dict().items():
            for bk in backbone_keys:
                if key.startswith(bk):
                    state_dict[key] = val
                    break
        return state_dict


# =============================================================================
# Stage 1: DR Classification Model
# =============================================================================

class SurgiRepFormerClassifier(nn.Module):
    """
    SurgiRepFormer with classification head for DR grading (Stage 1).
    
    The classification head is temporary - after Stage 1 training,
    only the backbone weights are kept for Stage 2.
    """
    
    def __init__(self, num_classes: int = 5, img_size: int = 256):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        
        # Same backbone as SurgiRepFormerHeatmap
        self.stem = nn.Sequential(
            RepConvBN(3, 16, kernel_size=3, stride=2),
            nn.GELU(),
            RepConvBN(16, 32, kernel_size=3, stride=2),
            nn.GELU(),
        )
        
        self.stage1 = nn.Sequential(
            RepFormerBlock(32, mlp_ratio=2.0),
            RepFormerBlock(32, mlp_ratio=2.0),
        )
        
        self.down1 = Downsample(32, 64)
        self.stage2 = nn.Sequential(
            RepFormerBlock(64, mlp_ratio=2.0),
            RepFormerBlock(64, mlp_ratio=2.0),
            RepFormerBlock(64, mlp_ratio=2.0),
        )
        
        self.down2 = Downsample(64, 128)
        self.stage3 = nn.Sequential(
            RepFormerBlock(128, mlp_ratio=2.0),
            RepFormerBlock(128, mlp_ratio=2.0),
            RepFormerBlock(128, mlp_ratio=2.0),
        )
        
        self.down3 = Downsample(128, 256)
        self.stage4 = nn.Sequential(
            RepFormerBlock(256, mlp_ratio=2.0),
            RepFormerBlock(256, mlp_ratio=2.0),
        )
        
        # Classification head: GlobalAvgPool -> FC
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
        
        self._count_parameters()
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger = get_logger()
        logger.info(f"[SurgiRepFormer] Classifier Total parameters: {total:,}")
        logger.info(f"[SurgiRepFormer] Classifier Trainable parameters: {trainable:,}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        # Encoder
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        
        # Classification
        logits = self.classifier(x)
        return logits
    
    def get_backbone_state_dict(self) -> Dict[str, Any]:
        """Get backbone state dict for Stage 1 -> Stage 2 transfer."""
        backbone_keys = ['stem', 'stage1', 'down1', 'stage2', 'down2', 'stage3', 'down3', 'stage4']
        state_dict = {}
        for key, val in self.state_dict().items():
            for bk in backbone_keys:
                if key.startswith(bk):
                    state_dict[key] = val
                    break
        return state_dict


# =============================================================================
# Dataset
# =============================================================================

class DRImageFolderDataset(Dataset):
    """
    DR (Diabetic Retinopathy) Classification Dataset from ImageFolder structure.
    
    Expected directory structure:
    dr_imagefolder_dir/
        DR/
            image1.png
            image2.png
            ...
        No_DR/
            image1.png
            image2.png
            ...
    Or for multi-class:
        dr_imagefolder_dir/
            class_0/
            class_1/
            ...
            class_{num_classes-1}/
    """
    
    def __init__(
        self,
        imagefolder_dir: str,
        img_size: int = 256,
        is_train: bool = True,
        val_ratio: float = 0.2,
        seed: int = 42,
        num_classes: int = 2
    ):
        self.imagefolder_dir = imagefolder_dir
        self.img_size = img_size
        self.is_train = is_train
        self.num_classes = num_classes
        
        # Find all class directories
        self.class_names = sorted([d for d in os.listdir(imagefolder_dir) 
                                   if os.path.isdir(os.path.join(imagefolder_dir, d))])
        
        if len(self.class_names) == 0:
            raise ValueError(f"No class directories found in {imagefolder_dir}")
        
        # Map class name to index
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Collect all samples
        all_samples = []
        for class_name in self.class_names:
            class_dir = os.path.join(imagefolder_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    all_samples.append({
                        'path': os.path.join(class_dir, fname),
                        'label': class_idx
                    })
        
        # Split into train/val
        rng = random.Random(seed)
        rng.shuffle(all_samples)
        
        n_val = int(len(all_samples) * val_ratio)
        if is_train:
            self.samples = all_samples[n_val:]
        else:
            self.samples = all_samples[:n_val]
        
        # Data augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        
        logger = get_logger()
        logger.info(f"[DRImageFolderDataset] {'Train' if is_train else 'Val'}: {len(self.samples)} samples, {len(self.class_names)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Data augmentation
        if self.is_train:
            # Horizontal flip
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
            
            # Convert to tensor and apply color jitter
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image_tensor = self.color_jitter(image_tensor)
        else:
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image_tensor,
            'label': sample['label'],
            'path': sample['path']
        }


class FOVEAPairedDataset(Dataset):
    """
    FOVEA Paired Dataset for Stage 2: Cross-Domain Contrastive Learning.
    
    Loads pre-operative and intra-operative image pairs for multi-granularity
    cross-domain contrastive bridging.
    
    JSON format:
    [{"patient_id": "...", "pre_img": "path", "intra_img": "path", ...}, ...]
    """
    
    def __init__(
        self,
        pairs: List[Dict],
        img_size: int = 256,
        augment: bool = True
    ):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
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
        
        # Synchronized augmentation (same flip for both)
        if self.augment and random.random() > 0.5:
            pre = TF.hflip(pre)
            intra = TF.hflip(intra)
        
        # Convert to tensor and normalize
        pre = TF.to_tensor(pre)
        intra = TF.to_tensor(intra)
        pre = self.normalize(pre)
        intra = self.normalize(intra)
        
        return pre, intra, p["patient_id"]


class ContrastiveProjector(nn.Module):
    """
    Multi-scale Contrastive Projector for cross-domain feature alignment.
    
    Projects features from multiple scales to a shared embedding space
    for contrastive learning.
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
        """
        Project multi-scale features to embedding space.
        
        Args:
            features_list: List of feature tensors from encoder stages
            
        Returns:
            List of L2-normalized embeddings
        """
        embeddings = []
        for proj, feat in zip(self.projectors, features_list):
            emb = proj(feat)
            emb = F.normalize(emb, dim=1)
            embeddings.append(emb)
        return embeddings


def multi_scale_info_nce(
    pre_embs: List[torch.Tensor],
    intra_embs: List[torch.Tensor],
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Multi-scale InfoNCE contrastive loss.
    
    Computes bidirectional contrastive loss between pre-operative and
    intra-operative embeddings at multiple scales.
    
    Args:
        pre_embs: List of pre-operative embeddings
        intra_embs: List of intra-operative embeddings
        temperature: Temperature scaling factor
        
    Returns:
        Averaged contrastive loss across all scales
    """
    total_loss = 0.0
    for pre_e, intra_e in zip(pre_embs, intra_embs):
        # Compute similarity matrix
        logits = torch.mm(pre_e, intra_e.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Bidirectional contrastive loss
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
        no_crop: bool = False
    ):
        self.manifest_path = manifest_path
        self.sequences = sequences
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.is_train = is_train
        self.augment = augment and is_train
        self.no_crop = no_crop
        
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
        
        print(f"[RMITTipDataset] Loaded {len(self.samples)} samples from {len(sequences)} sequences")
    
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

def compute_pck(errors: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
    """Compute PCK@threshold."""
    pck = {}
    for thresh in thresholds:
        pck[f'pck@{thresh}'] = (errors < thresh).mean()
    return pck


# =============================================================================
# Stage 1: DR Classification Training
# =============================================================================

def train_stage1_dr(args) -> None:
    """Stage 1: DR Classification training."""
    logger = get_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[SurgiRepFormer] Stage 1 DR - Using device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Datasets
    train_dataset = DRImageFolderDataset(
        imagefolder_dir=args.dr_imagefolder_dir,
        img_size=args.img_size,
        is_train=True,
        val_ratio=args.dr_val_ratio,
        seed=args.seed,
        num_classes=args.dr_num_classes
    )
    
    val_dataset = DRImageFolderDataset(
        imagefolder_dir=args.dr_imagefolder_dir,
        img_size=args.img_size,
        is_train=False,
        val_ratio=args.dr_val_ratio,
        seed=args.seed,
        num_classes=args.dr_num_classes
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = SurgiRepFormerClassifier(num_classes=args.dr_num_classes, img_size=args.img_size).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.dr_lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.dr_epochs, eta_min=1e-6)
    
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.dr_epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        logger.info(f"[SurgiRepFormer] Stage 1 DR - epoch {epoch}/{args.dr_epochs} "
                    f"train_loss={train_loss/len(train_loader):.4f} train_acc={train_acc:.3f} "
                    f"val_loss={val_loss/len(val_loader):.4f} val_acc={val_acc:.3f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            # Save backbone weights
            backbone_path = os.path.join(args.out_dir, 'dr_backbone.pth')
            torch.save(model.get_backbone_state_dict(), backbone_path)
            logger.info(f"[SurgiRepFormer] Stage 1 DR - Saved best backbone (acc: {best_acc:.3f})")
    
    logger.info(f"[SurgiRepFormer] Stage 1 DR - Completed. Best val_acc: {best_acc:.3f} at epoch {best_epoch}")
    
    # Save results
    results = {
        'stage': 'dr',
        'best_val_acc': best_acc,
        'best_epoch': best_epoch,
        'dr_epochs': args.dr_epochs,
        'dr_num_classes': args.dr_num_classes,
    }
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


# =============================================================================
# Stage 2: FOVEA Cross-Domain Contrastive Bridging
# =============================================================================

def train_stage2_fovea(args) -> None:
    """
    Stage 2: Multi-Granularity Cross-Domain Contrastive Bridging.
    
    Uses pre-operative and intra-operative image pairs to learn cross-domain
    feature alignment through multi-scale InfoNCE contrastive loss.
    """
    logger = get_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[SurgiRepFormer] Stage 2 FOVEA Contrastive - Using device: {device}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load FOVEA pairs JSON
    fovea_json = getattr(args, 'fovea_pairs_json', None) or args.fovea_manifest_json
    if not fovea_json or not os.path.exists(fovea_json):
        logger.error(f"[SurgiRepFormer] Stage 2 FOVEA - fovea_pairs_json not found: {fovea_json}")
        return
    
    with open(fovea_json, 'r') as f:
        all_pairs = json.load(f)
    
    logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Loaded {len(all_pairs)} image pairs")
    
    # Split by patient_id (80/20)
    patient_ids = list(set(p["patient_id"] for p in all_pairs))
    random.seed(args.seed)
    random.shuffle(patient_ids)
    
    val_ratio = getattr(args, 'fovea_val_ratio', 0.2)
    n_val = max(1, int(len(patient_ids) * val_ratio))
    val_patients = set(patient_ids[:n_val])
    train_patients = set(patient_ids[n_val:])
    
    train_pairs = [p for p in all_pairs if p["patient_id"] in train_patients]
    val_pairs = [p for p in all_pairs if p["patient_id"] in val_patients]
    
    logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Train: {len(train_pairs)} pairs from {len(train_patients)} patients")
    logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Val: {len(val_pairs)} pairs from {len(val_patients)} patients")
    
    # Dataset and DataLoader
    img_size = getattr(args, 'fovea_img_size', 256)
    batch_size = getattr(args, 'fovea_batch_size', 8)
    
    train_dataset = FOVEAPairedDataset(train_pairs, img_size=img_size, augment=True)
    val_dataset = FOVEAPairedDataset(val_pairs, img_size=img_size, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    # Model (backbone only)
    model = SurgiRepFormerHeatmap(heatmap_size=64).to(device)
    
    # Load pretrained backbone from Stage 1
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Loading pretrained backbone from {args.pretrained_weights}")
        backbone_state = torch.load(args.pretrained_weights, map_location=device, weights_only=False)
        model_state = model.state_dict()
        loaded_keys = 0
        for key, val in backbone_state.items():
            if key in model_state and model_state[key].shape == val.shape:
                model_state[key] = val
                loaded_keys += 1
        model.load_state_dict(model_state)
        logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Loaded {loaded_keys} backbone weights")
    else:
        logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Training from scratch (no pretrained weights)")
    
    # Contrastive projector (channels: [64, 128, 256] from stage2, stage3, stage4)
    embed_dim = getattr(args, 'fovea_embed_dim', 128)
    projector = ContrastiveProjector([64, 128, 256], embed_dim=embed_dim).to(device)
    
    # Optimizer (backbone + projector)
    lr = getattr(args, 'fovea_lr', 1e-4)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()),
        lr=lr
    )
    
    # Training config
    epochs = getattr(args, 'fovea_epochs', 30)
    temperature = getattr(args, 'fovea_temperature', 0.07)
    patience = getattr(args, 'patience', 10)
    
    best_val_loss = float('inf')
    best_val_retrieval_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Config: epochs={epochs}, lr={lr}, "
                f"batch_size={batch_size}, embed_dim={embed_dim}, temperature={temperature}")
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        projector.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        for pre_imgs, intra_imgs, patient_ids in train_loader:
            pre_imgs = pre_imgs.to(device)
            intra_imgs = intra_imgs.to(device)
            
            # Extract multi-scale features
            pre_feats = model.extract_multi_scale(pre_imgs)
            intra_feats = model.extract_multi_scale(intra_imgs)
            
            # Project to embedding space
            pre_embs = projector(pre_feats)
            intra_embs = projector(intra_feats)
            
            # Multi-scale InfoNCE loss
            loss = multi_scale_info_nce(pre_embs, intra_embs, temperature=temperature)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
        
        train_loss = train_loss_sum / max(train_batches, 1)
        
        # Validation
        model.eval()
        projector.eval()
        val_loss_sum = 0.0
        val_batches = 0
        val_retrieval_acc_sum = 0.0
        val_retrieval_batches = 0
        
        with torch.no_grad():
            for pre_imgs, intra_imgs, patient_ids in val_loader:
                pre_imgs = pre_imgs.to(device)
                intra_imgs = intra_imgs.to(device)
                
                pre_feats = model.extract_multi_scale(pre_imgs)
                intra_feats = model.extract_multi_scale(intra_imgs)
                
                pre_embs = projector(pre_feats)
                intra_embs = projector(intra_feats)
                
                loss = multi_scale_info_nce(pre_embs, intra_embs, temperature=temperature)
                val_loss_sum += loss.item()
                val_batches += 1
                
                # Compute retrieval accuracy
                retrieval_acc = compute_retrieval_accuracy(pre_embs, intra_embs)
                val_retrieval_acc_sum += retrieval_acc
                val_retrieval_batches += 1
        
        val_loss = val_loss_sum / max(val_batches, 1)
        val_retrieval_acc = val_retrieval_acc_sum / max(val_retrieval_batches, 1)
        
        logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - epoch {epoch}/{epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                    f"val_retrieval_acc={val_retrieval_acc:.1%}")
        
        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_retrieval_acc = val_retrieval_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save full model state_dict (includes BN running_mean/running_var buffers)
            backbone_path = os.path.join(args.out_dir, 'fovea_backbone.pth')
            torch.save(model.state_dict(), backbone_path)
            logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Saved best backbone (val_loss={best_val_loss:.4f}, val_retrieval_acc={val_retrieval_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Early stopping at epoch {epoch}")
                break
    
    logger.info(f"[SurgiRepFormer] Stage 2 FOVEA - Completed. Best val_loss={best_val_loss:.4f}, val_retrieval_acc={best_val_retrieval_acc:.1%} at epoch {best_epoch}")
    
    # Save results
    results = {
        'stage': 'fovea_contrastive',
        'best_val_loss': float(best_val_loss),
        'best_val_retrieval_acc': float(best_val_retrieval_acc),
        'best_epoch': best_epoch,
        'fovea_epochs': epochs,
        'embed_dim': embed_dim,
        'temperature': temperature,
        'train_pairs': len(train_pairs),
        'val_pairs': len(val_pairs),
    }
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)


# =============================================================================
# Stage 3: RMIT Tip Tracking Training
# =============================================================================

def train_stage3_rmit(args) -> float:
    """Stage 3: RMIT tip tracking training (modified to support pretrained weights)."""
    logger = get_logger()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Using device: {device}")
    
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
    
    logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Split mode: {args.split_mode}, fold: {args.fold}")
    logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Train sequences: {train_seqs}")
    logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Test sequence: {test_seq}")
    
    # Datasets
    train_dataset = RMITTipDataset(
        manifest_path=args.rmit_manifest_json,
        sequences=train_seqs,
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        is_train=True,
        augment=True,
        no_crop=args.no_crop
    )
    
    val_dataset = RMITTipDataset(
        manifest_path=args.rmit_manifest_json,
        sequences=[test_seq],
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        is_train=False,
        augment=False,
        no_crop=args.no_crop
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
    model = SurgiRepFormerHeatmap(
        heatmap_size=args.heatmap_size
    ).to(device)
    
    # Load pretrained weights (from Stage 2)
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Loading pretrained weights from {args.pretrained_weights}")
        ckpt = torch.load(args.pretrained_weights, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        # Use strict=False because Stage 2 may only have encoder weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Loaded pretrained weights: {len(state_dict)} keys")
        logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Missing keys (will be randomly initialized): {len(missing)}")
        if len(missing) > 0 and len(missing) <= 10:
            logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Missing keys: {missing[:10]}")
        logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Unexpected keys (ignored): {len(unexpected)}")
    else:
        logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Training from scratch (no pretrained weights)")
    
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
        
        # Log
        logger.info(f"[SurgiRepFormer] Stage 3 RMIT - epoch {epoch}/{args.epochs} "
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
            logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Saved best model (error: {best_val_error:.1f}px)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Early stopping at epoch {epoch}")
                break
        
        # Save latest checkpoint
        latest_path = os.path.join(args.out_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_path)
    
    logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Training completed. Best error: {best_val_error:.1f}px")
    
    # Fuse model for inference optimization
    logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Fusing model for inference...")
    model.fuse()
    
    # Save fused model
    fused_path = os.path.join(args.out_dir, 'fused_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'fused': True
    }, fused_path)
    logger.info(f"[SurgiRepFormer] Stage 3 RMIT - Saved fused model to {fused_path}")
    
    # Save final results
    results = {
        'stage': 'rmit',
        'best_val_error_px': float(best_val_error),
        'fold': args.fold,
        'test_seq': test_seq,
        'train_seqs': train_seqs,
        'img_size': args.img_size,
        'epochs_trained': epoch,
        'orig_best_val_error_px': float(best_val_error),
    }
    
    # Try to load best model metrics
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


def train(args):
    """Main training loop."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SurgiRepFormer] Using device: {device}")
    
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
    
    print(f"[SurgiRepFormer] Split mode: {args.split_mode}, fold: {args.fold}")
    print(f"[SurgiRepFormer] Train sequences: {train_seqs}")
    print(f"[SurgiRepFormer] Test sequence: {test_seq}")
    
    # Datasets
    train_dataset = RMITTipDataset(
        manifest_path=args.rmit_manifest_json,
        sequences=train_seqs,
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        is_train=True,
        augment=True,
        no_crop=args.no_crop
    )
    
    val_dataset = RMITTipDataset(
        manifest_path=args.rmit_manifest_json,
        sequences=[test_seq],
        img_size=args.img_size,
        heatmap_size=args.heatmap_size,
        is_train=False,
        augment=False,
        no_crop=args.no_crop
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
    model = SurgiRepFormerHeatmap(
        heatmap_size=args.heatmap_size
    ).to(device)
    
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
        print(f"[SurgiRepFormer] epoch {epoch}/{args.epochs} "
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
            print(f"[SurgiRepFormer] Saved best model (error: {best_val_error:.1f} px)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[SurgiRepFormer] Early stopping at epoch {epoch}")
                break
        
        # Save latest checkpoint
        latest_path = os.path.join(args.out_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_path)
    
    print(f"[SurgiRepFormer] Training completed. Best error: {best_val_error:.1f} px")
    
    # Fuse model for inference optimization
    print(f"[SurgiRepFormer] Fusing model for inference...")
    model.fuse()
    
    # Save fused model
    fused_path = os.path.join(args.out_dir, 'fused_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'fused': True
    }, fused_path)
    print(f"[SurgiRepFormer] Saved fused model to {fused_path}")
    
    # Save final results with both original and input space metrics
    results = {
        'best_val_error_px': float(best_val_error),
        'fold': args.fold,
        'test_seq': test_seq,
        'train_seqs': train_seqs,
        'img_size': args.img_size,
        'epochs_trained': epoch,
        # Original space metrics
        'orig_best_val_error_px': float(best_val_error),
        # Input space metrics (need to reload best checkpoint to get these)
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
    parser = argparse.ArgumentParser(description='SurgiRepFormer Three-Stage Training')
    
    # Stage selection
    parser.add_argument('--stage', type=int, default=3, choices=[1, 2, 3],
                        help='Training stage: 1=DR, 2=FOVEA, 3=RMIT (default: 3)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--run_all_stages', action='store_true',
                        help='Run all three stages sequentially')
    
    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Pretrained weights path (for Stage 2/3)')
    
    # General training configuration
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--heatmap_size', type=int, default=64,
                        help='Output heatmap size')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_crop', action='store_true',
                        help='Skip FOV crop, directly resize to img_size')
    
    # Stage 1: DR Classification parameters
    parser.add_argument('--dr_imagefolder_dir', type=str, default=None,
                        help='DR dataset path (ImageFolder format)')
    parser.add_argument('--dr_num_classes', type=int, default=5,
                        help='DR classification classes (default: 5)')
    parser.add_argument('--dr_epochs', type=int, default=30,
                        help='DR training epochs (default: 30)')
    parser.add_argument('--dr_lr', type=float, default=1e-3,
                        help='DR learning rate (default: 1e-3)')
    parser.add_argument('--dr_val_ratio', type=float, default=0.2,
                        help='DR validation split ratio')
    
    # Stage 2: FOVEA Cross-Domain Contrastive Bridging parameters
    parser.add_argument('--fovea_manifest_json', type=str, default=None,
                        help='FOVEA manifest JSON path (deprecated, use --fovea_pairs_json)')
    parser.add_argument('--fovea_pairs_json', type=str, default=None,
                        help='FOVEA pairs JSON path (pre/intra image pairs)')
    parser.add_argument('--fovea_img_size', type=int, default=256,
                        help='FOVEA input image size (default: 256)')
    parser.add_argument('--fovea_epochs', type=int, default=30,
                        help='FOVEA training epochs (default: 30)')
    parser.add_argument('--fovea_lr', type=float, default=1e-4,
                        help='FOVEA learning rate (default: 1e-4)')
    parser.add_argument('--fovea_batch_size', type=int, default=8,
                        help='FOVEA batch size (default: 8)')
    parser.add_argument('--fovea_embed_dim', type=int, default=128,
                        help='FOVEA contrastive embedding dimension (default: 128)')
    parser.add_argument('--fovea_temperature', type=float, default=0.07,
                        help='FOVEA InfoNCE temperature (default: 0.07)')
    parser.add_argument('--fovea_val_ratio', type=float, default=0.2,
                        help='FOVEA validation split ratio')
    
    # Stage 3: RMIT Tracking parameters
    parser.add_argument('--rmit_manifest_json', type=str, default=None,
                        help='RMIT manifest JSON path')
    parser.add_argument('--split_mode', type=str, default='loso',
                        choices=['loso'], help='Split mode')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2],
                        help='Fold index for LOSO')
    parser.add_argument('--epochs', type=int, default=100,
                        help='RMIT training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='RMIT learning rate')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup logger
    global _logger
    _logger = setup_logger(args.out_dir, name='train')
    logger = get_logger()
    
    # Print configuration
    logger.info("[SurgiRepFormer] Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    
    # Run training based on stage
    if args.run_all_stages:
        # Run all stages sequentially
        logger.info("[SurgiRepFormer] Running all three stages...")
        
        # Stage 1: DR
        if args.dr_imagefolder_dir:
            stage1_dir = os.path.join(args.out_dir, 'stage1_dr')
            args_stage1 = argparse.Namespace(**vars(args))
            args_stage1.out_dir = stage1_dir
            _logger = setup_logger(stage1_dir, name='train')
            train_stage1_dr(args_stage1)
            args.pretrained_weights = os.path.join(stage1_dir, 'dr_backbone.pth')
        else:
            logger.warning("[SurgiRepFormer] Skipping Stage 1: --dr_imagefolder_dir not provided")
        
        # Stage 2: FOVEA Cross-Domain Contrastive Bridging
        fovea_json = args.fovea_pairs_json or args.fovea_manifest_json
        if fovea_json:
            stage2_dir = os.path.join(args.out_dir, 'stage2_fovea')
            args_stage2 = argparse.Namespace(**vars(args))
            args_stage2.out_dir = stage2_dir
            # Ensure fovea_pairs_json is set for the new contrastive training
            args_stage2.fovea_pairs_json = fovea_json
            _logger = setup_logger(stage2_dir, name='train')
            train_stage2_fovea(args_stage2)
            # Stage 2 now outputs fovea_backbone.pth for Stage 3
            args.pretrained_weights = os.path.join(stage2_dir, 'fovea_backbone.pth')
        else:
            logger.warning("[SurgiRepFormer] Skipping Stage 2: --fovea_pairs_json not provided")
        
        # Stage 3: RMIT
        if args.rmit_manifest_json:
            stage3_dir = os.path.join(args.out_dir, 'stage3_rmit')
            args_stage3 = argparse.Namespace(**vars(args))
            args_stage3.out_dir = stage3_dir
            _logger = setup_logger(stage3_dir, name='train')
            train_stage3_rmit(args_stage3)
        else:
            logger.error("[SurgiRepFormer] Stage 3 requires --rmit_manifest_json")
    else:
        # Run single stage
        if args.stage == 1:
            if not args.dr_imagefolder_dir:
                logger.error("[SurgiRepFormer] Stage 1 requires --dr_imagefolder_dir")
                return
            train_stage1_dr(args)
        elif args.stage == 2:
            fovea_json = args.fovea_pairs_json or args.fovea_manifest_json
            if not fovea_json:
                logger.error("[SurgiRepFormer] Stage 2 requires --fovea_pairs_json")
                return
            # Ensure fovea_pairs_json is set
            args.fovea_pairs_json = fovea_json
            train_stage2_fovea(args)
        elif args.stage == 3:
            # Stage 3: RMIT (backward compatible - rmit_manifest_json not strictly required if using old train())
            if args.rmit_manifest_json:
                train_stage3_rmit(args)
            else:
                # Fallback to old train() for backward compatibility
                logger.info("[SurgiRepFormer] Using legacy train() function (no --rmit_manifest_json provided)")
                train(args)


if __name__ == '__main__':
    main()
