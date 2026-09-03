#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SurgiMamba for RMIT Surgical Instrument Tip Tracking (Task #14)

A Vision Mamba model using Selective State Space Models (SSM) for tip detection.
- Pure PyTorch implementation (no mamba-ssm dependency)
- Bidirectional scanning for global context
- U-Net style skip connections for multi-scale features

Supports three-stage training:
- Stage 1 (DR): Diabetic Retinopathy classification (pre-trains backbone)
- Stage 2 (FOVEA): Fovea localization (pre-trains heatmap decoder)
- Stage 3 (RMIT): Surgical instrument tip tracking (final task)
"""

import os
import sys
import json
import argparse
import logging
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder


# =============================================================================
# Logging Setup
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


def get_logger(name: str = 'train') -> logging.Logger:
    """Get existing logger or create a basic one."""
    return logging.getLogger(name)


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
# Mamba Building Blocks (Pure PyTorch Implementation)
# =============================================================================

class SimpleMambaBlock(nn.Module):
    """
    简化版 Mamba 块 - 纯 PyTorch 实现
    核心思想: 选择性状态空间模型 (S6)
    - 输入依赖的 A, B, C 参数（选择性机制）
    - 1D 因果扫描替代全局注意力
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        inner_dim = int(dim * expand)
        self.inner_dim = inner_dim
        
        # 输入投影
        self.in_proj = nn.Linear(dim, inner_dim * 2)  # x 和 z 两路
        
        # 1D 因果卷积（局部上下文）
        self.conv1d = nn.Conv1d(inner_dim, inner_dim, d_conv, 
                                padding=d_conv-1, groups=inner_dim)
        
        # SSM 参数投影（选择性机制的核心）
        self.x_proj = nn.Linear(inner_dim, d_state * 2 + 1, bias=False)  # B, C, dt
        
        # 可学习的 A 参数（对角化）
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A))  # log 空间稳定训练
        self.D = nn.Parameter(torch.ones(inner_dim))  # skip connection
        
        # 输出投影
        self.out_proj = nn.Linear(inner_dim, dim)
        self.norm = nn.LayerNorm(dim)
    
    def ssm_scan(self, x):
        """
        选择性扫描 (Selective Scan)
        x: (B, L, D) where D=inner_dim
        """
        B_batch, L, D = x.shape
        
        # 生成 SSM 参数 (选择性: 依赖于输入)
        x_proj = self.x_proj(x)  # (B, L, 2*d_state + 1)
        dt = F.softplus(x_proj[..., :1])  # 步长 delta (B, L, 1)
        B_ssm = x_proj[..., 1:1+self.d_state]  # (B, L, d_state)
        C_ssm = x_proj[..., 1+self.d_state:]   # (B, L, d_state)
        
        # 离散化 A
        A = -torch.exp(self.A_log)  # (d_state,) 负值保证稳定
        
        # 扫描（简化版：直接循环，小序列可接受）
        h = torch.zeros(B_batch, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(L):
            # h = A_bar * h + B_bar * x_t
            dA = torch.exp(dt[:, t] * A.unsqueeze(0))  # (B, d_state)
            dA = dA.unsqueeze(1).expand(-1, D, -1)      # (B, D, d_state)
            dB = (dt[:, t] * B_ssm[:, t]).unsqueeze(1).expand(-1, D, -1)
            h = dA * h + dB * x[:, t].unsqueeze(-1)
            # y_t = C * h
            y_t = (h * C_ssm[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # (B, L, D)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)  # skip connection
        return y
    
    def forward(self, x):
        """x: (B, L, D)"""
        residual = x
        x = self.norm(x)
        
        # 双路投影
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        
        # 1D 卷积 (因果)
        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[..., :x.shape[1]]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)
        
        # SSM 扫描
        y = self.ssm_scan(x_branch)
        
        # 门控
        y = y * F.silu(z)
        
        return residual + self.out_proj(y)


class VisionMambaBlock(nn.Module):
    """
    视觉 Mamba 块：2D → 1D 扫描 → 2D
    使用双向扫描（前向+后向）捕获全局上下文
    """
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.mamba_fwd = SimpleMambaBlock(dim, d_state)
        self.mamba_bwd = SimpleMambaBlock(dim, d_state)  # 反向扫描
        self.proj = nn.Linear(dim * 2, dim)  # 融合双向
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 展平为序列
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        # 双向扫描
        y_fwd = self.mamba_fwd(x_seq)
        y_bwd = self.mamba_bwd(x_seq.flip(1)).flip(1)
        # 融合
        y = self.proj(torch.cat([y_fwd, y_bwd], dim=-1))
        # 恢复 2D
        return y.transpose(1, 2).reshape(B, C, H, W)


class ConvBlock(nn.Module):
    """轻量卷积块，用于早期 stage (序列太长，Mamba 循环慢)"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim * 4, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim * 4, dim, 1)
    
    def forward(self, x):
        return x + self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))


class PatchMerge(nn.Module):
    """下采样模块：空间分辨率减半，通道数增加"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        return self.norm(self.conv(x))


# =============================================================================
# Stage 1: DR Classification Head and Dataset
# =============================================================================

class DRClassificationHead(nn.Module):
    """
    Classification head for DR stage.
    Applied on backbone output to get class logits.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) -> (B, num_classes)"""
        x = self.pool(x)  # (B, C, 1, 1)
        x = x.flatten(1)  # (B, C)
        return self.fc(x)  # (B, num_classes)


class DRImageFolderDataset(Dataset):
    """
    Dataset for DR classification from ImageFolder format.
    Directory structure: root/DR/*.jpg, root/No_DR/*.jpg
    """
    def __init__(
        self, 
        root_dir: str, 
        img_size: int = 256, 
        indices: Optional[List[int]] = None, 
        augment: bool = True,
        label_map: Optional[Dict[str, int]] = None
    ):
        self.base = ImageFolder(root=root_dir)
        self.img_size = img_size
        self.augment = augment
        self.indices = list(indices) if indices is not None else list(range(len(self.base.samples)))
        self.label_map = dict(label_map) if label_map is not None else {
            name: idx for name, idx in self.base.class_to_idx.items()
        }
        
        self.items = []
        for sample_idx in self.indices:
            path, orig_label = self.base.samples[sample_idx]
            class_name = self.base.classes[orig_label]
            self.items.append((path, self.label_map[class_name]))
        
        # Data augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        )
        
        print(f"[SurgiMamba] DR Dataset: {len(self.items)} samples")
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[idx]
        
        # Load image
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Augmentation
        if self.augment:
            # Random flip
            if random.random() < 0.5:
                img_tensor = torch.flip(img_tensor, [2])
            # Color jitter
            img_tensor = self.color_jitter(img_tensor)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)


def build_dr_datasets(
    root_dir: str, 
    img_size: int = 256, 
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[DRImageFolderDataset, DRImageFolderDataset, int]:
    """
    Build train/val DR datasets with stratified split.
    
    Returns:
        train_ds, val_ds, num_classes
    """
    random.seed(seed)
    
    base_ds = ImageFolder(root=root_dir)
    
    # Determine label mapping
    if set(base_ds.classes) == {"No_DR", "DR"}:
        label_map = {"No_DR": 0, "DR": 1}
    else:
        label_map = {name: idx for idx, name in enumerate(sorted(base_ds.classes))}
    
    num_classes = len(label_map)
    
    # Group indices by label (stratified split)
    by_label: Dict[int, List[int]] = {}
    for i, (_path, label) in enumerate(base_ds.samples):
        mapped = label_map[base_ds.classes[label]]
        by_label.setdefault(mapped, []).append(i)
    
    train_indices, val_indices = [], []
    for label, indices in sorted(by_label.items()):
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
    
    train_ds = DRImageFolderDataset(root_dir, img_size, train_indices, augment=True, label_map=label_map)
    val_ds = DRImageFolderDataset(root_dir, img_size, val_indices, augment=False, label_map=label_map)
    
    return train_ds, val_ds, num_classes


# =============================================================================
# Stage 2: FOVEA Cross-Domain Contrastive Learning
# =============================================================================

class FOVEAPairedDataset(Dataset):
    """
    Dataset for FOVEA cross-domain contrastive learning.
    Loads pre-operative and intra-operative image pairs.
    """
    def __init__(
        self,
        pairs: List[dict],
        img_size: int = 256,
        augment: bool = True
    ):
        self.pairs = list(pairs)
        self.img_size = img_size
        self.augment = augment
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        print(f"[SurgiMamba] FOVEA Paired Dataset: {len(self.pairs)} pairs")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        p = self.pairs[idx]
        
        # Load pre-operative and intra-operative images
        pre = Image.open(p["pre_img"]).convert("RGB")
        intra = Image.open(p["intra_img"]).convert("RGB")
        
        # Resize to target size
        pre = pre.resize((self.img_size, self.img_size))
        intra = intra.resize((self.img_size, self.img_size))
        
        # Synchronized augmentation (same flip for both images)
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
    Multi-scale contrastive projection head.
    Projects features from multiple scales to a common embedding space.
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
    
    Args:
        pre_embs: List of pre-operative embeddings at different scales
        intra_embs: List of intra-operative embeddings at different scales
        temperature: Softmax temperature
    
    Returns:
        Average InfoNCE loss across all scales
    """
    total_loss = 0.0
    for pre_e, intra_e in zip(pre_embs, intra_embs):
        # Compute similarity matrix
        logits = torch.mm(pre_e, intra_e.t()) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Symmetric loss
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


def load_fovea_pairs(
    pairs_json: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Load FOVEA pairs and split by patient_id.
    
    Args:
        pairs_json: Path to fovea_pairs.json
        val_ratio: Validation ratio
        seed: Random seed
    
    Returns:
        train_pairs, val_pairs
    """
    random.seed(seed)
    
    with open(pairs_json, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    # Split by patient_id
    patient_ids = list(set(p["patient_id"] for p in pairs))
    random.shuffle(patient_ids)
    
    if len(patient_ids) == 1:
        return pairs, pairs
    
    n_val = max(1, int(round(len(patient_ids) * val_ratio)))
    n_val = min(n_val, len(patient_ids) - 1)
    val_ids = set(patient_ids[:n_val])
    
    train_pairs = [p for p in pairs if p["patient_id"] not in val_ids]
    val_pairs = [p for p in pairs if p["patient_id"] in val_ids]
    
    return train_pairs, val_pairs


# =============================================================================
# SurgiMamba Model Architecture
# =============================================================================

class SurgiMambaHeatmap(nn.Module):
    """
    SurgiMamba: Vision Mamba for Surgical Instrument Tip Detection
    
    Architecture (~3-5M params):
    - Stem: Conv layers for initial feature extraction
    - Stage 1-2: ConvBlock (序列太长，用卷积更快)
    - Stage 3-4: VisionMambaBlock (序列较短，Mamba优势大)
    - Heatmap Head: U-Net style decoder with skip connections
    """
    
    def __init__(
        self,
        heatmap_size: int = 64,
        dims: List[int] = [32, 64, 128, 256],   # Reduced for ~3-4M params
        depths: List[int] = [2, 2, 2, 2],       # Reduced depth
        d_state: int = 8                        # Reduced state dim
    ):
        super().__init__()
        
        self.heatmap_size = heatmap_size
        self.dims = dims
        
        # Stem: 256x256x3 → 64x64x dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, dims[0], 3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )
        
        # Stage 1: 64x64x64 → 64x64x64 (ConvBlock, 序列4096太长)
        self.stage1 = nn.Sequential(*[ConvBlock(dims[0]) for _ in range(depths[0])])
        
        # Stage 2: 64x64x64 → 32x32x128 (ConvBlock)
        self.down1 = PatchMerge(dims[0], dims[1])
        self.stage2 = nn.Sequential(*[ConvBlock(dims[1]) for _ in range(depths[1])])
        
        # Stage 3: 32x32x128 → 16x16x256 (VisionMambaBlock, 序列256)
        self.down2 = PatchMerge(dims[1], dims[2])
        self.stage3 = nn.Sequential(*[VisionMambaBlock(dims[2], d_state) for _ in range(depths[2])])
        
        # Stage 4: 16x16x256 → 8x8x512 (VisionMambaBlock, 序列64)
        self.down3 = PatchMerge(dims[2], dims[3])
        self.stage4 = nn.Sequential(*[VisionMambaBlock(dims[3], d_state) for _ in range(depths[3])])
        
        # Heatmap Head (U-Net style decoder with skip connections)
        # 8x8x512 → 16x16x256
        self.up1 = nn.ConvTranspose2d(dims[3], dims[2], 4, stride=2, padding=1)
        self.up1_norm = nn.BatchNorm2d(dims[2])
        self.up1_conv = nn.Sequential(
            nn.Conv2d(dims[2] * 2, dims[2], 3, padding=1),
            nn.BatchNorm2d(dims[2]),
            nn.GELU(),
        )
        
        # 16x16x256 → 32x32x128
        self.up2 = nn.ConvTranspose2d(dims[2], dims[1], 4, stride=2, padding=1)
        self.up2_norm = nn.BatchNorm2d(dims[1])
        self.up2_conv = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], 3, padding=1),
            nn.BatchNorm2d(dims[1]),
            nn.GELU(),
        )
        
        # 32x32x128 → 64x64x64
        self.up3 = nn.ConvTranspose2d(dims[1], dims[0], 4, stride=2, padding=1)
        self.up3_norm = nn.BatchNorm2d(dims[0])
        self.up3_conv = nn.Sequential(
            nn.Conv2d(dims[0] * 2, dims[0], 3, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )
        
        # Final heatmap: 64x64x64 → 64x64x1
        self.heatmap_head = nn.Conv2d(dims[0], 1, 1)
        
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
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SurgiMamba] Total parameters: {total:,}")
        print(f"[SurgiMamba] Trainable parameters: {trainable:,}")
        print(f"[SurgiMamba] Model size: {total * 4 / 1024 / 1024:.2f} MB")
        
        # Estimate VRAM
        vram_estimate = total * 4 * 2 / 1024 / 1024 + 150  # params + gradients + activations
        print(f"[SurgiMamba] Estimated VRAM (per sample): ~{vram_estimate:.0f} MB")
    
    def forward(self, x: torch.Tensor, return_backbone_feat: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W) input image (256x256)
            return_backbone_feat: If True, return backbone features (for DR classification)
        
        Returns:
            dict with keys:
                - heatmap: (B, 1, 64, 64)
                - coords: (B, 2) decoded coordinates in [0, 1]
                - backbone_feat: (B, 512, 8, 8) if return_backbone_feat=True
        """
        # Encoder
        x = self.stem(x)        # (B, 64, 64, 64)
        
        s1 = self.stage1(x)     # (B, 64, 64, 64) - skip connection
        
        x = self.down1(s1)      # (B, 128, 32, 32)
        s2 = self.stage2(x)     # (B, 128, 32, 32) - skip connection
        
        x = self.down2(s2)      # (B, 256, 16, 16)
        s3 = self.stage3(x)     # (B, 256, 16, 16) - skip connection
        
        x = self.down3(s3)      # (B, 512, 8, 8)
        x = self.stage4(x)      # (B, 512, 8, 8)
        
        # Return backbone features for DR classification
        if return_backbone_feat:
            return {'backbone_feat': x}
        
        # Decoder with skip connections
        x = F.gelu(self.up1_norm(self.up1(x)))  # (B, 256, 16, 16)
        x = self.up1_conv(torch.cat([x, s3], dim=1))  # + skip
        
        x = F.gelu(self.up2_norm(self.up2(x)))  # (B, 128, 32, 32)
        x = self.up2_conv(torch.cat([x, s2], dim=1))  # + skip
        
        x = F.gelu(self.up3_norm(self.up3(x)))  # (B, 64, 64, 64)
        x = self.up3_conv(torch.cat([x, s1], dim=1))  # + skip
        
        # Heatmap
        heatmap = self.heatmap_head(x)  # (B, 1, 64, 64)
        
        # Decode coordinates
        coords = soft_argmax_2d(heatmap, temperature=0.1)  # (B, 2)
        
        return {
            'heatmap': heatmap,
            'coords': coords
        }
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to backbone features. Used for DR classification."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        return x  # (B, 256, 8, 8)
    
    def extract_multi_scale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from encoder stages.
        Used for FOVEA cross-domain contrastive learning.
        
        Returns:
            List of 3 feature maps from stage 2, 3, 4:
            - stage2: (B, 64, 32, 32)
            - stage3: (B, 128, 16, 16)
            - stage4: (B, 256, 8, 8)
        """
        x = self.stem(x)        # (B, 32, 64, 64)
        x = self.stage1(x)      # (B, 32, 64, 64)
        
        x = self.down1(x)       # (B, 64, 32, 32)
        s2 = self.stage2(x)     # (B, 64, 32, 32)
        
        x = self.down2(s2)      # (B, 128, 16, 16)
        s3 = self.stage3(x)     # (B, 128, 16, 16)
        
        x = self.down3(s3)      # (B, 256, 8, 8)
        s4 = self.stage4(x)     # (B, 256, 8, 8)
        
        return [s2, s3, s4]  # channels: [64, 128, 256]


# =============================================================================
# Dataset
# =============================================================================

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


# =============================================================================
# Stage 1: DR Classification Training
# =============================================================================

def train_stage_dr(args, device: torch.device, logger: logging.Logger):
    """
    Stage 1: Train DR classification.
    Saves backbone weights for Stage 2.
    """
    logger.info(f"[SurgiMamba] === Stage 1: DR Classification ===")
    logger.info(f"[SurgiMamba] DR ImageFolder: {args.dr_imagefolder_dir}")
    
    # Build datasets
    train_ds, val_ds, num_classes = build_dr_datasets(
        args.dr_imagefolder_dir,
        img_size=args.img_size,
        val_ratio=args.dr_val_ratio,
        seed=args.seed
    )
    
    logger.info(f"[SurgiMamba] DR classes: {num_classes}, train: {len(train_ds)}, val: {len(val_ds)}")
    
    # Compute class weights for imbalanced data
    train_labels = [label for _, label in train_ds.items]
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / np.maximum(counts, 1.0)
    sample_weights = [weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    model = SurgiMambaHeatmap(
        heatmap_size=args.heatmap_size,
        dims=[32, 64, 128, 256],
        depths=[2, 2, 2, 2],
        d_state=8
    ).to(device)
    
    # Classification head
    cls_head = DRClassificationHead(in_channels=256, num_classes=num_classes).to(device)
    
    # Optimizer
    optimizer = AdamW(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=args.dr_lr,
        weight_decay=args.wd
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.dr_epochs, eta_min=1e-6)
    
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # Training loop
    best_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.dr_epochs + 1):
        # Train
        model.train()
        cls_head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            backbone_feat = model.encode(images)  # (B, 512, 8, 8)
            logits = cls_head(backbone_feat)
            
            # Loss
            loss = F.cross_entropy(logits, labels, weight=class_weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
        
        scheduler.step()
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Validate
        model.eval()
        cls_head.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                backbone_feat = model.encode(images)
                logits = cls_head(backbone_feat)
                
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.numel()
        
        val_acc = val_correct / val_total
        
        logger.info(f"[SurgiMamba] DR epoch {epoch}/{args.dr_epochs} "
                   f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                   f"val_acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            
            # Save backbone weights only
            backbone_path = os.path.join(args.out_dir, 'dr_backbone.pth')
            torch.save(model.state_dict(), backbone_path)
            logger.info(f"[SurgiMamba] Saved best backbone (val_acc={val_acc:.4f})")
    
    logger.info(f"[SurgiMamba] Stage 1 complete. Best val_acc={best_acc:.4f} at epoch {best_epoch}")
    
    # Return backbone path for Stage 2
    return os.path.join(args.out_dir, 'dr_backbone.pth')


# =============================================================================
# Stage 2: FOVEA Cross-Domain Contrastive Training
# =============================================================================

def train_stage_fovea(args, device: torch.device, logger: logging.Logger):
    """
    Stage 2: Multi-Granularity Cross-Domain Contrastive Bridging.
    
    Uses shared backbone to encode pre-operative and intra-operative images,
    applies InfoNCE contrastive loss at multiple feature scales to learn
    domain-invariant representations.
    
    Saves backbone weights for Stage 3.
    """
    logger.info(f"[SurgiMamba] === Stage 2: FOVEA Cross-Domain Contrastive Learning ===")
    logger.info(f"[SurgiMamba] FOVEA pairs JSON: {args.fovea_pairs_json}")
    
    # Load FOVEA pairs
    train_pairs, val_pairs = load_fovea_pairs(
        args.fovea_pairs_json,
        val_ratio=args.fovea_val_ratio,
        seed=args.seed
    )
    
    if len(train_pairs) == 0:
        raise ValueError("FOVEA train set is empty")
    
    logger.info(f"[SurgiMamba] FOVEA pairs: train={len(train_pairs)}, val={len(val_pairs)}")
    
    # Build datasets
    train_ds = FOVEAPairedDataset(
        train_pairs, 
        img_size=args.fovea_img_size, 
        augment=True
    )
    val_ds = FOVEAPairedDataset(
        val_pairs, 
        img_size=args.fovea_img_size, 
        augment=False
    )
    
    # DataLoaders with drop_last=True for contrastive learning
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.fovea_batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.fovea_batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True,
        drop_last=True
    )
    
    # Model (backbone only for feature extraction)
    model = SurgiMambaHeatmap(
        heatmap_size=64,
        dims=[32, 64, 128, 256],
        depths=[2, 2, 2, 2],
        d_state=8
    ).to(device)
    
    # Load backbone weights from Stage 1
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        logger.info(f"[SurgiMamba] Loading backbone from Stage 1: {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location=device, weights_only=True)
        # Use strict=False because Stage 1 only has encoder + classifier head
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"[SurgiMamba] Backbone loaded: {len(state_dict)} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")
    
    # Contrastive projector for multi-scale features
    # Channels from extract_multi_scale: stage2=64, stage3=128, stage4=256
    in_channels_list = [64, 128, 256]
    projector = ContrastiveProjector(
        in_channels_list=in_channels_list,
        embed_dim=args.fovea_embed_dim
    ).to(device)
    
    logger.info(f"[SurgiMamba] ContrastiveProjector: in_channels={in_channels_list}, embed_dim={args.fovea_embed_dim}")
    
    # Optimizer (only backbone encoder parameters, not decoder)
    # For simplicity, we train the entire model but focus on encoder
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()),
        lr=args.fovea_lr
    )
    
    # Training loop
    best_loss = float('inf')
    best_val_retrieval_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.fovea_epochs + 1):
        # Training
        model.train()
        projector.train()
        total_loss = 0.0
        num_batches = 0
        
        for pre_imgs, intra_imgs, patient_ids in train_loader:
            pre_imgs = pre_imgs.to(device)
            intra_imgs = intra_imgs.to(device)
            
            optimizer.zero_grad()
            
            # Extract multi-scale features
            pre_feats = model.extract_multi_scale(pre_imgs)
            intra_feats = model.extract_multi_scale(intra_imgs)
            
            # Project to embedding space
            pre_embs = projector(pre_feats)
            intra_embs = projector(intra_feats)
            
            # Multi-scale InfoNCE loss
            loss = multi_scale_info_nce(
                pre_embs, 
                intra_embs, 
                temperature=args.fovea_temperature
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        train_loss = total_loss / max(num_batches, 1)
        
        # Validation
        model.eval()
        projector.eval()
        val_loss = 0.0
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
                
                loss = multi_scale_info_nce(
                    pre_embs, 
                    intra_embs, 
                    temperature=args.fovea_temperature
                )
                val_loss += loss.item()
                val_batches += 1
                
                # Compute retrieval accuracy
                retrieval_acc = compute_retrieval_accuracy(pre_embs, intra_embs)
                val_retrieval_acc_sum += retrieval_acc
                val_retrieval_batches += 1
        
        val_loss = val_loss / max(val_batches, 1)
        val_retrieval_acc = val_retrieval_acc_sum / max(val_retrieval_batches, 1)
        
        logger.info(f"[SurgiMamba] FOVEA epoch {epoch}/{args.fovea_epochs} "
                   f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                   f"val_retrieval_acc={val_retrieval_acc:.1%}")
        
        # Early stopping and save best
        if val_loss < best_loss:
            best_loss = val_loss
            best_val_retrieval_acc = val_retrieval_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save backbone weights only (not projector, not decoder)
            backbone_path = os.path.join(args.out_dir, 'fovea_backbone.pth')
            torch.save(model.state_dict(), backbone_path)
            logger.info(f"[SurgiMamba] Saved best backbone (val_loss={val_loss:.4f}, val_retrieval_acc={val_retrieval_acc:.1%})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"[SurgiMamba] Early stopping at epoch {epoch}")
                break
    
    logger.info(f"[SurgiMamba] Stage 2 complete. Best val_loss={best_loss:.4f}, val_retrieval_acc={best_val_retrieval_acc:.1%} at epoch {best_epoch}")
    
    # Return backbone path for Stage 3
    return os.path.join(args.out_dir, 'fovea_backbone.pth')


# =============================================================================
# Stage 3: RMIT Tip Tracking Training
# =============================================================================

def train(args):
    """Main training loop."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[SurgiMamba] Using device: {device}")
    
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
    
    print(f"[SurgiMamba] Split mode: {args.split_mode}, fold: {args.fold}")
    print(f"[SurgiMamba] Train sequences: {train_seqs}")
    print(f"[SurgiMamba] Test sequence: {test_seq}")
    
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
    model = SurgiMambaHeatmap(
        heatmap_size=args.heatmap_size,
        dims=[32, 64, 128, 256],
        depths=[2, 2, 2, 2],
        d_state=8
    ).to(device)
    
    # Load pretrained weights from Stage 2 (if provided)
    if args.pretrained_weights and os.path.exists(args.pretrained_weights):
        print(f"[SurgiMamba] Loading pretrained weights: {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location=device, weights_only=True)
        # Use strict=False because Stage 2 may only have encoder weights
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[SurgiMamba] Loaded pretrained weights: {len(state_dict)} keys, missing: {len(missing)}, unexpected: {len(unexpected)}")
    
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
        print(f"[SurgiMamba] epoch {epoch}/{args.epochs} "
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
            print(f"[SurgiMamba] Saved best model (error: {best_val_error:.1f} px)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"[SurgiMamba] Early stopping at epoch {epoch}")
                break
        
        # Save latest checkpoint
        latest_path = os.path.join(args.out_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_path)
    
    print(f"[SurgiMamba] Training completed. Best error: {best_val_error:.1f} px")
    
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
    parser = argparse.ArgumentParser(description='SurgiMamba Three-Stage Training')
    
    # Stage and output
    parser.add_argument('--stage', type=int, default=3, choices=[1, 2, 3],
                        help='Training stage: 1=DR, 2=FOVEA, 3=RMIT (default: 3)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--run_all_stages', action='store_true',
                        help='Run all three stages sequentially')
    
    # Pretrained weights
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Pretrained weights from previous stage')
    
    # Training configuration
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--heatmap_size', type=int, default=64,
                        help='Output heatmap size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wd', type=float, default=1e-2,
                        help='Weight decay')
    
    # Stage 1: DR Classification
    parser.add_argument('--dr_imagefolder_dir', type=str, default=None,
                        help='DR dataset path (ImageFolder format: DR/ and No_DR/ subdirs)')
    parser.add_argument('--dr_num_classes', type=int, default=2,
                        help='Number of DR classes (default: 2 for binary)')
    parser.add_argument('--dr_epochs', type=int, default=30,
                        help='DR training epochs')
    parser.add_argument('--dr_lr', type=float, default=1e-3,
                        help='DR learning rate')
    parser.add_argument('--dr_val_ratio', type=float, default=0.2,
                        help='DR validation ratio')
    
    # Stage 2: FOVEA Cross-Domain Contrastive Learning
    parser.add_argument('--fovea_pairs_json', type=str, default=None,
                        help='FOVEA pairs JSON path (pre_img, intra_img, patient_id)')
    parser.add_argument('--fovea_img_size', type=int, default=256,
                        help='FOVEA image size (default: 256)')
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
                        help='FOVEA validation ratio (default: 0.2)')
    
    # Stage 3: RMIT Tracking
    parser.add_argument('--rmit_manifest_json', type=str, default=None,
                        help='RMIT manifest JSON path')
    parser.add_argument('--split_mode', type=str, default='loso',
                        choices=['loso'], help='Split mode')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2],
                        help='Fold index for LOSO')
    parser.add_argument('--epochs', type=int, default=100,
                        help='RMIT training epochs')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='RMIT learning rate')
    parser.add_argument('--no_crop', action='store_true',
                        help='Skip FOV crop, directly resize to img_size')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(args.out_dir, name='train')
    
    # Print configuration
    logger.info("[SurgiMamba] Configuration:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info(f"[SurgiMamba] Device: {device}")
    
    # Run all stages or single stage
    if args.run_all_stages:
        # Stage 1: DR
        logger.info("[SurgiMamba] === Running all stages ===")
        args.stage = 1
        stage1_path = train_stage_dr(args, device, logger)
        
        # Stage 2: FOVEA
        args.stage = 2
        args.pretrained_weights = stage1_path
        stage2_path = train_stage_fovea(args, device, logger)
        
        # Stage 3: RMIT
        args.stage = 3
        args.pretrained_weights = stage2_path
        train(args)
    else:
        # Single stage
        if args.stage == 1:
            if not args.dr_imagefolder_dir:
                logger.error("--dr_imagefolder_dir is required for Stage 1")
                sys.exit(1)
            train_stage_dr(args, device, logger)
        elif args.stage == 2:
            if not args.fovea_pairs_json:
                logger.error("--fovea_pairs_json is required for Stage 2")
                sys.exit(1)
            train_stage_fovea(args, device, logger)
        elif args.stage == 3:
            # For Stage 3, rmit_manifest_json is not required if only loading weights
            # but we need either rmit_manifest_json or pretrained_weights
            if not args.rmit_manifest_json and not args.pretrained_weights:
                logger.error("--rmit_manifest_json or --pretrained_weights is required for Stage 3")
                sys.exit(1)
            train(args)


if __name__ == '__main__':
    main()
