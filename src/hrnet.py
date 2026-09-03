#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNet-W32 for RMIT Surgical Instrument Tip Tracking (Task #10)

Based on resnet.py framework, backbone replaced with HRNet-W32.
- Backbone: HRNet-W32 (via timm)
- Head: Heatmap head with soft-argmax decoding
- Preprocessing: Circular FOV detection and cropping
"""

import os
import json
import argparse
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

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[HRNet] Warning: timm not installed. Please run: pip install timm")


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
# Model Architecture
# =============================================================================

class HRNetW32Heatmap(nn.Module):
    """
    HRNet-W32 backbone with heatmap head for tip detection.
    Uses timm for the backbone implementation.
    """
    
    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: str = None,
        heatmap_size: int = 64,
        use_offset_head: bool = False
    ):
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm is required for HRNet. Install with: pip install timm")
        
        self.heatmap_size = heatmap_size
        self.use_offset_head = use_offset_head
        
        # Load HRNet-W32 backbone via timm (features_only mode)
        # This returns multi-scale features from different stages
        # If pretrained_path is provided, load from local file; otherwise download if pretrained=True
        self.backbone = timm.create_model(
            'hrnet_w32',
            pretrained=False,  # We'll load weights manually
            features_only=True,
            out_indices=(0, 1, 2, 3)  # Get all 4 scale outputs
        )
        
        # Load pretrained weights
        if pretrained or pretrained_path:
            if pretrained_path and os.path.isfile(pretrained_path):
                print(f"[HRNet] Loading pretrained weights from: {pretrained_path}")
                state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
                self._load_pretrained_weights(state_dict)
            elif pretrained:
                print("[HRNet] Loading pretrained weights from timm...")
                # Load from timm's pretrained model
                pretrained_model = timm.create_model('hrnet_w32', pretrained=True)
                state_dict = pretrained_model.state_dict()
                self._load_pretrained_weights(state_dict)
            else:
                print("[HRNet] Warning: No pretrained weights loaded, training from scratch")
        
        # HRNet-W32 output channels at different scales:
        # Stage 0: 32 channels at 1/4 resolution
        # Stage 1: 64 channels at 1/8 resolution
        # Stage 2: 128 channels at 1/16 resolution
        # Stage 3: 256 channels at 1/32 resolution
        # But in features_only mode with hrnet_w32, the channels are:
        # [32, 64, 128, 256] for stages 0-3
        
        # Get actual feature info from the backbone
        feature_info = self.backbone.feature_info
        self.feature_channels = [info['num_chs'] for info in feature_info]
        print(f"[HRNet] Feature channels: {self.feature_channels}")
        
        # Use highest resolution feature (stage 0) for heatmap generation
        # Channel count is typically 32 for HRNet-W32 at highest resolution
        high_res_channels = self.feature_channels[0]
        
        # Heatmap head: upsample from 1/4 resolution to heatmap_size (64x64)
        # For 256x256 input, 1/4 resolution = 64x64, so we may need minimal upsampling
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(high_res_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        
        # Optional offset head (using all scale features)
        if use_offset_head:
            total_channels = sum(self.feature_channels)
            self.offset_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(total_channels, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
                nn.Tanh()  # Output in [-1, 1]
            )
        
        # Count parameters
        self._count_parameters()
    
    def _load_pretrained_weights(self, state_dict: dict):
        """Load pretrained weights into backbone, handling key name differences."""
        backbone_state = self.backbone.state_dict()
        loaded_keys = []
        missing_keys = []
        
        for key, param in state_dict.items():
            # Map pretrained keys to backbone keys
            # timm HRNet uses prefix like 'conv1.', 'bn1.', 'layer1.', etc.
            # features_only mode uses same structure
            if key in backbone_state:
                if backbone_state[key].shape == param.shape:
                    backbone_state[key] = param
                    loaded_keys.append(key)
                else:
                    print(f"[HRNet] Shape mismatch for {key}: {backbone_state[key].shape} vs {param.shape}")
            else:
                missing_keys.append(key)
        
        # Load the filtered state dict
        self.backbone.load_state_dict(backbone_state)
        print(f"[HRNet] Loaded {len(loaded_keys)} pretrained parameters")
        if missing_keys:
            print(f"[HRNet] Note: {len(missing_keys)} keys not found in backbone (expected for classifier head, etc.)")
    
    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[HRNetW32Heatmap] Total parameters: {total:,}")
        print(f"[HRNetW32Heatmap] Trainable parameters: {trainable:,}")
        print(f"[HRNetW32Heatmap] Model size: {total * 4 / 1024 / 1024:.2f} MB")
        
        # Estimate VRAM
        vram_estimate = total * 4 * 2 / 1024 / 1024 + 200  # params + gradients + activations
        print(f"[HRNetW32Heatmap] Estimated VRAM (per sample): ~{vram_estimate:.0f} MB")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W) input image
        
        Returns:
            dict with keys:
                - heatmap: (B, 1, heatmap_size, heatmap_size)
                - offset: (B, 2) if use_offset_head
                - coords: (B, 2) decoded coordinates in [0, 1]
        """
        # Get multi-scale features from backbone
        features = self.backbone(x)  # List of feature maps at different scales
        
        # Use highest resolution feature (stage 0) for heatmap
        high_res_feat = features[0]  # (B, 32, H/4, W/4)
        
        # Generate heatmap
        heatmap = self.heatmap_head(high_res_feat)  # (B, 1, H/4, W/4)
        
        # Resize heatmap to target size if needed
        if heatmap.shape[2] != self.heatmap_size or heatmap.shape[3] != self.heatmap_size:
            heatmap = F.interpolate(
                heatmap,
                size=(self.heatmap_size, self.heatmap_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Decode coordinates using soft-argmax
        coords = soft_argmax_2d(heatmap, temperature=0.1)  # (B, 2)
        
        output = {
            'heatmap': heatmap,
            'coords': coords
        }
        
        # Optional offset head
        if self.use_offset_head:
            # Concatenate all scale features after global pooling
            pooled_features = []
            for feat in features:
                pooled = F.adaptive_avg_pool2d(feat, 1)
                pooled_features.append(pooled)
            concat_features = torch.cat(pooled_features, dim=1)
            offset = self.offset_head(concat_features)
            output['offset'] = offset
        
        return output


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


def train(args):
    """Main training loop."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[HRNet] Using device: {device}")
    
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
    
    print(f"[HRNet] Split mode: {args.split_mode}, fold: {args.fold}")
    print(f"[HRNet] Train sequences: {train_seqs}")
    print(f"[HRNet] Test sequence: {test_seq}")
    
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
    model = HRNetW32Heatmap(
        pretrained=False,  # Use local pretrained_path instead
        pretrained_path=args.pretrained_path,
        heatmap_size=args.heatmap_size,
        use_offset_head=False
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
    # patience_counter = 0  # 早停机制已禁用
    
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
        print(f"[HRNet] epoch {epoch}/{args.epochs} "
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
            # patience_counter = 0  # 早停机制已禁用
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.out_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': best_val_error,
                'metrics': metrics
            }, checkpoint_path)
            print(f"[HRNet] Saved best model (error: {best_val_error:.1f} px)")
        # else:
        #     patience_counter += 1
        #     if patience_counter >= args.patience:
        #         print(f"[HRNet] Early stopping at epoch {epoch}")
        #         break
        
        # Save latest checkpoint
        latest_path = os.path.join(args.out_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, latest_path)
    
    print(f"[HRNet] Training completed. Best error: {best_val_error:.1f} px")
    
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
    parser = argparse.ArgumentParser(description='HRNet-W32 for RMIT Tip Tracking')
    
    # Required arguments
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--rmit_manifest_json', type=str, required=True,
                        help='Path to RMIT manifest JSON')
    
    # Training configuration
    parser.add_argument('--split_mode', type=str, default='loso',
                        choices=['loso'], help='Split mode')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2],
                        help='Fold index for LOSO')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size after cropping')
    parser.add_argument('--heatmap_size', type=int, default=64,
                        help='Output heatmap size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default 8 for HRNet)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--no_crop', action='store_true',
                        help='Skip FOV crop, directly resize to img_size')
    parser.add_argument('--pretrained_path', type=str, 
                        default='/root/autodl-tmp/longfei/image_classification/models/hrnetv2_w32-90d8c5fb.pth',
                        help='Path to local pretrained HRNet weights (default: use local model)')
    
    args = parser.parse_args()
    
    # Print configuration
    print("[HRNet] Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    # Train
    train(args)


if __name__ == '__main__':
    main()
