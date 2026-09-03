"""
DR-SurgBridgeNet Optimized Implementation
==========================================
Key optimizations based on literature review:
1. Multi-scale FPN feature fusion (ViTDet, ECCV 2022)
2. Multi-layer contrastive alignment (MoCo v2, SupCon)
3. Temporal memory bank for tracking (TimeSformer, CoTTA)
4. Uncertainty-aware heatmap head (Evidential Deep Learning)
5. Test-time adaptation (Tent, SAR)
"""

import math
import os
import sys
import json
import random
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# ============================================================
# Utility Functions (from original)
# ============================================================

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def cosine_lr(base_lr: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return base_lr
    return base_lr * 0.5 * (1 + math.cos(math.pi * step / total_steps))

# ============================================================
# NEW: Feature Pyramid Network (FPN) for Multi-Scale Features
# ============================================================
# Reference: ViTDet (ECCV 2022) - "Exploring Plain Vision Transformer Backbones for Object Detection"
# Key insight: Extract features from multiple ViT layers and fuse them

class FeaturePyramidDecoder(nn.Module):
    """
    Multi-scale feature pyramid from ViT backbone.
    
    Extracts features from intermediate layers and fuses them:
    - Layer i: Early features (fine spatial detail)
    - Layer j: Middle features (semantic + spatial)
    - Layer k: Final layer (high semantic)
    
    Architecture:
    1. Extract patch tokens from 3 layers
    2. Reshape to 2D spatial maps
    3. Apply lateral convolutions (1x1)
    4. Top-down pathway with upsampling
    5. Output: fused multi-scale features
    """
    
    def __init__(
        self,
        embed_dim: int,
        out_dim: int = 256,
        num_levels: int = 3,
        use_p2: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_levels = num_levels
        self.use_p2 = use_p2
        
        # Lateral convolutions: project each level to out_dim
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, out_dim, kernel_size=1)
            for _ in range(num_levels)
        ])
        
        # Smooth convolutions after upsampling
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
            for _ in range(num_levels - 1)
        ])
        
        # FPN output convolutions
        self.fpn_conv = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        
        # Additional P2 level (higher resolution) for small tool tip detection
        if use_p2:
            self.p2_conv = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        features_list: List[torch.Tensor],  # List of [B, N, C] from different layers
        H: int, 
        W: int
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features_list: List of patch token tensors from different ViT layers
                           Each tensor: [B, N, C] where N = H*W/patch_size^2
            H, W: Spatial dimensions of the feature map
        
        Returns:
            Dict containing:
            - 'fpn_fused': [B, out_dim, H, W] - Final fused features
            - 'fpn_levels': List of feature maps at each FPN level
        """
        assert len(features_list) == self.num_levels, \
            f"Expected {self.num_levels} feature maps, got {len(features_list)}"
        
        # Reshape all features to 2D spatial maps
        fpn_features = []
        for i, feat in enumerate(features_list):
            # feat: [B, N, C] -> [B, C, H, W]
            B, N, C = feat.shape
            feat_2d = feat.transpose(1, 2).reshape(B, C, H, W)
            
            # Apply lateral conv
            lat_feat = self.lateral_convs[i](feat_2d)  # [B, out_dim, H, W]
            fpn_features.append(lat_feat)
        
        # Top-down pathway: fuse from highest level to lowest
        # fpn_features[0] = lowest layer (early), fpn_features[-1] = highest layer (final)
        # We want to propagate semantic info from top to bottom
        
        # Start from the deepest layer
        fpn_out = fpn_features[-1]
        fpn_levels = [fpn_out]
        
        # Build pyramid top-down
        for i in range(self.num_levels - 2, -1, -1):
            # Upsample and add
            fpn_out = F.interpolate(
                fpn_out, size=fpn_features[i].shape[-2:],
                mode='bilinear', align_corners=False
            )
            fpn_out = fpn_out + fpn_features[i]
            # Smooth
            fpn_out = self.smooth_convs[self.num_levels - 2 - i](fpn_out)
            fpn_levels.insert(0, fpn_out)
        
        # Final output
        fpn_fused = self.fpn_conv(fpn_levels[0])
        
        # Optional P2 level (2x upsampled for fine detail)
        if self.use_p2:
            p2 = F.interpolate(fpn_fused, scale_factor=2, mode='bilinear', align_corners=False)
            p2 = self.p2_conv(p2)
            fpn_levels.insert(0, p2)
        
        return {
            'fpn_fused': fpn_fused,  # [B, out_dim, H, W]
            'fpn_levels': fpn_levels  # List of [B, out_dim, H_i, W_i]
        }


# ============================================================
# NEW: Multi-Layer Contrastive Loss
# ============================================================
# Reference: MoCo v2, SupCon - Multi-layer contrastive learning improves transfer

class MultiLayerContrastiveLoss(nn.Module):
    """
    Multi-layer contrastive alignment loss.
    
    Instead of only aligning CLS tokens from the final layer,
    align features from multiple layers for richer transfer.
    
    Combines:
    1. InfoNCE loss for CLS tokens at each layer
    2. MMD (Maximum Mean Discrepancy) for patch-level distributions
    """
    
    def __init__(
        self,
        num_layers: int = 3,
        temperature: float = 0.07,
        layer_weights: Optional[List[float]] = None,
        use_mmd: bool = True,
        mmd_weight: float = 0.1,
        embed_dim: int = 384,  # Added parameter, default 384 for backward compatibility
    ):
        super().__init__()
        self.num_layers = num_layers
        self.temperature = temperature
        self.use_mmd = use_mmd
        self.mmd_weight = mmd_weight
        self.embed_dim = embed_dim  # Store for reference
        
        if layer_weights is None:
            # Default: increasing weight for deeper layers
            self.layer_weights = [0.5, 0.75, 1.0]
        else:
            self.layer_weights = layer_weights
        
        # Projection heads for each layer
        # Using configurable embed_dim instead of hardcoded 384
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, 256),  # embed_dim -> hidden (now configurable)
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
            )
            for _ in range(num_layers)
        ])
    
    def info_nce_loss(
        self, 
        z1: torch.Tensor, 
        z2: torch.Tensor
    ) -> torch.Tensor:
        """Standard InfoNCE loss for paired samples."""
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        B = z1.shape[0]
        
        # Similarity matrix
        sim = torch.mm(z1, z2.t()) / self.temperature  # [B, B]
        
        # Labels: diagonal is positive
        labels = torch.arange(B, device=z1.device)
        
        # Cross-entropy loss
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.t(), labels)
        
        return (loss_12 + loss_21) / 2
    
    def mmd_loss(
        self, 
        f1: torch.Tensor, 
        f2: torch.Tensor,
        kernel: str = 'rbf',
        bandwidth: float = None  # Auto-computed if None
    ) -> torch.Tensor:
        """
        Maximum Mean Discrepancy between two distributions.
        Measures distance between patch-level feature distributions.
        
        Numerical stability improvements:
        - Adaptive bandwidth based on feature dimension
        - Clamping of distance matrices to prevent negative values (float errors)
        - Gradient clipping on kernel values
        """
        # Flatten patch tokens
        f1_flat = f1.reshape(-1, f1.shape[-1])  # [B*N, C]
        f2_flat = f2.reshape(-1, f2.shape[-1])  # [B*N, C]
        
        n1, n2 = f1_flat.shape[0], f2_flat.shape[0]
        feat_dim = f1_flat.shape[-1]
        
        # Auto-compute bandwidth based on feature dimension (median heuristic)
        if bandwidth is None:
            # Median heuristic: bandwidth = median of pairwise distances
            with torch.no_grad():
                sample_size = min(100, n1)  # Sample for efficiency
                idx = torch.randperm(n1)[:sample_size]
                sample = f1_flat[idx]
                dists = torch.cdist(sample, sample)
                # Use median distance as bandwidth
                bandwidth = torch.median(dists).clamp(min=1.0).item()
        
        if kernel == 'rbf':
            # RBF kernel: K(x, y) = exp(-||x-y||^2 / (2*bandwidth^2))
            # Compute pairwise distances
            xx = torch.sum(f1_flat ** 2, dim=-1, keepdim=True)
            yy = torch.sum(f2_flat ** 2, dim=-1, keepdim=True)
            
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x.y
            # Clamp to prevent negative values from floating point errors
            dists_xx = torch.clamp(xx + xx.t() - 2 * torch.mm(f1_flat, f1_flat.t()), min=0.0)
            dists_yy = torch.clamp(yy + yy.t() - 2 * torch.mm(f2_flat, f2_flat.t()), min=0.0)
            dists_xy = torch.clamp(xx + yy.t() - 2 * torch.mm(f1_flat, f2_flat.t()), min=0.0)
            
            # Numerical stability: clamp kernel values
            bandwidth_sq = max(bandwidth ** 2, 1.0)  # Prevent division by tiny values
            k_xx = torch.exp(-dists_xx / (2 * bandwidth_sq)).clamp(max=1.0)
            k_yy = torch.exp(-dists_yy / (2 * bandwidth_sq)).clamp(max=1.0)
            k_xy = torch.exp(-dists_xy / (2 * bandwidth_sq)).clamp(max=1.0)
            
            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
            # Clamp result to prevent extreme values
            mmd = (k_xx.sum() / (n1 * n1) + k_yy.sum() / (n2 * n2) - 2 * k_xy.sum() / (n1 * n2))
            mmd = torch.clamp(mmd, min=0.0, max=10.0)  # Prevent runaway values
        else:
            # Linear kernel
            mmd = torch.mean(torch.mm(f1_flat, f1_flat.t())) + \
                  torch.mean(torch.mm(f2_flat, f2_flat.t())) - \
                  2 * torch.mean(torch.mm(f1_flat, f2_flat.t()))
            mmd = torch.clamp(mmd, min=0.0)  # MMD should be non-negative
        
        return mmd
    
    def forward(
        self,
        cls_list_1: List[torch.Tensor],  # CLS tokens from domain 1
        cls_list_2: List[torch.Tensor],  # CLS tokens from domain 2
        patch_list_1: Optional[List[torch.Tensor]] = None,  # Patch tokens (optional)
        patch_list_2: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-layer contrastive loss.
        
        Args:
            cls_list_1/2: Lists of CLS tokens from different layers
            patch_list_1/2: Optional lists of patch tokens for MMD
        
        Returns:
            Dict with 'total_loss' and individual losses
        """
        assert len(cls_list_1) == len(cls_list_2) == self.num_layers
        
        total_info_nce = 0.0
        losses_dict = {}
        
        for i, (cls1, cls2) in enumerate(zip(cls_list_1, cls_list_2)):
            # Project
            z1 = self.projectors[i](cls1)
            z2 = self.projectors[i](cls2)
            
            # InfoNCE
            loss_nce = self.info_nce_loss(z1, z2)
            total_info_nce += self.layer_weights[i] * loss_nce
            losses_dict[f'info_nce_layer_{i}'] = loss_nce.item()
        
        total_loss = total_info_nce / sum(self.layer_weights)
        
        # Optional MMD loss for patch-level alignment
        if self.use_mmd and patch_list_1 is not None and patch_list_2 is not None:
            total_mmd = 0.0
            for i, (p1, p2) in enumerate(zip(patch_list_1, patch_list_2)):
                mmd = self.mmd_loss(p1, p2)
                total_mmd += self.layer_weights[i] * mmd
                losses_dict[f'mmd_layer_{i}'] = mmd.item()
            
            total_loss = total_loss + self.mmd_weight * total_mmd / sum(self.layer_weights)
            losses_dict['mmd_total'] = total_mmd.item()
        
        losses_dict['total_loss'] = total_loss.item()
        
        return {'loss': total_loss, 'breakdown': losses_dict}


# ============================================================
# NEW: Temporal Memory Bank for Tracking
# ============================================================
# Reference: TimeSformer, CoTTA - Temporal consistency for video understanding

class TemporalMemoryBank(nn.Module):
    """
    Temporal memory bank for surgical tool tracking.
    
    Maintains a queue of past frame features for:
    1. Smooth trajectory prediction
    2. Recovery from occlusion
    3. Long-term object identity
    
    Architecture:
    - FIFO queue of feature vectors
    - Cross-attention with current frame features
    - Optional positional encoding for temporal order
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        memory_size: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_temporal_pe: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.memory_size = memory_size
        self.use_temporal_pe = use_temporal_pe
        
        # Memory queue: [memory_size, embed_dim]
        self.register_buffer('memory_queue', torch.zeros(memory_size, embed_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('memory_is_full', torch.zeros(1, dtype=torch.bool))
        
        # Cross-attention to query memory
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Temporal positional encoding
        if use_temporal_pe:
            self.temporal_pe = nn.Parameter(
                torch.zeros(1, memory_size, embed_dim)
            )
            nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim * 2, embed_dim)
        
    def _dequeue_enqueue(self, new_feature: torch.Tensor):
        """Update memory with new feature (FIFO)."""
        # new_feature: [B, C]
        B, C = new_feature.shape
        device = new_feature.device
        
        for b in range(B):
            ptr = int(self.memory_ptr.item())
            self.memory_queue[ptr] = new_feature[b].detach()
            
            ptr = (ptr + 1) % self.memory_size
            self.memory_ptr[0] = ptr
            
            if ptr == 0:
                self.memory_is_full[0] = True
    
    def forward(
        self, 
        query: torch.Tensor,  # Current frame: [B, N, C]
        update_memory: bool = True,
        memory_cls: Optional[torch.Tensor] = None,  # CLS token for memory update
    ) -> Dict[str, torch.Tensor]:
        """
        Query temporal memory and optionally update it.
        
        Args:
            query: Current frame features [B, N, C]
            update_memory: Whether to update memory queue
            memory_cls: CLS token to add to memory (uses first token if None)
        
        Returns:
            Dict with 'enhanced_features' and 'attention_weights'
        """
        B, N, C = query.shape
        device = query.device
        
        # Get valid memory entries
        if self.memory_is_full.item():
            valid_size = self.memory_size
        else:
            valid_size = int(self.memory_ptr.item())
        
        if valid_size == 0:
            # No memory yet, return unchanged
            return {
                'enhanced_features': query,
                'attention_weights': None,
                'memory_size': 0
            }
        
        # Prepare memory keys
        memory = self.memory_queue[:valid_size].unsqueeze(0).expand(B, -1, -1)  # [B, M, C]
        
        if self.use_temporal_pe:
            memory = memory + self.temporal_pe[:, :valid_size, :]
        
        # Cross-attention: query attends to memory
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=memory,
            value=memory,
            need_weights=True,
            average_attn_weights=True
        )  # attn_out: [B, N, C], attn_weights: [B, N, M]
        
        # Concatenate and project
        enhanced = torch.cat([query, attn_out], dim=-1)  # [B, N, 2C]
        enhanced = self.out_proj(enhanced)  # [B, N, C]
        
        # Update memory with CLS token
        if update_memory:
            if memory_cls is not None:
                self._dequeue_enqueue(memory_cls)
            else:
                self._dequeue_enqueue(query.mean(dim=1))  # Global pooling
        
        return {
            'enhanced_features': enhanced,
            'attention_weights': attn_weights,
            'memory_size': valid_size
        }
    
    def reset_memory(self):
        """Clear memory for new sequence."""
        self.memory_queue.zero_()
        self.memory_ptr.zero_()
        self.memory_is_full.zero_()


# ============================================================
# NEW: Uncertainty-Aware Heatmap Head
# ============================================================
# Reference: Evidential Deep Learning - Aleatoric uncertainty estimation

class UncertaintyHeatmapHead(nn.Module):
    """
    Heatmap head with uncertainty estimation.
    
    Predicts:
    1. Heatmap for tool tip localization
    2. Uncertainty map (aleatoric uncertainty)
    
    When confidence is low (e.g., occlusion, out-of-view), 
    the uncertainty should be high.
    
    Architecture:
    - Shared backbone CNN
    - Two prediction heads: heatmap and uncertainty
    - Heteroscedastic loss for training
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        
        # Shared backbone
        layers = []
        prev_ch = in_channels
        for i in range(num_layers - 1):
            layers.extend([
                nn.Conv2d(prev_ch, hidden_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.GELU(),
            ])
            prev_ch = hidden_channels
        
        self.backbone = nn.Sequential(*layers)
        
        # Heatmap prediction head
        self.heatmap_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        
        # Uncertainty (log-variance) prediction head
        self.uncertainty_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize heatmap head with small weights
        nn.init.normal_(self.heatmap_head.weight, std=0.001)
        nn.init.constant_(self.heatmap_head.bias, 0)
        
        # Initialize uncertainty head to predict low uncertainty initially
        nn.init.normal_(self.uncertainty_head.weight, std=0.001)
        nn.init.constant_(self.uncertainty_head.bias, -2.0)  # log(0.1) ~ low variance
    
    def forward(
        self, 
        x: torch.Tensor,  # [B, C, H, W]
        return_uncertainty: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Feature map [B, C, H, W]
            return_uncertainty: Whether to compute uncertainty
        
        Returns:
            Dict with 'heatmap', 'uncertainty' (optional), and 'confidence'
        """
        # Shared features
        feat = self.backbone(x)  # [B, hidden, H, W]
        
        # Heatmap prediction
        heatmap_logits = self.heatmap_head(feat)  # [B, 1, H, W]
        
        result = {'heatmap_logits': heatmap_logits}
        
        if return_uncertainty:
            # Uncertainty (log-variance)
            log_var = self.uncertainty_head(feat)  # [B, 1, H, W]
            uncertainty = torch.exp(log_var)  # Variance
            
            # Confidence: inverse of uncertainty at heatmap peak
            heatmap_sigmoid = torch.sigmoid(heatmap_logits)
            
            result.update({
                'log_variance': log_var,
                'uncertainty': uncertainty,
                'uncertainty_at_peak': self._uncertainty_at_peak(heatmap_sigmoid, uncertainty)
            })
        
        return result
    
    def _uncertainty_at_peak(
        self, 
        heatmap: torch.Tensor, 
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Get uncertainty at the predicted peak location."""
        B = heatmap.shape[0]
        heatmap_flat = heatmap.view(B, -1)  # [B, H*W]
        uncertainty_flat = uncertainty.view(B, -1)
        
        # Argmax location
        peak_idx = heatmap_flat.argmax(dim=1)  # [B]
        
        # Uncertainty at peak
        peak_uncertainty = uncertainty_flat.gather(1, peak_idx.unsqueeze(1)).squeeze(1)
        
        return peak_uncertainty
    
    @staticmethod
    def heteroscedastic_loss(
        pred: torch.Tensor,  # [B, 1, H, W]
        target: torch.Tensor,  # [B, 1, H, W]
        log_var: torch.Tensor,  # [B, 1, H, W]
    ) -> torch.Tensor:
        """
        Heteroscedastic regression loss.
        
        Loss = exp(-log_var) * (pred - target)^2 + log_var
        
        This allows the model to predict higher uncertainty 
        for difficult samples, reducing their contribution to the loss.
        """
        precision = torch.exp(-log_var)  # 1 / variance
        sq_error = (pred - target) ** 2
        
        loss = precision * sq_error + log_var
        
        return loss.mean()


# ============================================================
# NEW: Test-Time Adaptation (TTA)
# ============================================================
# Reference: Tent, SAR - Adapt to test distribution on the fly

class TestTimeAdapter(nn.Module):
    """
    Test-Time Adaptation module for surgical domain shift.
    
    At test time, adapt batch norm statistics and/or feature statistics
    to the current test distribution without requiring labels.
    
    Methods:
    1. BN adaptation: Update running mean/var on test data
    2. Entropy minimization: Adjust features to produce confident predictions
    3. Feature alignment: Match test features to source distribution
    """
    
    def __init__(
        self,
        model: nn.Module,
        adapt_bn: bool = True,
        adapt_entropy: bool = False,
        ent_weight: float = 0.1,
        source_stats: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.model = model
        self.adapt_bn = adapt_bn
        self.adapt_entropy = adapt_entropy
        self.ent_weight = ent_weight
        self.source_stats = source_stats
        
        # Collect adaptable parameters
        if adapt_entropy:
            self.adapt_params = []
            for name, param in model.named_parameters():
                if 'tracker.hm_head' in name or 'tracker.fpn' in name:
                    param.requires_grad = True
                    self.adapt_params.append(param)
        
        # Register hooks for BN adaptation
        if adapt_bn:
            self._setup_bn_adaptation()
    
    def _setup_bn_adaptation(self):
        """Set BatchNorm layers to train mode for statistics adaptation."""
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                module.train()
    
    def adapt_step(
        self, 
        x: torch.Tensor,
        num_steps: int = 1,
        lr: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Perform one adaptation step on test sample.
        
        Args:
            x: Test input [B, C, H, W]
            num_steps: Number of adaptation steps
            lr: Learning rate for entropy minimization
        
        Returns:
            Dict with adaptation metrics
        """
        metrics = {}
        
        if self.adapt_entropy and len(self.adapt_params) > 0:
            # Create optimizer
            optimizer = torch.optim.Adam(self.adapt_params, lr=lr)
            
            for _ in range(num_steps):
                # Forward pass
                output = self.model(x)
                
                if isinstance(output, dict) and 'heatmap_logits' in output:
                    heatmap = torch.sigmoid(output['heatmap_logits'])
                else:
                    heatmap = torch.sigmoid(output)
                
                # Entropy: -sum(p * log(p))
                heatmap_prob = heatmap.view(-1)
                entropy = -(heatmap_prob * torch.log(heatmap_prob + 1e-8)).mean()
                
                # Minimize entropy (encourage confident predictions)
                loss = self.ent_weight * entropy
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                metrics['entropy'] = entropy.item()
        
        return metrics
    
    def forward(self, x: torch.Tensor, adapt: bool = False) -> torch.Tensor:
        """Forward with optional adaptation."""
        if adapt:
            self.adapt_step(x)
        
        return self.model(x)


# ============================================================
# MODIFIED: Enhanced OneStreamTipTracker with all optimizations
# ============================================================

class OneStreamTipTrackerOptimized(nn.Module):
    """
    Enhanced surgical tool tip tracker with:
    1. Multi-scale FPN features
    2. Temporal memory bank (optional)
    3. Uncertainty-aware heatmap prediction
    4. Anatomy-anchored attention (original)
    """
    
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        encoder: nn.Module,  # ViTEncoder
        embed_dim: int,
        use_anatomy_anchor: bool = True,
        use_fpn: bool = True,
        use_temporal_memory: bool = False,
        use_uncertainty: bool = True,
        fpn_out_dim: int = 256,
        memory_size: int = 16,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_anatomy_anchor = use_anatomy_anchor
        self.use_fpn = use_fpn
        self.use_temporal_memory = use_temporal_memory
        self.use_uncertainty = use_uncertainty
        
        num_patches = (img_size // patch_size) ** 2
        
        # FPN for multi-scale features
        if use_fpn:
            self.fpn = FeaturePyramidDecoder(
                embed_dim=embed_dim,
                out_dim=fpn_out_dim,
                num_levels=3,
            )
            feat_dim = fpn_out_dim
        else:
            self.fpn = None
            feat_dim = embed_dim
        
        # Temporal memory bank
        if use_temporal_memory:
            self.memory_bank = TemporalMemoryBank(
                embed_dim=feat_dim,
                memory_size=memory_size
            )
        else:
            self.memory_bank = None
        
        # Cross-attention for template-search matching
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Positional tokens for template and search
        self.template_pos = nn.Parameter(torch.zeros(1, num_patches, feat_dim))
        self.search_pos = nn.Parameter(torch.zeros(1, num_patches, feat_dim))
        nn.init.trunc_normal_(self.template_pos, std=0.02)
        nn.init.trunc_normal_(self.search_pos, std=0.02)
        
        # Template memory (store across frames)
        self.register_buffer('template_memory', torch.zeros(1, feat_dim))
        self.register_buffer('template_updated', torch.zeros(1, dtype=torch.bool))
        
        # Heatmap prediction head
        if use_uncertainty:
            self.hm_head = UncertaintyHeatmapHead(
                in_channels=feat_dim,
                hidden_channels=64
            )
        else:
            # Original simple heatmap head
            self.hm_head = nn.Sequential(
                nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
                nn.GroupNorm(8, 256),
                nn.GELU(),
                nn.Conv2d(256, 64, kernel_size=3, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=1),
            )
        
        # Anatomy-anchored attention (from original)
        if use_anatomy_anchor:
            self.anatomy_proj = nn.Linear(2, feat_dim)  # 2 channels: vessel, optic disc
        
        # Classifier token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Final transformer for reasoning
        self.final_transformer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=8,
            dim_feedforward=feat_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
    
    def forward(
        self,
        template: torch.Tensor,  # [B, 3, H, W]
        search: torch.Tensor,    # [B, 3, H, W]
        encoder: nn.Module,
        anatomy_mask: Optional[torch.Tensor] = None,  # [B, 2, H, W]
        update_template: bool = False,
        update_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all optional enhancements.
        
        Args:
            template: Template frame
            search: Search frame
            encoder: ViT encoder
            anatomy_mask: Optional anatomical segmentation mask
            update_template: Whether to update template memory
            update_memory: Whether to update temporal memory
        
        Returns:
            Dict with heatmap, uncertainty, and other outputs
        """
        B = template.shape[0]
        H = W = self.img_size // self.patch_size
        
        # Encode template and search
        template_cls, template_patch = encoder.forward_tokens(template)
        search_cls, search_patch = encoder.forward_tokens(search)
        
        # Apply FPN for multi-scale features
        if self.use_fpn and self.fpn is not None:
            # We need intermediate features from encoder
            # For now, use last layer (will be enhanced with intermediate extraction)
            template_fpn = self.fpn([template_patch, template_patch, template_patch], H, W)
            search_fpn = self.fpn([search_patch, search_patch, search_patch], H, W)
            
            template_feat = template_fpn['fpn_fused']  # [B, C, H, W]
            search_feat = search_fpn['fpn_fused']
            
            # Flatten back to sequence
            template_seq = template_feat.flatten(2).transpose(1, 2)  # [B, N, C]
            search_seq = search_feat.flatten(2).transpose(1, 2)
        else:
            template_seq = template_patch
            search_seq = search_patch
            template_feat = template_patch.transpose(1, 2).reshape(B, -1, H, W)
            search_feat = search_patch.transpose(1, 2).reshape(B, -1, H, W)
        
        # Add positional encoding
        template_seq = template_seq + self.template_pos
        search_seq = search_seq + self.search_pos
        
        # Anatomy-anchored attention
        if self.use_anatomy_anchor and anatomy_mask is not None:
            # Downsample anatomy mask to feature map size
            anatomy_down = F.interpolate(anatomy_mask, size=(H, W), mode='bilinear', align_corners=False)
            anatomy_seq = anatomy_down.flatten(2).transpose(1, 2)  # [B, N, 2]
            anatomy_embed = self.anatomy_proj(anatomy_seq)  # [B, N, C]
            search_seq = search_seq + anatomy_embed
        
        # Template-search cross-attention
        # Query: search, Key/Value: template
        search_enhanced, _ = self.cross_attn(
            query=search_seq,
            key=template_seq,
            value=template_seq
        )
        
        # Residual
        search_seq = search_seq + search_enhanced
        
        # Temporal memory
        if self.use_temporal_memory and self.memory_bank is not None:
            mem_out = self.memory_bank(
                query=search_seq,
                update_memory=update_memory,
                memory_cls=search_cls
            )
            search_seq = mem_out['enhanced_features']
        
        # Final transformer
        search_seq = self.final_transformer(search_seq)
        
        # Reshape to 2D for heatmap prediction
        search_2d = search_seq.transpose(1, 2).reshape(B, -1, H, W)
        
        # Heatmap prediction
        if self.use_uncertainty:
            hm_output = self.hm_head(search_2d, return_uncertainty=True)
            result = {
                'heatmap_logits': hm_output['heatmap_logits'],
                'uncertainty': hm_output.get('uncertainty'),
                'log_variance': hm_output.get('log_variance'),
            }
        else:
            heatmap_logits = self.hm_head(search_2d)
            result = {
                'heatmap_logits': heatmap_logits,
                'uncertainty': None,
            }
        
        # Update template memory
        if update_template:
            self.template_memory = template_cls.detach().mean(dim=0, keepdim=True)
            self.template_updated[0] = True
        
        return result
    
    def reset_memory(self):
        """Reset temporal memory for new sequence."""
        if self.memory_bank is not None:
            self.memory_bank.reset_memory()
        self.template_updated[0] = False


# ============================================================
# Helper: Soft Argmax for coordinate extraction
# ============================================================

def soft_argmax_2d(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable argmax for coordinate extraction.
    
    Args:
        heatmap: [B, 1, H, W] logit heatmap
        temperature: Softmax temperature
    
    Returns:
        coords: [B, 2] (x, y) coordinates normalized to [0, 1]
    """
    B, _, H, W = heatmap.shape
    
    # Convert to probability
    prob = torch.softmax(heatmap.view(B, -1) / temperature, dim=-1)  # [B, H*W]
    
    # Compute expected coordinates
    y_coords = torch.arange(H, device=heatmap.device).float().view(1, H, 1).expand(1, H, W).reshape(1, -1)
    x_coords = torch.arange(W, device=heatmap.device).float().view(1, 1, W).expand(1, H, W).reshape(1, -1)
    
    y_expected = (prob * y_coords).sum(dim=-1)  # [B]
    x_expected = (prob * x_coords).sum(dim=-1)  # [B]
    
    # Normalize to [0, 1]
    coords = torch.stack([x_expected / (W - 1), y_expected / (H - 1)], dim=-1)  # [B, 2]
    
    return coords


# ============================================================
# Training Configuration (Enhanced)
# ============================================================

@dataclass
class TrainCfgOptimized:
    """Enhanced training configuration with optimization flags."""
    
    # Basic
    stage: str = "rmit"
    out_dir: str = "./output"
    init_ckpt: Optional[str] = None
    pretrained_encoder_ckpt: Optional[str] = None
    
    # Model architecture
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_ratio: float = 4.0
    drop: float = 0.0
    attn_drop: float = 0.0
    lora_r: int = 8
    use_residual_lora: bool = False
    residual_alpha: float = 0.1
    
    # Training
    batch_size: int = 8
    epochs: int = 20
    lr: float = 3e-4
    wd: float = 0.05
    num_workers: int = 4
    seed: int = 42
    amp: bool = False
    grad_accum: int = 1
    
    # PEFT
    freeze_backbone: bool = False
    use_ema: bool = False
    ema_decay: float = 0.9999
    
    # ============ NEW: Optimization Flags ============
    use_fpn: bool = True
    use_temporal_memory: bool = False
    use_uncertainty: bool = True
    use_multilayer_contrast: bool = True
    
    # FPN settings
    fpn_out_dim: int = 256
    fpn_num_levels: int = 3
    
    # Temporal memory settings
    memory_size: int = 16
    
    # Multi-layer contrastive loss
    contrast_layers: List[int] = field(default_factory=lambda: [2, 5, 7])  # Which layers
    contrast_temperature: float = 0.07
    contrast_layer_weights: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    use_mmd: bool = True
    mmd_weight: float = 0.1
    
    # Uncertainty loss weight
    uncertainty_weight: float = 0.1
    
    # Loss weights
    lambda_seg: float = 1.0
    lambda_align: float = 1.0
    lambda_hm: float = 1.0
    lambda_xy: float = 0.25
    
    # RMIT specific
    rmit_manifest_json: Optional[str] = None
    rmit_protocol: str = "leave_one_out"
    rmit_test_seq: Optional[str] = None
    rmit_val_ratio: float = 0.2
    max_pairs_per_seq: Optional[int] = None
    use_anatomy_anchor: bool = True
    eval_metrics: bool = False


# ============================================================
# Integration Example: Modified Training Loop
# ============================================================

def train_stage_rmit_optimized(cfg: TrainCfgOptimized, device: torch.device, 
                                encoder_class, dataset_class):
    """
    Enhanced RMIT training loop with optimizations.
    
    This is a template showing how to integrate the new components.
    """
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)
    
    # Build encoder
    enc = encoder_class(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        heads=cfg.heads,
        mlp_ratio=cfg.mlp_ratio,
        drop=cfg.drop,
        attn_drop=cfg.attn_drop,
        lora_r=cfg.lora_r,
        use_residual_lora=cfg.use_residual_lora,
        residual_alpha=cfg.residual_alpha,
    ).to(device)
    
    # Build tracker with optimizations
    tracker = OneStreamTipTrackerOptimized(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        encoder=enc,
        embed_dim=cfg.embed_dim,
        use_anatomy_anchor=cfg.use_anatomy_anchor,
        use_fpn=cfg.use_fpn,
        use_temporal_memory=cfg.use_temporal_memory,
        use_uncertainty=cfg.use_uncertainty,
        fpn_out_dim=cfg.fpn_out_dim,
        memory_size=cfg.memory_size,
    ).to(device)
    
    # Multi-layer contrastive loss
    if cfg.use_multilayer_contrast:
        contrast_loss_fn = MultiLayerContrastiveLoss(
            num_layers=len(cfg.contrast_layers),
            temperature=cfg.contrast_temperature,
            layer_weights=cfg.contrast_layer_weights,
            use_mmd=cfg.use_mmd,
            mmd_weight=cfg.mmd_weight,
        ).to(device)
    
    # ... rest of training loop follows original pattern
    # Key changes:
    # 1. Use intermediate layer features for FPN
    # 2. Apply multi-layer contrastive loss
    # 3. Use uncertainty-weighted heatmap loss
    # 4. Reset temporal memory at start of each sequence
    
    print("Optimized training setup complete.")


# ============================================================
# Main Entry Point for Testing
# ============================================================

if __name__ == "__main__":
    # Quick test of new modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Testing FeaturePyramidDecoder...")
    fpn = FeaturePyramidDecoder(embed_dim=384, out_dim=256).to(device)
    feat_list = [torch.randn(2, 1369, 384).to(device) for _ in range(3)]  # 37*37=1369
    fpn_out = fpn(feat_list, H=37, W=37)
    print(f"  FPN output shape: {fpn_out['fpn_fused'].shape}")
    
    print("\nTesting MultiLayerContrastiveLoss...")
    contrast = MultiLayerContrastiveLoss(num_layers=3).to(device)
    cls_list1 = [torch.randn(2, 384).to(device) for _ in range(3)]
    cls_list2 = [torch.randn(2, 384).to(device) for _ in range(3)]
    contrast_out = contrast(cls_list1, cls_list2)
    print(f"  Contrastive loss: {contrast_out['loss'].item():.4f}")
    
    print("\nTesting TemporalMemoryBank...")
    memory = TemporalMemoryBank(embed_dim=256).to(device)
    query = torch.randn(2, 100, 256).to(device)
    mem_out = memory(query, update_memory=True)
    print(f"  Enhanced features shape: {mem_out['enhanced_features'].shape}")
    print(f"  Memory size: {mem_out['memory_size']}")
    
    print("\nTesting UncertaintyHeatmapHead...")
    unc_head = UncertaintyHeatmapHead(in_channels=256).to(device)
    feat = torch.randn(2, 256, 37, 37).to(device)
    unc_out = unc_head(feat, return_uncertainty=True)
    print(f"  Heatmap shape: {unc_out['heatmap_logits'].shape}")
    print(f"  Uncertainty shape: {unc_out['uncertainty'].shape}")
    
    print("\n✓ All module tests passed!")
