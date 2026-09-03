#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DR-SurgBridgeNet (Full): DR-aware pretraining -> FOVEA paired bridging -> RMIT vitreoretinal tip tracking

This script is designed to be *runnable end-to-end* with publicly available datasets/manifests:
  - DR: CSV with columns [image_path,label]
  - FOVEA: JSON list of paired samples with vessel/OD masks (pre-op + intra-op)
  - RMIT: manifest JSON with sequences (frames_dir + ann_csv tip coords)

Key design choices (research-friendly and reproducible):
  - ViT encoder with Domain-Specific LoRA (PEFT); optional residual/cumulative LoRA (innovation).
  - Stage-2: segmentation (vessel/OD) + paired pre-op/intra-op contrastive alignment (FOVEA).
  - Stage-3: one-stream tracking (template+search) + optional anatomy-anchored attention using
            *predicted* vessel/OD masks from Stage-2 segmentation head.
  - Correct tracking augmentation with synchronized GT coordinate transforms (CRITICAL FIX).
  - TensorBoard logging, gradient accumulation, AMP, EMA, checkpointing.

Usage examples:
  # Stage-1: DR grading
  python dr_surgbridgenet_full.py --stage dr \
    --dr_train_csv data/dr/train.csv --dr_val_csv data/dr/val.csv \
    --out_dir runs/dr --epochs 20 --img_size 512 --patch_size 16 --amp

  # Stage-2: FOVEA bridging (use init from stage-1)
  python dr_surgbridgenet_full.py --stage fovea \
    --fovea_pairs_json data/fovea/fovea_pairs.json \
    --init_ckpt runs/dr/best.pt --freeze_backbone --use_ema \
    --out_dir runs/fovea --epochs 50 --img_size 512 --patch_size 16 --amp

  # Stage-3: RMIT tracking (use init from stage-2)
  python dr_surgbridgenet_full.py --stage rmit \
    --rmit_manifest_json data/rmit/rmit_manifest.json \
    --init_ckpt runs/fovea/best.pt --freeze_backbone --use_ema \
    --use_anatomy_anchor --out_dir runs/rmit --epochs 80 --img_size 256 --patch_size 16 --amp --eval_metrics

Notes:
  - This is a research codebase: it expects you to prepare manifests pointing to your local files.
  - Pretrained weight loading is supported via --pretrained_encoder_ckpt, but only if architectures match.
"""

import os
import math
import json
import time
import copy
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from torchvision.datasets import ImageFolder
except Exception:
    ImageFolder = None


import sys

# ============================================================
# Utils
# ============================================================
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# Losses
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance with hard negative mining.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Weight per class (tensor of shape [num_classes])
        gamma: Focusing parameter (default 2.0). Higher gamma = more focus on hard examples.
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute cross-entropy per sample (no reduction)
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        # Compute pt = probability of correct class
        # For numerical stability, use softmax probs directly
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def append_epoch_log(out_dir: str, msg: str) -> None:
    with open(os.path.join(out_dir, "epoch.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def pil_resize(img: Image.Image, size: int, is_mask: bool = False) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((size, size), resample=resample)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def normalize_img(x: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype)[:, None, None]
    std_t = torch.tensor(std, dtype=x.dtype)[:, None, None]
    return (x - mean_t) / (std_t + 1e-6)


def binarize_mask(m: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    # m: [1,H,W] in [0,1]
    return (m > thr).float()


def dice_loss_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2 * (probs * target).sum(dim=(2, 3))
    den = (probs + target).sum(dim=(2, 3)) + eps
    dice = 1 - (num + eps) / den
    return dice.mean()


def bce_loss_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss for addressing class imbalance.
    logits: [B, C]
    target: [B]
    """
    ce_loss = F.cross_entropy(logits, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()


def soft_argmax_2d(heatmap: torch.Tensor) -> torch.Tensor:
    """
    heatmap: [B,1,H,W] logits
    returns [B,2] coords (x,y) in pixel units
    """
    B, _, H, W = heatmap.shape
    hm = heatmap.view(B, -1)
    hm = F.softmax(hm, dim=-1)
    ys = torch.linspace(0, H - 1, H, device=heatmap.device)
    xs = torch.linspace(0, W - 1, W, device=heatmap.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # [HW,2]
    coords = (hm.unsqueeze(-1) * grid.unsqueeze(0)).sum(dim=1)
    return coords


def make_gaussian_heatmap(H: int, W: int, cx: float, cy: float, sigma: float = 3.0, device: str = "cpu") -> torch.Tensor:
    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    g = g / (g.max() + 1e-6)
    return g[None, None, :, :]  # [1,1,H,W]


# ============================================================
# Enhanced Augmentations (tracking augmentation returns metadata)
# ============================================================
class EnhancedAugmentation:
    @staticmethod
    def augment_dr(img: Image.Image, is_training: bool = True) -> Image.Image:
        if not is_training:
            return img
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.2:
            angle = random.uniform(-8, 8)
            img = img.rotate(angle, resample=Image.BILINEAR)
        if random.random() < 0.5:
            enh = ImageEnhance.Brightness(img)
            img = enh.enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.5:
            enh = ImageEnhance.Contrast(img)
            img = enh.enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        return img

    @staticmethod
    def augment_fovea_pair(
        pre: Image.Image, intra: Image.Image,
        pre_v: Image.Image, pre_od: Image.Image,
        intra_v: Image.Image, intra_od: Image.Image,
        is_training: bool = True
    ) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
        if not is_training:
            return pre, intra, pre_v, pre_od, intra_v, intra_od

        if random.random() < 0.5:
            pre = pre.transpose(Image.FLIP_LEFT_RIGHT)
            intra = intra.transpose(Image.FLIP_LEFT_RIGHT)
            pre_v = pre_v.transpose(Image.FLIP_LEFT_RIGHT)
            pre_od = pre_od.transpose(Image.FLIP_LEFT_RIGHT)
            intra_v = intra_v.transpose(Image.FLIP_LEFT_RIGHT)
            intra_od = intra_od.transpose(Image.FLIP_LEFT_RIGHT)

        # pre-op domain style
        if random.random() < 0.4:
            enh = ImageEnhance.Brightness(pre)
            pre = enh.enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.3:
            pre = pre.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

        # intra-op domain style
        if random.random() < 0.4:
            enh = ImageEnhance.Contrast(intra)
            intra = enh.enhance(random.uniform(0.9, 1.3))
        if random.random() < 0.3:
            enh = ImageEnhance.Brightness(intra)
            intra = enh.enhance(random.uniform(0.95, 1.15))
        return pre, intra, pre_v, pre_od, intra_v, intra_od

    @staticmethod
    def augment_tracking_pair(
        template: Image.Image, search: Image.Image,
        is_training: bool = True
    ) -> Tuple[Image.Image, Image.Image, bool, float]:
        """
        Temporal-consistent augmentations for tracking; returns (template, search, flip, angle_deg).
        CRITICAL: We return metadata so GT coords can be transformed consistently.
        """
        flip = False
        angle = 0.0
        if not is_training:
            return template, search, flip, angle

        if random.random() < 0.5:
            flip = True
            template = template.transpose(Image.FLIP_LEFT_RIGHT)
            search = search.transpose(Image.FLIP_LEFT_RIGHT)

        # Optional rotation (small); if you don't want to handle coord rotation, set prob to 0
        if random.random() < 0.2:
            angle = random.uniform(-5.0, 5.0)
            template = template.rotate(angle, resample=Image.BILINEAR)
            search = search.rotate(angle, resample=Image.BILINEAR)

        # shared brightness
        if random.random() < 0.3:
            factor = random.uniform(0.95, 1.05)
            template = ImageEnhance.Brightness(template).enhance(factor)
            search = ImageEnhance.Brightness(search).enhance(factor)

        return template, search, flip, angle


def _apply_flip_rot_xy(x: float, y: float, W: int, H: int, flip: bool, angle_deg: float) -> Tuple[float, float]:
    # Flip horizontally
    if flip:
        x = (W - 1) - x

    # Rotate around center (PIL rotates CCW around image center)
    if abs(angle_deg) > 1e-6:
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x0, y0 = x - cx, y - cy
        xr = cos_t * x0 - sin_t * y0
        yr = sin_t * x0 + cos_t * y0
        x, y = xr + cx, yr + cy
    return x, y


# ============================================================
# EMA
# ============================================================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.module = copy.deepcopy(model)
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad = False

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()
            for k, v in esd.items():
                if k in msd and torch.is_floating_point(v):
                    v.mul_(self.decay).add_(msd[k], alpha=(1.0 - self.decay))
    def state_dict(self) -> Dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict) -> None:
        self.module.load_state_dict(state_dict, strict=False)


# ============================================================
# Datasets
# ============================================================
class DRDataset(Dataset):
    """CSV: image_path,label (int)."""

    def __init__(self, csv_path: str, img_size: int, augment: bool = True):
        self.items = []
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            idx_img = header.index("image_path")
            idx_lab = header.index("label")
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < max(idx_img, idx_lab) + 1:
                    continue
                img_path, label = parts[idx_img], int(parts[idx_lab])
                if os.path.exists(img_path):
                    self.items.append((img_path, label))
        self.img_size = img_size
        self.augment = augment
        print(f"[DRDataset] {csv_path}: {len(self.items)} samples")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = pil_resize(load_image_rgb(path), self.img_size, False)
        img = EnhancedAugmentation.augment_dr(img, is_training=self.augment)
        x = normalize_img(pil_to_tensor(img))
        return x, torch.tensor(y, dtype=torch.long)


class DRImageFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        img_size: int,
        indices: Optional[List[int]] = None,
        augment: bool = True,
        label_map: Optional[Dict[str, int]] = None,
    ):
        if ImageFolder is None:
            raise ImportError("torchvision is required for --dr_imagefolder_dir")
        self.base = ImageFolder(root=root_dir)
        self.img_size = img_size
        self.augment = augment
        self.indices = list(indices) if indices is not None else list(range(len(self.base.samples)))
        self.label_map = dict(label_map) if label_map is not None else {
            class_name: idx for class_name, idx in self.base.class_to_idx.items()
        }
        self.items = []
        for sample_idx in self.indices:
            path, original_label = self.base.samples[sample_idx]
            class_name = self.base.classes[original_label]
            label = self.label_map[class_name]
            self.items.append((path, label))
        print(f"[DRImageFolderDataset] {root_dir}: {len(self.items)} samples augment={self.augment}")
        print(f"[DRImageFolderDataset] label_map={self.label_map}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = pil_resize(load_image_rgb(path), self.img_size, False)
        img = EnhancedAugmentation.augment_dr(img, is_training=self.augment)
        x = normalize_img(pil_to_tensor(img))
        return x, torch.tensor(y, dtype=torch.long)


def build_dr_datasets(cfg: "TrainCfg") -> Tuple[Dataset, Dataset]:
    if cfg.dr_imagefolder_dir:
        if not os.path.isdir(cfg.dr_imagefolder_dir):
            raise ValueError(f"dr_imagefolder_dir not found: {cfg.dr_imagefolder_dir}")
        if ImageFolder is None:
            raise ImportError("torchvision is required for --dr_imagefolder_dir")

        base = ImageFolder(root=cfg.dr_imagefolder_dir)
        if len(base.samples) == 0:
            raise ValueError(f"Empty ImageFolder dataset: {cfg.dr_imagefolder_dir}")

        if set(base.classes) == {"No_DR", "DR"}:
            label_map = {"No_DR": 0, "DR": 1}
        else:
            label_map = {class_name: idx for idx, class_name in enumerate(sorted(base.classes))}

        by_label: Dict[int, List[int]] = {}
        for sample_idx, (_path, original_label) in enumerate(base.samples):
            class_name = base.classes[original_label]
            mapped_label = label_map[class_name]
            by_label.setdefault(mapped_label, []).append(sample_idx)

        train_indices, val_indices = [], []
        for mapped_label, sample_indices in sorted(by_label.items()):
            sample_indices = sample_indices.copy()
            random.shuffle(sample_indices)
            if len(sample_indices) == 1:
                train_split = sample_indices
                val_split = sample_indices
            else:
                n_val = max(1, int(round(len(sample_indices) * cfg.dr_val_ratio)))
                n_val = min(n_val, len(sample_indices) - 1)
                val_split = sample_indices[:n_val]
                train_split = sample_indices[n_val:]
            train_indices.extend(train_split)
            val_indices.extend(val_split)

        if len(train_indices) == 0 or len(val_indices) == 0:
            raise ValueError(f"Invalid ImageFolder split: train={len(train_indices)} val={len(val_indices)}")

        print(f"[DR] Using ImageFolder from {cfg.dr_imagefolder_dir}")
        print(f"[DR] Classes discovered: {base.classes}")
        print(f"[DR] Stratified split with val_ratio={cfg.dr_val_ratio}: train={len(train_indices)} val={len(val_indices)}")
        train_ds = DRImageFolderDataset(cfg.dr_imagefolder_dir, cfg.img_size, indices=train_indices, augment=True, label_map=label_map)
        val_ds = DRImageFolderDataset(cfg.dr_imagefolder_dir, cfg.img_size, indices=val_indices, augment=False, label_map=label_map)
        return train_ds, val_ds

    if not cfg.dr_train_csv or not cfg.dr_val_csv:
        raise ValueError("Need either --dr_imagefolder_dir or both --dr_train_csv and --dr_val_csv")
    train_ds = DRDataset(cfg.dr_train_csv, cfg.img_size, augment=True)
    val_ds = DRDataset(cfg.dr_val_csv, cfg.img_size, augment=False)
    return train_ds, val_ds


class FOVEAPairedDataset(Dataset):
    """JSON list of paired samples with patient_id and masks."""

    def __init__(self, pairs: List[dict], img_size: int, augment: bool = True):
        self.pairs = pairs
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        it = self.pairs[idx]
        pre = pil_resize(load_image_rgb(it["pre_img"]), self.img_size, False)
        intra = pil_resize(load_image_rgb(it["intra_img"]), self.img_size, False)
        pre_v = pil_resize(load_mask(it["pre_vessel"]), self.img_size, True)
        pre_od = pil_resize(load_mask(it["pre_od"]), self.img_size, True)
        intra_v = pil_resize(load_mask(it["intra_vessel"]), self.img_size, True)
        intra_od = pil_resize(load_mask(it["intra_od"]), self.img_size, True)

        pre, intra, pre_v, pre_od, intra_v, intra_od = EnhancedAugmentation.augment_fovea_pair(
            pre, intra, pre_v, pre_od, intra_v, intra_od, is_training=self.augment
        )

        pre_x = normalize_img(pil_to_tensor(pre))
        intra_x = normalize_img(pil_to_tensor(intra))

        pre_v_t = binarize_mask(pil_to_tensor(pre_v))
        pre_od_t = binarize_mask(pil_to_tensor(pre_od))
        intra_v_t = binarize_mask(pil_to_tensor(intra_v))
        intra_od_t = binarize_mask(pil_to_tensor(intra_od))

        pre_y = torch.cat([pre_v_t, pre_od_t], dim=0)      # [2,H,W]
        intra_y = torch.cat([intra_v_t, intra_od_t], dim=0)
        return {
            "patient_id": str(it["patient_id"]),
            "pre_x": pre_x, "pre_y": pre_y,
            "intra_x": intra_x, "intra_y": intra_y,
        }


class RMITTrackingDataset(Dataset):
    """
    Frame-pair tracking dataset: prev as template, current as search.
    Includes CRITICAL FIX: apply same augmentation transforms to GT coordinates.
    """

    def __init__(
        self,
        manifest_json: str,
        img_size: int,
        is_training: bool = True,
        max_pairs_per_seq: Optional[int] = None,
        protocol: str = "leave_one_out",
        test_seq: Optional[str] = None,
        val_ratio: float = 0.2,
    ):
        with open(manifest_json, "r", encoding="utf-8") as f:
            man = json.load(f)

        self.img_size = img_size
        self.is_training = is_training
        self.samples = []  # tuples: (prev_path, cur_path, prev_xy, cur_xy, seq_name)

        all_by_seq = {}
        for seq in man["sequences"]:
            frames_dir = seq["frames_dir"]
            ann_csv = seq["ann_csv"]
            seq_name = seq.get("name", os.path.basename(frames_dir))

            # annotations
            ann = {}
            with open(ann_csv, "r", encoding="utf-8") as fcsv:
                header = fcsv.readline().strip().split(",")
                idx_f = header.index("frame")
                idx_x = header.index("x")
                idx_y = header.index("y")
                for line in fcsv:
                    p = line.strip().split(",")
                    if len(p) < max(idx_f, idx_x, idx_y) + 1:
                        continue
                    ann[p[idx_f]] = (float(p[idx_x]), float(p[idx_y]))

            frames = sorted([fn for fn in os.listdir(frames_dir) if fn.lower().endswith((".png", ".jpg", ".jpeg"))])
            pairs = []
            for i in range(1, len(frames)):
                f_prev, f_cur = frames[i - 1], frames[i]
                if f_prev in ann and f_cur in ann:
                    pairs.append((os.path.join(frames_dir, f_prev),
                                  os.path.join(frames_dir, f_cur),
                                  ann[f_prev], ann[f_cur],
                                  seq_name))
            if max_pairs_per_seq is not None and len(pairs) > max_pairs_per_seq:
                pairs = random.sample(pairs, max_pairs_per_seq)
            all_by_seq[seq_name] = pairs

        # protocol split
        seq_names = sorted(list(all_by_seq.keys()))
        if protocol == "leave_one_out" and len(seq_names) >= 2:
            if test_seq is None:
                test_seq = seq_names[0]
            if test_seq not in all_by_seq:
                raise ValueError(f"--rmit_test_seq {test_seq} not in sequences: {seq_names}")
            if is_training:
                use_seqs = [s for s in seq_names if s != test_seq]
            else:
                use_seqs = [test_seq]
            for s in use_seqs:
                self.samples.extend(all_by_seq[s])
        else:
            # random split on all pairs
            all_pairs = []
            for s in seq_names:
                all_pairs.extend(all_by_seq[s])
            random.shuffle(all_pairs)
            n_val = int(len(all_pairs) * val_ratio)
            if is_training:
                self.samples = all_pairs[n_val:]
            else:
                self.samples = all_pairs[:n_val]

        print(f"[RMITTrackingDataset] {'train' if is_training else 'val'} samples: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        prev_path, cur_path, prev_xy, cur_xy, _seq = self.samples[idx]
        prev_img = load_image_rgb(prev_path)
        cur_img = load_image_rgb(cur_path)

        W0, H0 = prev_img.size
        prev_img_r = pil_resize(prev_img, self.img_size, False)
        cur_img_r = pil_resize(cur_img, self.img_size, False)

        sx = self.img_size / float(W0)
        sy = self.img_size / float(H0)
        prev_x, prev_y = prev_xy[0] * sx, prev_xy[1] * sy
        cur_x, cur_y = cur_xy[0] * sx, cur_xy[1] * sy

        # Apply augmentation and synchronize coords
        prev_img_r, cur_img_r, flip, angle = EnhancedAugmentation.augment_tracking_pair(
            prev_img_r, cur_img_r, is_training=self.is_training
        )
        W = H = self.img_size
        prev_x, prev_y = _apply_flip_rot_xy(prev_x, prev_y, W, H, flip, angle)
        cur_x, cur_y = _apply_flip_rot_xy(cur_x, cur_y, W, H, flip, angle)

        template = normalize_img(pil_to_tensor(prev_img_r))
        search = normalize_img(pil_to_tensor(cur_img_r))
        gt_hm = make_gaussian_heatmap(self.img_size, self.img_size, cur_x, cur_y, sigma=4.0, device="cpu")[0]  # [1,H,W]

        return {
            "template": template,
            "search": search,
            "gt_xy": torch.tensor([cur_x, cur_y], dtype=torch.float32),
            "gt_hm": gt_hm,
        }


# ============================================================
# Position embedding interpolation (for pretrained loading)
# ============================================================
def resize_pos_embed(posemb: torch.Tensor, new_grid_size: int) -> torch.Tensor:
    """
    posemb: [1, 1+oldN, C]
    new_grid_size: int (newH == newW)
    """
    if posemb.ndim != 3:
        raise ValueError("posemb must be [1,1+N,C]")
    cls_tok = posemb[:, :1]
    grid_tok = posemb[:, 1:]
    oldN = grid_tok.shape[1]
    C = grid_tok.shape[2]
    old_grid = int(math.sqrt(oldN))
    if old_grid * old_grid != oldN:
        raise ValueError(f"Cannot sqrt old grid: oldN={oldN}")

    grid_2d = grid_tok.reshape(1, old_grid, old_grid, C).permute(0, 3, 1, 2)  # [1,C,H,W]
    grid_2d_new = F.interpolate(grid_2d, size=(new_grid_size, new_grid_size), mode="bicubic", align_corners=False)
    grid_new = grid_2d_new.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, C)
    return torch.cat([cls_tok, grid_new], dim=1)


# ============================================================
# Model: ViT encoder with Domain-Specific (Residual) LoRA
# ============================================================
class DomainLoRALinear(nn.Module):
    """
    Linear with domain-specific low-rank adapters.
    Supports optional cumulative/residual LoRA.
    """
    def __init__(self, in_features: int, out_features: int, r: int, domains: List[str],
                 bias: bool = True, use_residual: bool = True, residual_alpha: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.domains = list(domains)
        self.current_domain = self.domains[0]
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.A = nn.ParameterDict()
        self.B = nn.ParameterDict()
        for d in self.domains:
            self.A[d] = nn.Parameter(torch.zeros(r, in_features))
            self.B[d] = nn.Parameter(torch.zeros(out_features, r))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        for d in self.domains:
            nn.init.kaiming_uniform_(self.A[d], a=math.sqrt(5))
            nn.init.zeros_(self.B[d])

    def set_domain(self, domain: str) -> None:
        if domain not in self.domains:
            raise ValueError(f"Unknown domain '{domain}'. Available: {self.domains}")
        self.current_domain = domain

    def copy_domain_params(self, src_domain: str, dst_domain: str) -> None:
        """Copy LoRA params from src to dst to fix cold-start issues."""
        with torch.no_grad():
            self.A[dst_domain].copy_(self.A[src_domain])
            self.B[dst_domain].copy_(self.B[src_domain])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)

        if self.use_residual:
            # Accumulate LoRA from previous domains + current domain
            try:
                current_idx = self.domains.index(self.current_domain)
            except ValueError:
                current_idx = len(self.domains) - 1
            lora_sum = 0.0
            for i, d in enumerate(self.domains[:current_idx + 1]):
                A = self.A[d]
                B = self.B[d]
                w = 1.0 if i == current_idx else self.residual_alpha
                lora_sum = lora_sum + w * F.linear(F.linear(x, A, None), B, None)
        else:
            A = self.A[self.current_domain]
            B = self.B[self.current_domain]
            lora_sum = F.linear(F.linear(x, A, None), B, None)
        return base + lora_sum


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float, lora_r: int, domains: List[str], use_residual_lora: bool):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = DomainLoRALinear(dim, hidden, r=lora_r, domains=domains, use_residual=use_residual_lora)
        self.fc2 = DomainLoRALinear(hidden, dim, r=lora_r, domains=domains, use_residual=use_residual_lora)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def set_domain(self, domain: str) -> None:
        self.fc1.set_domain(domain)
        self.fc2.set_domain(domain)

    def copy_domain_params(self, src: str, dst: str) -> None:
        self.fc1.copy_domain_params(src, dst)
        self.fc2.copy_domain_params(src, dst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float, proj_drop: float,
                 lora_r: int, domains: List[str], use_residual_lora: bool):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = DomainLoRALinear(dim, dim * 3, r=lora_r, domains=domains, use_residual=use_residual_lora)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = DomainLoRALinear(dim, dim, r=lora_r, domains=domains, use_residual=use_residual_lora)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_domain(self, domain: str) -> None:
        self.qkv.set_domain(domain)
        self.proj.set_domain(domain)

    def copy_domain_params(self, src: str, dst: str) -> None:
        self.qkv.copy_domain_params(src, dst)
        self.proj.copy_domain_params(src, dst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, drop: float, attn_drop: float,
                 lora_r: int, domains: List[str], use_residual_lora: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, attn_drop, drop, lora_r, domains, use_residual_lora)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop, lora_r, domains, use_residual_lora)

    def set_domain(self, domain: str) -> None:
        self.attn.set_domain(domain)
        self.mlp.set_domain(domain)

    def copy_domain_params(self, src: str, dst: str) -> None:
        self.attn.copy_domain_params(src, dst)
        self.mlp.copy_domain_params(src, dst)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, depth: int, num_heads: int,
                 mlp_ratio: float, drop: float, attn_drop: float, lora_r: int, domains: List[str],
                 use_residual_lora: bool):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch.num_patches, embed_dim))
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop, lora_r, domains, use_residual_lora)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.domains = list(domains)
        self.current_domain = self.domains[0]

    def set_domain(self, domain: str) -> None:
        if domain not in self.domains:
            raise ValueError(f"Unknown domain '{domain}', available: {self.domains}")
        self.current_domain = domain
        for b in self.blocks:
            b.set_domain(domain)

    def copy_domain_params(self, src: str, dst: str) -> None:
        print(f"[ViT] Copying LoRA params: {src} -> {dst}")
        for b in self.blocks:
            b.copy_domain_params(src, dst)

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        t = self.patch(x)
        cls = self.cls.expand(B, -1, -1)
        t = torch.cat([cls, t], dim=1)
        t = t + self.pos[:, : t.size(1)]
        t = self.drop(t)
        for b in self.blocks:
            t = b(t)
        t = self.norm(t)
        cls_out = t[:, 0]
        patch_out = t[:, 1:]
        return cls_out, patch_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_out, _ = self.forward_tokens(x)
        return cls_out


class SegDecoder(nn.Module):
    """
    Patch-token segmentation decoder that works for ANY patch_size.
    It predicts logits on token grid and upsamples to img_size.
    """
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, out_ch: int = 2):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, out_ch, 1)
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B, N, C = patch_tokens.shape
        feat = patch_tokens.transpose(1, 2).reshape(B, C, self.grid, self.grid)
        logits_g = self.proj(feat)  # [B,out,g,g]
        logits = F.interpolate(logits_g, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return logits


class DRHead(nn.Module):
    """Enhanced classification head for DR staging with deeper MLP.

    Upgraded from single linear layer to deeper MLP to better capture complex
    DR features. Uses dropout for regularization and LayerNorm for stability.
    """

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        # Initialize weights for better gradient flow
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.mlp(cls_token)


class ContrastiveHead(nn.Module):
    def __init__(self, embed_dim: int, proj_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x)
        z = F.normalize(z, dim=-1)
        return z


class OneStreamTipTracker(nn.Module):
    """
    One-stream tracker with optional Anatomy-Anchored attention.
    """
    def __init__(self, img_size: int, patch_size: int, encoder: ViTEncoder, embed_dim: int, use_anatomy_anchor: bool = True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = encoder
        self.use_anatomy_anchor = use_anatomy_anchor

        grid = img_size // patch_size
        self.search_grid = grid
        self.num_search = grid * grid
        self.num_template = self.num_search

        self.pos_ts = nn.Parameter(torch.zeros(1, 1 + self.num_template + self.num_search, embed_dim))
        nn.init.trunc_normal_(self.pos_ts, std=0.02)

        self.norm = nn.LayerNorm(embed_dim)
        self.hm_head = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 1),
            nn.GELU(),
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1)
        )

        if use_anatomy_anchor:
            self.anatomy_attention = nn.Sequential(
                nn.Conv2d(embed_dim + 2, embed_dim, 1),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, 1)
            )

    def forward(self, template: torch.Tensor, search: torch.Tensor, anatomy_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.encoder.set_domain("surg")

        B = template.size(0)
        t_tok = self.encoder.patch(template)
        s_tok = self.encoder.patch(search)
        cls = self.encoder.cls.expand(B, -1, -1)
        x = torch.cat([cls, t_tok, s_tok], dim=1)
        x = x + self.pos_ts[:, : x.size(1)]
        x = self.encoder.drop(x)
        for b in self.encoder.blocks:
            x = b(x)
        x = self.norm(x)

        s = x[:, 1 + self.num_template: 1 + self.num_template + self.num_search]
        C = s.size(-1)
        feat = s.transpose(1, 2).reshape(B, C, self.search_grid, self.search_grid)

        if self.use_anatomy_anchor and anatomy_mask is not None:
            anatomy_resized = F.interpolate(anatomy_mask, size=(self.search_grid, self.search_grid), mode="bilinear", align_corners=False)
            feat = self.anatomy_attention(torch.cat([feat, anatomy_resized], dim=1))

        hm = self.hm_head(feat)
        hm = F.interpolate(hm, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return hm


# ============================================================
# Losses & Metrics
# ============================================================
def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    B = z1.size(0)
    logits = (z1 @ z2.t()) / temperature
    labels = torch.arange(B, device=z1.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss1 + loss2)


def compute_tracking_metrics(pred_coords: torch.Tensor, gt_coords: torch.Tensor, thresholds: List[int] = [5, 10, 20]) -> Dict[str, float]:
    distances = torch.sqrt(((pred_coords - gt_coords) ** 2).sum(dim=1))
    mse = distances.pow(2).mean().item()
    metrics = {"mse": mse, "mean_error": distances.mean().item(), "std_error": distances.std().item()}
    for thr in thresholds:
        metrics[f"precision@{thr}px"] = (distances < thr).float().mean().item()
    metrics["failure_rate"] = (distances > 20).float().mean().item()
    return metrics


# ============================================================
# PEFT utilities
# ============================================================
def get_lora_and_head_params(model: nn.Module, verbose: bool = False) -> List[nn.Parameter]:
    """
    Freeze backbone weights, return LoRA adapters and task heads (+ tracker.pos_ts).
    """
    params = []
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if "A." in name or "B." in name:
            param.requires_grad = True
            params.append(param)
            trainable += 1
            if verbose:
                print(f"  [LoRA] {name}")
        elif any(k in name for k in [
            "seg.", "con.", "head.", "tracker.hm_head", "tracker.norm", "tracker.anatomy_attention", "tracker.pos_ts"
        ]):
            param.requires_grad = True
            params.append(param)
            trainable += 1
            if verbose:
                print(f"  [Head] {name}")
        else:
            param.requires_grad = False
            frozen += 1
    print(f"[PEFT] Frozen params={frozen}, Trainable params={trainable}")
    return params


def verify_lora_frozen(model: nn.Module) -> bool:
    ok = True
    for name, p in model.named_parameters():
        if ("A." in name) or ("B." in name):
            if not p.requires_grad:
                print(f"❌ LoRA param frozen: {name}")
                ok = False
        # backbone should be frozen except heads
        if ("enc." in name or "encoder." in name) and p.requires_grad:
            # allow if it is LoRA A/B (handled above)
            if ("A." not in name) and ("B." not in name):
                print(f"❌ Backbone unexpectedly trainable: {name}")
                ok = False
    if ok:
        print("✅ PEFT freezing verification passed")
    return ok


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Pretrained weight loading
# ============================================================
def _strip_prefix(k: str) -> str:
    for pref in ["module.", "backbone.", "student.", "teacher.", "model.", "encoder."]:
        if k.startswith(pref):
            k = k[len(pref):]
    return k


def load_pretrained_encoder(encoder: ViTEncoder, ckpt_path: str) -> None:
    """
    Load external pretrained ViT weights into our encoder (best-effort).
    Works when architecture matches (depth/heads/embed_dim/patch_size).
    """
    print(f"[Pretrained] Loading encoder weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict):
        sd = ckpt
    else:
        raise ValueError("Unsupported checkpoint format")

    new_sd = {}
    for k, v in sd.items():
        k2 = _strip_prefix(k)

        # common renames
        k2 = k2.replace("patch_embed.proj.", "patch.proj.")
        k2 = k2.replace("cls_token", "cls")
        k2 = k2.replace("pos_embed", "pos")

        # Some checkpoints name mlp layers differently; keep best-effort
        new_sd[k2] = v

    # Handle pos embedding resize if needed
    if "pos" in new_sd:
        if new_sd["pos"].shape != encoder.pos.shape:
            g_new = encoder.patch.grid
            try:
                new_sd["pos"] = resize_pos_embed(new_sd["pos"], g_new)
                print(f"[Pretrained] Resized pos embed to grid {g_new}x{g_new}")
            except Exception as e:
                print(f"[Pretrained] WARNING: pos resize failed: {e}. Skipping pos.")
                new_sd.pop("pos", None)

    # Patch embedding mismatch: skip if shape mismatch
    if "patch.proj.weight" in new_sd and new_sd["patch.proj.weight"].shape != encoder.patch.proj.weight.shape:
        print("[Pretrained] WARNING: patch.proj.weight shape mismatch. Skipping patch embed weights.")
        new_sd.pop("patch.proj.weight", None)
    if "patch.proj.bias" in new_sd and encoder.patch.proj.bias is not None:
        if new_sd["patch.proj.bias"].shape != encoder.patch.proj.bias.shape:
            print("[Pretrained] WARNING: patch.proj.bias shape mismatch. Skipping patch embed bias.")
            new_sd.pop("patch.proj.bias", None)

    missing, unexpected = encoder.load_state_dict(new_sd, strict=False)
    print(f"[Pretrained] Loaded with missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) < 15:
        print(f"  missing keys (partial): {missing[:15]}")
    if len(unexpected) < 15:
        print(f"  unexpected keys (partial): {unexpected[:15]}")


# ============================================================
# Training helpers
# ============================================================
def save_ckpt(path: str, payload: dict) -> None:
    torch.save(payload, path)


def _infer_pos_resize_grid(model: nn.Module, key: str) -> Optional[int]:
    try:
        if key == "pos" and hasattr(model, "patch"):
            return int(model.patch.grid)
        if key.endswith(".pos"):
            prefix = key[:-len(".pos")]
            submodule = model.get_submodule(prefix) if prefix else model
            if hasattr(submodule, "patch"):
                return int(submodule.patch.grid)
    except Exception:
        return None
    return None


def _filter_compatible_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    target_state = model.state_dict()
    filtered_state = {}
    resized_keys = []
    skipped_keys = []

    for key, value in state_dict.items():
        if key not in target_state:
            continue

        target_value = target_state[key]
        if value.shape == target_value.shape:
            filtered_state[key] = value
            continue

        if value.ndim == 3 and (key == "pos" or key.endswith(".pos")):
            new_grid = _infer_pos_resize_grid(model, key)
            if new_grid is not None:
                try:
                    resized_value = resize_pos_embed(value, new_grid)
                    if resized_value.shape == target_value.shape:
                        filtered_state[key] = resized_value
                        resized_keys.append(f"{key}: {tuple(value.shape)} -> {tuple(resized_value.shape)}")
                        continue
                except Exception as e:
                    skipped_keys.append(f"{key}: resize failed ({e})")
                    continue

        skipped_keys.append(f"{key}: ckpt {tuple(value.shape)} != model {tuple(target_value.shape)}")

    return filtered_state, resized_keys, skipped_keys


def load_ckpt(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer] = None,
              scaler: Optional[torch.cuda.amp.GradScaler] = None, ema: Optional[ModelEMA] = None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    compatible_state, resized_keys, skipped_keys = _filter_compatible_state_dict(model, state_dict)
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    print(f"[CKPT] Loaded {len(compatible_state)} tensors from {path}")
    if resized_keys:
        print(f"[CKPT] Resized position embeddings ({len(resized_keys)}):")
        for item in resized_keys[:10]:
            print(f"  {item}")
    if skipped_keys:
        print(f"[CKPT] Skipped incompatible tensors ({len(skipped_keys)}):")
        for item in skipped_keys[:10]:
            print(f"  {item}")
    if missing:
        print(f"[CKPT] Missing keys after load: {len(missing)}")
    if unexpected:
        print(f"[CKPT] Unexpected keys after load: {len(unexpected)}")
    if optim is not None and "optim" in ckpt:
        try:
            optim.load_state_dict(ckpt["optim"])
        except Exception:
            pass
    if scaler is not None and "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            pass
    if ema is not None and "ema" in ckpt and ckpt["ema"] is not None:
        ema.load_state_dict(ckpt["ema"])
    return ckpt


def cosine_lr(base_lr: float, step: int, total_steps: int, min_lr_ratio: float = 0.05) -> float:
    t = step / max(1, total_steps)
    return min_lr_ratio * base_lr + 0.5 * (base_lr - min_lr_ratio * base_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def eval_fovea_seg(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    dices = []
    for batch in loader:
        pre_x = batch["pre_x"].to(device)
        pre_y = batch["pre_y"].to(device)
        intra_x = batch["intra_x"].to(device)
        intra_y = batch["intra_y"].to(device)

        enc: ViTEncoder = model["enc"]
        seg: SegDecoder = model["seg"]

        enc.set_domain("pre")
        _, pre_tok = enc.forward_tokens(pre_x)
        pre_logits = seg(pre_tok)

        enc.set_domain("intra")
        _, intra_tok = enc.forward_tokens(intra_x)
        intra_logits = seg(intra_tok)

        for logits, y in [(pre_logits, pre_y), (intra_logits, intra_y)]:
            probs = torch.sigmoid(logits)
            # dice per channel, then average
            num = 2 * (probs * y).sum(dim=(2, 3))
            den = (probs + y).sum(dim=(2, 3)) + 1e-6
            dice = ((num + 1e-6) / den).mean(dim=0)  # [2]
            dices.append(dice.cpu())
    if len(dices) == 0:
        return {"dice_vessel": 0.0, "dice_od": 0.0, "dice_mean": 0.0}
    dices = torch.stack(dices, dim=0).mean(dim=0)  # [2]
    return {"dice_vessel": float(dices[0]), "dice_od": float(dices[1]), "dice_mean": float(dices.mean())}


@torch.no_grad()
def eval_rmit(model: nn.Module, loader: DataLoader, device: torch.device, use_anatomy_anchor: bool, seg_for_anchor: Optional[SegDecoder]) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []
    for batch in loader:
        template = batch["template"].to(device)
        search = batch["search"].to(device)
        gt_xy = batch["gt_xy"].to(device)

        anatomy_mask = None
        if use_anatomy_anchor and seg_for_anchor is not None:
            enc: ViTEncoder = model["enc"]
            enc.set_domain("surg")
            _, tok = enc.forward_tokens(search)
            logits = seg_for_anchor(tok)  # [B,2,H,W]
            anatomy_mask = torch.sigmoid(logits)

        hm = model["tracker"](template, search, anatomy_mask=anatomy_mask)  # [B,1,H,W]
        pred_xy = soft_argmax_2d(hm)
        preds.append(pred_xy.cpu())
        gts.append(gt_xy.cpu())

    if len(preds) == 0:
        return {"mse": 0.0, "mean_error": 0.0, "precision@5px": 0.0, "precision@10px": 0.0, "precision@20px": 0.0, "failure_rate": 0.0}

    preds = torch.cat(preds, dim=0)
    gts = torch.cat(gts, dim=0)
    return compute_tracking_metrics(preds, gts)


# ============================================================
# Config
# ============================================================
@dataclass
class TrainCfg:
    stage: str
    out_dir: str
    init_ckpt: Optional[str]
    pretrained_encoder_ckpt: Optional[str]

    img_size: int
    patch_size: int
    embed_dim: int
    depth: int
    heads: int
    mlp_ratio: float
    drop: float
    attn_drop: float
    lora_r: int
    use_residual_lora: bool
    residual_alpha: float

    batch_size: int
    epochs: int
    lr: float
    wd: float
    num_workers: int
    seed: int
    amp: bool
    grad_accum: int

    freeze_backbone: bool
    use_ema: bool
    ema_decay: float

    # DR
    dr_imagefolder_dir: Optional[str]
    dr_train_csv: Optional[str]
    dr_val_csv: Optional[str]
    dr_val_ratio: float
    dr_num_classes: int

    # FOVEA
    fovea_pairs_json: Optional[str]
    fovea_val_ratio: float
    lambda_align: float
    lambda_seg: float

    # RMIT
    rmit_manifest_json: Optional[str]
    rmit_protocol: str
    rmit_test_seq: Optional[str]
    rmit_val_ratio: float
    max_pairs_per_seq: Optional[int]
    use_anatomy_anchor: bool
    eval_metrics: bool


def build_encoder(cfg: TrainCfg) -> ViTEncoder:
    domains = ["dr", "pre", "intra", "surg"]
    enc = ViTEncoder(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.heads,
        mlp_ratio=cfg.mlp_ratio,
        drop=cfg.drop,
        attn_drop=cfg.attn_drop,
        lora_r=cfg.lora_r,
        domains=domains,
        use_residual_lora=cfg.use_residual_lora,
    )
    return enc


# ============================================================
# Training: Stage-1 DR
# ============================================================
def train_stage_dr(cfg: TrainCfg, device: torch.device):
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    enc = build_encoder(cfg).to(device)
    if cfg.pretrained_encoder_ckpt:
        load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)

    train_ds, val_ds = build_dr_datasets(cfg)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty DR dataset split: train={len(train_ds)} val={len(val_ds)}")
    train_labels = [label for _, label in train_ds.items]
    val_labels = [label for _, label in val_ds.items]
    all_labels = train_labels + val_labels
    if len(all_labels) == 0:
        raise ValueError("Empty DR dataset.")
    inferred_num_classes = int(max(all_labels)) + 1
    if inferred_num_classes != cfg.dr_num_classes:
        print(f"[DR] Override dr_num_classes from {cfg.dr_num_classes} to inferred {inferred_num_classes}")

    head = DRHead(cfg.embed_dim, inferred_num_classes).to(device)
    model = nn.ModuleDict({"enc": enc, "head": head})

    # Compute class counts first (needed for sampler and class weights)
    counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long), minlength=inferred_num_classes).float()
    
    # WeightedRandomSampler for severe minority oversampling
    sample_weights = [1.0 / counts[label].item() if counts[label].item() > 0 else 0.0 for label in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True
    )
    print(f"[DR] Using WeightedRandomSampler with {len(train_ds)} samples (minority oversampling)")
    
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    present = counts > 0
    class_weights = torch.ones_like(counts)
    if present.any():
        present_counts = counts[present]
        # Inverse frequency weighting with BOOST for minority class
        w_present = present_counts.sum() / (present_counts + 1e-6)
        # Apply square root to smooth and then boost minority further
        w_present = torch.sqrt(w_present) * 1.5  # Boost minority weight
        w_present = w_present / w_present.mean()  # Normalize
        class_weights[present] = w_present
    class_weights[~present] = 0.0
    class_weights = class_weights.to(device)
    print(f"[DR] Class counts: {counts.tolist()}, class_weights: {class_weights.tolist()}")
    
    # FocalLoss for hard negative mining
    focal_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean').to(device)

    if cfg.freeze_backbone:
        # Safety check: warn if freezing backbone without pretraining
        if not cfg.pretrained_encoder_ckpt and not cfg.init_ckpt:
            print("[WARNING] You are freezing the backbone without loading a pretrained checkpoint! "
                  "The encoder will remain random, and the model will likely not learn anything useful.")
        params = get_lora_and_head_params(model, verbose=True)
        verify_lora_frozen(model)
        optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    best_acc = -1.0
    global_step = 0
    total_steps = cfg.epochs * max(1, len(train_ld)) // cfg.grad_accum

    for epoch in range(cfg.epochs):
        model.train()
        correct, total = 0, 0
        running_loss = 0.0

        optim.zero_grad(set_to_none=True)

        iter_train = enumerate(train_ld)
        if tqdm is not None:
            iter_train = tqdm(iter_train, total=len(train_ld), desc=f"DR {epoch+1}/{cfg.epochs}", leave=False)

        # 调试：监控前3个batch的预测分布
        debug_batch_count = 0

        for it, (x, y) in iter_train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            enc.set_domain("dr")
            lr_now = cosine_lr(cfg.lr, global_step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                cls = enc(x)
                logits = head(cls)
                # FocalLoss with class weighting for hard negative mining
                loss = focal_loss_fn(logits, y) / max(1, cfg.grad_accum)

            scaler.scale(loss).backward()

            running_loss += loss.item() * max(1, cfg.grad_accum)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

            # 调试：打印前3个batch的预测分布
            if debug_batch_count < 3 and (it + 1) % cfg.grad_accum == 0:
                with torch.no_grad():
                    pred_dist = torch.bincount(pred.cpu(), minlength=inferred_num_classes)
                    gt_dist = torch.bincount(y.cpu(), minlength=inferred_num_classes)
                    print(f"  [Debug Batch {debug_batch_count}] Loss: {loss.item() * max(1, cfg.grad_accum):.4f}, "
                          f"Pred: {pred_dist.tolist()}, GT: {gt_dist.tolist()}")
                debug_batch_count += 1

            if (it + 1) % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                
                # Logging and step update should happen after actual parameter update
                if global_step % 50 == 0:
                    writer.add_scalar("dr/train_loss", loss.item() * max(1, cfg.grad_accum), global_step)
                    writer.add_scalar("dr/lr", lr_now, global_step)
                
                global_step += 1
        if len(train_ld) % cfg.grad_accum != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1

        train_acc = correct / max(1, total)

        # val
        def _eval(m: nn.Module) -> Tuple[float, float]:
            m.eval()
            v_correct, v_total = 0, 0
            v_cls_correct = torch.zeros(inferred_num_classes, dtype=torch.float64)
            v_cls_total = torch.zeros(inferred_num_classes, dtype=torch.float64)
            with torch.no_grad():
                for x, y in val_ld:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    enc = m["enc"]; head = m["head"]
                    enc.set_domain("dr")
                    cls = enc(x)
                    logits = head(cls)
                    pred = logits.argmax(dim=1)
                    v_correct += (pred == y).sum().item()
                    v_total += y.numel()
                    for c in range(inferred_num_classes):
                        mask = (y == c)
                        cnt = mask.sum().item()
                        if cnt > 0:
                            v_cls_total[c] += cnt
                            v_cls_correct[c] += (pred[mask] == y[mask]).sum().item()
            acc = v_correct / max(1, v_total)
            present_cls = v_cls_total > 0
            if present_cls.any():
                bal_acc = float((v_cls_correct[present_cls] / v_cls_total[present_cls]).mean().item())
            else:
                bal_acc = 0.0
            return acc, bal_acc

        val_acc, val_bal_acc = _eval(model)
        if ema is not None:
            ema_acc, ema_bal_acc = _eval(ema.module)
        else:
            ema_acc = None
            ema_bal_acc = None

        writer.add_scalar("dr/train_acc", train_acc, epoch)
        writer.add_scalar("dr/val_acc", val_acc, epoch)
        writer.add_scalar("dr/val_bal_acc", val_bal_acc, epoch)
        if ema_acc is not None:
            writer.add_scalar("dr/ema_val_acc", ema_acc, epoch)
        if ema_bal_acc is not None:
            writer.add_scalar("dr/ema_val_bal_acc", ema_bal_acc, epoch)

        msg = f"[DR] epoch {epoch+1}/{cfg.epochs} loss={running_loss/len(train_ld):.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f} val_bal_acc={val_bal_acc:.4f}" + (f" ema_val_acc={ema_acc:.4f} ema_val_bal_acc={ema_bal_acc:.4f}" if ema_acc is not None else "")
        print(msg)
        append_epoch_log(cfg.out_dir, msg)

        metric = ema_bal_acc if ema_bal_acc is not None else val_bal_acc
        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best_acc,
            "ema": ema.state_dict() if ema is not None else None,
            "extra": {"stage": "dr"}
        }
        save_ckpt(os.path.join(cfg.out_dir, "last.pt"), payload)
        if metric > best_acc:
            best_acc = metric
            payload["best_metric"] = best_acc
            save_ckpt(os.path.join(cfg.out_dir, "best.pt"), payload)

    writer.close()


# ============================================================
# Training: Stage-2 FOVEA
# ============================================================
def _split_fovea_by_patient(pairs: List[dict], val_ratio: float) -> Tuple[List[dict], List[dict]]:
    # patient-level split
    by_pid = {}
    for it in pairs:
        pid = str(it["patient_id"])
        by_pid.setdefault(pid, []).append(it)
    pids = list(by_pid.keys())
    random.shuffle(pids)
    n_val = max(1, int(len(pids) * val_ratio))
    val_pids = set(pids[:n_val])
    train, val = [], []
    for pid, items in by_pid.items():
        (val if pid in val_pids else train).extend(items)
    return train, val


def train_stage_fovea(cfg: TrainCfg, device: torch.device):
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    enc = build_encoder(cfg).to(device)
    seg = SegDecoder(cfg.img_size, cfg.patch_size, cfg.embed_dim, out_ch=2).to(device)
    con = ContrastiveHead(cfg.embed_dim, proj_dim=256).to(device)
    model = nn.ModuleDict({"enc": enc, "seg": seg, "con": con})

    if cfg.pretrained_encoder_ckpt:
        load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)

    if cfg.init_ckpt:
        print(f"[FOVEA] Loading init ckpt: {cfg.init_ckpt}")
        load_ckpt(cfg.init_ckpt, model, optim=None)

    # data
    with open(cfg.fovea_pairs_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    train_pairs, val_pairs = _split_fovea_by_patient(pairs, cfg.fovea_val_ratio)
    if len(train_pairs) == 0 and len(val_pairs) > 1:
        train_pairs = val_pairs[1:]
        val_pairs = val_pairs[:1]
    if len(val_pairs) == 0 and len(train_pairs) > 1:
        val_pairs = train_pairs[:1]
        train_pairs = train_pairs[1:]

    train_ds = FOVEAPairedDataset(train_pairs, cfg.img_size, augment=True)
    val_ds = FOVEAPairedDataset(val_pairs, cfg.img_size, augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty FOVEA dataset split: train={len(train_ds)} val={len(val_ds)}")
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    if cfg.freeze_backbone:
        params = get_lora_and_head_params(model, verbose=False)
        verify_lora_frozen(model)
        optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    best = -1.0  # use dice_mean

    global_step = 0
    total_steps = cfg.epochs * max(1, len(train_ld)) // cfg.grad_accum

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        optim.zero_grad(set_to_none=True)

        iter_train = enumerate(train_ld)
        if tqdm is not None:
            iter_train = tqdm(iter_train, total=len(train_ld), desc=f"FOVEA {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iter_train:
            pre_x = batch["pre_x"].to(device, non_blocking=True)
            pre_y = batch["pre_y"].to(device, non_blocking=True)
            intra_x = batch["intra_x"].to(device, non_blocking=True)
            intra_y = batch["intra_y"].to(device, non_blocking=True)

            lr_now = cosine_lr(cfg.lr, global_step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                # pre
                enc.set_domain("pre")
                pre_cls, pre_tok = enc.forward_tokens(pre_x)
                pre_logits = seg(pre_tok)
                # intra
                enc.set_domain("intra")
                intra_cls, intra_tok = enc.forward_tokens(intra_x)
                intra_logits = seg(intra_tok)

                seg_loss = 0.5 * (
                    (dice_loss_logits(pre_logits, pre_y) + bce_loss_logits(pre_logits, pre_y)) +
                    (dice_loss_logits(intra_logits, intra_y) + bce_loss_logits(intra_logits, intra_y))
                )

                z_pre = con(pre_cls)
                z_intra = con(intra_cls)
                align_loss = info_nce_loss(z_pre, z_intra, temperature=0.07)

                loss = (cfg.lambda_seg * seg_loss + cfg.lambda_align * align_loss) / max(1, cfg.grad_accum)

            scaler.scale(loss).backward()

            running += loss.item() * max(1, cfg.grad_accum)

            if (it + 1) % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                
                # Logging and step update should happen after actual parameter update
                if global_step % 20 == 0:
                    writer.add_scalar("fovea/train_loss", loss.item() * max(1, cfg.grad_accum), global_step)
                    writer.add_scalar("fovea/seg_loss", seg_loss.item(), global_step)
                    writer.add_scalar("fovea/align_loss", align_loss.item(), global_step)
                    writer.add_scalar("fovea/lr", lr_now, global_step)
                
                global_step += 1
        if len(train_ld) % cfg.grad_accum != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1

        # eval dice
        metrics = eval_fovea_seg(model, val_ld, device)
        if ema is not None:
            ema_metrics = eval_fovea_seg(ema.module, val_ld, device)
        else:
            ema_metrics = None

        writer.add_scalar("fovea/val_dice_mean", metrics["dice_mean"], epoch)
        writer.add_scalar("fovea/val_dice_vessel", metrics["dice_vessel"], epoch)
        writer.add_scalar("fovea/val_dice_od", metrics["dice_od"], epoch)
        if ema_metrics is not None:
            writer.add_scalar("fovea/ema_val_dice_mean", ema_metrics["dice_mean"], epoch)

        msg = f"[FOVEA] epoch {epoch+1}/{cfg.epochs} train_loss={running/len(train_ld):.4f} val_dice_mean={metrics['dice_mean']:.4f}"
        if ema_metrics is not None:
            msg += f" ema_val_dice_mean={ema_metrics['dice_mean']:.4f}"
        print(msg)
        append_epoch_log(cfg.out_dir, msg)

        metric = ema_metrics["dice_mean"] if ema_metrics is not None else metrics["dice_mean"]

        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "ema": ema.state_dict() if ema is not None else None,
            "extra": {"stage": "fovea"}
        }
        save_ckpt(os.path.join(cfg.out_dir, "last.pt"), payload)
        if metric > best:
            best = metric
            payload["best_metric"] = best
            save_ckpt(os.path.join(cfg.out_dir, "best.pt"), payload)

    writer.close()


# ============================================================
# Training: Stage-3 RMIT
# ============================================================
def train_stage_rmit(cfg: TrainCfg, device: torch.device):
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    enc = build_encoder(cfg).to(device)
    seg = SegDecoder(cfg.img_size, cfg.patch_size, cfg.embed_dim, out_ch=2).to(device)  # used for anatomy anchor
    tracker = OneStreamTipTracker(cfg.img_size, cfg.patch_size, enc, cfg.embed_dim, use_anatomy_anchor=cfg.use_anatomy_anchor).to(device)
    model = nn.ModuleDict({"enc": enc, "seg": seg, "tracker": tracker})

    if cfg.pretrained_encoder_ckpt:
        load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)

    if cfg.init_ckpt:
        print(f"[RMIT] Loading init ckpt: {cfg.init_ckpt}")
        load_ckpt(cfg.init_ckpt, model, optim=None)

    # CRITICAL FIX: Initialize 'surg' LoRA from 'intra' LoRA
    # The seg head is frozen and expects 'intra'-like features.
    # If 'surg' starts from 0, the seg head will fail (domain shift).
    print("[RMIT] Initializing 'surg' domain from 'intra' domain for continuity...")
    enc.copy_domain_params(src="intra", dst="surg")

    # data
    train_ds = RMITTrackingDataset(
        cfg.rmit_manifest_json, cfg.img_size, is_training=True,
        max_pairs_per_seq=cfg.max_pairs_per_seq,
        protocol=cfg.rmit_protocol, test_seq=cfg.rmit_test_seq, val_ratio=cfg.rmit_val_ratio
    )
    val_ds = RMITTrackingDataset(
        cfg.rmit_manifest_json, cfg.img_size, is_training=False,
        max_pairs_per_seq=None,
        protocol=cfg.rmit_protocol, test_seq=cfg.rmit_test_seq, val_ratio=cfg.rmit_val_ratio
    )
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty RMIT dataset split: train={len(train_ds)} val={len(val_ds)}")

    # If anatomy anchor enabled, we typically freeze seg parameters (use as teacher)
    if cfg.use_anatomy_anchor:
        for p in seg.parameters():
            p.requires_grad = False

    if cfg.freeze_backbone:
        # PEFT for RMIT: train LoRA + tracker heads (+pos_ts). Keep seg frozen (teacher for anatomy anchor).
        params: List[nn.Parameter] = []
        frozen, trainable = 0, 0
        for name, p in model.named_parameters():
            is_lora = ('A.' in name) or ('B.' in name)
            is_seg = name.startswith('seg.')
            is_tracker = name.startswith('tracker.')
            if cfg.use_anatomy_anchor and is_seg:
                p.requires_grad = False
                frozen += 1
                continue
            if is_lora or is_tracker or ('tracker.pos_ts' in name):
                p.requires_grad = True
                params.append(p)
                trainable += 1
            else:
                p.requires_grad = False
                frozen += 1
        print(f'[PEFT-RMIT] Frozen params={frozen}, Trainable params={trainable}')
        optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    else:
        # Full fine-tune; still keep seg frozen if used only for anatomy mask inference.
        if cfg.use_anatomy_anchor:
            for p in seg.parameters():
                p.requires_grad = False
        optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    best = 1e9  # minimize mean_error

    global_step = 0
    total_steps = cfg.epochs * max(1, len(train_ld)) // cfg.grad_accum

    for epoch in range(cfg.epochs):
        model.train()
        if cfg.use_anatomy_anchor:
            seg.eval()
        running = 0.0
        optim.zero_grad(set_to_none=True)

        iter_train = enumerate(train_ld)
        if tqdm is not None:
            iter_train = tqdm(iter_train, total=len(train_ld), desc=f"RMIT {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iter_train:
            template = batch["template"].to(device, non_blocking=True)
            search = batch["search"].to(device, non_blocking=True)
            gt_hm = batch["gt_hm"].to(device, non_blocking=True)
            gt_xy = batch["gt_xy"].to(device, non_blocking=True)

            lr_now = cosine_lr(cfg.lr, global_step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                anatomy_mask = None
                if cfg.use_anatomy_anchor:
                    with torch.no_grad():
                        enc.set_domain("surg")
                        _, tok = enc.forward_tokens(search)
                        logits = seg(tok)
                        anatomy_mask = torch.sigmoid(logits).detach()

                pred_hm = tracker(template, search, anatomy_mask=anatomy_mask)

                loss_hm = F.mse_loss(torch.sigmoid(pred_hm), gt_hm)
                pred_xy = soft_argmax_2d(pred_hm)
                loss_xy = F.smooth_l1_loss(pred_xy, gt_xy)
                loss = (loss_hm + 0.25 * loss_xy) / max(1, cfg.grad_accum)

            scaler.scale(loss).backward()

            running += loss.item() * max(1, cfg.grad_accum)

            if (it + 1) % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                
                # Logging and step update should happen after actual parameter update
                if global_step % 20 == 0:
                    writer.add_scalar("rmit/train_loss", loss.item() * max(1, cfg.grad_accum), global_step)
                    writer.add_scalar("rmit/loss_hm", loss_hm.item(), global_step)
                    writer.add_scalar("rmit/loss_xy", loss_xy.item(), global_step)
                    writer.add_scalar("rmit/lr", lr_now, global_step)
                
                global_step += 1
        if len(train_ld) % cfg.grad_accum != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1

        # eval
        metrics = eval_rmit(model, val_ld, device, cfg.use_anatomy_anchor, seg if cfg.use_anatomy_anchor else None)
        if ema is not None:
            seg_anchor_ema = None
            if cfg.use_anatomy_anchor:
                try:
                    seg_anchor_ema = ema.module['seg']
                except Exception:
                    seg_anchor_ema = seg
            ema_metrics = eval_rmit(ema.module, val_ld, device, cfg.use_anatomy_anchor, seg_anchor_ema)
        else:
            ema_metrics = None
        writer.add_scalar("rmit/val_mean_error", metrics["mean_error"], epoch)
        if cfg.eval_metrics:
            writer.add_scalar("rmit/val_precision@5px", metrics["precision@5px"], epoch)
            writer.add_scalar("rmit/val_precision@10px", metrics["precision@10px"], epoch)
            writer.add_scalar("rmit/val_precision@20px", metrics["precision@20px"], epoch)
        if ema_metrics is not None:
            writer.add_scalar("rmit/ema_val_mean_error", ema_metrics["mean_error"], epoch)

        msg = f"[RMIT] epoch {epoch+1}/{cfg.epochs} train_loss={running/len(train_ld):.4f} val_mean_error={metrics['mean_error']:.3f}"
        if cfg.eval_metrics:
            msg += f" p@10={metrics['precision@10px']:.3f}"
        if ema_metrics is not None:
            msg += f" ema_val_mean_error={ema_metrics['mean_error']:.3f}"
        print(msg)
        append_epoch_log(cfg.out_dir, msg)

        metric = ema_metrics["mean_error"] if ema_metrics is not None else metrics["mean_error"]

        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "ema": ema.state_dict() if ema is not None else None,
            "extra": {"stage": "rmit"}
        }
        save_ckpt(os.path.join(cfg.out_dir, "last.pt"), payload)
        if metric < best:
            best = metric
            payload["best_metric"] = best
            save_ckpt(os.path.join(cfg.out_dir, "best.pt"), payload)

    writer.close()


# ============================================================
# CLI
# ============================================================
def parse_args() -> TrainCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, required=True, choices=["dr", "fovea", "rmit"])
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--init_ckpt", type=str, default=None)
    p.add_argument("--pretrained_encoder_ckpt", type=str, default=None)

    p.add_argument("--img_size", type=int, default=518)  # 518 is better for DINOv2 (14*37)
    p.add_argument("--patch_size", type=int, default=14) # DINOv2 default
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--attn_drop", type=float, default=0.0)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--use_residual_lora", action="store_true")
    p.add_argument("--residual_alpha", type=float, default=0.1)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_accum", type=int, default=1)

    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.9999)

    # DR
    p.add_argument("--dr_imagefolder_dir", type=str, default=None)
    p.add_argument("--dr_train_csv", type=str, default=None)
    p.add_argument("--dr_val_csv", type=str, default=None)
    p.add_argument("--dr_val_ratio", type=float, default=0.2)
    p.add_argument("--dr_num_classes", type=int, default=5)

    # FOVEA
    p.add_argument("--fovea_pairs_json", type=str, default=None)
    p.add_argument("--fovea_val_ratio", type=float, default=0.2)
    p.add_argument("--lambda_align", type=float, default=1.0)
    p.add_argument("--lambda_seg", type=float, default=1.0)

    # RMIT
    p.add_argument("--rmit_manifest_json", type=str, default=None)
    p.add_argument("--rmit_protocol", type=str, default="leave_one_out", choices=["leave_one_out", "random"])
    p.add_argument("--rmit_test_seq", type=str, default=None)
    p.add_argument("--rmit_val_ratio", type=float, default=0.2)
    p.add_argument("--max_pairs_per_seq", type=int, default=None)
    p.add_argument("--use_anatomy_anchor", action="store_true")
    p.add_argument("--eval_metrics", action="store_true")

    a = p.parse_args()

    cfg = TrainCfg(
        stage=a.stage,
        out_dir=a.out_dir,
        init_ckpt=a.init_ckpt,
        pretrained_encoder_ckpt=a.pretrained_encoder_ckpt,
        img_size=a.img_size,
        patch_size=a.patch_size,
        embed_dim=a.embed_dim,
        depth=a.depth,
        heads=a.heads,
        mlp_ratio=a.mlp_ratio,
        drop=a.drop,
        attn_drop=a.attn_drop,
        lora_r=a.lora_r,
        use_residual_lora=a.use_residual_lora,
        residual_alpha=a.residual_alpha,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        wd=a.wd,
        num_workers=a.num_workers,
        seed=a.seed,
        amp=a.amp,
        grad_accum=max(1, a.grad_accum),
        freeze_backbone=a.freeze_backbone,
        use_ema=a.use_ema,
        ema_decay=a.ema_decay,
        dr_imagefolder_dir=a.dr_imagefolder_dir,
        dr_train_csv=a.dr_train_csv,
        dr_val_csv=a.dr_val_csv,
        dr_val_ratio=a.dr_val_ratio,
        dr_num_classes=a.dr_num_classes,
        fovea_pairs_json=a.fovea_pairs_json,
        fovea_val_ratio=a.fovea_val_ratio,
        lambda_align=a.lambda_align,
        lambda_seg=a.lambda_seg,
        rmit_manifest_json=a.rmit_manifest_json,
        rmit_protocol=a.rmit_protocol,
        rmit_test_seq=a.rmit_test_seq,
        rmit_val_ratio=a.rmit_val_ratio,
        max_pairs_per_seq=a.max_pairs_per_seq,
        use_anatomy_anchor=a.use_anatomy_anchor,
        eval_metrics=a.eval_metrics,
    )
    return cfg


def main():
    cfg = parse_args()
    seed_all(cfg.seed)
    ensure_dir(cfg.out_dir)

    # Redirect print to both stdout and file
    sys.stdout = Logger(os.path.join(cfg.out_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(cfg.out_dir, "train.err"), sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage: {cfg.stage}")

    # Minimal argument checks
    if cfg.stage == "dr":
        train_stage_dr(cfg, device)
    elif cfg.stage == "fovea":
        assert cfg.fovea_pairs_json, "Need --fovea_pairs_json"
        train_stage_fovea(cfg, device)
    elif cfg.stage == "rmit":
        assert cfg.rmit_manifest_json, "Need --rmit_manifest_json"
        train_stage_rmit(cfg, device)
    else:
        raise ValueError(cfg.stage)


if __name__ == "__main__":
    main()
