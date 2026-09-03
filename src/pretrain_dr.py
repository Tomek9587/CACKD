#!/usr/bin/env python
"""
pretrain_dr.py — Stage 1: pretrain ResNet18 encoder on Diabetic Retinopathy classification.
"""
import argparse
import os
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# 0. utils
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


def normalize_img(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device).view(3, 1, 1)
    return (x - mean) / std


# =============================================================================
# 1. dataset
# =============================================================================
CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferate_DR", "Severe"]


class DRClassificationDataset(Dataset):
    def __init__(self, root: str, img_size: int = 224, is_training: bool = True):
        self.root = root
        self.img_size = img_size
        self.is_training = is_training
        self.samples = []
        for label, cls in enumerate(CLASS_NAMES):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_dir, fn), label))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.is_training:
            img = self._augment(img)
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return normalize_img(torch.from_numpy(arr)), label


# =============================================================================
# 2. model
# =============================================================================
class DRClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=("IMAGENET1K_V1" if pretrained else None))
        self.encoder = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def encoder_state_dict(self):
        return self.encoder.state_dict()


# =============================================================================
# 3. train
# =============================================================================
def train_dr(cfg, device: torch.device):
    ensure_dir(cfg.out_dir)

    model = DRClassifier(num_classes=5, pretrained=cfg.pretrained).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[DR] Trainable: {trainable:,} / {total:,}")

    ds = DRClassificationDataset(cfg.data_dir, img_size=cfg.img_size, is_training=True)
    val_split = int(len(ds) * cfg.val_ratio)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [len(ds) - val_split, val_split],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    # disable augmentation for val
    val_ds.dataset.is_training = False

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print(f"[DR] Train: {len(train_ds)}, Val: {len(val_ds)}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        correct = total_seen = 0
        iterator = enumerate(train_ld)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_ld), desc=f"DR {epoch+1}/{cfg.epochs}", leave=False)

        for it, (img, label) in iterator:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
                logits = model(img)
                loss = criterion(logits, label)

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == label).sum().item()
            total_seen += label.size(0)

        # validation
        model.eval()
        val_correct = val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for img, label in val_ld:
                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                logits = model(img)
                val_loss += F.cross_entropy(logits, label).item()
                val_correct += (logits.argmax(dim=1) == label).sum().item()
                val_total += label.size(0)

        train_acc = correct / max(total_seen, 1)
        val_acc = val_correct / max(val_total, 1)
        print(f"[DR] epoch {epoch+1}/{cfg.epochs} loss={running_loss/max(len(train_ld),1):.4f} "
              f"train_acc={train_acc:.4f} val_loss={val_loss/max(len(val_ld),1):.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "encoder": model.encoder_state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
            }, os.path.join(cfg.out_dir, "best.pt"))

    print(f"[DR] Best val acc: {best_acc:.4f}")


# =============================================================================
# 4. args
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain ResNet18 on DR classification")
    parser.add_argument("--out_dir", type=str, default="logs/pretrain_dr")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/colored_images")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
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
    train_dr(cfg, device)


if __name__ == "__main__":
    main()
