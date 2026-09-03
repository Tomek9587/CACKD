#!/usr/bin/env python
"""
pretrain_fovea.py — Stage 2: pretrain encoder+decoder on FOVEA surgical-frame segmentation.
Uses weights from Stage 1 (DR) to initialize the encoder.
"""
import argparse
import glob
import os
import random
from typing import Dict, List, Tuple

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


def letterbox_resize(
    img: Image.Image,
    target_size: int,
    masks: List[Image.Image] = None,
    fill: Tuple[int, int, int] = (0, 0, 0),
):
    orig_w, orig_h = img.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    new_img = Image.new("RGB", (target_size, target_size), fill)
    new_img.paste(img, (pad_left, pad_top))

    if masks is not None:
        new_masks = []
        for m in masks:
            m = m.resize((new_w, new_h), Image.NEAREST)
            new_m = Image.new("L", (target_size, target_size), 0)
            new_m.paste(m, (pad_left, pad_top))
            new_masks.append(new_m)
        return new_img, new_masks, scale, pad_top, pad_left
    return new_img, None, scale, pad_top, pad_left


# =============================================================================
# 1. dataset
# =============================================================================
class FOVEASegmentationDataset(Dataset):
    """
    使用 FOVEA 的 intraoperative 帧 (_i_img.png) 和对应 mask (_i_od_1.png, _i_ve_1.png)。
    """
    def __init__(self, root: str, img_size: int = 512, is_training: bool = True):
        self.root = root
        self.img_size = img_size
        self.is_training = is_training
        self.samples = sorted(glob.glob(os.path.join(root, "FOVEA*_i_img.png")))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: Image.Image, masks: List[Image.Image]) -> Tuple[Image.Image, List[Image.Image]]:
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-0.15, 0.15))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-0.15, 0.15))
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            masks = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in masks]
        return img, masks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.samples[idx]
        base = img_path.replace("_i_img.png", "")
        od_path = base + "_i_od_1.png"
        ve_path = base + "_i_ve_1.png"

        img = Image.open(img_path).convert("RGB")
        od = Image.open(od_path).convert("L")
        ve = Image.open(ve_path).convert("L")

        img, masks, _, _, _ = letterbox_resize(img, self.img_size, [od, ve])
        if self.is_training:
            img, masks = self._augment(img, masks)

        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mask_od = np.asarray(masks[0], dtype=np.float32) / 255.0
        mask_ve = np.asarray(masks[1], dtype=np.float32) / 255.0
        mask = np.stack([mask_od, mask_ve], axis=0)  # [2, H, W]

        return {
            "img": normalize_img(torch.from_numpy(arr)),
            "mask": torch.from_numpy(mask).clamp(0, 1),
        }


# =============================================================================
# 2. model
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


class FOVEASegNet(nn.Module):
    """
    ResNet18 encoder + U-Net decoder + 2-channel segmentation head.
    Compatible with RMIT V8 decoder structure.
    """
    def __init__(self, num_classes: int = 2, decoder_channels: List[int] = [256, 128, 64]):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)          # /4
        self.enc3 = resnet.layer2                                          # /8
        self.enc4 = resnet.layer3                                          # /16

        self.up1 = UpBlock(256, 128, decoder_channels[0])
        self.up2 = UpBlock(decoder_channels[0], 64, decoder_channels[1])
        self.bottleneck = ConvBlock(decoder_channels[1], decoder_channels[2])

        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels[2], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d1 = self.up1(e4, e3)
        d2 = self.up2(d1, e2)
        d3 = self.bottleneck(d2)
        out = self.seg_head(d3)
        out = F.interpolate(out, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        return out

    def encoder_state_dict(self):
        return {
            "enc1": self.enc1.state_dict(),
            "enc2": self.enc2.state_dict(),
            "enc3": self.enc3.state_dict(),
            "enc4": self.enc4.state_dict(),
        }

    def decoder_state_dict(self):
        return {
            "up1": self.up1.state_dict(),
            "up2": self.up2.state_dict(),
            "bottleneck": self.bottleneck.state_dict(),
            "seg_head": self.seg_head.state_dict(),
        }


def load_encoder_from_dr(model: FOVEASegNet, dr_checkpoint: str):
    ckpt = torch.load(dr_checkpoint, map_location="cpu")
    enc_state = ckpt.get("encoder", ckpt.get("model"))
    if enc_state is None:
        raise ValueError("DR checkpoint missing encoder weights")

    # DRClassifier saved encoder as a Sequential of [conv1+bn1+relu, maxpool+layer1, layer2, layer3, layer4]
    # Build a temporary Sequential with the same structure and load the full state,
    # then copy parameters to our enc1..enc4.
    from torchvision import models
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
    print("[FOVEA] Loaded encoder from DR checkpoint")


# =============================================================================
# 3. train
# =============================================================================
def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return ((2 * inter + smooth) / (union + smooth)).mean().item()


def train_fovea(cfg, device: torch.device):
    ensure_dir(cfg.out_dir)

    model = FOVEASegNet(num_classes=2).to(device)
    if cfg.dr_checkpoint and os.path.exists(cfg.dr_checkpoint):
        load_encoder_from_dr(model, cfg.dr_checkpoint)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[FOVEA] Trainable: {trainable:,} / {total:,}")

    ds = FOVEASegmentationDataset(cfg.data_dir, img_size=cfg.img_size, is_training=True)
    n_val = max(1, int(len(ds) * cfg.val_ratio))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed)
    )
    val_ds.dataset.is_training = False

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print(f"[FOVEA] Train: {len(train_ds)}, Val: {len(val_ds)}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))

    best_dice = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        iterator = enumerate(train_ld)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_ld), desc=f"FOVEA {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iterator:
            img = batch["img"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
                logits = model(img)
                bce = F.binary_cross_entropy_with_logits(logits, mask)
                pred = torch.sigmoid(logits)
                dice = 1 - (2 * (pred * mask).sum() + 1e-6) / (pred.sum() + mask.sum() + 1e-6)
                loss = bce + dice

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()

        # validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_ld:
                img = batch["img"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                logits = model(img)
                bce = F.binary_cross_entropy_with_logits(logits, mask)
                val_loss += bce.item()
                val_dice += dice_score(logits, mask)

        avg_loss = running_loss / max(len(train_ld), 1)
        avg_val_loss = val_loss / max(len(val_ld), 1)
        avg_val_dice = val_dice / max(len(val_ld), 1)
        print(f"[FOVEA] epoch {epoch+1}/{cfg.epochs} loss={avg_loss:.4f} "
              f"val_bce={avg_val_loss:.4f} val_dice={avg_val_dice:.4f}")

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save({
                "model": model.state_dict(),
                "encoder": model.encoder_state_dict(),
                "decoder": model.decoder_state_dict(),
                "epoch": epoch,
                "val_dice": avg_val_dice,
            }, os.path.join(cfg.out_dir, "best.pt"))

    print(f"[FOVEA] Best val dice: {best_dice:.4f}")


# =============================================================================
# 4. args
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain encoder+decoder on FOVEA segmentation")
    parser.add_argument("--out_dir", type=str, default="logs/pretrain_fovea")
    parser.add_argument("--data_dir", type=str, default="/root/autodl-tmp/FOVEA")
    parser.add_argument("--dr_checkpoint", type=str, default="logs/pretrain_dr/best.pt")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()
    return args


def main():
    cfg = parse_args()
    seed_all(cfg.seed)
    ensure_dir(cfg.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_fovea(cfg, device)


if __name__ == "__main__":
    main()
