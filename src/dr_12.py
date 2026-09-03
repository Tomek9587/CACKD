#!/usr/bin/env python
"""
dr_12.py — V12: Multi-decoder ensemble (SAR + heatmap) + temporal modeling.

Two orthogonal improvements over V11 (dr_11.py):
1. Ensemble decoder: simultaneously run SAR head and heatmap(+offset) head on the
   shared decoder feature; train with combined loss, predict with averaged xy.
   - heatmap branch has Gaussian supervision -> stronger localization prior,
     helps when SAR confidence collapses on out-of-distribution frames (kp0 on seq1).
2. Temporal modeling: a lightweight 1D temporal conv (kernel=3) over the decoder
   feature map exploits inter-frame coherence. Requires sequence-sampling dataset.
   -相邻帧的器械位置强相关，单帧定位失败时可用时序补偿。

Both can be toggled independently via --decoder_type ensemble and --use_temporal.
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
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Reuse V11 building blocks
_src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _src_dir)
from dr_11 import (
    seed_all,
    ensure_dir,
    load_image_rgb,
    normalize_img,
    pil_to_tensor,
    letterbox_resize,
    make_gaussian,
    soft_argmax_2d,
    sar_decode,
    SARHead,
    HeatmapKeypointNet,
    mse_heatmap_loss,
    xy_regression_loss,
    total_loss,
    sar_loss,
    compute_metrics,
    warmup_cosine_lr,
    load_pretrained_weights,
    resolve_loso_split,
    resolve_within_seq_split,
    load_rmit_manifest_sequences,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =============================================================================
# Temporal dataset (sequence-sampling)
# =============================================================================
class TemporalRMITDataset(Dataset):
    """Sequence-sampling dataset. Returns clips of T consecutive frames.

    Training: stride=1, all full-length windows (no padding).
    Eval:     stride=seq_len (non-overlapping), last clip padded with last frame.
    """

    def __init__(
        self,
        manifest_json: str,
        img_size: int,
        split_names: set,
        is_training: bool,
        seq_len: int = 8,
        stride: Optional[int] = None,
        sigma: float = 2.0,
        augment: bool = True,
        aug_color: float = 0.15,
        aug_rotate: float = 5.0,
        aug_blur: float = 0.0,
        aug_noise: float = 0.0,
        aug_cutout: float = 0.0,
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.seq_len = seq_len
        self.sigma = sigma
        self.augment = augment
        self.aug_color = aug_color
        self.aug_rotate = aug_rotate
        self.aug_blur = aug_blur
        self.aug_noise = aug_noise
        self.aug_cutout = aug_cutout

        if stride is None:
            stride = 1 if is_training else seq_len
        self.stride = stride

        sequences = load_rmit_manifest_sequences(manifest_json)
        selected = [s for s in sequences if s["name"] in split_names]

        self.sequences: List[List[dict]] = []
        for seq in selected:
            recs = self._load_sequence(seq["frames_dir"], seq["ann_csv"])
            recs.sort(key=lambda r: r["path"])
            self.sequences.append(recs)

        self.clips: List[Tuple[int, int]] = []  # (seq_idx, start)
        for si, recs in enumerate(self.sequences):
            n = len(recs)
            if n == 0:
                continue
            if is_training:
                for start in range(0, n - seq_len + 1, stride):
                    self.clips.append((si, start))
            else:
                for start in range(0, n, stride):
                    self.clips.append((si, start))

    def _load_sequence(self, frames_dir, ann_csv):
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

    def _augment_frame(self, img, xy):
        if self.aug_color > 0:
            if random.random() < 0.5:
                img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-self.aug_color, self.aug_color))
            if random.random() < 0.5:
                img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-self.aug_color, self.aug_color))
            if random.random() < 0.3:
                img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-self.aug_color, self.aug_color))
        if self.aug_blur > 0 and random.random() < self.aug_blur:
            img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.4, 1.8))
        if self.aug_noise > 0 and random.random() < self.aug_noise:
            arr = np.asarray(img, dtype=np.float32)
            noise = np.random.normal(0.0, self.aug_noise * 255.0, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 255.0).astype(np.uint8)
            img = Image.fromarray(arr)
        if self.aug_cutout > 0 and random.random() < self.aug_cutout:
            w, h = img.size
            cw = max(8, random.randint(w // 10, w // 5))
            ch = max(8, random.randint(h // 10, h // 5))
            x0 = random.randint(0, max(1, w - cw))
            y0 = random.randint(0, max(1, h - ch))
            img.paste((0, 0, 0), (x0, y0, x0 + cw, y0 + ch))
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

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        si, start = self.clips[idx]
        recs = self.sequences[si]
        n = len(recs)
        end = min(start + self.seq_len, n)
        actual = end - start

        imgs, xys, hms = [], [], []
        for fi in range(start, end):
            rec = recs[fi]
            img = load_image_rgb(rec["path"])
            xy = rec["xy"].copy()
            img, xy, _, _, _ = letterbox_resize(img, self.img_size, xy)
            if self.is_training and self.augment:
                img, xy = self._augment_frame(img, xy)
            xy = np.clip(xy, 0, self.img_size - 1)
            xy_norm = xy / float(self.img_size - 1)
            gt_hm = torch.zeros(4, self.img_size, self.img_size, dtype=torch.float32)
            for k in range(4):
                gt_hm[k] = make_gaussian(self.img_size, xy[k, 0], xy[k, 1], self.sigma)
            imgs.append(normalize_img(pil_to_tensor(img)))
            xys.append(torch.from_numpy(xy_norm).float())
            hms.append(gt_hm)

        while len(imgs) < self.seq_len:
            imgs.append(imgs[-1].clone())
            xys.append(xys[-1].clone())
            hms.append(hms[-1].clone())

        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask[:actual] = True

        return {
            "img": torch.stack(imgs),      # [T, 3, H, W]
            "gt_xy": torch.stack(xys),     # [T, 4, 2]
            "gt_hm": torch.stack(hms),     # [T, 4, H, W]
            "mask": mask,                  # [T]
        }


# =============================================================================
# Temporal conv
# =============================================================================
class TemporalConv(nn.Module):
    """1D temporal conv over T dim of [B, T, C, H, W]."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1),
                              padding=(pad, 0, 0))
        self.bn = nn.BatchNorm3d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        x = self.act(self.bn(self.conv(x)))
        return x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]


# =============================================================================
# V12 network
# =============================================================================
class HeatmapKeypointNetV12(HeatmapKeypointNet):
    """V12: ensemble head (SAR + heatmap) + optional temporal conv.

    decoder_type:
      - "sar"            : SAR only (same as V11)
      - "heatmap_offset" : heatmap only (same as V11)
      - "ensemble"       : SAR + heatmap(+offset), predict averaged xy
    use_temporal: if True, add 1D temporal conv on decoder feature.
    """

    def __init__(
        self,
        img_size: int,
        pretrained: bool = True,
        backbone: str = "resnet18",
        decoder_channels: List[int] = [256, 128, 64],
        decoder_stride: int = 4,
        use_aspp: bool = False,
        use_offset: bool = True,
        decoder_type: str = "ensemble",
        bcir_beta: float = 10.0,
        sar_decode_mode: str = "softmax",
        sar_topk: int = 9,
        sar_temperature: float = 0.5,
        use_temporal: bool = False,
        temporal_kernel: int = 3,
        ensemble_alpha: float = 0.5,
    ):
        is_ensemble = decoder_type == "ensemble"
        parent_type = "sar" if is_ensemble else decoder_type
        super().__init__(
            img_size=img_size,
            pretrained=pretrained,
            backbone=backbone,
            decoder_channels=decoder_channels,
            decoder_stride=decoder_stride,
            use_aspp=use_aspp,
            use_offset=use_offset and (parent_type in ("heatmap_offset", "ensemble")),
            decoder_type=parent_type,
            bcir_beta=bcir_beta,
            sar_decode_mode=sar_decode_mode,
            sar_topk=sar_topk,
            sar_temperature=sar_temperature,
        )
        self.decoder_type = decoder_type  # restore "ensemble"
        self.use_temporal = use_temporal
        self.ensemble_alpha = ensemble_alpha  # weight for xy_hm in fused prediction

        if use_temporal:
            self.temporal_conv = TemporalConv(decoder_channels[2], temporal_kernel)
        else:
            self.temporal_conv = None

        if is_ensemble:
            # parent (sar) did not build hm_head/offset_head; build them now
            self.use_offset = use_offset  # override parent's False
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
                    nn.Conv2d(64, 8, 1),
                )

    def _backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, 3, H, W] -> feat: [N, C, h, w]"""
        e2, e3, e4 = self.encoder(x)
        e4 = self.aspp(e4)
        d1 = self.up1(e4, e3)
        d2 = self.up2(d1, e2)
        d3 = self.bottleneck(d2)
        if self.up3 is not None:
            d3 = self.up3(d3)
        return d3

    def _heads_forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """feat: [N, C, h, w] -> dict of outputs (N = B*T or B)."""
        if self.decoder_type == "ensemble":
            # --- SAR branch ---
            sar_out = self.sar_head(feat)
            xy_sar = sar_decode(
                sar_out["logit"], sar_out["keypoint"],
                mode=self.sar_decode_mode,
                topk=self.sar_topk,
                temperature=self.sar_temperature,
            )
            # --- heatmap (+offset) branch ---
            hm_logits = self.hm_head(feat)
            hm_logits = F.interpolate(hm_logits, size=(self.img_size, self.img_size),
                                       mode="bilinear", align_corners=False)
            coarse_xy = soft_argmax_2d(hm_logits, temperature=1.0)
            coarse_xy_norm = coarse_xy / float(self.img_size - 1)
            if self.use_offset:
                B = feat.size(0)
                offset_logits = self.offset_head(feat)
                offset_logits = F.interpolate(offset_logits, size=(self.img_size, self.img_size),
                                               mode="bilinear", align_corners=False)
                offset = torch.tanh(offset_logits).view(B, 4, 2, self.img_size, self.img_size)
                grid = coarse_xy_norm * 2 - 1
                grid = grid.unsqueeze(2).unsqueeze(3)
                sampled = []
                for k in range(4):
                    s = F.grid_sample(offset[:, k], grid[:, k], mode="bilinear",
                                      padding_mode="border", align_corners=True)
                    sampled.append(s.squeeze(-1).squeeze(-1))
                offset_xy = torch.stack(sampled, dim=1) * 0.05
                xy_hm = (coarse_xy_norm + offset_xy).clamp(0, 1)
            else:
                xy_hm = coarse_xy_norm

            xy_fused = (1.0 - self.ensemble_alpha) * xy_sar + self.ensemble_alpha * xy_hm
            return {
                "xy": xy_fused,           # final prediction (used by evaluate)
                "xy_hm": xy_hm,           # for total_loss xy regression
                "xy_sar": xy_sar,
                "coarse_xy": coarse_xy_norm,
                "logit": sar_out["logit"],
                "keypoint": sar_out["keypoint"],
                "hm_logits": hm_logits,
            }
        elif self.decoder_type == "sar":
            sar_out = self.sar_head(feat)
            xy = sar_decode(
                sar_out["logit"], sar_out["keypoint"],
                mode=self.sar_decode_mode, topk=self.sar_topk,
                temperature=self.sar_temperature,
            )
            return {"xy": xy, "logit": sar_out["logit"], "keypoint": sar_out["keypoint"]}
        else:  # heatmap_offset
            hm_logits = self.hm_head(feat)
            hm_logits = F.interpolate(hm_logits, size=(self.img_size, self.img_size),
                                       mode="bilinear", align_corners=False)
            coarse_xy = soft_argmax_2d(hm_logits, temperature=1.0)
            coarse_xy_norm = coarse_xy / float(self.img_size - 1)
            if self.use_offset:
                B = feat.size(0)
                offset_logits = self.offset_head(feat)
                offset_logits = F.interpolate(offset_logits, size=(self.img_size, self.img_size),
                                               mode="bilinear", align_corners=False)
                offset = torch.tanh(offset_logits).view(B, 4, 2, self.img_size, self.img_size)
                grid = coarse_xy_norm * 2 - 1
                grid = grid.unsqueeze(2).unsqueeze(3)
                sampled = []
                for k in range(4):
                    s = F.grid_sample(offset[:, k], grid[:, k], mode="bilinear",
                                      padding_mode="border", align_corners=True)
                    sampled.append(s.squeeze(-1).squeeze(-1))
                offset_xy = torch.stack(sampled, dim=1) * 0.05
                xy = (coarse_xy_norm + offset_xy).clamp(0, 1)
            else:
                xy = coarse_xy_norm
            return {"hm_logits": hm_logits, "coarse_xy": coarse_xy_norm, "xy": xy}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 5:
            # temporal: [B, T, 3, H, W]
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feat = self._backbone_forward(x)
            if self.temporal_conv is not None:
                feat = feat.reshape(B, T, *feat.shape[1:])
                feat = self.temporal_conv(feat)
                feat = feat.reshape(B * T, *feat.shape[2:])
            out = self._heads_forward(feat)
            out = {k: (v.reshape(B, T, *v.shape[1:]) if v.dim() >= 2 else v)
                   for k, v in out.items()}
            return out
        else:
            # single frame [B, 3, H, W]
            feat = self._backbone_forward(x)
            if self.temporal_conv is not None:
                B = feat.size(0)
                feat = feat.reshape(B, 1, *feat.shape[1:])
                feat = self.temporal_conv(feat)
                feat = feat.reshape(B, *feat.shape[2:])
            return self._heads_forward(feat)


# =============================================================================
# Loss
# =============================================================================
def ensemble_loss(
    pred: Dict[str, torch.Tensor],
    gt_hm: torch.Tensor,
    gt_xy: torch.Tensor,
    img_size: int,
    sar_scale: float = 16.0,
    sar_weight: float = 1.0,
    hm_xy_weight: float = 10.0,
    hm_coarse_weight: float = 5.0,
    kp_weights: Optional[torch.Tensor] = None,
    xy_loss_type: str = "l1",
    wing_w: float = 10.0,
    wing_epsilon: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combined SAR + heatmap loss for ensemble decoder."""
    loss_sar, _ = sar_loss(pred, gt_xy, img_size, sar_scale=sar_scale, kp_weights=kp_weights)
    # total_loss uses pred["xy"] for xy regression; swap in xy_hm for the heatmap branch
    pred_for_hm = {**pred, "xy": pred["xy_hm"]}
    loss_hm, hm_terms = total_loss(
        pred_for_hm, gt_hm, gt_xy,
        kp_weights=kp_weights,
        xy_loss_type=xy_loss_type, wing_w=wing_w, wing_epsilon=wing_epsilon,
        xy_weight=hm_xy_weight, coarse_weight=hm_coarse_weight,
    )
    loss = sar_weight * loss_sar + loss_hm
    return loss, {
        "sar": loss_sar.item(),
        "hm": loss_hm.item(),
        "hm_xy": hm_terms["xy"],
        "hm_coarse": hm_terms["coarse"],
    }


# =============================================================================
# Eval
# =============================================================================
@torch.no_grad()
def adapt_bn_v12(model: nn.Module, loader: DataLoader, device: torch.device, passes: int = 1) -> None:
    """AdaBN: feed target-domain frames (flattened from clips) to update BN stats."""
    was_training = model.training
    model.train()
    for _ in range(passes):
        for batch in loader:
            img = batch["img"].to(device, non_blocking=True)  # [B, T, 3, H, W]
            B, T = img.shape[:2]
            img_flat = img.view(B * T, *img.shape[2:])  # [B*T, 3, H, W]
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                _ = model(img_flat)  # 4D path updates all BN
    if not was_training:
        model.eval()
    print(f"[AdaBN] updated BN stats with {passes} pass(es)")


@torch.no_grad()
def evaluate_v12(model: nn.Module, loader: DataLoader, device: torch.device,
                 img_size: int, adabn: bool = False, adabn_passes: int = 1) -> Dict[str, float]:
    if adabn:
        adapt_bn_v12(model, loader, device, passes=adabn_passes)
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        img = batch["img"].to(device)          # [B, T, 3, H, W]
        gt_xy = batch["gt_xy"].to(device)      # [B, T, 4, 2]
        mask = batch["mask"].to(device)        # [B, T]
        pred = model(img)
        pred_xy = pred["xy"]                   # [B, T, 4, 2]
        B, T = pred_xy.shape[:2]
        pred_flat = pred_xy.reshape(B * T, 4, 2)
        gt_flat = gt_xy.reshape(B * T, 4, 2)
        mask_flat = mask.reshape(B * T)
        all_pred.append(pred_flat[mask_flat].cpu())
        all_gt.append(gt_flat[mask_flat].cpu())
    all_pred = torch.cat(all_pred) * (img_size - 1)
    all_gt = torch.cat(all_gt) * (img_size - 1)
    return compute_metrics(all_pred, all_gt, img_size)


# =============================================================================
# Train
# =============================================================================
def train_v12(cfg, device: torch.device) -> None:
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    model = HeatmapKeypointNetV12(
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
        use_temporal=cfg.use_temporal,
        temporal_kernel=cfg.temporal_kernel,
        ensemble_alpha=cfg.ensemble_alpha,
    ).to(device)

    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        load_pretrained_weights(model, cfg.checkpoint)

    if cfg.freeze_backbone:
        for p in model.encoder.parameters():
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[V12] Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    print(f"[V12] decoder_type={cfg.decoder_type} use_temporal={cfg.use_temporal} seq_len={cfg.seq_len}")

    if cfg.eval_only:
        ckpt_path = cfg.checkpoint if cfg.checkpoint else os.path.join(cfg.out_dir, "best.pt")
        load_pretrained_weights(model, ckpt_path)
        if cfg.protocol == "loso":
            _, val_names, test_name = resolve_loso_split(cfg.manifest, cfg.fold)
        else:
            _, val_names, test_name = resolve_within_seq_split(cfg.manifest, cfg.seq)
        print(f"[V12 eval] protocol={cfg.protocol} fold={cfg.fold} seq={test_name}")
        val_ds = TemporalRMITDataset(
            cfg.manifest, cfg.img_size, val_names, False,
            seq_len=cfg.seq_len, sigma=cfg.sigma, augment=False,
        )
        val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        metrics = evaluate_v12(model, val_ld, device, cfg.img_size, adabn=cfg.adabn, adabn_passes=cfg.adabn_passes)
        print(f"[V12 eval] {test_name}: err={metrics['mean_error_px']:.2f}px pck@20={metrics['pck@20']:.3f} pck@50={metrics['pck@50']:.3f}")
        print(json.dumps(metrics, indent=2))
        return

    kp_weights = None
    if cfg.kp_adaptive:
        init_w = cfg.kp_weights if cfg.kp_weights is not None else [1.0, 1.0, 1.0, 1.0]
        kp_weights = torch.tensor(init_w, dtype=torch.float32)
        print(f"[V12] Adaptive kp weights init: {kp_weights.tolist()}")
    elif cfg.kp_weights is not None:
        kp_weights = torch.tensor(cfg.kp_weights, dtype=torch.float32)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.wd,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))

    if cfg.protocol == "loso":
        train_names, val_names, test_name = resolve_loso_split(cfg.manifest, cfg.fold)
        print(f"[V12] LOSO fold {cfg.fold}: train={sorted(train_names)}, val={test_name}")
    else:
        train_names, val_names, test_name = resolve_within_seq_split(cfg.manifest, cfg.seq)
        print(f"[V12] Within-seq {cfg.seq}")

    train_ds = TemporalRMITDataset(
        cfg.manifest, cfg.img_size, train_names, True,
        seq_len=cfg.seq_len, sigma=cfg.sigma, augment=True,
        aug_color=cfg.aug_color, aug_rotate=cfg.aug_rotate,
        aug_blur=cfg.aug_blur, aug_noise=cfg.aug_noise, aug_cutout=cfg.aug_cutout,
    )
    val_ds = TemporalRMITDataset(
        cfg.manifest, cfg.img_size, val_names, False,
        seq_len=cfg.seq_len, sigma=cfg.sigma, augment=False,
    )
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    print(f"[V12] Train clips: {len(train_ds)}, Val clips: {len(val_ds)}")

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
            iterator = tqdm(iterator, total=len(train_ld), desc=f"V12 {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iterator:
            img = batch["img"].to(device, non_blocking=True)        # [B, T, 3, H, W]
            gt_xy = batch["gt_xy"].to(device, non_blocking=True)    # [B, T, 4, 2]
            gt_hm = batch["gt_hm"].to(device, non_blocking=True)    # [B, T, 4, H, W]
            B, T = img.shape[:2]
            gt_xy_flat = gt_xy.view(B * T, 4, 2)
            gt_hm_flat = gt_hm.view(B * T, 4, cfg.img_size, cfg.img_size)

            lr_now = warmup_cosine_lr(cfg.lr, global_step, total_steps, warmup_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
                pred = model(img)  # values are [B, T, ...]
                pred_flat = {k: v.reshape(B * T, *v.shape[2:]) for k, v in pred.items()}
                if cfg.decoder_type == "ensemble":
                    loss, loss_dict = ensemble_loss(
                        pred_flat, gt_hm_flat, gt_xy_flat, cfg.img_size,
                        sar_scale=cfg.sar_scale, sar_weight=cfg.sar_weight,
                        hm_xy_weight=cfg.xy_weight, hm_coarse_weight=cfg.coarse_weight,
                        kp_weights=kp_weights,
                        xy_loss_type=cfg.xy_loss_type, wing_w=cfg.wing_w, wing_epsilon=cfg.wing_epsilon,
                    )
                elif cfg.decoder_type == "sar":
                    loss, loss_dict = sar_loss(pred_flat, gt_xy_flat, cfg.img_size,
                                               sar_scale=cfg.sar_scale, kp_weights=kp_weights)
                else:  # heatmap_offset
                    loss, loss_dict = total_loss(
                        pred_flat, gt_hm_flat, gt_xy_flat, kp_weights=kp_weights,
                        xy_loss_type=cfg.xy_loss_type, wing_w=cfg.wing_w, wing_epsilon=cfg.wing_epsilon,
                        xy_weight=cfg.xy_weight, coarse_weight=cfg.coarse_weight,
                    )

            if not torch.isfinite(loss):
                print(f"[V12] skip non-finite loss")
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
                writer.add_scalar("v12/train_loss", loss.item(), global_step)
                for lk, lv in loss_dict.items():
                    writer.add_scalar(f"v12/train_{lk}", lv, global_step)
                writer.add_scalar("v12/lr", lr_now, global_step)

        metrics = evaluate_v12(model, val_ld, device, cfg.img_size,
                               adabn=cfg.adabn, adabn_passes=cfg.adabn_passes)
        for k, v in metrics.items():
            writer.add_scalar(f"v12/val_{k}", v, epoch)

        avg_loss = running_loss / max(len(train_ld), 1)
        avg_terms = {k: v / max(len(train_ld), 1) for k, v in running_terms.items()}

        if cfg.kp_adaptive:
            per_kp_err = torch.tensor([metrics[f"kp{k}_mean_error_px"] for k in range(4)], dtype=torch.float32)
            mean_err = per_kp_err.mean().clamp_min(1e-6)
            rel = per_kp_err / mean_err
            mom = cfg.kp_adapt_momentum
            kp_weights = mom * kp_weights + (1.0 - mom) * rel
            kp_weights = torch.clamp(kp_weights, cfg.kp_adapt_min_weight, cfg.kp_adapt_max_weight)
            if cfg.kp_adapt_normalize:
                kp_weights = kp_weights / kp_weights.sum() * 4.0
            print(f"[V12] adaptive kp_weights: {kp_weights.tolist()}")

        loss_terms = " ".join([f"{k}={v:.4f}" for k, v in avg_terms.items()])
        print(
            f"[V12 {cfg.decoder_type}] epoch {epoch+1}/{cfg.epochs} "
            f"loss={avg_loss:.4f} {loss_terms} "
            f"val_err={metrics['mean_error_px']:.2f}px "
            f"pck@20={metrics['pck@20']:.3f} pck@50={metrics['pck@50']:.3f} "
            f"kp0={metrics['kp0_mean_error_px']:.1f}"
        )

        metric = metrics["mean_error_px"]
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                        "epoch": epoch, "metric": metric},
                       os.path.join(cfg.out_dir, "best.pt"))
        else:
            patience_counter += 1

        torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                    "epoch": epoch, "metric": metric},
                   os.path.join(cfg.out_dir, "last.pt"))

        if patience_counter >= cfg.patience:
            print(f"[V12] Early stopping at epoch {epoch+1}")
            break

    writer.close()
    print(f"[V12] Best val error: {best_metric:.2f}px")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="V12: ensemble + temporal keypoint tracker")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str,
                        default="/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking/data/manifests/rmit_manifest.json")
    parser.add_argument("--protocol", type=str, default="within_seq", choices=["loso", "within_seq"])
    parser.add_argument("--seq", type=str, default="seq1", choices=["seq1", "seq2", "seq3"])

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--decoder_channels", type=int, nargs=3, default=[256, 128, 64])
    parser.add_argument("--decoder_stride", type=int, default=4, choices=[2, 4])
    parser.add_argument("--use_aspp", action="store_true")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--decoder_type", type=str, default="ensemble",
                        choices=["heatmap_offset", "sar", "ensemble"])
    parser.add_argument("--bcir_beta", type=float, default=10.0)
    parser.add_argument("--sar_scale", type=float, default=16.0)
    parser.add_argument("--sar_decode_mode", type=str, default="softmax", choices=["argmax", "softmax"])
    parser.add_argument("--sar_topk", type=int, default=9)
    parser.add_argument("--sar_temperature", type=float, default=0.5)
    parser.add_argument("--ensemble_alpha", type=float, default=0.5,
                        help="weight of heatmap branch in fused xy (0=SAR only, 1=heatmap only)")

    # temporal
    parser.add_argument("--use_temporal", action="store_true", help="enable 1D temporal conv on decoder feature")
    parser.add_argument("--temporal_kernel", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=8, help="clip length for temporal sampling")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--batch_size", type=int, default=4, help="batch size in clips (each clip = seq_len frames)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)

    # loss / kp reweighting
    parser.add_argument("--kp_weights", type=float, nargs=4, default=None)
    parser.add_argument("--kp_adaptive", action="store_true")
    parser.add_argument("--kp_adapt_momentum", type=float, default=0.9)
    parser.add_argument("--kp_adapt_max_weight", type=float, default=8.0)
    parser.add_argument("--kp_adapt_min_weight", type=float, default=0.3)
    parser.add_argument("--kp_adapt_normalize", action="store_true")
    parser.add_argument("--xy_loss_type", type=str, default="l1", choices=["l1", "smooth_l1", "wing"])
    parser.add_argument("--wing_w", type=float, default=10.0)
    parser.add_argument("--wing_epsilon", type=float, default=2.0)
    parser.add_argument("--xy_weight", type=float, default=10.0)
    parser.add_argument("--coarse_weight", type=float, default=5.0)
    parser.add_argument("--sar_weight", type=float, default=1.0, help="weight of SAR loss in ensemble")

    # AdaBN
    parser.add_argument("--adabn", action="store_true")
    parser.add_argument("--adabn_passes", type=int, default=1)

    # augmentation
    parser.add_argument("--aug_color", type=float, default=0.3)
    parser.add_argument("--aug_rotate", type=float, default=10)
    parser.add_argument("--aug_blur", type=float, default=0.3)
    parser.add_argument("--aug_noise", type=float, default=0.05)
    parser.add_argument("--aug_cutout", type=float, default=0.1)

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
    train_v12(cfg, device)


if __name__ == "__main__":
    main()
