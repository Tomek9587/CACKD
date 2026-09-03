#!/usr/bin/env python
"""
dr_7.py — Rigid Instrument Tracker for RMIT

设计理念（V7 大刀阔斧版）：
1. 不用 ViT-LoRA，改用 ResNet18 + MLP
2. 不玩三阶段预训练，单阶段直接在 RMIT 上训练
3. 不把 640×480 硬拉成 512×512，改为保持长宽比 + padding
4. 4 个点不平铺预测，而是预测一个刚性位姿：中心 + 朝向 + 长度 + 相对偏移
5. 重建 4 个关键点，用 L1 直接监督
"""
import argparse
import csv
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

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
    """ImageNet normalization."""
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
    """
    保持长宽比 resize + padding。

    Returns:
        img: target_size x target_size 的图像
        gt_xy: 变换后的坐标 (如果输入为 None 则返回 None)
        scale: 缩放比例
        pad_top: 顶部 padding
        pad_left: 左侧 padding
    """
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


def denormalize_xy(
    xy: torch.Tensor,
    scale: float,
    pad_top: int,
    pad_left: int,
) -> torch.Tensor:
    """把 padded 图像坐标反变换回原图坐标。"""
    xy = xy.clone()
    xy[:, 0] = (xy[:, 0] - pad_left) / scale
    xy[:, 1] = (xy[:, 1] - pad_top) / scale
    return xy


# =============================================================================
# 1. 刚性器械模型：参数化 ↔ 重建
# =============================================================================
class RigidPose:
    """5 个独立参数 + 2 个衍生参数（cos/sin），共 7 个网络输出。"""
    def __init__(self, cx: float, cy: float, theta: float, L: float, alpha: float, beta: float):
        self.cx = cx
        self.cy = cy
        self.theta = theta
        self.L = L
        self.alpha = alpha
        self.beta = beta

    @property
    def axis(self) -> np.ndarray:
        return np.array([math.cos(self.theta), math.sin(self.theta)], dtype=np.float32)

    def reconstruct(self) -> np.ndarray:
        """重建 4 个点：left_tip, right_tip, junction, midpoint。"""
        c = np.array([self.cx, self.cy], dtype=np.float32)
        a = self.axis
        half = self.L / 2.0
        points = np.zeros((4, 2), dtype=np.float32)
        points[0] = c + half * a              # left tip
        points[1] = c - half * a              # right tip
        points[2] = c + self.alpha * half * a # junction
        points[3] = c + self.beta * half * a  # midpoint
        return points

    @staticmethod
    def from_points(points: np.ndarray) -> "RigidPose":
        """从 4 个 GT 点反解位姿参数。"""
        left = points[0]
        right = points[1]
        c = (left + right) / 2.0
        diff = left - right
        L = float(np.linalg.norm(diff))
        theta = float(math.atan2(diff[1], diff[0]))
        a = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        half = L / 2.0 if L > 1e-6 else 1e-6

        def proj(p):
            return float(np.dot(p - c, a) / half)

        alpha = proj(points[2])
        beta = proj(points[3])
        return RigidPose(float(c[0]), float(c[1]), theta, L, alpha, beta)


# =============================================================================
# 2. 数据集
# =============================================================================
def load_rmit_manifest_sequences(manifest_json: str) -> List[dict]:
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["sequences"] if isinstance(manifest, dict) and "sequences" in manifest else manifest


class RMITRigidDataset(Dataset):
    """
    每个样本是一张图 + 4 个点的坐标。
    可选择返回连续帧对（用于时序一致性）。
    """
    def __init__(
        self,
        manifest_json: str,
        img_size: int,
        split_names: set,
        is_training: bool,
        use_pairs: bool = False,
        max_frame_gap: int = 1,
        augment: bool = True,
        within_seq_val_ratio: float = 0.2,
        within_seq_seed: int = 42,
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.use_pairs = use_pairs
        self.max_frame_gap = max_frame_gap
        self.augment = augment

        sequences = load_rmit_manifest_sequences(manifest_json)
        selected = [s for s in sequences if s["name"] in split_names]

        self.records: List[dict] = []
        for seq in selected:
            self.records.extend(self._load_sequence(seq["frames_dir"], seq["ann_csv"]))
        self.records.sort(key=lambda r: r["path"])

        # 同序列时序划分
        if len(split_names) == 1 and within_seq_val_ratio > 0:
            n = len(self.records)
            n_val = max(1, int(round(n * within_seq_val_ratio)))
            # 训练取前段，验证取后段，保证时序不泄露
            if is_training:
                self.records = self.records[: n - n_val]
            else:
                self.records = self.records[n - n_val:]

        self.items: List[dict] = []
        if use_pairs:
            for gap in range(1, max_frame_gap + 1):
                for i in range(gap, len(self.records)):
                    self.items.append({
                        "template": self.records[i - gap],
                        "target": self.records[i],
                    })
        else:
            for r in self.records:
                self.items.append({"target": r})

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
                records.append({
                    "path": path,
                    "xy": np.array(pts, dtype=np.float32),  # [4,2]
                })
        return records

    def __len__(self) -> int:
        return len(self.items)

    def _process(self, record: dict) -> Tuple[torch.Tensor, torch.Tensor, RigidPose]:
        img = load_image_rgb(record["path"])
        xy = record["xy"].copy()

        img, xy, scale, pad_top, pad_left = letterbox_resize(img, self.img_size, xy)

        # 数据增强：轻微颜色抖动 + 小角度旋转（同步变换坐标）
        if self.is_training and self.augment:
            img, xy = self._augment(img, xy)

        # clamp
        xy = np.clip(xy, 0, self.img_size - 1)

        # 归一化到 [0,1]
        xy_norm = xy / float(self.img_size - 1)
        pose = RigidPose.from_points(xy_norm)

        tensor = normalize_img(pil_to_tensor(img))
        return tensor, torch.from_numpy(xy_norm).float(), pose

    def _augment(self, img: Image.Image, xy: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        # 颜色抖动
        from PIL import ImageEnhance, ImageFilter
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-0.1, 0.1))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-0.1, 0.1))

        # 小角度旋转
        if random.random() < 0.3:
            angle = random.uniform(-5.0, 5.0)
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        target_tensor, target_xy, target_pose = self._process(item["target"])

        out = {
            "img": target_tensor,
            "gt_xy": target_xy,                 # [4,2] normalized
            "path": item["target"]["path"],
        }

        if self.use_pairs:
            template_tensor, _, _ = self._process(item["template"])
            out["template"] = template_tensor

        return out


# =============================================================================
# 3. 网络：ResNet18 + MLP
# =============================================================================
class RigidInstrumentNet(nn.Module):
    """
    输出 7 个参数：
      - cx, cy (sigmoid -> [0,1])
      - cosθ, sinθ (L2 normalize -> 单位向量)
      - logL (exp -> 长度，再除以 img_size 归一化)
      - alpha, beta (线性)
    """
    def __init__(
        self,
        img_size: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.img_size = img_size

        if backbone == "resnet18":
            net = models.resnet18(weights=("IMAGENET1K_V1" if pretrained else None))
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            feat_dim = 512
        elif backbone == "resnet34":
            net = models.resnet34(weights=("IMAGENET1K_V1" if pretrained else None))
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            feat_dim = 512
        else:
            raise ValueError(backbone)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 7),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
            cx, cy: [B]
            axis: [B, 2] 单位向量
            L_norm: [B] 归一化长度
            alpha, beta: [B]
        """
        feat = self.backbone(x)
        out = self.head(feat)

        cx = torch.sigmoid(out[:, 0])
        cy = torch.sigmoid(out[:, 1])
        raw_axis = out[:, 2:4]
        axis = F.normalize(raw_axis, dim=-1, eps=1e-6)
        logL = out[:, 4]
        L_norm = torch.exp(logL).clamp_min(1e-4)
        alpha = out[:, 5]
        beta = out[:, 6]

        return {
            "cx": cx,
            "cy": cy,
            "axis": axis,
            "L_norm": L_norm,
            "alpha": alpha,
            "beta": beta,
        }


def reconstruct_points(pred: Dict[str, torch.Tensor]) -> torch.Tensor:
    """从网络输出重建 4 个归一化坐标点。"""
    cx = pred["cx"]
    cy = pred["cy"]
    axis = pred["axis"]
    L_norm = pred["L_norm"]
    alpha = pred["alpha"]
    beta = pred["beta"]

    center = torch.stack([cx, cy], dim=-1)          # [B, 2]
    half = L_norm.unsqueeze(-1) / 2.0               # [B, 1]

    pts = []
    pts.append(center + half * axis)                # left tip
    pts.append(center - half * axis)                # right tip
    pts.append(center + (alpha.unsqueeze(-1) * half) * axis)  # junction
    pts.append(center + (beta.unsqueeze(-1) * half) * axis)   # midpoint
    return torch.stack(pts, dim=1)                  # [B, 4, 2]


# =============================================================================
# 4. 损失函数
# =============================================================================
def pose_parameter_loss(pred: Dict[str, torch.Tensor], gt_pose: RigidPose, device: torch.device) -> torch.Tensor:
    """把 GT 反解的参数和网络输出对齐。"""
    gt_cx = torch.tensor(gt_pose.cx, dtype=torch.float32, device=device)
    gt_cy = torch.tensor(gt_pose.cy, dtype=torch.float32, device=device)
    gt_axis = torch.tensor(gt_pose.axis, dtype=torch.float32, device=device)
    gt_L = torch.tensor(gt_pose.L, dtype=torch.float32, device=device)
    gt_alpha = torch.tensor(gt_pose.alpha, dtype=torch.float32, device=device)
    gt_beta = torch.tensor(gt_pose.beta, dtype=torch.float32, device=device)

    loss_c = F.l1_loss(torch.stack([pred["cx"], pred["cy"]], dim=-1),
                       torch.stack([gt_cx, gt_cy], dim=-1))
    loss_axis = (1.0 - (pred["axis"] * gt_axis).sum(dim=-1)).mean()
    loss_L = F.l1_loss(pred["L_norm"], gt_L)
    loss_alpha = F.l1_loss(pred["alpha"], gt_alpha)
    loss_beta = F.l1_loss(pred["beta"], gt_beta)

    return loss_c + loss_axis + loss_L + loss_alpha + loss_beta


def point_loss(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_xy, gt_xy)


def total_loss(pred: Dict[str, torch.Tensor], gt_xy: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    pred_xy = reconstruct_points(pred)
    loss_pts = point_loss(pred_xy, gt_xy)

    # 正则：让 alpha/beta 不要太离谱（器械结构先验）
    reg_alpha = F.relu(pred["alpha"].abs() - 1.5).mean()
    reg_beta = F.relu(pred["beta"].abs() - 1.5).mean()
    reg_L = F.relu(-pred["L_norm"]).mean()  # 长度非负已由 exp 保证

    loss = loss_pts + 0.05 * (reg_alpha + reg_beta + reg_L)
    return loss, {"pts": loss_pts.item(), "reg": (reg_alpha + reg_beta).item()}


# =============================================================================
# 5. 评估
# =============================================================================
def compute_metrics(pred_xy_px: torch.Tensor, gt_xy_px: torch.Tensor, img_size: int) -> Dict[str, float]:
    diff = pred_xy_px - gt_xy_px
    dist = torch.sqrt((diff ** 2).sum(dim=-1))  # [B, 4]

    metrics = {
        "mean_error_px": float(dist.mean().item()),
        "max_error_px": float(dist.max().item()),
        "pck@5": float((dist < 5).float().mean().item()),
        "pck@10": float((dist < 10).float().mean().item()),
        "pck@20": float((dist < 20).float().mean().item()),
    }
    for k in range(4):
        metrics[f"kp{k}_mean_error_px"] = float(dist[:, k].mean().item())
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
) -> Dict[str, float]:
    model.eval()
    all_pred, all_gt = [], []
    for batch in loader:
        img = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)  # [B,4,2] normalized
        pred = model(img)
        pred_xy = reconstruct_points(pred)
        all_pred.append(pred_xy.cpu())
        all_gt.append(gt_xy.cpu())

    all_pred = torch.cat(all_pred, dim=0) * (img_size - 1)
    all_gt = torch.cat(all_gt, dim=0) * (img_size - 1)
    return compute_metrics(all_pred, all_gt, img_size)


# =============================================================================
# 6. LOSO 划分
# =============================================================================
def resolve_loso_split(manifest_json: str, fold: int) -> Tuple[set, set, str]:
    sequences = load_rmit_manifest_sequences(manifest_json)
    names = sorted([s["name"] for s in sequences])
    fold = fold % len(names)
    test_name = names[fold]
    train_names = {n for n in names if n != test_name}
    return train_names, {test_name}, test_name


def resolve_within_seq_split(
    manifest_json: str, seq_name: str, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[set, set, str]:
    """
    同序列按时间顺序划分：前 (1-val_ratio) 训练，后 val_ratio 验证。
    返回 split_names 都是 {seq_name}，但内部在 dataset 中做时序切分。
    """
    return {seq_name}, {seq_name}, seq_name


# =============================================================================
# 7. 训练
# =============================================================================
def warmup_cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_rmit(cfg, device: torch.device) -> None:
    ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    model = RigidInstrumentNet(
        img_size=cfg.img_size,
        backbone=cfg.backbone,
        pretrained=cfg.pretrained,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    if cfg.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
        print(f"[V7] Freeze backbone, trainable head params: {trainable:,}")
    else:
        print(f"[V7] Full fine-tune, trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.wd,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))

    if cfg.protocol == "loso":
        train_names, val_names, test_name = resolve_loso_split(cfg.manifest, cfg.fold)
        print(f"[V7] LOSO fold {cfg.fold}: train={sorted(train_names)}, val={test_name}")
    elif cfg.protocol == "within_seq":
        if not cfg.seq:
            raise ValueError("--protocol within_seq requires --seq")
        train_names, val_names, test_name = resolve_within_seq_split(cfg.manifest, cfg.seq)
        print(f"[V7] Within-seq {cfg.seq}: temporal split 80/20")
    else:
        raise ValueError(cfg.protocol)

    train_ds = RMITRigidDataset(
        cfg.manifest, cfg.img_size, train_names,
        is_training=True, use_pairs=False, augment=True,
        within_seq_val_ratio=cfg.within_seq_val_ratio,
    )
    val_ds = RMITRigidDataset(
        cfg.manifest, cfg.img_size, val_names,
        is_training=False, use_pairs=False, augment=False,
        within_seq_val_ratio=cfg.within_seq_val_ratio,
    )

    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    print(f"[V7] Train: {len(train_ds)}, Val: {len(val_ds)}")

    total_steps = max(1, len(train_ld)) * cfg.epochs
    warmup_steps = max(1, len(train_ld)) * cfg.warmup_epochs
    global_step = 0
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        running_pts = 0.0

        iterator = enumerate(train_ld)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(train_ld), desc=f"V7 epoch {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iterator:
            img = batch["img"].to(device, non_blocking=True)
            gt_xy = batch["gt_xy"].to(device, non_blocking=True)

            lr_now = warmup_cosine_lr(cfg.lr, global_step, total_steps, warmup_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
                pred = model(img)
                loss, loss_dict = total_loss(pred, gt_xy)

            if not torch.isfinite(loss):
                print(f"[V7] skip non-finite loss at epoch={epoch+1} iter={it+1}")
                optim.zero_grad(set_to_none=True)
                continue

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim)
            scaler.update()

            running_loss += loss.item()
            running_pts += loss_dict["pts"]
            global_step += 1

            if global_step % 20 == 0:
                writer.add_scalar("v7/train_loss", loss.item(), global_step)
                writer.add_scalar("v7/train_pts_loss", loss_dict["pts"], global_step)
                writer.add_scalar("v7/lr", lr_now, global_step)

        # Validation
        metrics = evaluate(model, val_ld, device, cfg.img_size)
        for k, v in metrics.items():
            writer.add_scalar(f"v7/val_{k}", v, epoch)

        avg_loss = running_loss / max(len(train_ld), 1)
        avg_pts = running_pts / max(len(train_ld), 1)
        msg = (
            f"[V7] epoch {epoch+1}/{cfg.epochs} "
            f"train_loss={avg_loss:.4f} pts={avg_pts:.4f} "
            f"val_err={metrics['mean_error_px']:.2f}px "
            f"pck@5={metrics['pck@5']:.3f} pck@10={metrics['pck@10']:.3f} pck@20={metrics['pck@20']:.3f}"
        )
        print(msg)

        # Early stopping on mean error
        metric = metrics["mean_error_px"]
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch,
                "metric": metric,
            }, os.path.join(cfg.out_dir, "best.pt"))
        else:
            patience_counter += 1

        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "epoch": epoch,
            "metric": metric,
        }, os.path.join(cfg.out_dir, "last.pt"))

        if patience_counter >= cfg.patience:
            print(f"[V7] Early stopping at epoch {epoch+1}")
            break

    writer.close()
    print(f"[V7] Best val error: {best_metric:.2f}px")


# =============================================================================
# 8. 命令行
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DR-SurgBridgeNet V7: Rigid Instrument Tracker")
    parser.add_argument("--stage", type=str, default="rmit", choices=["rmit"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default="/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking/data/manifests/rmit_manifest.json")
    parser.add_argument("--protocol", type=str, default="loso", choices=["loso", "within_seq"])
    parser.add_argument("--seq", type=str, default=None, choices=["seq1", "seq2", "seq3"])
    parser.add_argument("--within_seq_val_ratio", type=float, default=0.2)

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)

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
