"""
dr_6.py — DR-SurgBridgeNet V6: Multi-Keypoint RMIT Tip Tracking

V5→V6 核心改动:
1. 多点跟踪: 1点 → 4关键点 (Left Tip, Right Tip, Junction, Midpoint)
2. 梯度修复: soft_argmax_from_prob 替代 logit-space soft-argmax (消除 1/(H*W) 梯度稀释)
3. 损失修复: pure_mse_heatmap_loss 替代 BCE+MSE; wing_loss 替代 smooth_l1
4. 几何约束: geometric_consistency_loss (刚体距离一致性 + 方向一致性)
5. 增强降低: flip=0, jitter=0.05, rotate=3°, scale=0.03, blur/erase=0
6. Stage2 隔离: --skip_stage2_backbone 跳过 FOVEA 污染的 backbone 权重
7. Offset head: scale=0.2, weight std=0.02, 去掉 L2 正则化
"""
import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import surgical_tracking.src.dr_2 as base
from surgical_tracking.src.dr_2 import *

# =============================================================================
# 常量
# =============================================================================
RMIT_REFERENCE_SIZE = 512.0
RMIT_REFERENCE_P10 = 20.0
NUM_KEYPOINTS = 4  # Left Tip, Right Tip, Junction, Midpoint
KEYPOINT_NAMES = ["left_tip", "right_tip", "junction", "midpoint"]
PRIMARY_KEYPOINT = 0  # Left Tip 作为主跟踪点


# =============================================================================
# 1. 新坐标提取函数 (替代 logit-space soft-argmax)
# =============================================================================
def soft_argmax_from_prob(
    heatmap_logits: torch.Tensor,
    beta: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对 sigmoid 概率图做幂次锐化后求期望坐标。
    避免 logit 空间 softmax 的 1/(H×W) 梯度稀释问题。

    Args:
        heatmap_logits: [B, K, H, W] 原始 logits
        beta: 锐化幂次 (越大越尖锐)
        eps: 数值稳定项

    Returns:
        xy_px: [B, K, 2] 像素坐标 (x, y)
    """
    bsz, K, H, W = heatmap_logits.shape
    prob = torch.sigmoid(heatmap_logits)
    # 幂次锐化: 让峰值更突出
    prob_sharp = prob ** beta
    prob_flat = prob_sharp.view(bsz, K, -1)  # [B, K, H*W]
    prob_norm = prob_flat / (prob_flat.sum(dim=-1, keepdim=True) + eps)  # [B, K, H*W]

    ys = torch.linspace(0, H - 1, H, device=heatmap_logits.device, dtype=heatmap_logits.dtype)
    xs = torch.linspace(0, W - 1, W, device=heatmap_logits.device, dtype=heatmap_logits.dtype)

    prob_2d = prob_norm.view(bsz, K, H, W)
    y = (prob_2d * ys.view(1, 1, H, 1)).sum(dim=(2, 3))  # [B, K]
    x = (prob_2d * xs.view(1, 1, 1, W)).sum(dim=(2, 3))  # [B, K]

    return torch.stack([x, y], dim=-1)  # [B, K, 2]


def argmax_from_prob(heatmap_logits: torch.Tensor) -> torch.Tensor:
    """
    对 sigmoid 概率图取 argmax 坐标 (不可微，仅用于推理/评估)。

    Args:
        heatmap_logits: [B, K, H, W]

    Returns:
        xy_px: [B, K, 2]
    """
    bsz, K, H, W = heatmap_logits.shape
    prob = torch.sigmoid(heatmap_logits)
    flat = prob.view(bsz, K, -1)  # [B, K, H*W]
    idx = flat.argmax(dim=-1)  # [B, K]
    y = (idx // W).float()
    x = (idx % W).float()
    return torch.stack([x, y], dim=-1)  # [B, K, 2]


# =============================================================================
# 2. 新损失函数
# =============================================================================
def pure_mse_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    纯 MSE heatmap loss — 避免 BCE 的大梯度主导。
    sigmoid → MSE，梯度量级 ~0.01，与 xy loss 匹配。
    """
    pred = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    return F.mse_loss(pred, target)


def wing_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    w: float = 5.0,
    epsilon: float = 0.5,
) -> torch.Tensor:
    """
    Wing Loss (CVPR 2018) — 对小误差更敏感的定位损失。

    |x| < w:  w * ln(1 + |x|/epsilon)
    |x| >= w: |x| - C   (C = w - w*ln(1 + w/epsilon))
    """
    diff = (pred - target).abs()
    C = w - w * math.log(1.0 + w / epsilon)
    loss = torch.where(
        diff < w,
        w * torch.log(1.0 + diff / epsilon),
        diff - C,
    )
    return loss.mean()


def geometric_consistency_loss(
    pred_xy: torch.Tensor,
    gt_xy: torch.Tensor,
    lambda_dist: float = 1.0,
    lambda_dir: float = 0.5,
) -> torch.Tensor:
    """
    几何一致性损失: 4点刚体约束。

    Args:
        pred_xy: [B, K, 2] 预测坐标
        gt_xy: [B, K, 2] GT 坐标
        lambda_dist: 成对距离一致性权重
        lambda_dir: 方向一致性权重

    Returns:
        loss: 标量
    """
    B, K, _ = pred_xy.shape
    if K < 2:
        return torch.tensor(0.0, device=pred_xy.device)

    loss = torch.tensor(0.0, device=pred_xy.device)

    # 成对距离一致性
    if lambda_dist > 0:
        dist_loss = torch.tensor(0.0, device=pred_xy.device)
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                pred_d = torch.norm(pred_xy[:, i] - pred_xy[:, j], dim=-1)  # [B]
                gt_d = torch.norm(gt_xy[:, i] - gt_xy[:, j], dim=-1)  # [B]
                dist_loss = dist_loss + F.smooth_l1_loss(pred_d, gt_d)
                count += 1
        if count > 0:
            loss = loss + lambda_dist * dist_loss / count

    # 方向一致性 (以 Left Tip→Right Tip 为主轴方向)
    if lambda_dir > 0 and K >= 2:
        pred_dir = F.normalize(pred_xy[:, 1] - pred_xy[:, 0], dim=-1)  # [B, 2]
        gt_dir = F.normalize(gt_xy[:, 1] - gt_xy[:, 0], dim=-1)
        cos_sim = (pred_dir * gt_dir).sum(dim=-1)  # [B]
        dir_loss = (1.0 - cos_sim).mean()
        loss = loss + lambda_dir * dir_loss

    return loss


# =============================================================================
# 3. DynamicWeightAverage (保留)
# =============================================================================
class DynamicWeightAverage:
    def __init__(self, task_names: List[str], temperature: float = 2.0):
        self.task_names = list(task_names)
        self.temperature = temperature
        self.prev_losses: Optional[List[float]] = None

    def __call__(self, losses_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        current = [losses_dict[k].detach().item() for k in self.task_names]
        if self.prev_losses is None:
            weights = [1.0 / len(self.task_names)] * len(self.task_names)
        else:
            rates = [c / (p + 1e-8) for c, p in zip(current, self.prev_losses)]
            exp_rates = [math.exp(r / self.temperature) for r in rates]
            s = sum(exp_rates)
            weights = [e / s for e in exp_rates]
        self.prev_losses = current
        total = sum(w * losses_dict[k] for w, k in zip(weights, self.task_names))
        return total, {k: w for k, w in zip(self.task_names, weights)}


# =============================================================================
# 4. 评估指标 (V6: 支持多点)
# =============================================================================
def compute_tracking_metrics_v6(
    pred_xy_px: torch.Tensor,
    gt_xy_px: torch.Tensor,
    image_size: int,
    reference_size: float = RMIT_REFERENCE_SIZE,
    reference_p10: float = RMIT_REFERENCE_P10,
) -> Dict[str, float]:
    """
    多点 tracking 评估。

    Args:
        pred_xy_px: [N, 2] 像素坐标
        gt_xy_px: [N, 2]
    """
    diff_px = pred_xy_px.float() - gt_xy_px.float()
    dist_px = torch.sqrt((diff_px ** 2).sum(dim=1))

    norm_scale = max(float(image_size - 1), 1.0)
    dist_norm = torch.sqrt(((diff_px / norm_scale) ** 2).sum(dim=1))
    reference_dist = dist_norm * reference_size
    scaled_p10_px = reference_p10 * float(image_size) / reference_size

    return {
        "mean_error_px": float(dist_px.mean().item()),
        "mean_error_ref": float(reference_dist.mean().item()),
        "precision@5px": float((dist_px <= 5.0).float().mean().item()),
        "precision@10px": float((dist_px <= 10.0).float().mean().item()),
        "precision@20px": float((dist_px <= 20.0).float().mean().item()),
        "precision@ref10": float((dist_px <= scaled_p10_px).float().mean().item()),
        "pck@10": float((dist_px < 10.0).float().mean().item()),
        "pck@20": float((dist_px < 20.0).float().mean().item()),
        "pck@30": float((dist_px < 30.0).float().mean().item()),
        "pck@50": float((dist_px < 50.0).float().mean().item()),
    }


# =============================================================================
# 5. 辅助函数 (保留)
# =============================================================================
def resolve_rmit_reference_protocol(img_size: int) -> Tuple[float, float]:
    if img_size >= 512:
        return 512.0, 20.0
    return 256.0, 10.0


def load_rmit_manifest_sequences(manifest_json: str) -> List[dict]:
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["sequences"] if isinstance(manifest, dict) and "sequences" in manifest else manifest


def resolve_loso_split(manifest_json: str, fold: int) -> Tuple[set, set, str]:
    sequences = load_rmit_manifest_sequences(manifest_json)
    seq_names = sorted([seq["name"] for seq in sequences])
    if len(seq_names) != 3:
        print(f"[V6] Warning: LOSO expects 3 sequences, found {len(seq_names)}")
    fold = fold % len(seq_names)
    test_seq_name = seq_names[fold]
    val_names = {test_seq_name}
    train_names = {name for name in seq_names if name != test_seq_name}
    return train_names, val_names, test_seq_name


def resolve_rmit_split_names(
    manifest_json: str, protocol: str, test_seq: Optional[str],
    val_ratio: float, seed: int, exclude_seqs: Optional[List[str]] = None,
) -> Tuple[set, set]:
    sequences = load_rmit_manifest_sequences(manifest_json)
    seq_names = [seq["name"] for seq in sequences]
    if exclude_seqs:
        seq_names = [n for n in seq_names if n not in set(exclude_seqs)]
    if protocol == "leave_one_out" and test_seq:
        val_names = {test_seq} if test_seq in seq_names else ({seq_names[0]} if seq_names else set())
        train_names = {n for n in seq_names if n not in val_names}
        return train_names, val_names
    rng = random.Random(seed)
    rng.shuffle(seq_names)
    n_val = max(1, int(round(len(seq_names) * val_ratio))) if len(seq_names) > 1 else 1
    val_names = set(seq_names[:n_val])
    train_names = {n for n in seq_names if n not in val_names}
    if not train_names and val_names:
        moved = next(iter(val_names))
        val_names.remove(moved)
        train_names.add(moved)
    return train_names, val_names


def warmup_cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# 6. 数据增强 V6 (4点 + 极低强度)
# =============================================================================
def augment_tracking_v6(
    template: Image.Image,
    search: Image.Image,
    gt_xy_all: torch.Tensor,  # [K, 2] 像素坐标
    is_training: bool = True,
    jitter_strength: float = 0.05,
    flip_prob: float = 0.0,       # 禁用翻转 (器械有方向性)
    rotate_prob: float = 0.1,
    rotate_range: float = 3.0,    # 3°
    scale_prob: float = 0.1,
    scale_range: float = 0.03,    # 3%
) -> Tuple[Image.Image, Image.Image, torch.Tensor]:
    """
    V6 数据增强: 4关键点同步变换，极低强度。
    """
    if not is_training:
        return template, search, gt_xy_all

    img_size = search.size[0]
    gt_xy_all = gt_xy_all.clone()

    # 颜色抖动 (不影响坐标)
    template = base.EnhancedAugmentation._jitter(template, s=jitter_strength)
    search = base.EnhancedAugmentation._jitter(search, s=jitter_strength)

    # 旋转
    if random.random() < rotate_prob:
        angle = random.uniform(-rotate_range, rotate_range)
        center = img_size / 2
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        template = template.rotate(angle, resample=Image.BILINEAR, expand=False)
        search = search.rotate(angle, resample=Image.BILINEAR, expand=False)
        for k in range(gt_xy_all.size(0)):
            x, y = gt_xy_all[k, 0].item() - center, gt_xy_all[k, 1].item() - center
            gt_xy_all[k, 0] = x * cos_a - y * sin_a + center
            gt_xy_all[k, 1] = x * sin_a + y * cos_a + center

    # 缩放
    if random.random() < scale_prob:
        scale = random.uniform(1.0 - scale_range, 1.0 + scale_range)
        new_size = int(img_size * scale)
        template_scaled = template.resize((new_size, new_size), Image.BILINEAR)
        search_scaled = search.resize((new_size, new_size), Image.BILINEAR)
        if scale > 1.0:
            start = (new_size - img_size) // 2
            template = template_scaled.crop((start, start, start + img_size, start + img_size))
            search = search_scaled.crop((start, start, start + img_size, start + img_size))
            for k in range(gt_xy_all.size(0)):
                gt_xy_all[k, 0] = gt_xy_all[k, 0] * scale - start
                gt_xy_all[k, 1] = gt_xy_all[k, 1] * scale - start
        else:
            pad = (img_size - new_size) // 2
            t_new = Image.new('RGB', (img_size, img_size), (0, 0, 0))
            s_new = Image.new('RGB', (img_size, img_size), (0, 0, 0))
            t_new.paste(template_scaled, (pad, pad))
            s_new.paste(search_scaled, (pad, pad))
            template, search = t_new, s_new
            for k in range(gt_xy_all.size(0)):
                gt_xy_all[k, 0] = gt_xy_all[k, 0] * scale + pad
                gt_xy_all[k, 1] = gt_xy_all[k, 1] * scale + pad

    # 翻转
    if random.random() < flip_prob:
        template = template.transpose(Image.FLIP_LEFT_RIGHT)
        search = search.transpose(Image.FLIP_LEFT_RIGHT)
        for k in range(gt_xy_all.size(0)):
            gt_xy_all[k, 0] = search.size[0] - 1 - gt_xy_all[k, 0]

    return template, search, gt_xy_all


# =============================================================================
# 7. Token Correlation (保留)
# =============================================================================
def token_correlation(
    template_tokens: torch.Tensor,
    search_tokens: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    t_norm = F.normalize(template_tokens, dim=-1)
    s_norm = F.normalize(search_tokens, dim=-1)
    corr = torch.bmm(s_norm, t_norm.transpose(1, 2))
    attn = F.softmax(corr / temperature, dim=-1)
    matched = torch.bmm(attn, template_tokens)
    return search_tokens + matched


# =============================================================================
# 8. MultiKeypointOffsetHead (4点 offset)
# =============================================================================
class MultiKeypointOffsetHead(nn.Module):
    """
    多关键点 offset head: 共享特征提取，独立预测每个关键点的 offset。

    改进 (V6):
    - scale=0.2 (修正范围 ±102px@512)
    - weight std=0.02 (更快的初始学习)
    - 去掉 L2 正则化
    """
    def __init__(self, feat_dim: int, num_keypoints: int = 4, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.num_keypoints = num_keypoints
        # 共享特征提取
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 每个关键点独立的 offset 预测
        self.offset_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
                nn.Tanh(),
            )
            for _ in range(num_keypoints)
        ])
        self.scale = nn.Parameter(torch.tensor(0.2))

        # 初始化
        for m in self.shared.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for head in self.offset_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, feat_map: torch.Tensor, coarse_xy_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: [B, C, H, W]
            coarse_xy_norm: [B, K, 2] 粗坐标 (归一化)

        Returns:
            offset_norm: [B, K, 2]
        """
        bsz, C, H, W = feat_map.shape
        K = coarse_xy_norm.size(1)

        offsets = []
        for k in range(K):
            grid = coarse_xy_norm[:, k] * 2 - 1  # [B, 2] → [-1, 1]
            grid = grid.view(bsz, 1, 1, 2)
            sampled = F.grid_sample(feat_map, grid, mode='bilinear', padding_mode='border', align_corners=True)
            sampled = sampled.view(bsz, -1)
            shared_feat = self.shared(sampled)
            offset = self.offset_heads[k](shared_feat) * self.scale
            offsets.append(offset)

        return torch.stack(offsets, dim=1)  # [B, K, 2]


# =============================================================================
# 9. RMITTrackingDatasetV5 (4关键点)
# =============================================================================
class RMITTrackingDatasetV5(Dataset):
    """
    RMIT Tracking Dataset V5 — 4关键点版本。

    CSV 格式: frame, x0, y0, x1, y1, x2, y2, x3, y3
    (0=LeftTip, 1=RightTip, 2=Junction, 3=Midpoint)
    """
    def __init__(
        self,
        manifest_json: str,
        img_size: int,
        is_training: bool,
        max_pairs_per_seq: Optional[int],
        split_names: set,
        max_frame_gap: int = 1,
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.items: List[dict] = []

        sequences = load_rmit_manifest_sequences(manifest_json)
        selected = [seq for seq in sequences if seq["name"] in split_names]

        for seq in selected:
            seq_name = seq["name"]
            records = self._load_sequence(seq["frames_dir"], seq["ann_csv"])
            pairs = []
            for gap in range(1, max_frame_gap + 1):
                for i in range(gap, len(records)):
                    prev_rec, curr_rec = records[i - gap], records[i]
                    pairs.append({
                        "template_path": prev_rec["path"],
                        "search_path": curr_rec["path"],
                        "search_xy_all": curr_rec["xy_all"],  # [K, 2]
                        "seq_name": seq_name,
                        "frame_idx": i,
                    })
            if max_pairs_per_seq is not None and is_training:
                random.shuffle(pairs)
                pairs = pairs[:max_pairs_per_seq]
            self.items.extend(pairs)

    def _load_sequence(self, frames_dir: str, ann_csv: str) -> List[dict]:
        """加载 4 点 CSV 标注。"""
        records = []
        with open(ann_csv, "r", encoding="utf-8") as f:
            reader = base.csv.DictReader(f)
            for row in reader:
                frame_key = base._find_col(row, ["frame", "frame_id", "image", "image_name", "filename", "file"])
                if frame_key is None:
                    continue
                # 尝试 4 点格式
                xy_all = []
                for k in range(NUM_KEYPOINTS):
                    xk = base._find_col(row, [f"x{k}", f"tip_x{k}", f"cx{k}"])
                    yk = base._find_col(row, [f"y{k}", f"tip_y{k}", f"cy{k}"])
                    if xk is not None and yk is not None:
                        xy_all.append([float(row[xk]), float(row[yk])])
                # 退化: 单点格式
                if not xy_all:
                    x_key = base._find_col(row, ["x", "tip_x", "cx"])
                    y_key = base._find_col(row, ["y", "tip_y", "cy"])
                    if x_key is not None and y_key is not None:
                        xy_all = [[float(row[x_key]), float(row[y_key])]]

                if not xy_all:
                    continue

                frame_name = row[frame_key]
                path = frame_name if os.path.isabs(frame_name) else os.path.join(frames_dir, frame_name)
                if not os.path.exists(path):
                    continue

                # 补齐到 NUM_KEYPOINTS 个点 (如果不足则复制最后一个)
                while len(xy_all) < NUM_KEYPOINTS:
                    xy_all.append(xy_all[-1][:])

                records.append({
                    "path": path,
                    "xy_all": torch.tensor(xy_all[:NUM_KEYPOINTS], dtype=torch.float32),  # [K, 2]
                })
        records.sort(key=lambda r: os.path.basename(r["path"]))
        return records

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        template_raw = base.load_image_rgb(item["template_path"])
        search_raw = base.load_image_rgb(item["search_path"])
        gt_xy_all = item["search_xy_all"].clone()  # [K, 2]

        # 缩放
        scale_x = self.img_size / max(search_raw.size[0], 1)
        scale_y = self.img_size / max(search_raw.size[1], 1)
        gt_xy_px_all = torch.zeros(NUM_KEYPOINTS, 2, dtype=torch.float32)
        for k in range(NUM_KEYPOINTS):
            gt_xy_px_all[k, 0] = gt_xy_all[k, 0] * scale_x
            gt_xy_px_all[k, 1] = gt_xy_all[k, 1] * scale_y

        template = base.pil_resize(template_raw, self.img_size, False)
        search = base.pil_resize(search_raw, self.img_size, False)
        search_for_vis = search.copy() if not self.is_training else None

        # 增强
        if self.is_training:
            template, search, gt_xy_px_all = augment_tracking_v6(
                template, search, gt_xy_px_all, True
            )

        gt_xy_px_all = gt_xy_px_all.clamp(0, self.img_size - 1)
        gt_xy_norm_all = gt_xy_px_all / max(float(self.img_size - 1), 1.0)

        # 生成 4 通道 heatmap
        sigma = max(10.0, self.img_size / 20.0)
        gt_hm_all = torch.zeros(NUM_KEYPOINTS, self.img_size, self.img_size, dtype=torch.float32)
        for k in range(NUM_KEYPOINTS):
            gt_hm_all[k] = base._make_gaussian(
                self.img_size, gt_xy_px_all[k, 0].item(), gt_xy_px_all[k, 1].item(), sigma=sigma
            ).squeeze(0)

        result = {
            "template": base.normalize_img(base.pil_to_tensor(template)),
            "search": base.normalize_img(base.pil_to_tensor(search)),
            "gt_xy_px_all": gt_xy_px_all,      # [K, 2]
            "gt_xy_norm_all": gt_xy_norm_all,   # [K, 2]
            "gt_hm_all": gt_hm_all,             # [K, H, W]
            "seq_name": item["seq_name"],
            "frame_idx": item["frame_idx"],
        }

        if search_for_vis is not None:
            result["search_for_vis"] = base.pil_to_tensor(search_for_vis)

        return result


# =============================================================================
# 10. OneStreamTipTrackerV5 (多关键点)
# =============================================================================
class OneStreamTipTrackerV5(nn.Module):
    """
    V5 多关键点跟踪器:
    - 4 通道 heatmap head
    - MultiKeypointOffsetHead
    - soft_argmax_from_prob 替代 logit-space soft-argmax
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        encoder: nn.Module,
        embed_dim: int,
        feature_layers: List[int],
        num_keypoints: int = NUM_KEYPOINTS,
        use_token_correlation: bool = True,
        fpn_out_dim: int = 256,
        offset_dropout: float = 0.2,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = encoder
        self.feature_layers = feature_layers
        self.num_keypoints = num_keypoints
        self.use_token_correlation = use_token_correlation

        self.search_grid = img_size // patch_size
        self.num_search = self.search_grid ** 2
        self.num_template = self.num_search

        self.pos_ts = nn.Parameter(torch.zeros(1, 1 + self.num_template + self.num_search, embed_dim))
        nn.init.trunc_normal_(self.pos_ts, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)

        self.fuser = base.MultiScaleTokenFuser(embed_dim, fpn_out_dim, len(feature_layers))

        # 多关键点 heatmap head (K 通道)
        self.hm_head = nn.Sequential(
            nn.Conv2d(fpn_out_dim, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, num_keypoints, 1),
        )

        # 多关键点 offset head
        self.offset_head = MultiKeypointOffsetHead(
            fpn_out_dim, num_keypoints=num_keypoints, hidden_dim=128, dropout=offset_dropout
        )

    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict:
                heatmap_logits: [B, K, H, W]
                coarse_xy_norm: [B, K, 2]
                offset_norm: [B, K, 2]
                xy_norm: [B, K, 2]
        """
        self.encoder.set_domain("surg")
        bsz = template.size(0)

        template_tokens = self.encoder.patch(template)
        search_tokens = self.encoder.patch(search)
        cls_token = self.encoder.cls.expand(bsz, -1, -1)

        hidden = torch.cat([cls_token, template_tokens, search_tokens], dim=1)
        hidden = hidden + self.pos_ts[:, :hidden.size(1)]
        hidden = self.encoder.drop(hidden)

        selected_template: Dict[int, torch.Tensor] = {}
        selected_search: Dict[int, torch.Tensor] = {}
        target = set(self.feature_layers)

        for idx, block in enumerate(self.encoder.blocks):
            hidden = block(hidden)
            if idx in target:
                normalized = self.norm(hidden)
                selected_template[idx] = normalized[:, 1:1+self.num_template]
                selected_search[idx] = normalized[:, 1+self.num_template:1+self.num_template+self.num_search]

        if self.use_token_correlation:
            search_multi = []
            for layer_idx in self.feature_layers:
                enhanced = token_correlation(selected_template[layer_idx], selected_search[layer_idx])
                search_multi.append(enhanced)
        else:
            search_multi = [selected_search[idx] for idx in self.feature_layers]

        feat = self.fuser(search_multi, self.search_grid)

        # Heatmap: [B, K, H', W'] → upsample → [B, K, H, W]
        heatmap_logits = self.hm_head(feat)
        heatmap_logits = F.interpolate(
            heatmap_logits, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )

        # 坐标提取: soft_argmax_from_prob
        coarse_xy_px = soft_argmax_from_prob(heatmap_logits, beta=5.0)  # [B, K, 2]
        coarse_xy_norm = coarse_xy_px / max(float(self.img_size - 1), 1.0)

        # Offset 精炼
        offset_norm = self.offset_head(feat, coarse_xy_norm)  # [B, K, 2]

        xy_norm = (coarse_xy_norm + offset_norm).clamp(0.0, 1.0)

        return {
            "heatmap_logits": heatmap_logits,
            "coarse_xy_norm": coarse_xy_norm,
            "offset_norm": offset_norm,
            "xy_norm": xy_norm,
        }


# =============================================================================
# 11. 评估函数 V6
# =============================================================================
@torch.no_grad()
def eval_rmit_v6(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    primary_keypoint: int = PRIMARY_KEYPOINT,
    reference_size: float = RMIT_REFERENCE_SIZE,
    reference_p10: float = RMIT_REFERENCE_P10,
    vis_dir: Optional[str] = None,
    epoch: int = -1,
    num_vis_per_seq: int = 5,
) -> Dict[str, float]:
    """V6 评估: 逐关键点统计 + 主关键点综合指标。"""
    model.eval()
    tracker = model["tracker"]

    all_preds = []   # [N_total, 2]  主关键点
    all_gts = []
    all_preds_all = []  # [N_total, K, 2] 全部关键点
    all_gts_all = []
    coarse_preds = []

    seq_vis_count = {}

    for batch in loader:
        template = batch["template"].to(device)
        search = batch["search"].to(device)
        gt_xy_px_all = batch["gt_xy_px_all"].to(device)  # [B, K, 2]
        seq_names = batch["seq_name"]
        frame_idxs = batch["frame_idx"]

        out = tracker(template, search)

        pred_xy_px_all = out["xy_norm"] * max(float(image_size - 1), 1.0)  # [B, K, 2]
        coarse_px = out["coarse_xy_norm"] * max(float(image_size - 1), 1.0)

        # 主关键点
        pk = primary_keypoint
        pred_pk = pred_xy_px_all[:, pk]  # [B, 2]
        gt_pk = gt_xy_px_all[:, pk]
        coarse_pk = coarse_px[:, pk]

        all_preds.append(pred_pk.cpu())
        all_gts.append(gt_pk.cpu())
        all_preds_all.append(pred_xy_px_all.cpu())
        all_gts_all.append(gt_xy_px_all.cpu())
        coarse_preds.append(coarse_pk.cpu())

    if not all_preds:
        return {k: 0.0 for k in [
            "mean_error_px", "mean_error_ref", "precision@5px", "precision@10px",
            "precision@20px", "precision@ref10", "pck@10", "pck@20", "pck@30", "pck@50",
            "coarse_mean_error_px",
        ]}

    preds_cat = torch.cat(all_preds, dim=0)
    gts_cat = torch.cat(all_gts, dim=0)
    coarse_cat = torch.cat(coarse_preds, dim=0)

    metrics = compute_tracking_metrics_v6(preds_cat, gts_cat, image_size, reference_size, reference_p10)

    coarse_dist = torch.sqrt(((coarse_cat - gts_cat) ** 2).sum(dim=1))
    metrics["coarse_mean_error_px"] = float(coarse_dist.mean().item())

    # 逐关键点误差
    preds_all_cat = torch.cat(all_preds_all, dim=0)  # [N, K, 2]
    gts_all_cat = torch.cat(all_gts_all, dim=0)
    for k in range(NUM_KEYPOINTS):
        d = torch.sqrt(((preds_all_cat[:, k] - gts_all_cat[:, k]) ** 2).sum(dim=1))
        metrics[f"kp{k}_mean_error_px"] = float(d.mean().item())
        metrics[f"kp{k}_pck@20"] = float((d < 20.0).float().mean().item())

    metrics[f"primary_keypoint"] = primary_keypoint
    return metrics


# =============================================================================
# 12. 训练函数 V6
# =============================================================================
def train_stage_rmit_v6(cfg, device: torch.device) -> None:
    """RMIT V6 多关键点训练。"""
    base.ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)

    enc = base.build_encoder(cfg).to(device)

    tracker = OneStreamTipTrackerV5(
        cfg.img_size, cfg.patch_size, enc, cfg.embed_dim, cfg.feature_layers,
        num_keypoints=NUM_KEYPOINTS,
        use_token_correlation=cfg.use_token_correlation,
        fpn_out_dim=cfg.fpn_out_dim,
        offset_dropout=cfg.offset_dropout,
    ).to(device)

    model = nn.ModuleDict({"enc": enc, "tracker": tracker})

    # 加载预训练权重
    if cfg.pretrained_encoder_ckpt:
        base.load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)

    if cfg.init_ckpt:
        skip_s2 = getattr(cfg, "skip_stage2_backbone", False)
        if skip_s2:
            # 跳过 Stage2 backbone: 只加载 surg LoRA 参数 (如果有)
            print(f"[V6] Loading init ckpt (skip_stage2_backbone=True, surg LoRA only): {cfg.init_ckpt}")
            state = torch.load(cfg.init_ckpt, map_location='cpu')
            if 'model' in state:
                surg_lora_state = {}
                for k, v in state['model'].items():
                    if k.startswith('enc.') and ('.A.surg' in k or '.B.surg' in k or '.M.surg' in k):
                        surg_lora_state[k.replace('enc.', '', 1)] = v
                if surg_lora_state:
                    missing, unexpected = enc.load_state_dict(surg_lora_state, strict=False)
                    print(f"[V6] Loaded {len(surg_lora_state)} surg LoRA params")
                else:
                    print("[V6] No surg LoRA params found in ckpt, using fresh init")
            else:
                print("[V6] No 'model' key in ckpt, skipping")
        else:
            # 正常加载: 共享 backbone + 非 surg LoRA
            print(f"[V6] Loading init ckpt (shared backbone, skip surg LoRA): {cfg.init_ckpt}")
            state = torch.load(cfg.init_ckpt, map_location='cpu')
            if 'model' in state:
                encoder_state = {}
                for k, v in state['model'].items():
                    if k.startswith('enc.'):
                        new_k = k.replace('enc.', '', 1)
                        if '.A.surg' in new_k or '.B.surg' in new_k or '.M.surg' in new_k:
                            continue
                        encoder_state[new_k] = v
                missing, unexpected = enc.load_state_dict(encoder_state, strict=False)
                print(f"[V6] Loaded {len(encoder_state)} shared params, missing={len(missing)}, unexpected={len(unexpected)}")
            else:
                base.load_ckpt(cfg.init_ckpt, model, optim=None)

    # 数据集划分
    split_mode = getattr(cfg, "split_mode", "loso")
    loso_fold = getattr(cfg, "loso_fold", 0)

    if split_mode == "loso":
        train_names, val_names, test_seq_name = resolve_loso_split(cfg.rmit_manifest_json, loso_fold)
        print(f"[V6] LOSO Fold {loso_fold}: train={sorted(train_names)}, test={test_seq_name}")
    else:
        train_names, val_names = resolve_rmit_split_names(
            cfg.rmit_manifest_json, cfg.rmit_protocol, cfg.rmit_test_seq,
            cfg.rmit_val_ratio, cfg.seed, getattr(cfg, "exclude_seqs", None),
        )
        print(f"[V6] Custom: train={sorted(train_names)}, val={sorted(val_names)}")

    train_ds = RMITTrackingDatasetV5(
        cfg.rmit_manifest_json, cfg.img_size, True,
        cfg.max_pairs_per_seq, train_names, cfg.max_frame_gap,
    )
    val_ds = RMITTrackingDatasetV5(
        cfg.rmit_manifest_json, cfg.img_size, False,
        None, val_names, 1,
    )

    train_ld = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    print(f"[V6] Train: {len(train_ds)}, Val: {len(val_ds)}")
    print(f"[V6] Token correlation: {cfg.use_token_correlation}")

    # 参数分组
    params = []
    for name, param in model.named_parameters():
        is_adapter = ("A." in name) or ("B." in name) or ("M." in name)
        is_tracker = name.startswith("tracker.")

        if cfg.unfreeze_all:
            param.requires_grad = True
            params.append(param)
        elif cfg.freeze_backbone:
            if is_adapter or is_tracker:
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False
        else:
            param.requires_grad = True
            params.append(param)

    trainable_count = sum(p.numel() for p in params)
    total_count = sum(p.numel() for p in model.parameters())
    mode_str = "Full FT" if cfg.unfreeze_all else ("LoRA+Tracker" if cfg.freeze_backbone else "All")
    print(f"[V6] Mode: {mode_str} | Trainable: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.1f}%)")

    rmit_lr = getattr(cfg, "rmit_lr", None) or cfg.lr
    print(f"[V6] LR: {rmit_lr}")

    optim = torch.optim.AdamW(params, lr=rmit_lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))
    ema = base.ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    if cfg.use_dwa:
        dwa = DynamicWeightAverage(["hm", "xy", "geo"], temperature=cfg.dwa_temperature)
    else:
        dwa = None

    best = 1e9
    patience_counter = 0
    global_step = 0
    steps_per_epoch = max(1, len(train_ld)) // cfg.grad_accum
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    coord_ref = max(float(getattr(cfg, "rmit_coord_ref_size", RMIT_REFERENCE_SIZE)), 1.0)
    ref_size, ref_p10 = resolve_rmit_reference_protocol(cfg.img_size)
    grad_clip = 5.0
    lambda_geo = getattr(cfg, "lambda_geo", 1.0)
    primary_kp = getattr(cfg, "primary_keypoint", PRIMARY_KEYPOINT)

    vis_dir = os.path.join(cfg.out_dir, "vis")
    base.ensure_dir(vis_dir)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        running_hm = 0.0
        running_xy = 0.0
        running_geo = 0.0
        optim.zero_grad(set_to_none=True)

        iterator = enumerate(train_ld)
        if base.tqdm is not None:
            iterator = base.tqdm(iterator, total=len(train_ld), desc=f"V6 {epoch+1}/{cfg.epochs}", leave=False)

        for it, batch in iterator:
            template = batch["template"].to(device, non_blocking=True)
            search = batch["search"].to(device, non_blocking=True)
            gt_hm_all = batch["gt_hm_all"].to(device, non_blocking=True)         # [B, K, H, W]
            gt_xy_px_all = batch["gt_xy_px_all"].to(device, non_blocking=True)   # [B, K, 2]
            gt_xy_norm_all = batch["gt_xy_norm_all"].to(device, non_blocking=True)  # [B, K, 2]

            lr_now = warmup_cosine_lr(rmit_lr, global_step, total_steps, warmup_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
                out = tracker(template, search)

                # --- Heatmap loss (pure MSE, 4 通道) ---
                loss_hm = pure_mse_heatmap_loss(out["heatmap_logits"], gt_hm_all)

                # --- XY loss (wing loss, 4 关键点) ---
                pred_xy_ref = out["xy_norm"] * coord_ref          # [B, K, 2]
                gt_xy_ref = gt_xy_norm_all * coord_ref
                loss_xy = wing_loss(pred_xy_ref, gt_xy_ref)

                # --- Coarse XY loss ---
                pred_coarse_ref = out["coarse_xy_norm"] * coord_ref
                loss_xy_coarse = wing_loss(pred_coarse_ref, gt_xy_ref)
                loss_xy = loss_xy + loss_xy_coarse

                # --- Geometric consistency loss ---
                pred_xy_px = out["xy_norm"] * max(float(cfg.img_size - 1), 1.0)
                loss_geo = geometric_consistency_loss(pred_xy_px, gt_xy_px_all)

                # --- 组合 ---
                if dwa is not None:
                    total_loss, w_dict = dwa({"hm": loss_hm, "xy": loss_xy, "geo": loss_geo * lambda_geo})
                else:
                    w_dict = {"hm": cfg.lambda_hm, "xy": cfg.lambda_xy, "geo": lambda_geo}
                    total_loss = cfg.lambda_hm * loss_hm + cfg.lambda_xy * loss_xy + lambda_geo * loss_geo

                loss = total_loss / max(1, cfg.grad_accum)

            # Debug log
            if global_step % 20 == 0:
                print(f"[V6-DBG] step={global_step} | hm={loss_hm.item():.4f} | "
                      f"xy={loss_xy.item():.4f} | geo={loss_geo.item():.4f} | "
                      f"off_scale={tracker.offset_head.scale.item():.4f} | lr={lr_now:.2e}")

            if not torch.isfinite(loss.detach()):
                print(f"[V6] skip non-finite loss at epoch={epoch+1} iter={it+1}")
                optim.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            running += loss.item() * max(1, cfg.grad_accum)
            running_hm += loss_hm.item()
            running_xy += loss_xy.item()
            running_geo += loss_geo.item()

            if (it + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if global_step % 20 == 0:
                    writer.add_scalar("v6/train_loss", loss.item() * max(1, cfg.grad_accum), global_step)
                    writer.add_scalar("v6/loss_hm", loss_hm.item(), global_step)
                    writer.add_scalar("v6/loss_xy", loss_xy.item(), global_step)
                    writer.add_scalar("v6/loss_geo", loss_geo.item(), global_step)
                    writer.add_scalar("v6/lr", lr_now, global_step)
                global_step += 1

        # Handle remaining gradients
        if len(train_ld) % cfg.grad_accum != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1

        # Validation
        metrics = eval_rmit_v6(
            model, val_ld, device, cfg.img_size, primary_kp, ref_size, ref_p10,
            vis_dir=vis_dir, epoch=epoch, num_vis_per_seq=cfg.num_vis_per_seq,
        )
        ema_metrics = eval_rmit_v6(
            ema.module, val_ld, device, cfg.img_size, primary_kp, ref_size, ref_p10,
        ) if ema is not None else None

        # Logging
        for key in ["mean_error_px", "mean_error_ref", "pck@10", "pck@20", "pck@30", "pck@50"]:
            writer.add_scalar(f"v6/val_{key}", metrics.get(key, 0.0), epoch)
        writer.add_scalar("v6/val_coarse_err", metrics.get("coarse_mean_error_px", 0.0), epoch)
        for k in range(NUM_KEYPOINTS):
            writer.add_scalar(f"v6/val_kp{k}_err", metrics.get(f"kp{k}_mean_error_px", 0.0), epoch)
            writer.add_scalar(f"v6/val_kp{k}_pck20", metrics.get(f"kp{k}_pck@20", 0.0), epoch)
        if ema_metrics:
            writer.add_scalar("v6/ema_mean_error_px", ema_metrics.get("mean_error_px", 0.0), epoch)
            writer.add_scalar("v6/ema_pck@20", ema_metrics.get("pck@20", 0.0), epoch)

        msg = (
            f"[V6] epoch {epoch+1}/{cfg.epochs} "
            f"loss={running/max(len(train_ld),1):.4f} "
            f"hm={running_hm/max(len(train_ld),1):.4f} "
            f"xy={running_xy/max(len(train_ld),1):.4f} "
            f"geo={running_geo/max(len(train_ld),1):.4f} "
            f"val_err={metrics['mean_error_px']:.2f}px "
            f"coarse={metrics['coarse_mean_error_px']:.2f}px "
            f"pck@10={metrics['pck@10']:.3f} pck@20={metrics['pck@20']:.3f} pck@50={metrics['pck@50']:.3f}"
        )
        for k in range(NUM_KEYPOINTS):
            msg += f" kp{k}={metrics.get(f'kp{k}_mean_error_px', 0):.1f}"
        if ema_metrics:
            msg += f" | EMA err={ema_metrics['mean_error_px']:.2f} pck@20={ema_metrics['pck@20']:.3f}"
        print(msg)
        base.append_epoch_log(cfg.out_dir, msg)

        # Early stopping
        metric = ema_metrics["mean_error_ref"] if ema_metrics else metrics["mean_error_ref"]
        if metric < best:
            best = metric
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg.patience:
            print(f"[V6] Early stopping at epoch {epoch+1}")
            break

        # Save
        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "ema": ema.state_dict() if ema else None,
            "extra": {"stage": "rmit_v6_multi_kp", "reference_size": ref_size},
        }
        base.save_ckpt(f"{cfg.out_dir}/last.pt", payload)
        if metric <= best:
            payload["best_metric"] = best
            base.save_ckpt(f"{cfg.out_dir}/best.pt", payload)

    writer.close()


# =============================================================================
# 13. 命令行参数
# =============================================================================
class TrainCfgV6:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def parse_args() -> TrainCfgV6:
    parser = argparse.ArgumentParser(description="DR-SurgBridgeNet V6 Multi-Keypoint Tracking")

    # Stage / output
    parser.add_argument("--stage", type=str, required=True, choices=["dr", "fovea", "rmit"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--pretrained_encoder_ckpt", type=str, default=None)

    # Architecture
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--attn_drop", type=float, default=0.0)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_rank_strategy", type=str, default="late_heavy")
    parser.add_argument("--lora_rank_min", type=int, default=0)
    parser.add_argument("--lora_rank_spec", type=str, default=None)
    parser.add_argument("--use_residual_lora", action="store_true")
    parser.add_argument("--residual_alpha", type=float, default=0.1)

    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze_all", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # DR / FOVEA (保持兼容)
    parser.add_argument("--dr_imagefolder_dir", type=str, default=None)
    parser.add_argument("--dr_train_csv", type=str, default=None)
    parser.add_argument("--dr_val_csv", type=str, default=None)
    parser.add_argument("--dr_val_ratio", type=float, default=0.2)
    parser.add_argument("--dr_num_classes", type=int, default=2)
    parser.add_argument("--fovea_pairs_json", type=str, default=None)
    parser.add_argument("--fovea_val_ratio", type=float, default=0.2)
    parser.add_argument("--fovea_temperature", type=float, default=0.07)
    parser.add_argument("--fovea_embed_dim", type=int, default=128)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_seg", type=float, default=2.0)

    # RMIT
    parser.add_argument("--rmit_manifest_json", type=str, default=None)
    parser.add_argument("--rmit_protocol", type=str, default="leave_one_out")
    parser.add_argument("--rmit_test_seq", type=str, default=None)
    parser.add_argument("--rmit_val_ratio", type=float, default=0.2)
    parser.add_argument("--rmit_lr", type=float, default=3e-5)
    parser.add_argument("--max_pairs_per_seq", type=int, default=None)
    parser.add_argument("--eval_metrics", action="store_true")
    parser.add_argument("--exclude_seqs", type=str, nargs="+", default=["seq2"])

    # LOSO
    parser.add_argument("--split_mode", type=str, default="loso", choices=["loso", "custom"])
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2])

    # V6 新增
    parser.add_argument("--skip_stage2_backbone", action="store_true",
                        help="跳过 Stage2 FOVEA 污染的 backbone 权重")
    parser.add_argument("--lambda_geo", type=float, default=1.0,
                        help="几何一致性损失权重")
    parser.add_argument("--primary_keypoint", type=int, default=PRIMARY_KEYPOINT,
                        help="主关键点索引 (默认 0=LeftTip)")

    # Feature / loss
    parser.add_argument("--feature_layers", type=str, default="2,5,7")
    parser.add_argument("--align_layer_weights", type=str, default=None)
    parser.add_argument("--fpn_out_dim", type=int, default=256)
    parser.add_argument("--use_token_correlation", action="store_true")
    parser.add_argument("--disable_token_correlation", action="store_true")
    parser.add_argument("--use_dwa", action="store_true")
    parser.add_argument("--dwa_temperature", type=float, default=2.0)
    parser.add_argument("--lambda_hm", type=float, default=1.0)
    parser.add_argument("--lambda_xy", type=float, default=1.0)
    parser.add_argument("--use_dora", action="store_true", default=True)
    parser.add_argument("--disable_dora", action="store_true")

    # 其他
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--max_frame_gap", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--offset_dropout", type=float, default=0.2)
    parser.add_argument("--num_vis_per_seq", type=int, default=5)
    parser.add_argument("--lora_last_n_layers", type=int, default=0)
    parser.add_argument("--use_task_uncertainty", action="store_true")
    parser.add_argument("--disable_task_uncertainty", action="store_true")
    parser.add_argument("--contrast_temperature", type=float, default=0.07)
    parser.add_argument("--proj_dim", type=int, default=256)

    args = parser.parse_args()

    feature_layers = base.parse_int_list(args.feature_layers, args.depth)
    align_layer_weights = base.resolve_layer_weights(len(feature_layers), args.align_layer_weights)

    if args.unfreeze_all:
        lora_layer_ranks = [0] * args.depth
    elif args.lora_last_n_layers > 0:
        lora_layer_ranks = [0] * (args.depth - args.lora_last_n_layers) + [args.lora_r] * args.lora_last_n_layers
    else:
        lora_layer_ranks = base.resolve_lora_layer_ranks(
            args.depth, args.lora_r, args.lora_rank_strategy, args.lora_rank_min, args.lora_rank_spec,
        )

    coord_ref, _ = resolve_rmit_reference_protocol(args.img_size)
    use_token_correlation = not args.disable_token_correlation

    return TrainCfgV6(
        stage=args.stage, out_dir=args.out_dir,
        init_ckpt=args.init_ckpt, pretrained_encoder_ckpt=args.pretrained_encoder_ckpt,
        img_size=args.img_size, patch_size=args.patch_size, embed_dim=args.embed_dim,
        depth=args.depth, heads=args.heads, mlp_ratio=args.mlp_ratio, drop=args.drop, attn_drop=args.attn_drop,
        lora_r=args.lora_r, lora_rank_strategy=args.lora_rank_strategy,
        lora_rank_min=args.lora_rank_min, lora_rank_spec=args.lora_rank_spec,
        lora_layer_ranks=lora_layer_ranks,
        use_residual_lora=args.use_residual_lora, residual_alpha=args.residual_alpha,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, wd=args.wd,
        num_workers=args.num_workers, seed=args.seed, amp=args.amp,
        grad_accum=max(1, args.grad_accum), freeze_backbone=args.freeze_backbone,
        unfreeze_all=args.unfreeze_all,
        use_ema=args.use_ema, ema_decay=args.ema_decay,
        dr_imagefolder_dir=args.dr_imagefolder_dir, dr_train_csv=args.dr_train_csv,
        dr_val_csv=args.dr_val_csv, dr_val_ratio=args.dr_val_ratio, dr_num_classes=args.dr_num_classes,
        fovea_pairs_json=args.fovea_pairs_json, fovea_val_ratio=args.fovea_val_ratio,
        fovea_temperature=args.fovea_temperature, fovea_embed_dim=args.fovea_embed_dim,
        lambda_align=args.lambda_align, lambda_seg=args.lambda_seg,
        rmit_manifest_json=args.rmit_manifest_json, rmit_protocol=args.rmit_protocol,
        rmit_test_seq=args.rmit_test_seq, rmit_val_ratio=args.rmit_val_ratio,
        rmit_lr=args.rmit_lr, max_pairs_per_seq=args.max_pairs_per_seq,
        eval_metrics=args.eval_metrics, exclude_seqs=args.exclude_seqs,
        split_mode=args.split_mode, loso_fold=args.fold,
        feature_layers=feature_layers, align_layer_weights=align_layer_weights,
        fpn_out_dim=args.fpn_out_dim,
        use_token_correlation=use_token_correlation,
        use_dwa=args.use_dwa, dwa_temperature=args.dwa_temperature,
        lambda_hm=args.lambda_hm, lambda_xy=args.lambda_xy,
        use_dora=not args.disable_dora,
        use_task_uncertainty=not args.disable_task_uncertainty,
        warmup_epochs=args.warmup_epochs, max_frame_gap=args.max_frame_gap,
        patience=args.patience, offset_dropout=args.offset_dropout,
        num_vis_per_seq=args.num_vis_per_seq, lora_last_n_layers=args.lora_last_n_layers,
        contrast_temperature=args.contrast_temperature, proj_dim=args.proj_dim,
        # V6 new
        skip_stage2_backbone=args.skip_stage2_backbone,
        lambda_geo=args.lambda_geo,
        primary_keypoint=args.primary_keypoint,
        rmit_coord_ref_size=coord_ref,
    )


# =============================================================================
# 14. Main
# =============================================================================
def main() -> None:
    cfg = parse_args()
    base.seed_all(cfg.seed)
    base.ensure_dir(cfg.out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage: {cfg.stage}")
    print(f"LoRA layer ranks: {cfg.lora_layer_ranks}")

    if cfg.stage == "dr":
        base.train_stage_dr(cfg, device)
    elif cfg.stage == "fovea":
        if not cfg.fovea_pairs_json:
            raise ValueError("Need --fovea_pairs_json")
        base.train_stage_fovea_v2(cfg, device)
    elif cfg.stage == "rmit":
        if not cfg.rmit_manifest_json:
            raise ValueError("Need --rmit_manifest_json")
        train_stage_rmit_v6(cfg, device)
    else:
        raise ValueError(cfg.stage)


if __name__ == "__main__":
    main()
