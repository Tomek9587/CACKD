"""
dr_4.py — DR-SurgBridgeNet V4: RMIT Tip Tracking (重构版)

核心改进:
1. β-NLL heatmap loss (ICLR 2022) 防止 variance collapse
2. Token Correlation 替代 Cross-Attention (零额外参数)
3. DynamicWeightAverage 替代 GradNorm (对负loss鲁棒)
4. 简化 offset head + dropout 正则化
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

# 确保项目根目录在 PYTHONPATH 中，以便 import surgical_tracking.src.dr_2
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
# 常量定义
# =============================================================================
RMIT_REFERENCE_SIZE = 512.0
RMIT_REFERENCE_P10 = 20.0
RMIT_HEATMAP_POS_WEIGHT = 32.0
RMIT_HEATMAP_MSE_WEIGHT = 0.25


# =============================================================================
# 1. 损失函数
# =============================================================================
def stable_heatmap_loss_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = RMIT_HEATMAP_POS_WEIGHT,
    mse_weight: float = RMIT_HEATMAP_MSE_WEIGHT,
) -> torch.Tensor:
    """稳定的 BCE + MSE heatmap 损失（来自 dr_3，已验证稳定）"""
    target = target.clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)
    weights = 1.0 + pos_weight * target
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
    mse = F.mse_loss(probs, target)
    return weighted_bce + mse_weight * mse


def beta_nll_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    β-NLL loss (ICLR 2022) 防止 variance collapse。
    
    关键：detach precision 阻止 variance collapse。
    """
    pred = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    variance = torch.exp(log_variance).clamp(min=1e-6, max=100.0)
    diff_sq = (pred - target) ** 2
    # 关键：detach precision 阻止 variance collapse
    precision_weighted = (1.0 / variance).detach() ** beta
    loss = precision_weighted * diff_sq + 0.5 * log_variance
    return loss.mean()


def mse_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    简单 MSE heatmap loss。
    """
    pred = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    return F.mse_loss(pred, target)


def focal_heatmap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss 变体 for heatmap。
    
    focal_weight = (1 - pred*target - (1-pred)*(1-target))^gamma * MSE
    
    Args:
        logits: 模型输出的 logits
        target: 目标 heatmap
        gamma: focal weight 的指数，默认 2.0
    """
    pred = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    
    # focal weight: (1 - pred*target - (1-pred)*(1-target))^gamma
    # 简化: 对于正样本(target=1)，weight = (1-pred)^gamma
    #       对于负样本(target=0)，weight = pred^gamma
    focal_weight = (1.0 - pred * target - (1.0 - pred) * (1.0 - target)) ** gamma
    
    # MSE loss
    mse = (pred - target) ** 2
    
    # weighted MSE
    loss = focal_weight * mse
    return loss.mean()


# =============================================================================
# 2. DynamicWeightAverage (基于 loss 变化率的动态权重)
# =============================================================================
class DynamicWeightAverage:
    """
    基于 loss 变化率的动态权重，无可学习参数。
    
    参考: End-to-End Multi-Task Learning with Attention (CVPR 2019)
    """
    def __init__(self, task_names: List[str], temperature: float = 2.0):
        self.task_names = list(task_names)
        self.temperature = temperature
        self.prev_losses: Optional[List[float]] = None
    
    def __call__(self, losses_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        current = [losses_dict[k].detach().item() for k in self.task_names]
        
        if self.prev_losses is None:
            # 第一次调用，使用均匀权重
            weights = [1.0 / len(self.task_names)] * len(self.task_names)
        else:
            # 变化率 = L_t / L_{t-1}
            rates = [c / (p + 1e-8) for c, p in zip(current, self.prev_losses)]
            # softmax over rates/temperature
            exp_rates = [math.exp(r / self.temperature) for r in rates]
            s = sum(exp_rates)
            weights = [e / s for e in exp_rates]
        
        self.prev_losses = current
        
        # 计算加权总损失
        total = sum(w * losses_dict[k] for w, k in zip(weights, self.task_names))
        weights_dict = {k: w for k, w in zip(self.task_names, weights)}
        
        return total, weights_dict


# =============================================================================
# 3. 评估指标函数
# =============================================================================
def compute_tracking_metrics_v4(
    pred_xy_px: torch.Tensor,
    gt_xy_px: torch.Tensor,
    image_size: int,
    reference_size: float = RMIT_REFERENCE_SIZE,
    reference_p10: float = RMIT_REFERENCE_P10,
) -> Dict[str, float]:
    """计算 tracking 评估指标，支持 reference 参数对齐。"""
    diff_px = pred_xy_px.float() - gt_xy_px.float()
    dist_px = torch.sqrt((diff_px ** 2).sum(dim=1))
    
    norm_scale = max(float(image_size - 1), 1.0)
    diff_norm = diff_px / norm_scale
    dist_norm = torch.sqrt((diff_norm ** 2).sum(dim=1))
    diag_norm = max((2.0 ** 0.5), 1e-6)
    
    reference_dist = dist_norm * reference_size
    scaled_p10_px = reference_p10 * float(image_size) / reference_size
    
    return {
        "mse_px": float((diff_px ** 2).mean().item()),
        "mean_error_px": float(dist_px.mean().item()),
        "mean_error_ref": float(reference_dist.mean().item()),
        "mean_error_width_norm": float(dist_norm.mean().item()),
        "mean_error_diag_norm": float((dist_norm / diag_norm).mean().item()),
        "precision@5px": float((dist_px <= 5.0).float().mean().item()),
        "precision@10px": float((dist_px <= 10.0).float().mean().item()),
        "precision@20px": float((dist_px <= 20.0).float().mean().item()),
        "precision@ref10": float((dist_px <= scaled_p10_px).float().mean().item()),
        "failure_rate@20px": float((dist_px > 20.0).float().mean().item()),
    }


# =============================================================================
# 4. 辅助函数
# =============================================================================
def resolve_rmit_reference_protocol(img_size: int) -> Tuple[float, float]:
    """根据图像大小解析 reference 协议。"""
    if img_size >= 512:
        return 512.0, 20.0
    return 256.0, 10.0


def load_rmit_manifest_sequences(manifest_json: str) -> List[dict]:
    """加载 RMIT manifest 序列。"""
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["sequences"] if isinstance(manifest, dict) and "sequences" in manifest else manifest


def resolve_rmit_split_names(
    manifest_json: str,
    protocol: str,
    test_seq: Optional[str],
    val_ratio: float,
    seed: int,
    exclude_seqs: Optional[List[str]] = None,
) -> Tuple[set, set]:
    """解析 RMIT 数据集的训练/验证划分。
    
    Args:
        manifest_json: manifest 文件路径
        protocol: 划分协议 ("leave_one_out" 或 "random")
        test_seq: leave_one_out 协议下的测试序列
        val_ratio: 验证集比例
        seed: 随机种子
        exclude_seqs: 要排除的序列名列表
    """
    sequences = load_rmit_manifest_sequences(manifest_json)
    seq_names = [seq["name"] for seq in sequences]
    
    # 过滤掉排除的序列
    if exclude_seqs:
        exclude_set = set(exclude_seqs)
        seq_names = [name for name in seq_names if name not in exclude_set]
    
    if protocol == "leave_one_out" and test_seq:
        # 如果 test_seq 被排除，则使用第一个可用序列作为验证集
        if test_seq not in seq_names:
            val_names = {seq_names[0]} if seq_names else set()
        else:
            val_names = {test_seq}
        train_names = {name for name in seq_names if name not in val_names}
        return train_names, val_names
    
    rng = random.Random(seed)
    names = list(seq_names)
    rng.shuffle(names)
    n_val = max(1, int(round(len(names) * val_ratio))) if len(names) > 1 else 1
    n_val = min(n_val, max(1, len(names)))
    val_names = set(names[:n_val])
    train_names = {name for name in names if name not in val_names}
    
    if not train_names and val_names:
        moved = next(iter(val_names))
        val_names.remove(moved)
        train_names.add(moved)
    
    return train_names, val_names


def resolve_loso_split(
    manifest_json: str,
    fold: int,
) -> Tuple[set, set, str]:
    """
    标准 LOSO (Leave-One-Sequence-Out) 交叉验证划分。
    
    RMIT 数据集标准评估协议：用 2 个序列训练，1 个序列测试，3 折交叉验证取平均。
    
    Args:
        manifest_json: manifest 文件路径
        fold: 折数 (0, 1, 2)
            - fold 0: 测试 seq1，训练 seq2+seq3
            - fold 1: 测试 seq2，训练 seq1+seq3
            - fold 2: 测试 seq3，训练 seq1+seq2
    
    Returns:
        train_names: 训练序列集合
        val_names: 验证/测试序列集合
        test_seq_name: 测试序列名
    """
    sequences = load_rmit_manifest_sequences(manifest_json)
    seq_names = sorted([seq["name"] for seq in sequences])  # 排序确保一致性
    
    if len(seq_names) != 3:
        print(f"[RMIT-V4] Warning: LOSO expects 3 sequences, found {len(seq_names)}: {seq_names}")
    
    # 确保 fold 在有效范围内
    fold = fold % len(seq_names)
    
    # 测试序列为第 fold 个序列
    test_seq_name = seq_names[fold]
    val_names = {test_seq_name}
    train_names = {name for name in seq_names if name != test_seq_name}
    
    return train_names, val_names, test_seq_name


def augment_tracking_v4(
    template: Image.Image,
    search: Image.Image,
    gt_xy: torch.Tensor,
    is_training: bool = True,
    # 增强参数（可调整）
    jitter_strength: float = 0.2,
    flip_prob: float = 0.3,
    shift_prob: float = 0.3,
    shift_range: float = 5.0,
    rotate_prob: float = 0.2,
    rotate_range: float = 15.0,
    scale_prob: float = 0.2,
    scale_range: float = 0.1,
    blur_prob: float = 0.1,
    erase_prob: float = 0.1,
) -> Tuple[Image.Image, Image.Image, torch.Tensor]:
    """
    修复版 augment_tracking：使用 search.size 而非 template.size。
    
    增强的数据增强策略（缓解过拟合）：
    - 颜色抖动 (ColorJitter) - 增强强度
    - 水平翻转
    - 随机平移抖动
    - 随机旋转
    - 随机缩放
    - 高斯模糊 (GaussianBlur)
    - 随机擦除 (RandomErasing) - 在 tensor 上实现
    """
    if not is_training:
        return template, search, gt_xy
    
    img_size = search.size[0]  # 假设正方形
    
    # 颜色抖动 - 增加强度
    template = base.EnhancedAugmentation._jitter(template, s=jitter_strength)
    search = base.EnhancedAugmentation._jitter(search, s=jitter_strength)
    
    # 随机旋转
    if random.random() < rotate_prob:
        angle = random.uniform(-rotate_range, rotate_range)
        # 旋转时需要调整 gt_xy
        center = img_size / 2
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        # 对 template 和 search 应用相同的旋转
        template = template.rotate(angle, resample=Image.BILINEAR, expand=False)
        search = search.rotate(angle, resample=Image.BILINEAR, expand=False)
        
        # 旋转 gt_xy（相对于中心点）
        gt_xy = gt_xy.clone()
        x, y = gt_xy[0].item() - center, gt_xy[1].item() - center
        new_x = x * cos_a - y * sin_a + center
        new_y = x * sin_a + y * cos_a + center
        gt_xy[0] = new_x
        gt_xy[1] = new_y
    
    # 随机缩放
    if random.random() < scale_prob:
        scale = random.uniform(1.0 - scale_range, 1.0 + scale_range)
        new_size = int(img_size * scale)
        
        # 缩放图像
        template_scaled = template.resize((new_size, new_size), Image.BILINEAR)
        search_scaled = search.resize((new_size, new_size), Image.BILINEAR)
        
        # 裁剪或填充回原始尺寸
        if scale > 1.0:
            # 中心裁剪
            start = (new_size - img_size) // 2
            template = template_scaled.crop((start, start, start + img_size, start + img_size))
            search = search_scaled.crop((start, start, start + img_size, start + img_size))
        else:
            # 填充
            pad = (img_size - new_size) // 2
            template_new = Image.new('RGB', (img_size, img_size), (0, 0, 0))
            search_new = Image.new('RGB', (img_size, img_size), (0, 0, 0))
            template_new.paste(template_scaled, (pad, pad))
            search_new.paste(search_scaled, (pad, pad))
            template, search = template_new, search_new
        
        # 调整 gt_xy
        gt_xy = gt_xy.clone()
        if scale > 1.0:
            gt_xy[0] = gt_xy[0] * scale - start
            gt_xy[1] = gt_xy[1] * scale - start
        else:
            gt_xy[0] = gt_xy[0] * scale + pad
            gt_xy[1] = gt_xy[1] * scale + pad
    
    # 水平翻转 (修复: 使用 search.size)
    if random.random() < flip_prob:
        template = template.transpose(Image.FLIP_LEFT_RIGHT)
        search = search.transpose(Image.FLIP_LEFT_RIGHT)
        gt_xy = gt_xy.clone()
        gt_xy[0] = search.size[0] - 1 - gt_xy[0]  # 修复: 用 search.size
    
    # 随机微小平移抖动（增加小数据集鲁棒性）
    if random.random() < shift_prob:
        shift_px = random.uniform(-shift_range, shift_range)
        shift_py = random.uniform(-shift_range, shift_range)
        gt_xy = gt_xy.clone()
        gt_xy[0] = gt_xy[0] + shift_px
        gt_xy[1] = gt_xy[1] + shift_py
    
    # 高斯模糊
    if random.random() < blur_prob:
        from PIL import ImageFilter
        radius = random.uniform(0.5, 1.5)
        template = template.filter(ImageFilter.GaussianBlur(radius=radius))
        search = search.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # 随机擦除 - 在 tensor 上实现（返回时处理）
    # 注意：随机擦除在 dataset 的 __getitem__ 中实现，因为需要 tensor 操作
    
    return template, search, gt_xy


def warmup_cosine_lr(base_lr: float, step: int, total_steps: int, warmup_steps: int) -> float:
    """Warmup + Cosine 学习率调度。"""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# 5. Token Correlation（零参数）
# =============================================================================
def token_correlation(
    template_tokens: torch.Tensor,
    search_tokens: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    零参数的 template-search 相关性融合，借鉴 OSTrack/SiamFC。
    
    Args:
        template_tokens: [B, N_t, D] template tokens
        search_tokens: [B, N_s, D] search tokens
        temperature: softmax 温度（较小温度产生更尖锐的注意力分布）
    
    Returns:
        enhanced_search: [B, N_s, D] 增强后的 search tokens
    """
    t_norm = F.normalize(template_tokens, dim=-1)
    s_norm = F.normalize(search_tokens, dim=-1)
    
    # 计算相关性: [B, N_s, N_t]
    corr = torch.bmm(s_norm, t_norm.transpose(1, 2))
    attn = F.softmax(corr / temperature, dim=-1)
    
    # 使用 template 特征加权融合
    matched = torch.bmm(attn, template_tokens)  # [B, N_s, D]
    
    # residual fusion
    return search_tokens + matched


# =============================================================================
# 6. 简化的 LightOffsetHead
# =============================================================================
class LightOffsetHead(nn.Module):
    """
    轻量级 offset 精炼，参数量 << 原 SpatialAwareOffsetHead。
    
    关键设计点:
    - 只在粗坐标位置采样单个点特征（而非 ws*ws 窗口）
    - scale 初始化为 0.1 → offset 初始接近 0
    - 添加 dropout 防止过拟合
    """
    def __init__(self, feat_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Tanh(),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))  # 初始化小 offset
        
        # 初始化接近零
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, feat_map: torch.Tensor, coarse_xy_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: [B, C, H, W]
            coarse_xy_norm: [B, 2] in [0, 1]
        
        Returns:
            offset_norm: [B, 2] offset in normalized space
        """
        bsz = feat_map.size(0)
        h, w = feat_map.shape[2:]
        
        # 将坐标转为 grid_sample 格式 [-1, 1]
        grid = coarse_xy_norm * 2 - 1  # [B, 2]
        grid = grid.view(bsz, 1, 1, 2)  # [B, 1, 1, 2]
        
        # 在 coarse 位置采样特征
        sampled = F.grid_sample(
            feat_map, grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        sampled = sampled.view(bsz, -1)  # [B, C]
        
        # 预测 offset
        offset = self.mlp(sampled) * self.scale  # [B, 2]
        
        # offset 在归一化空间，范围 ~ [-0.1, 0.1]
        offset_norm = offset / max(float(h), 1.0)  # 转换到 [0,1] 空间的步长
        
        return offset_norm


# =============================================================================
# 7. RMITTrackingDatasetV4
# =============================================================================
class RMITTrackingDatasetV4(Dataset):
    """
    RMIT Tracking Dataset V4.
    
    改进:
    - 使用修复后的 augment_tracking_v4
    - 支持跳帧采样（max_frame_gap）增加训练多样性
    - 支持返回序列名和帧索引用于可视化
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
            
            # 支持跳帧采样
            for gap in range(1, max_frame_gap + 1):
                for i in range(gap, len(records)):
                    prev_rec, curr_rec = records[i - gap], records[i]
                    pairs.append({
                        "template_path": prev_rec["path"],
                        "search_path": curr_rec["path"],
                        "search_xy": torch.tensor([curr_rec["x"], curr_rec["y"]], dtype=torch.float32),
                        "seq_name": seq_name,  # 添加序列名
                        "frame_idx": i,  # 添加帧索引
                    })
            
            if max_pairs_per_seq is not None and is_training:
                random.shuffle(pairs)
                pairs = pairs[:max_pairs_per_seq]
            
            self.items.extend(pairs)
    
    def _load_sequence(self, frames_dir: str, ann_csv: str) -> List[dict]:
        """加载序列帧和标注。"""
        records = []
        with open(ann_csv, "r", encoding="utf-8") as f:
            reader = base.csv.DictReader(f)
            for row in reader:
                frame_key = base._find_col(row, ["frame", "frame_id", "image", "image_name", "filename", "file"])
                x_key = base._find_col(row, ["x", "tip_x", "cx"])
                y_key = base._find_col(row, ["y", "tip_y", "cy"])
                if frame_key is None or x_key is None or y_key is None:
                    continue
                frame_name = row[frame_key]
                path = frame_name if os.path.isabs(frame_name) else os.path.join(frames_dir, frame_name)
                if not os.path.exists(path):
                    continue
                records.append({"path": path, "x": float(row[x_key]), "y": float(row[y_key])})
        records.sort(key=lambda r: os.path.basename(r["path"]))
        return records
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        
        # 加载图像
        template_raw = base.load_image_rgb(item["template_path"])
        search_raw = base.load_image_rgb(item["search_path"])
        gt_xy = item["search_xy"].clone()
        
        # 计算缩放比例
        scale_x = self.img_size / max(search_raw.size[0], 1)
        scale_y = self.img_size / max(search_raw.size[1], 1)
        gt_xy_px = torch.tensor([gt_xy[0] * scale_x, gt_xy[1] * scale_y], dtype=torch.float32)
        
        # 调整大小
        template = base.pil_resize(template_raw, self.img_size, False)
        search = base.pil_resize(search_raw, self.img_size, False)
        
        # 保存原始 search 图用于可视化（resize 后、归一化前）
        search_for_vis = search.copy() if not self.is_training else None
        
        # 数据增强 (使用修复后的版本)
        if self.is_training:
            template, search, gt_xy_px = augment_tracking_v4(template, search, gt_xy_px, True)
        
        # 确保不越界
        gt_xy_px = gt_xy_px.clamp(0, self.img_size - 1)
        
        # 归一化坐标
        gt_xy_norm = gt_xy_px / max(float(self.img_size - 1), 1.0)
        
        # 生成 heatmap（带 label smoothing）
        sigma = max(2.0, self.img_size / 64.0)
        gt_hm = base._make_gaussian(self.img_size, gt_xy_px[0].item(), gt_xy_px[1].item(), sigma=sigma)
        
        # Label smoothing for heatmap: target = target * 0.9 + 0.1 / num_pixels
        if self.is_training:
            num_pixels = self.img_size * self.img_size
            gt_hm = gt_hm * 0.9 + 0.1 / num_pixels
        
        result = {
            "template": base.normalize_img(base.pil_to_tensor(template)),
            "search": base.normalize_img(base.pil_to_tensor(search)),
            "gt_xy_px": gt_xy_px,
            "gt_xy_norm": gt_xy_norm,
            "gt_hm": gt_hm,
            "seq_name": item["seq_name"],  # 添加序列名
            "frame_idx": item["frame_idx"],  # 添加帧索引
        }
        
        # 随机擦除 - 在归一化后的 tensor 上实现
        if self.is_training and random.random() < 0.1:
            # 对 search tensor 进行随机擦除
            search_tensor = result["search"]
            c, h, w = search_tensor.shape
            area = h * w
            
            # 随机擦除区域大小 (2%-20% 的面积)
            erase_area = random.uniform(0.02, 0.2) * area
            erase_h = int(math.sqrt(erase_area * random.uniform(0.3, 3.0)))
            erase_w = int(erase_area / max(erase_h, 1))
            erase_h = min(erase_h, h)
            erase_w = min(erase_w, w)
            
            # 随机位置
            i = random.randint(0, h - erase_h)
            j = random.randint(0, w - erase_w)
            
            # 用随机值填充
            fill_val = torch.rand(c) * 2 - 1  # 归一化后的范围 [-1, 1]
            search_tensor[:, i:i+erase_h, j:j+erase_w] = fill_val.view(c, 1, 1)
            result["search"] = search_tensor
        
        # 非训练模式下返回原始 search 图用于可视化
        if search_for_vis is not None:
            result["search_for_vis"] = base.pil_to_tensor(search_for_vis)  # [3, H, W], 0-255
        
        return result


# =============================================================================
# 8. OneStreamTipTrackerV4（核心模型）
# =============================================================================
class OneStreamTipTrackerV4(nn.Module):
    """
    OneStream Tip Tracker V4 - 重构版架构。
    
    核心改进:
    1. Token Correlation 替代 Cross-Attention（零额外参数）
    2. 轻量级 LightOffsetHead 替代 SpatialAwareOffsetHead
    3. 主路径使用标准 soft-argmax（不使用 uncertainty 加权）
    4. uncertainty 仅在 β-NLL 损失中使用
    """
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        encoder: nn.Module,
        embed_dim: int,
        feature_layers: List[int],
        use_anatomy_anchor: bool = False,
        use_uncertainty_head: bool = False,
        use_token_correlation: bool = True,
        fpn_out_dim: int = 256,
        offset_dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = encoder
        self.feature_layers = feature_layers
        self.use_anatomy_anchor = use_anatomy_anchor
        self.use_uncertainty_head = use_uncertainty_head
        self.use_token_correlation = use_token_correlation
        
        self.search_grid = img_size // patch_size
        self.num_search = self.search_grid ** 2
        self.num_template = self.num_search
        
        # 位置编码
        self.pos_ts = nn.Parameter(torch.zeros(1, 1 + self.num_template + self.num_search, embed_dim))
        nn.init.trunc_normal_(self.pos_ts, std=0.02)
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 多尺度融合
        self.fuser = base.MultiScaleTokenFuser(embed_dim, fpn_out_dim, len(feature_layers))
        
        # 可选 anatomy attention
        if use_anatomy_anchor:
            self.anatomy_attention = nn.Sequential(
                nn.Conv2d(fpn_out_dim + 2, fpn_out_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(fpn_out_dim, fpn_out_dim, kernel_size=1),
            )
        
        # Heatmap head
        if use_uncertainty_head:
            self.hm_head = base.UncertaintyHeatmapHead(fpn_out_dim)
        else:
            self.hm_head = nn.Sequential(
                nn.Conv2d(fpn_out_dim, 256, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(256, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 1, 1),
            )
        
        # 轻量级 offset head
        self.offset_head = LightOffsetHead(fpn_out_dim, hidden_dim=64, dropout=offset_dropout)
    
    def forward(
        self,
        template: torch.Tensor,
        search: torch.Tensor,
        anatomy_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            template: [B, 3, H, W]
            search: [B, 3, H, W]
            anatomy_mask: [B, 2, H, W] optional anatomy mask
        
        Returns:
            dict with keys:
                - heatmap_logits: [B, 1, H, W]
                - log_variance: [B, 1, H, W] or None
                - coarse_xy_norm: [B, 2] soft-argmax 坐标
                - offset_norm: [B, 2] 精细 offset
                - xy_norm: [B, 2] 最终坐标 (coarse + offset)
        """
        self.encoder.set_domain("surg")
        bsz = template.size(0)
        
        # Patch embedding
        template_tokens = self.encoder.patch(template)  # [B, N_t, D]
        search_tokens = self.encoder.patch(search)      # [B, N_s, D]
        
        # CLS token
        cls_token = self.encoder.cls.expand(bsz, -1, -1)
        
        # Concatenate: [cls, template, search]
        hidden = torch.cat([cls_token, template_tokens, search_tokens], dim=1)
        hidden = hidden + self.pos_ts[:, :hidden.size(1)]
        hidden = self.encoder.drop(hidden)
        
        # Extract multi-layer features
        selected_template: Dict[int, torch.Tensor] = {}
        selected_search: Dict[int, torch.Tensor] = {}
        target = set(self.feature_layers)
        
        for idx, block in enumerate(self.encoder.blocks):
            hidden = block(hidden)
            if idx in target:
                normalized = self.norm(hidden)
                selected_template[idx] = normalized[:, 1:1+self.num_template]
                selected_search[idx] = normalized[:, 1+self.num_template:1+self.num_template+self.num_search]
        
        # Token correlation（零参数）
        if self.use_token_correlation:
            search_multi = []
            for layer_idx in self.feature_layers:
                enhanced = token_correlation(selected_template[layer_idx], selected_search[layer_idx])
                search_multi.append(enhanced)
        else:
            search_multi = [selected_search[idx] for idx in self.feature_layers]
        
        # 多尺度融合
        feat = self.fuser(search_multi, self.search_grid)
        
        # Anatomy attention
        if self.use_anatomy_anchor and anatomy_mask is not None:
            anatomy_resized = F.interpolate(
                anatomy_mask, size=(self.search_grid, self.search_grid),
                mode="bilinear", align_corners=False
            )
            feat = self.anatomy_attention(torch.cat([feat, anatomy_resized], dim=1))
        
        # Heatmap
        if self.use_uncertainty_head:
            hm_out = self.hm_head(feat)
            heatmap_logits = hm_out["heatmap_logits"]
            log_variance = hm_out["log_variance"]
        else:
            heatmap_logits = self.hm_head(feat)
            log_variance = None
        
        # 上采样到原始尺寸
        heatmap_logits = F.interpolate(
            heatmap_logits, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False
        )
        if log_variance is not None:
            log_variance = F.interpolate(
                log_variance, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False
            )
        
        # 主路径：标准 soft-argmax（不使用 uncertainty 加权）
        coarse_xy_px = base.soft_argmax_2d(heatmap_logits)
        coarse_xy_norm = coarse_xy_px / max(float(self.img_size - 1), 1.0)
        
        # Offset 精炼
        offset_norm = self.offset_head(feat, coarse_xy_norm)
        
        # Final refined coordinate
        xy_norm = (coarse_xy_norm + offset_norm).clamp(0.0, 1.0)
        
        return {
            "heatmap_logits": heatmap_logits,
            "log_variance": log_variance,
            "coarse_xy_norm": coarse_xy_norm,
            "offset_norm": offset_norm,
            "xy_norm": xy_norm,
        }


# =============================================================================
# 9. 评估函数
# =============================================================================
def save_tracking_visualization(
    search_img: torch.Tensor,
    gt_xy_px: torch.Tensor,
    pred_xy_px: torch.Tensor,
    seq_name: str,
    frame_idx: int,
    save_path: str,
    img_size: int,
) -> None:
    """
    保存单个 tracking 结果的可视化图片。
    
    Args:
        search_img: [3, H, W] 原始 search 图像 (0-1 范围，需乘以 255 显示)
        gt_xy_px: [2] GT 坐标 (像素)
        pred_xy_px: [2] 预测坐标 (像素)
        seq_name: 序列名
        frame_idx: 帧索引
        save_path: 保存路径
        img_size: 图像尺寸
    """
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    
    # 转换为 numpy，处理 [0, 1] 范围 -> [0, 255]
    img_np = search_img.permute(1, 2, 0).cpu().numpy()
    # pil_to_tensor 已经将值归一化到 [0, 1]，需要乘以 255
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    
    # 计算像素误差
    error_px = torch.sqrt(((pred_xy_px - gt_xy_px) ** 2).sum()).item()
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img_np)
    
    # 获取坐标值
    gt_x, gt_y = gt_xy_px[0].item(), gt_xy_px[1].item()
    pred_x, pred_y = pred_xy_px[0].item(), pred_xy_px[1].item()
    
    # 绘制 GT 点（绿色）
    ax.scatter(gt_x, gt_y, c='green', s=100, marker='o', edgecolors='white', linewidths=2, zorder=5)
    ax.annotate('GT', (gt_x, gt_y), xytext=(gt_x + 10, gt_y - 10), fontsize=12, color='green',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 绘制 Pred 点（红色）
    ax.scatter(pred_x, pred_y, c='red', s=100, marker='o', edgecolors='white', linewidths=2, zorder=5)
    ax.annotate('Pred', (pred_x, pred_y), xytext=(pred_x + 10, pred_y + 10), fontsize=12, color='red',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 绘制连接线（蓝色）
    ax.plot([gt_x, pred_x], [gt_y, pred_y], 'b-', linewidth=2, zorder=4)
    
    # 标注误差
    mid_x = (gt_x + pred_x) / 2
    mid_y = (gt_y + pred_y) / 2
    ax.annotate(f'{error_px:.1f}px', (mid_x, mid_y), fontsize=10, color='blue',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.8))
    
    # 设置标题
    ax.set_title(f'{seq_name} | Frame {frame_idx} | Error: {error_px:.1f}px', fontsize=14)
    ax.axis('off')
    
    # 保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


@torch.no_grad()
def eval_rmit_v4(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    feature_layers: List[int],
    use_anatomy_anchor: bool,
    image_size: int,
    reference_size: float = RMIT_REFERENCE_SIZE,
    reference_p10: float = RMIT_REFERENCE_P10,
    # 可视化参数
    vis_dir: Optional[str] = None,
    epoch: int = -1,
    num_vis_per_seq: int = 5,
) -> Dict[str, float]:
    """
    RMIT V4 评估函数。
    
    改进: 
    - 显式传入 reference 参数
    - 支持可视化保存
    """
    model.eval()
    preds, gts = [], []
    coarse_preds = []  # 同时记录 coarse 预测用于调试
    
    # 可视化收集
    vis_data = []  # [(seq_name, frame_idx, search_img, gt_xy_px, pred_xy_px), ...]
    seq_vis_count = {}  # 每个序列已保存的数量
    
    seg_module = model["seg"] if "seg" in model else None
    tracker = model["tracker"]
    enc = model["enc"]
    
    for batch in loader:
        template = batch["template"].to(device)
        search = batch["search"].to(device)
        gt_xy_px = batch["gt_xy_px"].to(device)
        seq_names = batch["seq_name"]
        frame_idxs = batch["frame_idx"]
        
        # 可视化数据
        search_for_vis = batch.get("search_for_vis")
        
        # Anatomy mask
        anatomy_mask = None
        if use_anatomy_anchor and seg_module is not None:
            _, search_tok = base.encoder_forward_selected(enc, search, "surg", feature_layers)
            logits = seg_module(search_tok)
            anatomy_mask = torch.sigmoid(logits)
        
        # Forward
        tracker_out = tracker(template, search, anatomy_mask=anatomy_mask)
        
        # 主输出：refined xy
        pred_xy_px = tracker_out["xy_norm"] * max(float(image_size - 1), 1.0)
        coarse_xy_px = tracker_out["coarse_xy_norm"] * max(float(image_size - 1), 1.0)
        
        preds.append(pred_xy_px.cpu())
        coarse_preds.append(coarse_xy_px.cpu())
        gts.append(gt_xy_px.cpu())
        
        # 收集可视化数据
        if vis_dir is not None and search_for_vis is not None:
            bsz = template.size(0)
            for i in range(bsz):
                seq_name = seq_names[i] if isinstance(seq_names, list) else seq_names[0]
                frame_idx = frame_idxs[i].item() if hasattr(frame_idxs[i], 'item') else frame_idxs[i]
                
                # 统计每个序列的可视化数量
                if seq_name not in seq_vis_count:
                    seq_vis_count[seq_name] = 0
                if seq_vis_count[seq_name] >= num_vis_per_seq:
                    continue
                
                vis_data.append({
                    "seq_name": seq_name,
                    "frame_idx": frame_idx,
                    "search_img": search_for_vis[i],
                    "gt_xy_px": gt_xy_px[i].cpu(),
                    "pred_xy_px": pred_xy_px[i].cpu(),
                })
                seq_vis_count[seq_name] += 1
    
    if not preds:
        return {k: 0.0 for k in ["mse_px", "mean_error_px", "mean_error_ref", "mean_error_width_norm",
                                  "mean_error_diag_norm", "precision@5px", "precision@10px",
                                  "precision@20px", "precision@ref10", "failure_rate@20px",
                                  "coarse_mean_error_px", "pck@10", "pck@20", "pck@30", "pck@50"]}
    
    preds_cat = torch.cat(preds, dim=0)
    gts_cat = torch.cat(gts, dim=0)
    coarse_cat = torch.cat(coarse_preds, dim=0)
    
    metrics = compute_tracking_metrics_v4(preds_cat, gts_cat, image_size, reference_size, reference_p10)
    
    # 添加 coarse 误差用于调试
    coarse_diff = coarse_cat - gts_cat
    coarse_dist = torch.sqrt((coarse_diff ** 2).sum(dim=1))
    metrics["coarse_mean_error_px"] = float(coarse_dist.mean().item())
    
    # 计算 PCK (Percentage of Correct Keypoints) 指标
    diff_px = preds_cat.float() - gts_cat.float()
    dist_px = torch.sqrt((diff_px ** 2).sum(dim=1))
    metrics["pck@10"] = float((dist_px < 10.0).float().mean().item())  # 误差 < 10px 的比例
    metrics["pck@20"] = float((dist_px < 20.0).float().mean().item())  # 误差 < 20px 的比例
    metrics["pck@30"] = float((dist_px < 30.0).float().mean().item())  # 误差 < 30px 的比例
    metrics["pck@50"] = float((dist_px < 50.0).float().mean().item())  # 误差 < 50px 的比例
    
    # 保存可视化图片
    vis_save_dir = None
    if vis_dir is not None and len(vis_data) > 0:
        vis_save_dir = os.path.join(vis_dir, f"epoch_{epoch:03d}")
        os.makedirs(vis_save_dir, exist_ok=True)
        
        for i, data in enumerate(vis_data):
            save_path = os.path.join(
                vis_save_dir, 
                f"{data['seq_name']}_frame{data['frame_idx']:04d}.png"
            )
            save_tracking_visualization(
                data["search_img"],
                data["gt_xy_px"],
                data["pred_xy_px"],
                data["seq_name"],
                data["frame_idx"],
                save_path,
                image_size,
            )
        
        print(f"[RMIT-V4] Saved {len(vis_data)} visualization images to {vis_save_dir}")
    
    metrics["_vis_save_dir"] = vis_save_dir  # 返回可视化保存路径
    
    return metrics


# =============================================================================
# 10. 训练函数
# =============================================================================
def train_stage_rmit_v4(cfg, device: torch.device) -> None:
    """RMIT V4 训练阶段。"""
    base.ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)
    
    # Build encoder
    enc = base.build_encoder(cfg).to(device)
    
    # Build seg decoder (for anatomy anchor)
    seg = (
        base.SegDecoderV2(cfg.img_size, cfg.patch_size, cfg.embed_dim, cfg.feature_layers, out_ch=2, fpn_out_dim=cfg.fpn_out_dim).to(device)
        if cfg.use_anatomy_anchor
        else None
    )
    
    # Build tracker V4
    tracker = OneStreamTipTrackerV4(
        cfg.img_size,
        cfg.patch_size,
        enc,
        cfg.embed_dim,
        cfg.feature_layers,
        use_anatomy_anchor=cfg.use_anatomy_anchor,
        use_uncertainty_head=cfg.use_uncertainty_head,
        use_token_correlation=cfg.use_token_correlation,
        fpn_out_dim=cfg.fpn_out_dim,
        offset_dropout=cfg.offset_dropout,
    ).to(device)
    
    # Model dict
    model_items = {"enc": enc, "tracker": tracker}
    if seg is not None:
        model_items["seg"] = seg
    model = nn.ModuleDict(model_items)
    
    # Load pretrained
    if cfg.pretrained_encoder_ckpt:
        base.load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)
    
    if cfg.init_ckpt:
        print(f"[RMIT-V4] Loading init ckpt: {cfg.init_ckpt}")
        base.load_ckpt(cfg.init_ckpt, model, optim=None)
        try:
            enc.copy_domain_params(src_domain="intra", dst_domain="surg")
        except Exception:
            pass
    
    # Split dataset - 根据 split_mode 选择划分方式
    split_mode = getattr(cfg, "split_mode", "loso")
    loso_fold = getattr(cfg, "loso_fold", 0)
    
    if split_mode == "loso":
        # 标准 LOSO 交叉验证模式
        train_names, val_names, test_seq_name = resolve_loso_split(
            manifest_json=cfg.rmit_manifest_json,
            fold=loso_fold,
        )
        print(f"[RMIT-V4] LOSO Fold {loso_fold}: train={sorted(train_names)}, test={test_seq_name}")
    else:
        # 自定义模式 (使用 exclude_seqs)
        train_names, val_names = resolve_rmit_split_names(
            manifest_json=cfg.rmit_manifest_json,
            protocol=cfg.rmit_protocol,
            test_seq=cfg.rmit_test_seq,
            val_ratio=cfg.rmit_val_ratio,
            seed=cfg.seed,
            exclude_seqs=getattr(cfg, "exclude_seqs", None),
        )
        print(f"[RMIT-V4] Custom mode - Excluded sequences: {getattr(cfg, 'exclude_seqs', None) or []}")
        print(f"[RMIT-V4] Train sequences: {sorted(train_names)}")
        print(f"[RMIT-V4] Val sequences: {sorted(val_names)}")
    print(f"[RMIT-V4] Token correlation enabled: {cfg.use_token_correlation}")
    print(f"[RMIT-V4] DWA enabled: {cfg.use_dwa}")
    print(f"[RMIT-V4] Uncertainty head enabled: {cfg.use_uncertainty_head}")
    
    # Datasets（使用 max_frame_gap 增加训练多样性）
    train_ds = RMITTrackingDatasetV4(
        cfg.rmit_manifest_json,
        cfg.img_size,
        is_training=True,
        max_pairs_per_seq=cfg.max_pairs_per_seq,
        split_names=train_names,
        max_frame_gap=cfg.max_frame_gap,
    )
    val_ds = RMITTrackingDatasetV4(
        cfg.rmit_manifest_json,
        cfg.img_size,
        is_training=False,
        max_pairs_per_seq=None,
        split_names=val_names,
        max_frame_gap=1,  # 验证集只用相邻帧
    )
    
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    # 打印数据集统计信息
    print(f"[RMIT-V4] Train sequences used: {sorted(train_names)}")
    print(f"[RMIT-V4] Val sequences used: {sorted(val_names)}")
    print(f"[RMIT-V4] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"[RMIT-V4] Total frames used: {len(train_ds) + len(val_ds)}")
    
    # Parameter grouping
    params = []
    for name, param in model.named_parameters():
        is_adapter = ("A." in name) or ("B." in name) or ("M." in name)
        is_tracker = name.startswith("tracker.")
        is_seg = name.startswith("seg.")
        
        # 方案A: 完全微调，所有参数可训练
        if cfg.unfreeze_all:
            if is_seg and cfg.use_anatomy_anchor and cfg.freeze_seg:
                param.requires_grad = False
            else:
                param.requires_grad = True
                params.append(param)
        # 如果 freeze_seg 为 True，冻结 seg 参数
        elif cfg.use_anatomy_anchor and is_seg and cfg.freeze_seg:
            param.requires_grad = False
        elif cfg.freeze_backbone:
            if is_adapter or is_tracker:
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False
        else:
            if is_seg and cfg.use_anatomy_anchor and cfg.freeze_seg:
                param.requires_grad = False
            else:
                param.requires_grad = True
                params.append(param)
    
    trainable_count = sum(p.numel() for p in params)
    total_count = sum(p.numel() for p in model.parameters())
    
    # 打印训练配置信息
    if cfg.unfreeze_all:
        print(f"[RMIT-V4] Mode: Full Fine-Tuning (all params unfrozen)")
    else:
        lora_layers = sum(1 for r in cfg.lora_layer_ranks if r > 0)
        print(f"[RMIT-V4] Mode: LoRA Fine-Tuning ({lora_layers}/{cfg.depth} layers with LoRA)")
        if cfg.lora_last_n_layers > 0:
            print(f"[RMIT-V4] LoRA config: last {cfg.lora_last_n_layers} layers only")
    
    print(f"[RMIT-V4] Loss type: {cfg.hm_loss_type}")
    print(f"[RMIT-V4] Trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.1f}%)")
    
    # 使用 RMIT 专用学习率（如果未指定则使用默认的 cfg.lr）
    rmit_lr = getattr(cfg, "rmit_lr", None) or cfg.lr
    print(f"[RMIT-V4] Learning rate: {rmit_lr}")
    
    # Optimizer
    optim = torch.optim.AdamW(params, lr=rmit_lr, weight_decay=cfg.wd)
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == "cuda"))
    ema = base.ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    
    # Loss balancer (DWA)
    if cfg.use_dwa:
        dwa = DynamicWeightAverage(["hm", "xy_coarse", "xy_refined"], temperature=cfg.dwa_temperature)
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
    
    # 可视化目录
    vis_dir = os.path.join(cfg.out_dir, "vis")
    base.ensure_dir(vis_dir)
    
    for epoch in range(cfg.epochs):
        model.train()
        if seg is not None and cfg.freeze_seg:
            seg.eval()
        
        running = 0.0
        running_hm = 0.0
        running_xy_coarse = 0.0
        running_xy_refined = 0.0
        
        optim.zero_grad(set_to_none=True)
        
        iterator = enumerate(train_ld)
        if base.tqdm is not None:
            iterator = base.tqdm(iterator, total=len(train_ld), desc=f"RMIT-V4 {epoch+1}/{cfg.epochs}", leave=False)
        
        for it, batch in iterator:
            template = batch["template"].to(device, non_blocking=True)
            search = batch["search"].to(device, non_blocking=True)
            gt_hm = batch["gt_hm"].to(device, non_blocking=True)
            gt_xy_px = batch["gt_xy_px"].to(device, non_blocking=True)
            gt_xy_norm = batch["gt_xy_norm"].to(device, non_blocking=True)
            
            # LR schedule
            lr_now = warmup_cosine_lr(rmit_lr, global_step, total_steps, warmup_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now
            
            with torch.amp.autocast('cuda', enabled=(cfg.amp and device.type == "cuda")):
                # Anatomy mask
                anatomy_mask = None
                if seg is not None:
                    with torch.no_grad():
                        _, search_tokens = base.encoder_forward_selected(enc, search, "surg", cfg.feature_layers)
                        anatomy_mask = torch.sigmoid(seg(search_tokens)).detach()
                
                # Forward
                tracker_out = tracker(template, search, anatomy_mask=anatomy_mask)
                
                # Losses
                pred_hm = tracker_out["heatmap_logits"]
                
                # Heatmap loss (根据 hm_loss_type 选择)
                if cfg.hm_loss_type == "beta_nll" and cfg.use_uncertainty_head and tracker_out["log_variance"] is not None:
                    loss_hm = beta_nll_heatmap_loss(pred_hm, gt_hm, tracker_out["log_variance"])
                elif cfg.hm_loss_type == "mse":
                    loss_hm = mse_heatmap_loss(pred_hm, gt_hm)
                elif cfg.hm_loss_type == "focal":
                    loss_hm = focal_heatmap_loss(pred_hm, gt_hm, gamma=2.0)
                else:
                    # 默认使用 stable heatmap loss
                    loss_hm = stable_heatmap_loss_logits(pred_hm, gt_hm)
                
                # Coarse xy loss
                pred_xy_coarse_ref = tracker_out["coarse_xy_norm"] * coord_ref
                gt_xy_ref = gt_xy_norm * coord_ref
                loss_xy_coarse = F.smooth_l1_loss(pred_xy_coarse_ref, gt_xy_ref)
                
                # Refined xy loss
                pred_xy_refined_ref = tracker_out["xy_norm"] * coord_ref
                loss_xy_refined = F.smooth_l1_loss(pred_xy_refined_ref, gt_xy_ref)
                
                # Offset L2 regularization
                offset_reg = (tracker_out["offset_norm"] ** 2).mean() * 0.1
                
                # Combine losses
                if dwa is not None:
                    total_loss, w_dict = dwa({
                        "hm": loss_hm,
                        "xy_coarse": loss_xy_coarse,
                        "xy_refined": loss_xy_refined,
                    })
                    total_loss = total_loss + offset_reg
                else:
                    total_loss = cfg.lambda_hm * loss_hm + cfg.lambda_xy * (loss_xy_coarse + loss_xy_refined) + offset_reg
                    w_dict = {"hm": cfg.lambda_hm, "xy_coarse": cfg.lambda_xy, "xy_refined": cfg.lambda_xy}
                
                loss = total_loss / max(1, cfg.grad_accum)
            
            # Skip non-finite loss
            if not torch.isfinite(loss.detach()):
                print(f"[RMIT-V4] skip non-finite loss at epoch={epoch+1} iter={it+1}")
                optim.zero_grad(set_to_none=True)
                continue
            
            # Backward
            scaler.scale(loss).backward()
            
            running += loss.item() * max(1, cfg.grad_accum)
            running_hm += loss_hm.item()
            running_xy_coarse += loss_xy_coarse.item()
            running_xy_refined += loss_xy_refined.item()
            
            # Update
            if (it + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                
                if ema is not None:
                    ema.update(model)
                
                # Logging
                if global_step % 20 == 0:
                    writer.add_scalar("rmit_v4/train_loss", loss.item() * max(1, cfg.grad_accum), global_step)
                    writer.add_scalar("rmit_v4/loss_hm", loss_hm.item(), global_step)
                    writer.add_scalar("rmit_v4/loss_xy_coarse", loss_xy_coarse.item(), global_step)
                    writer.add_scalar("rmit_v4/loss_xy_refined", loss_xy_refined.item(), global_step)
                    writer.add_scalar("rmit_v4/lr", lr_now, global_step)
                    writer.add_scalar("rmit_v4/offset_scale", tracker.offset_head.scale.item(), global_step)
                    
                    for name, w in w_dict.items():
                        writer.add_scalar(f"rmit_v4/weight_{name}", w, global_step)
                
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
        metrics = eval_rmit_v4(
            model, val_ld, device, cfg.feature_layers, cfg.use_anatomy_anchor, 
            cfg.img_size, ref_size, ref_p10,
            vis_dir=vis_dir, epoch=epoch, num_vis_per_seq=cfg.num_vis_per_seq
        )
        ema_metrics = eval_rmit_v4(
            ema.module, val_ld, device, cfg.feature_layers, cfg.use_anatomy_anchor, 
            cfg.img_size, ref_size, ref_p10,
            vis_dir=None, epoch=-1, num_vis_per_seq=0
        ) if ema is not None else None
        
        # Logging
        writer.add_scalar("rmit_v4/val_mean_error_px", metrics["mean_error_px"], epoch)
        writer.add_scalar("rmit_v4/val_mean_error_ref", metrics["mean_error_ref"], epoch)
        writer.add_scalar("rmit_v4/val_coarse_mean_error_px", metrics["coarse_mean_error_px"], epoch)
        writer.add_scalar("rmit_v4/val_precision_ref10", metrics["precision@ref10"], epoch)
        writer.add_scalar("rmit_v4/val_pck@10", metrics["pck@10"], epoch)
        writer.add_scalar("rmit_v4/val_pck@20", metrics["pck@20"], epoch)
        writer.add_scalar("rmit_v4/val_pck@30", metrics["pck@30"], epoch)
        writer.add_scalar("rmit_v4/val_pck@50", metrics["pck@50"], epoch)
        
        if cfg.eval_metrics:
            writer.add_scalar("rmit_v4/val_precision@10px", metrics["precision@10px"], epoch)
        
        if ema_metrics is not None:
            writer.add_scalar("rmit_v4/ema_pck@10", ema_metrics["pck@10"], epoch)
            writer.add_scalar("rmit_v4/ema_pck@20", ema_metrics["pck@20"], epoch)
            writer.add_scalar("rmit_v4/ema_mean_error_px", ema_metrics["mean_error_px"], epoch)
        
        msg = (
            f"[RMIT-V4] epoch {epoch+1}/{cfg.epochs} "
            f"train_loss={running/len(train_ld):.4f} "
            f"hm={running_hm/len(train_ld):.4f} "
            f"xy_c={running_xy_coarse/len(train_ld):.4f} "
            f"xy_r={running_xy_refined/len(train_ld):.4f} "
            f"val_err_px={metrics['mean_error_px']:.2f} "
            f"coarse_err={metrics['coarse_mean_error_px']:.2f} "
            f"pck@10={metrics['pck@10']:.3f} pck@20={metrics['pck@20']:.3f} pck@50={metrics['pck@50']:.3f}"
        )
        
        if ema_metrics is not None:
            msg += f" ema_err={ema_metrics['mean_error_px']:.2f} ema_pck@10={ema_metrics['pck@10']:.3f}"
        
        # 添加可视化路径到日志
        vis_save_dir = metrics.get("_vis_save_dir")
        if vis_save_dir:
            msg += f"\n[RMIT-V4] Visualization saved to: {vis_save_dir}"
        
        print(msg)
        base.append_epoch_log(cfg.out_dir, msg)
        
        # Early stopping
        metric = ema_metrics["mean_error_ref"] if ema_metrics is not None else metrics["mean_error_ref"]
        
        if metric < best:
            best = metric
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.patience:
            print(f"[RMIT-V4] Early stopping at epoch {epoch+1}")
            break
        
        # Save checkpoint
        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "ema": ema.state_dict() if ema is not None else None,
            "extra": {
                "stage": "rmit_v4_token_corr",
                "reference_size": ref_size,
                "reference_p10": ref_p10,
            },
        }
        
        base.save_ckpt(f"{cfg.out_dir}/last.pt", payload)
        
        if metric <= best:
            payload["best_metric"] = best
            base.save_ckpt(f"{cfg.out_dir}/best.pt", payload)
    
    writer.close()


# =============================================================================
# 11. 命令行参数解析
# =============================================================================
class TrainCfgV4:
    """V4 训练配置。"""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def parse_args() -> TrainCfgV4:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="DR Training V4 (Refactored)")
    
    # Stage and output
    parser.add_argument("--stage", type=str, required=True, choices=["dr", "fovea", "rmit"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--pretrained_encoder_ckpt", type=str, default=None)
    
    # Model architecture
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
    parser.add_argument("--lora_rank_strategy", type=str, default="late_heavy", choices=["uniform", "linear", "late_heavy"])
    parser.add_argument("--lora_rank_min", type=int, default=0)
    parser.add_argument("--lora_rank_spec", type=str, default=None)
    parser.add_argument("--use_residual_lora", action="store_true")
    parser.add_argument("--residual_alpha", type=float, default=0.1)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    
    # DR stage
    parser.add_argument("--dr_imagefolder_dir", type=str, default=None)
    parser.add_argument("--dr_train_csv", type=str, default=None)
    parser.add_argument("--dr_val_csv", type=str, default=None)
    parser.add_argument("--dr_val_ratio", type=float, default=0.2)
    parser.add_argument("--dr_num_classes", type=int, default=2)
    
    # FOVEA stage (纯跨域对比对齐，无分割)
    parser.add_argument("--fovea_pairs_json", type=str, default=None)
    parser.add_argument("--fovea_val_ratio", type=float, default=0.2)
    parser.add_argument("--fovea_temperature", type=float, default=0.07, help="FOVEA对比学习温度参数 (默认: 0.07)")
    parser.add_argument("--fovea_embed_dim", type=int, default=128, help="FOVEA对比学习嵌入维度 (默认: 128)")
    # 保留以下参数用于向后兼容（纯对比对齐模式下不使用）
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_seg", type=float, default=2.0)
    
    # RMIT stage
    parser.add_argument("--rmit_manifest_json", type=str, default=None)
    parser.add_argument("--rmit_protocol", type=str, default="leave_one_out", choices=["leave_one_out", "random"])
    parser.add_argument("--rmit_test_seq", type=str, default=None)
    parser.add_argument("--rmit_val_ratio", type=float, default=0.2)
    parser.add_argument("--rmit_lr", type=float, default=3e-5, help="RMIT stage learning rate (default: 3e-5)")
    parser.add_argument("--max_pairs_per_seq", type=int, default=None)
    parser.add_argument("--use_anatomy_anchor", action="store_true")
    parser.add_argument("--eval_metrics", action="store_true")
    parser.add_argument("--exclude_seqs", type=str, nargs="+", default=["seq2"], help="排除的序列名列表 (默认: seq2)")
    
    # LOSO 交叉验证参数
    parser.add_argument("--split_mode", type=str, default="loso", choices=["loso", "custom"],
                        help="数据集划分模式: loso=标准LOSO交叉验证, custom=使用exclude_seqs自定义模式 (默认: loso)")
    parser.add_argument("--fold", type=int, default=0, choices=[0, 1, 2],
                        help="LOSO折数: 0=测试seq1, 1=测试seq2, 2=测试seq3 (默认: 0)")
    
    # Feature layers
    parser.add_argument("--feature_layers", type=str, default="2,5,7")
    parser.add_argument("--align_layer_weights", type=str, default=None)
    parser.add_argument("--contrast_temperature", type=float, default=0.07)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--fpn_out_dim", type=int, default=256)
    
    # Loss weights
    parser.add_argument("--use_uncertainty_head", action="store_true")
    parser.add_argument("--lambda_hm", type=float, default=1.0)
    parser.add_argument("--lambda_xy", type=float, default=0.25)
    parser.add_argument("--use_dora", action="store_true", default=True)
    parser.add_argument("--disable_dora", action="store_true")
    parser.add_argument("--use_task_uncertainty", action="store_true")
    parser.add_argument("--disable_task_uncertainty", action="store_true")
    parser.add_argument("--lambda_edge", type=float, default=1.0)
    parser.add_argument("--edge_weight", type=float, default=4.0)
    parser.add_argument("--edge_kernel_size", type=int, default=5)
    
    # V4 新增参数
    parser.add_argument("--use_token_correlation", action="store_true", help="Enable token correlation (default: True)")
    parser.add_argument("--disable_token_correlation", action="store_true", help="Disable token correlation")
    parser.add_argument("--use_dwa", action="store_true", help="Enable DynamicWeightAverage")
    parser.add_argument("--dwa_temperature", type=float, default=2.0, help="DWA temperature")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Number of warmup epochs")
    parser.add_argument("--max_frame_gap", type=int, default=3, help="Max frame gap for training pairs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--freeze_seg", action="store_true", help="Freeze seg module")
    parser.add_argument("--offset_dropout", type=float, default=0.3, help="Offset head dropout")
    parser.add_argument("--num_vis_per_seq", type=int, default=5, help="Number of visualization images per sequence")
    
    # 对比实验参数 (V4.1)
    # 方案A: 完全微调
    parser.add_argument("--unfreeze_all", action="store_true", help="方案A: 完全微调，不使用LoRA，所有参数可训练")
    
    # 方案B: 冻结更多 + MSE Loss
    parser.add_argument("--lora_last_n_layers", type=int, default=0, help="只在backbone最后N层注入LoRA，默认0表示全层LoRA")
    parser.add_argument("--hm_loss_type", type=str, default="beta_nll", choices=["beta_nll", "mse", "focal"], help="heatmap loss类型")
    
    args = parser.parse_args()
    
    # Resolve derived values
    feature_layers = base.parse_int_list(args.feature_layers, args.depth)
    align_layer_weights = base.resolve_layer_weights(len(feature_layers), args.align_layer_weights)
    
    # 处理 LoRA rank 配置
    if args.unfreeze_all:
        # 方案A: 完全微调，不使用 LoRA (所有层 rank=0)
        lora_layer_ranks = [0] * args.depth
    elif args.lora_last_n_layers > 0:
        # 方案B: 只在最后 N 层注入 LoRA
        lora_layer_ranks = [0] * (args.depth - args.lora_last_n_layers) + [args.lora_r] * args.lora_last_n_layers
    else:
        # 默认: 使用策略配置 LoRA
        lora_layer_ranks = base.resolve_lora_layer_ranks(
            depth=args.depth,
            base_rank=args.lora_r,
            strategy=args.lora_rank_strategy,
            min_rank=args.lora_rank_min,
            spec=args.lora_rank_spec,
        )
    
    coord_ref, eval_ref_p10 = resolve_rmit_reference_protocol(args.img_size)
    
    # Token correlation 默认开启
    use_token_correlation = not args.disable_token_correlation
    
    return TrainCfgV4(
        stage=args.stage,
        out_dir=args.out_dir,
        init_ckpt=args.init_ckpt,
        pretrained_encoder_ckpt=args.pretrained_encoder_ckpt,
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        drop=args.drop,
        attn_drop=args.attn_drop,
        lora_r=args.lora_r,
        lora_rank_strategy=args.lora_rank_strategy,
        lora_rank_min=args.lora_rank_min,
        lora_rank_spec=args.lora_rank_spec,
        lora_layer_ranks=lora_layer_ranks,
        use_residual_lora=args.use_residual_lora,
        residual_alpha=args.residual_alpha,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        wd=args.wd,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=args.amp,
        grad_accum=max(1, args.grad_accum),
        freeze_backbone=args.freeze_backbone,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        dr_imagefolder_dir=args.dr_imagefolder_dir,
        dr_train_csv=args.dr_train_csv,
        dr_val_csv=args.dr_val_csv,
        dr_val_ratio=args.dr_val_ratio,
        dr_num_classes=args.dr_num_classes,
        fovea_pairs_json=args.fovea_pairs_json,
        fovea_val_ratio=args.fovea_val_ratio,
        fovea_temperature=args.fovea_temperature,
        fovea_embed_dim=args.fovea_embed_dim,
        lambda_align=args.lambda_align,
        lambda_seg=args.lambda_seg,
        rmit_manifest_json=args.rmit_manifest_json,
        rmit_protocol=args.rmit_protocol,
        rmit_test_seq=args.rmit_test_seq,
        rmit_val_ratio=args.rmit_val_ratio,
        rmit_lr=args.rmit_lr,
        max_pairs_per_seq=args.max_pairs_per_seq,
        use_anatomy_anchor=args.use_anatomy_anchor,
        eval_metrics=args.eval_metrics,
        exclude_seqs=args.exclude_seqs,
        split_mode=args.split_mode,
        loso_fold=args.fold,
        feature_layers=feature_layers,
        align_layer_weights=align_layer_weights,
        contrast_temperature=args.contrast_temperature,
        proj_dim=args.proj_dim,
        fpn_out_dim=args.fpn_out_dim,
        use_uncertainty_head=args.use_uncertainty_head,
        lambda_hm=args.lambda_hm,
        lambda_xy=args.lambda_xy,
        use_dora=not args.disable_dora,
        use_task_uncertainty=not args.disable_task_uncertainty,
        lambda_edge=args.lambda_edge,
        edge_weight=args.edge_weight,
        edge_kernel_size=args.edge_kernel_size,
        # V4 specific
        use_token_correlation=use_token_correlation,
        use_dwa=args.use_dwa,
        dwa_temperature=args.dwa_temperature,
        warmup_epochs=args.warmup_epochs,
        max_frame_gap=args.max_frame_gap,
        patience=args.patience,
        freeze_seg=args.freeze_seg,
        offset_dropout=args.offset_dropout,
        num_vis_per_seq=args.num_vis_per_seq,
        # V4.1 对比实验参数
        unfreeze_all=args.unfreeze_all,
        lora_last_n_layers=args.lora_last_n_layers,
        hm_loss_type=args.hm_loss_type,
        # Reference values
        rmit_coord_ref_size=coord_ref,
        rmit_eval_ref_p10=eval_ref_p10,
    )


# =============================================================================
# 12. Main 函数
# =============================================================================
def main() -> None:
    """主入口。"""
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
        train_stage_rmit_v4(cfg, device)
    else:
        raise ValueError(cfg.stage)


if __name__ == "__main__":
    main()
