"""
CAC-KD: Confidence-Aware Cross-Domain Keypoint Detection.

Three architectural innovations:
1. CEH (Confidence Estimation Head): predicts per-keypoint confidence from
   decoder features. Warm-started with offline CDT confidence, then transitions
   to self-consistency supervision.
2. CGFA (Confidence-Gated Feature Adapter): per-sample feature adaptation
   whose strength is gated by CEH confidence. Low-confidence aux samples get
   strong adaptation; target samples get none.
3. UMGR (Uncertainty-Modulated Geometry Refinement): graph-based keypoint
   refinement where each edge's contribution is modulated by the product of
   its endpoint confidences. Low-confidence keypoints (e.g. cross-instrument
   kp0 on RMIT) automatically receive weak geometry constraints.

Usage:
    python -m surgical_tracking.src.dr_cackd \
        --manifest data/manifests/rmit_cataract_manifest.json \
        --aux_manifest data/manifests/cataract_manifest.json \
        --baseline_ckpt logs/v11_fold0_heatmap_offset_efficientnet_v2_s/best.pt \
        --conf_cache logs/cdt_confidence_f0.json \
        --protocol loso --fold 0 \
        --out_dir logs/cackd_rmit_f0
"""

import os
import sys
import json
import math
import random
import argparse
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dr_11 import (
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    load_rmit_manifest_sequences,
    resolve_loso_split,
    warmup_cosine_lr,
    ensure_dir,
    soft_argmax_2d,
)

# Reuse CDT confidence computation + dataset wrapper + loss
from dr_xtransfer import (
    compute_aux_confidence,
    confidence_weighted_loss,
    evaluate,
)

# Reuse ASD edge discovery + edge list
from dr_geo import ALL_EDGES, discover_articulated_structure


# =============================================================================
# Module 1: CEH — Confidence Estimation Head
# =============================================================================

class ConfidenceEstimationHead(nn.Module):
    """Predict per-keypoint aleatoric uncertainty (log-variance) from features.

    Uses Kendall et al. (CVPR'18) uncertainty weighting: the log-variance is
    learned END-TO-END through the loss, with NO separate supervision target.

        L_k = exp(-log_var_k) * error_k + log_var_k

    - High error → optimal log_var = log(error) → low confidence
    - Low error  → optimal log_var = log(error) → high confidence
    - The log_var regulariser prevents collapse (unlike self-consistency).

    Confidence is derived: conf = sigmoid(-log_var).

    Input:  d3  [B, C, H/4, W/4]
    Output: log_var [B, K]  (unconstrained real values)
    """

    def __init__(self, in_channels: int, num_keypoints: int = 4, hidden: int = 64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, num_keypoints)
        # Init bias=0 → log_var=0 → var=1 → precision=1 (neutral start)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, d3: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(d3).flatten(1)
        x = F.relu(self.fc1(pooled), inplace=True)
        # Clamp to prevent extreme confidence.
        # min=-1.5 → max conf=sigmoid(1.5)=0.82 (prevent UMGR over-geometry)
        # max=2.0  → min conf=sigmoid(-2.0)=0.12 (still can express low conf)
        return self.fc2(x).clamp(-1.5, 2.0)  # log_var [B, K]


# =============================================================================
# Module 2: CGFA — Confidence-Gated Feature Adapter
# =============================================================================

class ConfidenceGatedFeatureAttention(nn.Module):
    """Lightweight channel attention gated by teacher confidence (~320 params).

    For aux-domain samples: channels are reweighted by a learned function of
    per-keypoint confidence. Low confidence → suppress uncertain channels.
    For target-domain samples: features pass through unchanged.

    Inspired by EgoPoseFormer v2 (CVPR 2026) lightweight gating design.
    """

    def __init__(self, num_keypoints: int = 4, channels: int = 64):
        super().__init__()
        self.fc = nn.Linear(num_keypoints, channels)
        # Init bias → sigmoid(4)≈0.98, near-identity at start
        nn.init.constant_(self.fc.bias, 4.0)

    def forward(self, feat: torch.Tensor, confidence: torch.Tensor,
                is_aux: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.fc(confidence))  # [B, C]
        gate = gate.unsqueeze(-1).unsqueeze(-1)    # [B, C, 1, 1]
        is_aux = is_aux.view(-1, 1, 1, 1).to(feat.dtype)
        # Target: unchanged. Aux: channel-weighted by confidence
        return feat * (1 - is_aux) + is_aux * feat * gate


# =============================================================================
# Module 3: UMGR — Uncertainty-Modulated Geometry Refinement
# =============================================================================

class UncertaintyModulatedGeometryRefinement(nn.Module):
    """Graph-based keypoint refinement with per-keypoint confidence modulation.

    Each edge (i, j) contributes a correction whose magnitude is gated by
    conf_i * conf_j. Low-confidence keypoints (e.g. cross-instrument kp0)
    automatically receive weak geometry corrections, avoiding the failure
    mode of global-sigma geometry modules.

    Learnable edge weights (softmax-normalised) further tune which edges
    matter, initialised uniformly.
    """

    def __init__(self, num_keypoints: int = 4, hidden_dim: int = 16):
        super().__init__()
        self.num_kp = num_keypoints
        self.hidden = hidden_dim

        self.edge_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )
        # Zero-init correction so refinement starts as identity
        nn.init.zeros_(self.node_update[-1].weight)
        nn.init.zeros_(self.node_update[-1].bias)

        # Learnable per-edge weights (uniform init -> all equal)
        self.edge_logits = nn.Parameter(torch.zeros(len(ALL_EDGES)))

    def get_edge_weights(self) -> torch.Tensor:
        return F.softmax(self.edge_logits, dim=-1)

    def forward(self, xy: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xy:          [B, K, 2]  (normalised or pixel, same scale in/out)
            confidence:  [B, K]      per-keypoint confidence
        Returns:
            refined xy:  [B, K, 2]
        """
        B, K, _ = xy.shape
        edge_w = self.get_edge_weights()  # [E]

        edge_feats = []
        for idx, (i, j) in enumerate(ALL_EDGES):
            diff = xy[:, i] - xy[:, j]  # [B, 2]
            dist = torch.sqrt((diff ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-6))
            angle = torch.atan2(diff[..., 1:2], diff[..., 0:1])
            feat = torch.cat([dist, angle], dim=-1)  # [B, 2]
            feat = self.edge_encoder(feat)           # [B, H]
            # Confidence gate: low conf at either endpoint -> weak edge
            gate = (confidence[:, i] * confidence[:, j]).unsqueeze(-1)  # [B, 1]
            feat = feat * edge_w[idx] * gate
            edge_feats.append((i, j, feat))

        node_agg = torch.zeros(B, K, self.edge_encoder[-1].out_features,
                               device=xy.device)
        count = torch.zeros(K, device=xy.device)
        for i, j, feat in edge_feats:
            node_agg[:, i] = node_agg[:, i] + feat
            node_agg[:, j] = node_agg[:, j] + feat
            count[i] += 1
            count[j] += 1
        node_agg = node_agg / count.clamp_min(1).unsqueeze(0).unsqueeze(-1)

        node_input = torch.cat([node_agg, xy], dim=-1)
        correction = self.node_update(node_input)
        return xy + correction


# =============================================================================
# Model wrapper: CACKeypointNet
# =============================================================================

class CACKeypointNet(nn.Module):
    """HeatmapKeypointNet + CEH + CGFA + UMGR.

    Re-implements the forward to insert the three modules at the right
    places without modifying the original dr_11.HeatmapKeypointNet.
    """

    _TRAIN_ONLY_PARAMS = {
        'use_krm', 'kp_adaptive', 'kp_adapt_momentum', 'kp_adapt_max_weight',
        'kp_adapt_min_weight', 'adabn', 'adabn_passes', 'edge_weights_init',
    }

    def __init__(self, num_keypoints: int = 4, **kwargs):
        super().__init__()
        edge_init = kwargs.pop('edge_weights_init', None)
        for p in self._TRAIN_ONLY_PARAMS:
            kwargs.pop(p, None)

        self.base = HeatmapKeypointNet(**kwargs)
        dec_c = kwargs.get('decoder_channels', [256, 128, 64])[2]

        # Module 1: CGFA-lite (channel attention, ~320 params)
        self.cgfa = ConfidenceGatedFeatureAttention(num_keypoints, dec_c)
        # Module 2: CMGR-lite (graph refinement, ~664 params)
        self.umgr = UncertaintyModulatedGeometryRefinement(num_keypoints, hidden_dim=16)

    def forward(self, x: torch.Tensor,
                confidence: Optional[torch.Tensor] = None,
                is_aux: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: input image [B, C, H, W]
            confidence: offline per-keypoint confidence [B, K]. If None, all 1.0
                        (target domain at inference). This is FIXED, not learned.
            is_aux: [B] flag, 1=aux domain, 0=target domain
        """
        B = x.size(0)
        if confidence is None:
            confidence = torch.ones(B, 4, device=x.device)
        if is_aux is None:
            is_aux = torch.zeros(B, device=x.device)

        # --- encoder + decoder (replicates HeatmapKeypointNet.forward) ---
        e2, e3, e4 = self.base.encoder(x)        # /4, /8, /16
        e4 = self.base.aspp(e4)
        d1 = self.base.up1(e4, e3)               # [B, dec0, H/8, W/8]
        d2 = self.base.up2(d1, e2)              # [B, dec1, H/4, W/4]
        d3 = self.base.bottleneck(d2)           # [B, dec2, H/4, W/4]

        # --- Module 2: CGFA adapts features (aux only, gated by OFFLINE conf) ---
        d3_adapted = self.cgfa(d3, confidence, is_aux)

        # --- heatmap + offset decode (uses adapted features) ---
        hm_logits = self.base.hm_head(d3_adapted)  # [B, 4, H/4, W/4]
        hm_logits = F.interpolate(
            hm_logits, size=(self.base.img_size, self.base.img_size),
            mode="bilinear", align_corners=False)

        coarse_xy = soft_argmax_2d(hm_logits, temperature=1.0)  # [B, 4, 2]
        coarse_xy_norm = coarse_xy / float(self.base.img_size - 1)

        if self.base.use_offset:
            offset_logits = self.base.offset_head(d3_adapted)
            offset_logits = F.interpolate(
                offset_logits, size=(self.base.img_size, self.base.img_size),
                mode="bilinear", align_corners=False)
            offset = torch.tanh(offset_logits).view(
                B, 4, 2, self.base.img_size, self.base.img_size)
            grid = (coarse_xy_norm * 2 - 1).unsqueeze(2).unsqueeze(3)
            sampled = []
            for k in range(4):
                ok = offset[:, k]  # [B, 2, H, W]
                s = F.grid_sample(ok, grid[:, k], mode="bilinear",
                                 padding_mode="border", align_corners=True)
                sampled.append(s.squeeze(-1).squeeze(-1))
            offset_xy = torch.stack(sampled, dim=1) * 0.05
            xy_norm = (coarse_xy_norm + offset_xy).clamp(0, 1)
        else:
            xy_norm = coarse_xy_norm

        # --- Module 3: UMGR refines xy using OFFLINE confidence ---
        edge_weights = self.umgr.get_edge_weights()  # [E]
        xy_refined = self.umgr(xy_norm, confidence)

        return {
            "hm_logits": hm_logits,
            "coarse_xy": coarse_xy_norm,
            "xy": xy_refined,
            "xy_pre_refine": xy_norm,
            "edge_weights": edge_weights,
        }

    @property
    def encoder(self):
        return self.base.encoder

    @property
    def backbone_name(self):
        return self.base.backbone_name


# =============================================================================
# Dataset wrapper: adds is_aux flag
# =============================================================================

class CacDataset(Dataset):
    """Wraps RMITHeatmapDataset with per-keypoint confidence + is_aux flag."""

    def __init__(self, base_dataset, confidence_map, num_keypoints=4):
        self.base = base_dataset
        self.confidence_map = confidence_map
        self.num_kp = num_keypoints

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        path = item["path"]
        if path in self.confidence_map:
            conf = self.confidence_map[path]
            is_aux = 1
        else:
            conf = np.ones(self.num_kp, dtype=np.float32)
            is_aux = 0
        item["confidence"] = torch.from_numpy(np.array(conf, dtype=np.float32))
        item["is_aux"] = torch.tensor(is_aux, dtype=torch.float32)
        return item


# =============================================================================
# Loss
# =============================================================================

def cac_loss(
    pred: Dict[str, torch.Tensor],
    gt_hm: torch.Tensor,
    gt_xy: torch.Tensor,
    offline_conf: torch.Tensor,    # [B, K] from baseline (CDT), FIXED
    is_aux: torch.Tensor,          # [B]
    img_size: int,
    geo_weight: float = 0.5,
    xy_weight: float = 10.0,
    coarse_weight: float = 5.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CAC-KD loss: CDT keypoint loss + confidence-modulated geometry loss.

    L = L_kp(offline_conf) + geo_weight * L_geo(offline_conf)

    Both terms use the SAME offline confidence (fixed, not learned).
    No feedback loop — training is as stable as CDT.
    """
    B, K = offline_conf.shape
    pred_conf = offline_conf  # fixed, not learned
    conf_sum = offline_conf.sum() + 1e-6

    # --- Keypoint loss (heatmap + xy + coarse), offline-confidence weighted ---
    pred_hm = torch.sigmoid(pred["hm_logits"])
    se = (pred_hm - gt_hm) ** 2
    per_kp_hm = se.view(B, K, -1).mean(dim=-1)  # [B, K]
    loss_hm = (offline_conf * per_kp_hm).sum() / conf_sum

    diff_xy = torch.abs(pred["xy"] - gt_xy).mean(dim=-1)       # [B, K]
    loss_xy = (offline_conf * diff_xy).sum() / conf_sum

    diff_coarse = torch.abs(pred["coarse_xy"] - gt_xy).mean(dim=-1)  # [B, K]
    loss_coarse = (offline_conf * diff_coarse).sum() / conf_sum

    loss_kp = loss_hm + xy_weight * loss_xy + coarse_weight * loss_coarse

    # --- Geometry loss, offline-confidence modulated ---
    pred_xy_px = pred["xy"] * (img_size - 1)
    gt_xy_px = gt_xy * (img_size - 1)
    edge_w = pred["edge_weights"]  # [E] from UMGR (learnable)

    geo_losses = []
    for idx, (i, j) in enumerate(ALL_EDGES):
        pd = torch.sqrt(((pred_xy_px[:, i] - pred_xy_px[:, j]) ** 2)
                        .sum(dim=-1).clamp_min(1e-6))
        gd = torch.sqrt(((gt_xy_px[:, i] - gt_xy_px[:, j]) ** 2)
                        .sum(dim=-1).clamp_min(1e-6))
        rel = torch.abs(pd / gd.clamp_min(1e-6) - 1.0)
        gate = (pred_conf[:, i] * pred_conf[:, j])  # [B]
        geo_losses.append(rel * gate * edge_w[idx])
    loss_geo = torch.stack(geo_losses, dim=-1).sum(dim=-1).mean()

    total = loss_kp + geo_weight * loss_geo
    return total, {
        "kp": loss_kp.item(), "hm": loss_hm.item(),
        "xy": loss_xy.item(), "coarse": loss_coarse.item(),
        "geo": loss_geo.item(),
        "mean_conf": pred_conf.mean().item(),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CAC-KD: Confidence-Aware Cross-Domain Keypoint Detection")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--aux_manifest", type=str, required=True)
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--conf_cache", type=str, default=None)
    parser.add_argument("--naive_aux", action="store_true")
    parser.add_argument("--protocol", type=str, default="loso")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--decoder_type", type=str, default="heatmap_offset")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--decoder_stride", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)

    # CAC-KD specific
    parser.add_argument("--geo_weight", type=float, default=0.5)
    parser.add_argument("--cdt_ckpt", type=str, default=None,
                        help="CDT checkpoint for two-stage training (Stage 2)")
    parser.add_argument("--freeze_backbone", action="store_true", default=False,
                        help="Freeze backbone, train only CGFA+UMGR (Stage 2)")
    parser.add_argument("--module_lr", type=float, default=1e-3,
                        help="LR for new modules when backbone is frozen")

    parser.add_argument("--aug_color", type=float, default=0.3)
    parser.add_argument("--aug_rotate", type=float, default=10)
    parser.add_argument("--aug_blur", type=float, default=0.3)
    parser.add_argument("--aug_noise", type=float, default=0.05)
    parser.add_argument("--aug_cutout", type=float, default=0.1)

    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    if args.no_pretrained:
        args.pretrained = False

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CAC] Device: {device}")
    ensure_dir(args.out_dir)

    # === Step 1: Confidence (offline CDT) ===
    if args.naive_aux:
        print("[CAC] Naive aux mode: all confidence=1.0")
        confidence_map = {}
    elif args.conf_cache and os.path.exists(args.conf_cache):
        print(f"[CAC] Loading cached confidence: {args.conf_cache}")
        raw = json.load(open(args.conf_cache))
        confidence_map = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
    else:
        confidence_map = compute_aux_confidence(
            args.baseline_ckpt, args.aux_manifest,
            args.img_size, device, args.backbone,
        )
        if args.conf_cache:
            json.dump({k: v.tolist() for k, v in confidence_map.items()},
                      open(args.conf_cache, "w"))
            print(f"[CAC] Confidence cached to {args.conf_cache}")

    # === Step 2: Data split ===
    if args.protocol == "loso":
        train_names, val_names, test_name = resolve_loso_split(args.manifest, args.fold)
        half_train, half_val = None, None
        print(f"[CAC] LOSO fold {args.fold}: test={test_name}")
    elif args.protocol == "half":
        seqs = load_rmit_manifest_sequences(args.manifest)
        train_names = val_names = {s["name"] for s in seqs}
        test_name = "half"
        half_train, half_val = "first", "second"
        print(f"[CAC] Half-split: train=first, val=second")

    train_ds = RMITHeatmapDataset(
        args.manifest, args.img_size, train_names, is_training=True,
        sigma=args.sigma, augment=True,
        aug_color=args.aug_color, aug_rotate=args.aug_rotate,
        aug_blur=args.aug_blur, aug_noise=args.aug_noise,
        aug_cutout=args.aug_cutout, half_split=half_train,
    )
    train_ds = CacDataset(train_ds, confidence_map)

    val_ds = RMITHeatmapDataset(
        args.manifest, args.img_size, val_names, is_training=False,
        sigma=args.sigma, augment=False, half_split=half_val,
    )

    print(f"[CAC] Train: {len(train_ds)}  Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # === Step 3: Model ===
    model = CACKeypointNet(
        img_size=args.img_size, pretrained=args.pretrained, backbone=args.backbone,
        decoder_stride=args.decoder_stride, decoder_type=args.decoder_type,
        use_offset=args.use_offset,
    ).to(device)

    # Load checkpoint into base.* (CDT checkpoint for Stage 2, or baseline for Stage 1)
    ckpt_path = args.cdt_ckpt if args.cdt_ckpt else args.baseline_ckpt
    state = torch.load(ckpt_path, map_location=device)
    model_state = state["model"] if "model" in state else state
    if any(k.startswith("base.") for k in model_state):
        model_state = {k.replace("base.", "", 1): v
                       for k, v in model_state.items() if k.startswith("base.")}
    base_state = {f"base.{k}": v for k, v in model_state.items()}
    model.load_state_dict(base_state, strict=False)
    print(f"[CAC] Loaded {'CDT' if args.cdt_ckpt else 'baseline'} backbone ({len(base_state)} tensors).")

    # Stage 2: freeze backbone, train only CGFA+UMGR
    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if name.startswith("base."):
                p.requires_grad = False
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[CAC] Stage 2: backbone FROZEN. Trainable params: {n_trainable}")

    n_params = sum(p.numel() for p in model.parameters())
    n_new = sum(p.numel() for n, p in model.named_parameters()
                if not n.startswith("base."))
    print(f"[CAC] Model: {n_params:,} params ({n_new} new in CGFA+UMGR)")

    # === Step 4: Training ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    train_lr = args.module_lr if args.freeze_backbone else args.lr
    optimizer = torch.optim.AdamW(trainable_params, lr=train_lr, weight_decay=args.wd)
    scaler = GradScaler("cuda")

    best_err = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            lr = train_lr * (epoch + 1) / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / max(
                1, args.epochs - args.warmup_epochs)
            lr = train_lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        sum_loss = sum_kp = sum_geo = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}", leave=False)
        for batch in pbar:
            imgs = batch["img"].to(device)
            gt_xy = batch["gt_xy"].to(device)
            gt_hm = batch["gt_hm"].to(device)
            offline_conf = batch["confidence"].to(device)
            is_aux = batch["is_aux"].to(device)
            B = imgs.size(0)

            optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(imgs, confidence=offline_conf, is_aux=is_aux)
                loss, m = cac_loss(
                    pred, gt_hm, gt_xy, offline_conf, is_aux,
                    args.img_size, geo_weight=args.geo_weight,
                    xy_weight=10.0, coarse_weight=5.0,
                )

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += loss.item() * B
            sum_kp += m["kp"] * B
            sum_geo += m["geo"] * B
            n += B
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                geo=f"{m['geo']:.3f}")

        val_metrics = evaluate(model, val_loader, device, args.img_size)
        val_err = val_metrics["mean_error"]

        print(f"[CAC] epoch {epoch+1}/{args.epochs} "
              f"loss={sum_loss/n:.4f} kp={sum_kp/n:.4f} geo={sum_geo/n:.4f} | "
              f"val_err={val_err:.2f}px pck@20={val_metrics['pck_20']:.3f} "
              f"pck@50={val_metrics['pck_50']:.3f}")
        for k in range(4):
            print(f"  kp{k}: {val_metrics['kp_errors'][k]:.2f}px")

        if val_err < best_err:
            best_err = val_err
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "metric": val_err, "args": vars(args),
                "val_metrics": val_metrics,
            }, os.path.join(args.out_dir, "best.pt"))
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"[CAC] Early stopping at epoch {epoch+1}")
            break

    print(f"[CAC] Best: {best_err:.2f}px at epoch {best_epoch+1}")


if __name__ == "__main__":
    main()
