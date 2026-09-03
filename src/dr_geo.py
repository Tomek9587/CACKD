"""
Learnable Geometry Module for Surgical Instrument Keypoint Tracking.

Three innovations:
1. ASD (Articulated Structure Discovery): Auto-discovers rigid/joint/free edges
   from training data using scale-normalized CV. Used as INITIALIZATION only.
2. LGM (Learnable Geometry Module): Replaces all manual hyperparameters with
   learnable parameters:
   - Edge weights: 6 learnable params (init from ASD), auto-learn which edges matter
   - Loss balancing: Uncertainty weighting (Kendall CVPR'18), auto-balance base/geo loss
   - No manual geo_max_weight, no warmup schedule, no conf_threshold needed
3. KRM (Keypoint Refinement Module): Graph-based refinement using learned edge weights.

Usage:
    python -m surgical_tracking.src.dr_geo \
        --manifest data/manifests/rmit_manifest.json \
        --protocol loso --fold 0 \
        --backbone efficientnet_v2_s --pretrained --use_krm
"""

import os
import sys
import csv
import json
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dr_11 import (
    HeatmapKeypointNet,
    RMITHeatmapDataset,
    load_rmit_manifest_sequences,
    resolve_loso_split,
    warmup_cosine_lr,
    load_surgical_pretrained,
    ensure_dir,
    total_loss,
)


# =============================================================================
# 1. ASD: Articulated Structure Discovery (initialization only)
# =============================================================================

EDGE_TYPES = {"RIGID": 0, "SOFT": 1, "FREE": 2}
ALL_EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]


def discover_articulated_structure(
    dataset: RMITHeatmapDataset,
    img_size: int,
    rigid_cv_threshold: float = 0.08,
    soft_cv_threshold: float = 0.20,
) -> Tuple[np.ndarray, Dict]:
    """
    Discover rigid/joint/free edges from training data.
    Returns initial edge weights (not fixed, will be learned).
    
    Returns:
        edge_weights_init: (6,) array of initial weights for each edge
            rigid → 1.0, soft → 0.5, free → 0.1
        info: dict with structure details
    """
    all_dists = {f"{i}-{j}": [] for i,j in ALL_EDGES}
    
    for rec in dataset.records:
        xy_px = rec["xy"] * (img_size - 1)
        frame_d = []
        for i, j in ALL_EDGES:
            d = np.sqrt(((xy_px[i] - xy_px[j]) ** 2).sum())
            frame_d.append(d)
        frame_mean = max(np.mean(frame_d), 1e-6)
        for idx, (i, j) in enumerate(ALL_EDGES):
            all_dists[f"{i}-{j}"].append(frame_d[idx] / frame_mean)
    
    edge_weights_init = np.zeros(len(ALL_EDGES))
    structure = {"rigid": [], "soft": [], "free": []}
    
    print("[ASD] Discovered structure:")
    for idx, (i, j) in enumerate(ALL_EDGES):
        arr = np.array(all_dists[f"{i}-{j}"])
        cv = arr.std() / max(arr.mean(), 1e-6)
        if cv < rigid_cv_threshold:
            etype = "rigid"
            edge_weights_init[idx] = 1.0
            structure["rigid"].append((i, j))
        elif cv < soft_cv_threshold:
            etype = "soft"
            edge_weights_init[idx] = 0.5
            structure["soft"].append((i, j))
        else:
            etype = "free"
            edge_weights_init[idx] = 0.1
            structure["free"].append((i, j))
        print(f"  kp{i}-kp{j}: CV={cv:.4f} → {etype} (init_weight={edge_weights_init[idx]:.1f})")
    
    return edge_weights_init, structure


# =============================================================================
# 2. LGM: Learnable Geometry Module
# =============================================================================

class LearnableGeometryModule(nn.Module):
    """
    Replaces ALL manual hyperparameters with learnable parameters.
    
    Learnable params:
    - edge_weights (6,): per-edge importance, init from ASD, softmax-normalized
    - log_sigma_base (1): controls base loss weight via uncertainty weighting
    - log_sigma_geo (1): controls geometry loss weight via uncertainty weighting
    
    Loss = base_loss / (2*σ_base²) + geo_loss / (2*σ_geo²) + log(σ_base) + log(σ_geo)
    
    If geometry constraint is harmful, σ_geo → large, geo weight → 0 (auto-disable).
    If geometry constraint helps, σ_geo → small, geo weight → large (auto-enable).
    """
    
    def __init__(self, edge_weights_init: np.ndarray):
        super().__init__()
        # Learnable edge weights (init from ASD discovery)
        self.edge_logits = nn.Parameter(
            torch.tensor(np.log(edge_weights_init + 1e-6), dtype=torch.float32)
        )
        
        # Uncertainty weighting (Kendall et al. CVPR 2018)
        # Initialize σ_base=1 (base loss weight=0.5), σ_geo=3 (geo loss weight~0.05, weak start)
        self.log_sigma_base = nn.Parameter(torch.tensor(0.0))  # σ=1, weight=0.5
        self.log_sigma_geo = nn.Parameter(torch.tensor(1.5))   # σ~4.5, weight~0.02
    
    def get_edge_weights(self) -> torch.Tensor:
        """Returns softmax-normalized edge weights (sum to 1)."""
        return F.softmax(self.edge_logits, dim=-1)
    
    def get_loss_weights(self) -> Tuple[float, float]:
        """Returns (base_weight, geo_weight) from uncertainty weighting."""
        sigma_base = torch.exp(self.log_sigma_base)
        sigma_geo = torch.exp(self.log_sigma_geo)
        base_w = 1.0 / (2.0 * sigma_base ** 2)
        geo_w = 1.0 / (2.0 * sigma_geo ** 2)
        return base_w.item(), geo_w.item()
    
    def compute_weighted_geo_loss(
        self, pred_xy: torch.Tensor, gt_xy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute geometry loss with learnable edge weights.
        
        Args:
            pred_xy: (B, K, 2) in pixels
            gt_xy: (B, K, 2) in pixels
        """
        B = pred_xy.size(0)
        edge_w = self.get_edge_weights()  # (6,) softmax
        
        # Compute scale-normalized distance error per edge
        losses = []
        for idx, (i, j) in enumerate(ALL_EDGES):
            pred_d = torch.sqrt(((pred_xy[:, i] - pred_xy[:, j]) ** 2).sum(dim=-1).clamp_min(1e-6))
            gt_d = torch.sqrt(((gt_xy[:, i] - gt_xy[:, j]) ** 2).sum(dim=-1).clamp_min(1e-6))
            # Relative error: |pred_d/gt_d - 1|
            rel_err = torch.abs(pred_d / gt_d.clamp_min(1e-6) - 1.0)
            losses.append(rel_err * edge_w[idx])
        
        return torch.stack(losses, dim=-1).sum(dim=-1).mean()
    
    def compute_total_loss(
        self, base_loss: torch.Tensor, geo_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Uncertainty-weighted multi-task loss (Kendall et al. CVPR 2018).
        
        L = base/(2σ_base²) + geo/(2σ_geo²) + log(σ_base) + log(σ_geo)
        """
        sigma_base_sq = torch.exp(2 * self.log_sigma_base)
        sigma_geo_sq = torch.exp(2 * self.log_sigma_geo)
        
        total = (base_loss / (2 * sigma_base_sq) 
                 + geo_loss / (2 * sigma_geo_sq)
                 + self.log_sigma_base + self.log_sigma_geo)
        
        base_w = (1.0 / (2 * sigma_base_sq)).item()
        geo_w = (1.0 / (2 * sigma_geo_sq)).item()
        edge_w = self.get_edge_weights().detach().cpu().numpy()
        
        metrics = {
            "base_loss": base_loss.item(),
            "geo_loss": geo_loss.item(),
            "base_weight": base_w,
            "geo_weight": geo_w,
            "sigma_base": math.exp(self.log_sigma_base.item()),
            "sigma_geo": math.exp(self.log_sigma_geo.item()),
            "edge_weights": edge_w,
        }
        return total, metrics


# =============================================================================
# 3. KRM: Keypoint Refinement Module (uses learned edge weights)
# =============================================================================

class KeypointRefinementModule(nn.Module):
    """Graph-based keypoint refinement using learned edge weights."""
    
    def __init__(self, num_keypoints: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.num_kp = num_keypoints
        
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
        
        # Initialize to zero correction
        nn.init.zeros_(self.node_update[-1].weight)
        nn.init.zeros_(self.node_update[-1].bias)
    
    def forward(self, xy: torch.Tensor, edge_weights: torch.Tensor = None) -> torch.Tensor:
        B, K, _ = xy.shape
        edges = [(i, j) for i in range(K) for j in range(i+1, K)]
        
        edge_feats = []
        for idx, (i, j) in enumerate(edges):
            diff = xy[:, i] - xy[:, j]
            dist = torch.sqrt((diff ** 2).sum(dim=-1, keepdim=True).clamp_min(1e-6))
            angle = torch.atan2(diff[..., 1:2], diff[..., 0:1])
            feat = torch.cat([dist, angle], dim=-1)
            feat = self.edge_encoder(feat)
            if edge_weights is not None:
                feat = feat * edge_weights[idx]
            edge_feats.append((i, j, feat))
        
        node_agg = torch.zeros(B, K, self.edge_encoder[-1].out_features, device=xy.device)
        count = torch.zeros(K, device=xy.device)
        for i, j, feat in edge_feats:
            node_agg[:, i] += feat
            node_agg[:, j] += feat
            count[i] += 1
            count[j] += 1
        node_agg = node_agg / count.clamp_min(1).unsqueeze(0).unsqueeze(-1)
        
        node_input = torch.cat([node_agg, xy], dim=-1)
        correction = self.node_update(node_input)
        return xy + correction


# =============================================================================
# 4. Model wrapper
# =============================================================================

class GeometryAwareKeypointNet(nn.Module):
    """Wraps HeatmapKeypointNet with LGM + KRM."""
    
    _TRAIN_ONLY_PARAMS = {
        'use_krm', 'kp_adaptive', 'kp_adapt_momentum', 'kp_adapt_max_weight',
        'kp_adapt_min_weight', 'adabn', 'adabn_passes', 'edge_weights_init',
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        edge_init = kwargs.pop('edge_weights_init', None)
        self.use_krm = kwargs.pop('use_krm', True)
        for p in self._TRAIN_ONLY_PARAMS:
            kwargs.pop(p, None)
        
        self.base = HeatmapKeypointNet(**kwargs)
        self.lgm = LearnableGeometryModule(
            edge_weights_init=edge_init if edge_init is not None else np.ones(6)
        )
        self.krm = KeypointRefinementModule(num_keypoints=4, hidden_dim=64)
    
    def forward(self, x):
        out = self.base(x)
        if self.use_krm:
            edge_w = self.lgm.get_edge_weights()
            out['xy_refined'] = self.krm(out['xy'], edge_w)
            out['xy'] = out['xy_refined']
        return out
    
    @property
    def encoder(self):
        return self.base.encoder
    
    @property
    def backbone_name(self):
        return self.base.backbone_name


# =============================================================================
# 5. Training & Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(model, loader, device, img_size):
    model.eval()
    all_kp = [[] for _ in range(4)]
    pck_20 = pck_50 = total = 0
    
    for batch in tqdm(loader, desc="Eval", leave=False):
        imgs = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        pred = model(imgs)
        pred_px = pred["xy"] * (img_size - 1)
        gt_px = gt_xy * (img_size - 1)
        err = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1))
        for k in range(4):
            all_kp[k].extend(err[:, k].cpu().numpy().tolist())
        pck_20 += (err < 20).float().sum().item()
        pck_50 += (err < 50).float().sum().item()
        total += err.numel()
    
    all_flat = [e for kp_errs in all_kp for e in kp_errs]
    return {
        "mean_error": np.mean(all_flat),
        "kp_errors": [np.mean(e) for e in all_kp],
        "pck_20": pck_20/total, "pck_50": pck_50/total,
    }


def main():
    parser = argparse.ArgumentParser(description="Learnable Geometry Module for keypoint tracking")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--protocol", type=str, default="loso", choices=["loso", "half"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_s")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--decoder_type", type=str, default="heatmap_offset")
    parser.add_argument("--use_offset", action="store_true", default=True)
    parser.add_argument("--decoder_stride", type=int, default=4)
    parser.add_argument("--surg_pretrained", type=str, default=None)
    
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
    
    parser.add_argument("--adabn", action="store_true")
    parser.add_argument("--adabn_passes", type=int, default=1)
    parser.add_argument("--kp_adaptive", action="store_true")
    parser.add_argument("--kp_adapt_momentum", type=float, default=0.9)
    parser.add_argument("--kp_adapt_max_weight", type=float, default=8.0)
    parser.add_argument("--kp_adapt_min_weight", type=float, default=0.3)
    
    parser.add_argument("--aug_color", type=float, default=0.3)
    parser.add_argument("--aug_rotate", type=float, default=10)
    parser.add_argument("--aug_blur", type=float, default=0.3)
    parser.add_argument("--aug_noise", type=float, default=0.05)
    parser.add_argument("--aug_cutout", type=float, default=0.1)
    
    # Model options (NO manual geo hyperparameters - all learnable!)
    parser.add_argument("--use_krm", action="store_true", help="Enable KRM refinement")
    parser.add_argument("--no_lgm", action="store_true", help="Disable LGM (baseline without geometry)")
    parser.add_argument("--geo_rigid_cv", type=float, default=0.08)
    parser.add_argument("--geo_soft_cv", type=float, default=0.20)
    
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    
    args = parser.parse_args()
    if args.no_pretrained:
        args.pretrained = False
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    ensure_dir(args.out_dir)
    
    # Data split
    if args.protocol == "loso":
        train_names, val_names, test_name = resolve_loso_split(args.manifest, args.fold)
        print(f"[Geo] LOSO fold {args.fold}: val={test_name}")
    elif args.protocol == "half":
        seqs = load_rmit_manifest_sequences(args.manifest)
        names = {s["name"] for s in seqs}
        train_names = val_names = names
        test_name = "half"
    
    train_ds = RMITHeatmapDataset(
        args.manifest, args.img_size, train_names, is_training=True,
        sigma=args.sigma, augment=True,
        aug_color=args.aug_color, aug_rotate=args.aug_rotate,
        aug_blur=args.aug_blur, aug_noise=args.aug_noise, aug_cutout=args.aug_cutout,
        half_split="first" if args.protocol == "half" else None,
    )
    val_ds = RMITHeatmapDataset(
        args.manifest, args.img_size, val_names, is_training=False,
        sigma=args.sigma, augment=False,
        half_split="second" if args.protocol == "half" else None,
    )
    print(f"[Geo] Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # === ASD: Discover structure (for initialization only) ===
    edge_weights_init, structure = discover_articulated_structure(
        train_ds, args.img_size,
        rigid_cv_threshold=args.geo_rigid_cv,
        soft_cv_threshold=args.geo_soft_cv,
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    
    # Model with learnable geometry
    model = GeometryAwareKeypointNet(
        img_size=args.img_size, pretrained=args.pretrained, backbone=args.backbone,
        decoder_stride=args.decoder_stride, decoder_type=args.decoder_type,
        use_offset=args.use_offset, use_krm=args.use_krm,
        edge_weights_init=edge_weights_init,
        kp_adaptive=args.kp_adaptive, kp_adapt_momentum=args.kp_adapt_momentum,
        kp_adapt_max_weight=args.kp_adapt_max_weight, kp_adapt_min_weight=args.kp_adapt_min_weight,
        adabn=args.adabn,
    ).to(device)
    
    if args.surg_pretrained:
        load_surgical_pretrained(model, args.surg_pretrained)
        model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_lgm = sum(p.numel() for p in model.lgm.parameters())
    n_krm = sum(p.numel() for p in model.krm.parameters())
    print(f"[Geo] Model: {n_params:,} params (LGM: {n_lgm}, KRM: {n_krm})")
    print(f"[Geo] NO manual geo hyperparameters - all learnable!")
    
    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(args.out_dir, "best.pt")
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        metrics = evaluate(model, val_loader, device, args.img_size)
        print(f"[Geo eval] {test_name}: err={metrics['mean_error']:.2f}px")
        return
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler("cuda")

    # === Two-phase training (auto-adaptive) ===
    # Phase 1 (first half): baseline only, no geo loss → establish strong base
    # Phase 2 (second half): load Phase 1 best, add LGM + geo loss
    # Early stopping selects best across BOTH phases → LGM auto-enabled only if it helps
    phase1_epochs = args.epochs // 2 if not args.no_lgm else 0

    best_err = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_phase1_err = float("inf")
    best_phase1_state = None

    for epoch in range(args.epochs):
        is_phase2 = (epoch >= phase1_epochs) and not args.no_lgm

        # Phase 2 transition: load Phase 1 best, reset optimizer
        if is_phase2 and epoch == phase1_epochs:
            if best_phase1_state is not None:
                model.load_state_dict(best_phase1_state)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
            scaler = GradScaler("cuda")
            print(f"[Geo] === Phase 2: LGM activated (Phase 1 best: {best_phase1_err:.2f}px) ===")

        # KRM only in Phase 2 (needs learned edge weights from LGM)
        model.use_krm = args.use_krm if is_phase2 else False

        model.train()
        sum_loss = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}{'*' if is_phase2 else ''}", leave=False)

        for batch in pbar:
            imgs = batch["img"].to(device)
            gt_xy = batch["gt_xy"].to(device)
            gt_hm = batch["gt_hm"].to(device)
            B = imgs.size(0)

            optimizer.zero_grad()
            with autocast("cuda"):
                pred = model(imgs)
                base_loss, _ = total_loss(pred, gt_hm, gt_xy)

                if not is_phase2:
                    # Phase 1 or baseline: no geometry
                    loss = base_loss
                    metrics = {"base_loss": base_loss.item(), "geo_loss": 0.0,
                               "base_weight": 1.0, "geo_weight": 0.0,
                               "sigma_base": 0.0, "sigma_geo": 0.0}
                else:
                    # Phase 2: LGM + geometry loss
                    pred_px = pred["xy"] * (args.img_size - 1)
                    gt_px = gt_xy * (args.img_size - 1)
                    geo_loss = model.lgm.compute_weighted_geo_loss(pred_px, gt_px)
                    loss, metrics = model.lgm.compute_total_loss(base_loss, geo_loss)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            sum_loss += loss.item() * B
            n += B
            pbar.set_postfix(base=f"{metrics['base_loss']:.3f}", geo=f"{metrics['geo_loss']:.4f}",
                            gw=f"{metrics['geo_weight']:.4f}")

        # Eval
        val_metrics = evaluate(model, val_loader, device, args.img_size)
        val_err = val_metrics["mean_error"]

        # Print
        if is_phase2:
            bw, gw = model.lgm.get_loss_weights()
            ew = model.lgm.get_edge_weights().detach().cpu().numpy()
            ew_str = " ".join([f"{e:.2f}" for e in ew])
            print(f"[Geo] epoch {epoch+1}/{args.epochs} (P2) loss={sum_loss/n:.4f} "
                  f"geo_w={gw:.4f} σ_geo={math.exp(model.lgm.log_sigma_geo.item()):.2f} "
                  f"edge_w=[{ew_str}] val_err={val_err:.2f}px pck@20={val_metrics['pck_20']:.3f}")
        else:
            tag = "(P1)" if not args.no_lgm else ""
            print(f"[Geo] epoch {epoch+1}/{args.epochs} {tag} loss={sum_loss/n:.4f} "
                  f"val_err={val_err:.2f}px pck@20={val_metrics['pck_20']:.3f}")
        for k in range(4):
            print(f"  kp{k}: {val_metrics['kp_errors'][k]:.2f}px")

        # Track Phase 1 best (for Phase 2 initialization)
        if not is_phase2 and val_err < best_phase1_err:
            best_phase1_err = val_err
            best_phase1_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Track global best across both phases
        if val_err < best_err:
            best_err = val_err
            best_epoch = epoch
            patience_counter = 0
            save_dict = {
                "epoch": epoch, "model": model.state_dict(),
                "metric": val_err, "args": vars(args),
                "structure": structure,
                "phase": 2 if is_phase2 else 1,
            }
            if is_phase2:
                save_dict["learned_edge_weights"] = ew
                save_dict["learned_sigmas"] = {
                    "base": math.exp(model.lgm.log_sigma_base.item()),
                    "geo": math.exp(model.lgm.log_sigma_geo.item()),
                }
            torch.save(save_dict, os.path.join(args.out_dir, "best.pt"))
        else:
            if args.no_lgm or is_phase2:
                patience_counter += 1

        # Early stopping (baseline: always; LGM: only in Phase 2)
        if (args.no_lgm or is_phase2) and patience_counter >= args.patience:
            print(f"[Geo] Early stopping at epoch {epoch+1}")
            break

    phase_tag = "Phase 2 (LGM)" if (not args.no_lgm and best_epoch >= phase1_epochs) else "Phase 1 (baseline)"
    print(f"[Geo] Best: {best_err:.2f}px at epoch {best_epoch+1} ({phase_tag})")

    if not args.no_lgm and best_epoch >= phase1_epochs:
        print("\n=== Learned Parameters ===")
        print(f"Edge weights: {ew_str}")
        for idx, (i, j) in enumerate(ALL_EDGES):
            init_w = edge_weights_init[idx]
            print(f"  kp{i}-kp{j}: init={init_w:.1f} → learned={ew[idx]:.3f}")
        print(f"σ_base={math.exp(model.lgm.log_sigma_base.item()):.3f} → base_weight={bw:.4f}")
        print(f"σ_geo={math.exp(model.lgm.log_sigma_geo.item()):.3f} → geo_weight={gw:.4f}")
    elif not args.no_lgm:
        print("\n=== LGM NOT selected (Phase 1 baseline won) ===")


if __name__ == "__main__":
    main()
