import argparse
import json
import os
import random
import sys
from typing import Dict, Optional

# 确保项目根目录在 PYTHONPATH 中，以便 import surgical_tracking.src.dr_2
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import surgical_tracking.src.dr_2 as base
from surgical_tracking.src.dr_2 import *


RMIT_REFERENCE_SIZE = 512.0
RMIT_REFERENCE_P10 = 20.0
RMIT_HEATMAP_POS_WEIGHT = 32.0
RMIT_HEATMAP_MSE_WEIGHT = 0.25


def compute_tracking_metrics_v3(
    pred_xy_px: torch.Tensor,
    gt_xy_px: torch.Tensor,
    image_size: int,
    reference_size: float = RMIT_REFERENCE_SIZE,
    reference_p10: float = RMIT_REFERENCE_P10,
) -> Dict[str, float]:
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


def stable_heatmap_loss_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = RMIT_HEATMAP_POS_WEIGHT,
    mse_weight: float = RMIT_HEATMAP_MSE_WEIGHT,
) -> torch.Tensor:
    target = target.clamp(0.0, 1.0)
    probs = torch.sigmoid(logits)
    weights = 1.0 + pos_weight * target
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weighted_bce = (bce * weights).sum() / weights.sum().clamp_min(1.0)
    mse = F.mse_loss(probs, target)
    return weighted_bce + mse_weight * mse


def resolve_rmit_reference_protocol(img_size: int):
    if img_size >= 512:
        return 512.0, 20.0
    return 256.0, 10.0


def load_rmit_manifest_sequences(manifest_json: str):
    with open(manifest_json, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["sequences"] if isinstance(manifest, dict) and "sequences" in manifest else manifest


def resolve_rmit_split_names(
    manifest_json: str,
    protocol: str,
    test_seq: Optional[str],
    val_ratio: float,
    seed: int,
):
    sequences = load_rmit_manifest_sequences(manifest_json)
    seq_names = [seq["name"] for seq in sequences]
    if protocol == "leave_one_out" and test_seq:
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


class RMITTrackingDatasetV3(base.Dataset):
    def __init__(
        self,
        manifest_json: str,
        img_size: int,
        is_training: bool,
        max_pairs_per_seq: Optional[int],
        split_names,
    ):
        self.img_size = img_size
        self.is_training = is_training
        self.items = []
        sequences = load_rmit_manifest_sequences(manifest_json)
        selected = [seq for seq in sequences if seq["name"] in split_names]
        for seq in selected:
            records = self._load_sequence(seq["frames_dir"], seq["ann_csv"])
            pairs = []
            for i in range(1, len(records)):
                prev_rec, curr_rec = records[i - 1], records[i]
                pairs.append({
                    "template_path": prev_rec["path"],
                    "search_path": curr_rec["path"],
                    "search_xy": torch.tensor([curr_rec["x"], curr_rec["y"]], dtype=torch.float32),
                })
            if max_pairs_per_seq is not None and is_training:
                random.shuffle(pairs)
                pairs = pairs[:max_pairs_per_seq]
            self.items.extend(pairs)

    def _load_sequence(self, frames_dir: str, ann_csv: str):
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        template_raw = base.load_image_rgb(item["template_path"])
        search_raw = base.load_image_rgb(item["search_path"])
        gt_xy = item["search_xy"].clone()
        scale_x = self.img_size / max(search_raw.size[0], 1)
        scale_y = self.img_size / max(search_raw.size[1], 1)
        gt_xy_px = torch.tensor([gt_xy[0] * scale_x, gt_xy[1] * scale_y], dtype=torch.float32)
        template = base.pil_resize(template_raw, self.img_size, False)
        search = base.pil_resize(search_raw, self.img_size, False)
        if self.is_training:
            template, search, gt_xy_px = base.EnhancedAugmentation.augment_tracking(template, search, gt_xy_px, True)
        gt_xy_norm = gt_xy_px / max(float(self.img_size - 1), 1.0)
        sigma = max(2.0, self.img_size / 64.0)
        gt_hm = base._make_gaussian(self.img_size, gt_xy_px[0].item(), gt_xy_px[1].item(), sigma=sigma)
        return {
            "template": base.normalize_img(base.pil_to_tensor(template)),
            "search": base.normalize_img(base.pil_to_tensor(search)),
            "gt_xy_px": gt_xy_px,
            "gt_xy_norm": gt_xy_norm,
            "gt_hm": gt_hm,
        }


class OneStreamTipTrackerV3(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        encoder: nn.Module,
        embed_dim: int,
        feature_layers,
        use_anatomy_anchor: bool = False,
        use_uncertainty_head: bool = False,
        fpn_out_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.encoder = encoder
        self.feature_layers = feature_layers
        self.use_anatomy_anchor = use_anatomy_anchor
        self.use_uncertainty_head = use_uncertainty_head
        self.search_grid = img_size // patch_size
        self.num_search = self.search_grid * self.search_grid
        self.num_template = self.num_search
        self.pos_ts = nn.Parameter(torch.zeros(1, 1 + self.num_template + self.num_search, embed_dim))
        nn.init.trunc_normal_(self.pos_ts, std=0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.fuser = base.MultiScaleTokenFuser(embed_dim, fpn_out_dim, len(feature_layers))
        if use_anatomy_anchor:
            self.anatomy_attention = nn.Sequential(
                nn.Conv2d(fpn_out_dim + 2, fpn_out_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(fpn_out_dim, fpn_out_dim, kernel_size=1),
            )
        if use_uncertainty_head:
            self.hm_head = base.UncertaintyHeatmapHead(fpn_out_dim)
        else:
            self.hm_head = nn.Sequential(
                nn.Conv2d(fpn_out_dim, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(256, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=1),
            )
        self.xy_head = nn.Sequential(
            nn.Conv2d(fpn_out_dim, fpn_out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(fpn_out_dim, max(64, fpn_out_dim // 2)),
            nn.GELU(),
            nn.Linear(max(64, fpn_out_dim // 2), 2),
            nn.Sigmoid(),
        )

    def forward(self, template: torch.Tensor, search: torch.Tensor, anatomy_mask: Optional[torch.Tensor] = None):
        self.encoder.set_domain("surg")
        bsz = template.size(0)
        template_tokens = self.encoder.patch(template)
        search_tokens = self.encoder.patch(search)
        cls_token = self.encoder.cls.expand(bsz, -1, -1)
        hidden = torch.cat([cls_token, template_tokens, search_tokens], dim=1)
        hidden = hidden + self.pos_ts[:, : hidden.size(1)]
        hidden = self.encoder.drop(hidden)
        selected = {}
        target = set(self.feature_layers)
        for idx, block in enumerate(self.encoder.blocks):
            hidden = block(hidden)
            if idx in target:
                normalized = self.norm(hidden)
                selected[idx] = normalized[:, 1 + self.num_template: 1 + self.num_template + self.num_search]
        search_multi = [selected[idx] for idx in self.feature_layers]
        feat = self.fuser(search_multi, self.search_grid)
        if self.use_anatomy_anchor and anatomy_mask is not None:
            anatomy_resized = F.interpolate(anatomy_mask, size=(self.search_grid, self.search_grid), mode="bilinear", align_corners=False)
            feat = self.anatomy_attention(torch.cat([feat, anatomy_resized], dim=1))
        xy_norm = self.xy_head(feat)
        if self.use_uncertainty_head:
            out = self.hm_head(feat)
            return {
                "heatmap_logits": F.interpolate(out["heatmap_logits"], size=(self.img_size, self.img_size), mode="bilinear", align_corners=False),
                "log_variance": F.interpolate(out["log_variance"], size=(self.img_size, self.img_size), mode="bilinear", align_corners=False),
                "uncertainty": F.interpolate(out["uncertainty"], size=(self.img_size, self.img_size), mode="bilinear", align_corners=False),
                "xy_norm": xy_norm,
            }
        heatmap_logits = self.hm_head(feat)
        return {
            "heatmap_logits": F.interpolate(heatmap_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False),
            "log_variance": None,
            "uncertainty": None,
            "xy_norm": xy_norm,
        }


@torch.no_grad()
def eval_rmit_v3(model: nn.Module, loader: DataLoader, device: torch.device, feature_layers, use_anatomy_anchor: bool, image_size: int) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []
    seg_module = model["seg"] if "seg" in model else None
    for batch in loader:
        template = batch["template"].to(device)
        search = batch["search"].to(device)
        gt_xy_px = batch["gt_xy_px"].to(device)
        anatomy_mask = None
        if use_anatomy_anchor and seg_module is not None:
            _, search_tok = base.encoder_forward_selected(model["enc"], search, "surg", feature_layers)
            logits = seg_module(search_tok)
            anatomy_mask = torch.sigmoid(logits)
        tracker_out = model["tracker"](template, search, anatomy_mask=anatomy_mask)
        if tracker_out.get("xy_norm") is not None:
            pred_xy_px = tracker_out["xy_norm"] * max(float(image_size - 1), 1.0)
        else:
            pred_xy_px = base.soft_argmax_2d(tracker_out["heatmap_logits"])
        preds.append(pred_xy_px.cpu())
        gts.append(gt_xy_px.cpu())
    if not preds:
        return {
            "mse_px": 0.0,
            "mean_error_px": 0.0,
            "mean_error_ref": 0.0,
            "mean_error_width_norm": 0.0,
            "mean_error_diag_norm": 0.0,
            "precision@5px": 0.0,
            "precision@10px": 0.0,
            "precision@20px": 0.0,
            "precision@ref10": 0.0,
            "failure_rate@20px": 0.0,
        }
    return compute_tracking_metrics_v3(torch.cat(preds, dim=0), torch.cat(gts, dim=0), image_size=image_size)


def train_stage_rmit_v3(cfg, device: torch.device) -> None:
    base.ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)
    enc = base.build_encoder(cfg).to(device)
    seg = (
        base.SegDecoderV2(cfg.img_size, cfg.patch_size, cfg.embed_dim, cfg.feature_layers, out_ch=2, fpn_out_dim=cfg.fpn_out_dim).to(device)
        if cfg.use_anatomy_anchor
        else None
    )
    tracker = OneStreamTipTrackerV3(
        cfg.img_size,
        cfg.patch_size,
        enc,
        cfg.embed_dim,
        cfg.feature_layers,
        use_anatomy_anchor=cfg.use_anatomy_anchor,
        use_uncertainty_head=cfg.use_uncertainty_head,
        fpn_out_dim=cfg.fpn_out_dim,
    ).to(device)
    task_weight = base.TaskUncertaintyWeighting(["hm", "xy"]).to(device) if cfg.use_task_uncertainty else None
    model_items = {"enc": enc, "tracker": tracker}
    if seg is not None:
        model_items["seg"] = seg
    if task_weight is not None:
        model_items["task_weight"] = task_weight
    model = nn.ModuleDict(model_items)
    if cfg.pretrained_encoder_ckpt:
        base.load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)
    if cfg.init_ckpt:
        print(f"[RMIT-V3] Loading init ckpt: {cfg.init_ckpt}")
        base.load_ckpt(cfg.init_ckpt, model, optim=None)
        try:
            enc.copy_domain_params(src_domain="intra", dst_domain="surg")
        except Exception:
            pass
    train_names, val_names = resolve_rmit_split_names(
        manifest_json=cfg.rmit_manifest_json,
        protocol=cfg.rmit_protocol,
        test_seq=cfg.rmit_test_seq,
        val_ratio=cfg.rmit_val_ratio,
        seed=cfg.seed,
    )
    print(f"[RMIT-V3] Train sequences: {sorted(train_names)}")
    print(f"[RMIT-V3] Val sequences: {sorted(val_names)}")
    print(f"[RMIT-V3] Anatomy anchor enabled: {cfg.use_anatomy_anchor}")
    train_ds = RMITTrackingDatasetV3(
        cfg.rmit_manifest_json,
        cfg.img_size,
        is_training=True,
        max_pairs_per_seq=cfg.max_pairs_per_seq,
        split_names=train_names,
    )
    val_ds = RMITTrackingDatasetV3(
        cfg.rmit_manifest_json,
        cfg.img_size,
        is_training=False,
        max_pairs_per_seq=None,
        split_names=val_names,
    )
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    params = []
    for name, param in model.named_parameters():
        is_adapter = ("A." in name) or ("B." in name) or ("M." in name)
        is_tracker = name.startswith("tracker.")
        is_seg = name.startswith("seg.")
        is_task_weight = name.startswith("task_weight.")
        if cfg.use_anatomy_anchor and is_seg:
            param.requires_grad = False
        elif cfg.freeze_backbone:
            if is_adapter or is_tracker or is_task_weight:
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False
        else:
            if is_seg and cfg.use_anatomy_anchor:
                param.requires_grad = False
            else:
                param.requires_grad = True
                params.append(param)
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = base.ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    best = 1e9
    global_step = 0
    total_steps = cfg.epochs * max(1, len(train_ld)) // cfg.grad_accum
    coord_ref = max(float(getattr(cfg, "rmit_coord_ref_size", RMIT_REFERENCE_SIZE)), 1.0)
    p10_ref = float(getattr(cfg, "rmit_eval_ref_p10", RMIT_REFERENCE_P10))
    grad_clip = 5.0
    for epoch in range(cfg.epochs):
        model.train()
        if seg is not None:
            seg.eval()
        running = 0.0
        running_hm = 0.0
        running_xy_ref = 0.0
        optim.zero_grad(set_to_none=True)
        iterator = enumerate(train_ld)
        if base.tqdm is not None:
            iterator = base.tqdm(iterator, total=len(train_ld), desc=f"RMIT-V3 {epoch+1}/{cfg.epochs}", leave=False)
        for it, batch in iterator:
            template = batch["template"].to(device, non_blocking=True)
            search = batch["search"].to(device, non_blocking=True)
            gt_hm = batch["gt_hm"].to(device, non_blocking=True)
            gt_xy_px = batch["gt_xy_px"].to(device, non_blocking=True)
            gt_xy_norm = batch["gt_xy_norm"].to(device, non_blocking=True)
            lr_now = base.cosine_lr(cfg.lr, global_step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                anatomy_mask = None
                if seg is not None:
                    with torch.no_grad():
                        _, search_tokens = base.encoder_forward_selected(enc, search, "surg", cfg.feature_layers)
                        anatomy_mask = torch.sigmoid(seg(search_tokens)).detach()
                tracker_out = tracker(template, search, anatomy_mask=anatomy_mask)
                pred_hm = tracker_out["heatmap_logits"]
                loss_hm = stable_heatmap_loss_logits(pred_hm, gt_hm)
                if tracker_out.get("xy_norm") is not None:
                    pred_xy_norm = tracker_out["xy_norm"]
                else:
                    pred_xy_px = base.soft_argmax_2d(pred_hm)
                    pred_xy_norm = pred_xy_px / max(float(cfg.img_size - 1), 1.0)
                pred_xy_ref = pred_xy_norm * coord_ref
                gt_xy_ref = gt_xy_norm * coord_ref
                loss_xy = F.smooth_l1_loss(pred_xy_ref, gt_xy_ref)
                if task_weight is not None:
                    total_loss, _ = task_weight({"hm": loss_hm, "xy": loss_xy})
                    loss = total_loss / max(1, cfg.grad_accum)
                else:
                    loss = (cfg.lambda_hm * loss_hm + cfg.lambda_xy * loss_xy) / max(1, cfg.grad_accum)
            if not torch.isfinite(loss.detach()):
                print(f"[RMIT-V3] skip non-finite loss at epoch={epoch+1} iter={it+1}")
                optim.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            running += loss.item() * max(1, cfg.grad_accum)
            running_hm += loss_hm.item()
            running_xy_ref += loss_xy.item()
            if (it + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if global_step % 20 == 0:
                    writer.add_scalar("rmit_v3/train_loss", loss.item() * max(1, cfg.grad_accum), global_step)
                    writer.add_scalar("rmit_v3/loss_hm", loss_hm.item(), global_step)
                    writer.add_scalar("rmit_v3/loss_xy_ref", loss_xy.item(), global_step)
                    if task_weight is not None:
                        writer.add_scalar("rmit_v3/log_var_hm", task_weight.log_vars["hm"].item(), global_step)
                        writer.add_scalar("rmit_v3/log_var_xy", task_weight.log_vars["xy"].item(), global_step)
                    writer.add_scalar("rmit_v3/lr", lr_now, global_step)
                global_step += 1
        if len(train_ld) % cfg.grad_accum != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1
        metrics = eval_rmit_v3(model, val_ld, device, cfg.feature_layers, cfg.use_anatomy_anchor, image_size=cfg.img_size)
        ema_metrics = eval_rmit_v3(ema.module, val_ld, device, cfg.feature_layers, cfg.use_anatomy_anchor, image_size=cfg.img_size) if ema is not None else None
        writer.add_scalar("rmit_v3/val_mean_error_px", metrics["mean_error_px"], epoch)
        writer.add_scalar("rmit_v3/val_mean_error_ref", metrics["mean_error_ref"], epoch)
        writer.add_scalar("rmit_v3/val_mean_error_width_norm", metrics["mean_error_width_norm"], epoch)
        writer.add_scalar("rmit_v3/val_precision_ref10", metrics["precision@ref10"], epoch)
        if cfg.eval_metrics:
            writer.add_scalar("rmit_v3/val_precision@10px", metrics["precision@10px"], epoch)
        msg = (
            f"[RMIT-V3] epoch {epoch+1}/{cfg.epochs} "
            f"train_loss={running/len(train_ld):.4f} "
            f"hm_loss={running_hm/len(train_ld):.4f} "
            f"xy_ref_loss={running_xy_ref/len(train_ld):.4f} "
            f"val_mean_error_px={metrics['mean_error_px']:.3f} "
            f"val_mean_error_ref={metrics['mean_error_ref']:.3f} "
            f"p@ref10={metrics['precision@ref10']:.3f}"
        )
        if cfg.eval_metrics:
            msg += f" p@10={metrics['precision@10px']:.3f}"
        if ema_metrics is not None:
            msg += f" ema_val_mean_error_ref={ema_metrics['mean_error_ref']:.3f}"
        print(msg)
        base.append_epoch_log(cfg.out_dir, msg)
        metric = ema_metrics["mean_error_ref"] if ema_metrics is not None else metrics["mean_error_ref"]
        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "ema": ema.state_dict() if ema is not None else None,
            "extra": {
                "stage": "rmit_v3_normalized",
                "reference_size": coord_ref,
                "reference_p10": p10_ref,
            },
        }
        base.save_ckpt(f"{cfg.out_dir}/last.pt", payload)
        if metric < best:
            best = metric
            payload["best_metric"] = best
            base.save_ckpt(f"{cfg.out_dir}/best.pt", payload)
    writer.close()


def parse_args():
    cfg = base.parse_args()
    coord_ref, eval_ref_p10 = resolve_rmit_reference_protocol(cfg.img_size)
    setattr(cfg, "rmit_coord_ref_size", coord_ref)
    setattr(cfg, "rmit_eval_ref_p10", eval_ref_p10)
    return cfg


def main() -> None:
    cfg = parse_args()
    base.seed_all(cfg.seed)
    base.ensure_dir(cfg.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage: {cfg.stage}")
    print(f"LoRA layer ranks: {cfg.lora_layer_ranks}")
    print(f"RMIT reference size: {cfg.rmit_coord_ref_size}")
    if cfg.stage == "dr":
        base.train_stage_dr(cfg, device)
    elif cfg.stage == "fovea":
        if not cfg.fovea_pairs_json:
            raise ValueError("Need --fovea_pairs_json")
        base.train_stage_fovea_v2(cfg, device)
    elif cfg.stage == "rmit":
        if not cfg.rmit_manifest_json:
            raise ValueError("Need --rmit_manifest_json")
        train_stage_rmit_v3(cfg, device)
    else:
        raise ValueError(cfg.stage)


if __name__ == "__main__":
    main()
