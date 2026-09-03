import argparse
import copy
import csv
import json
import math
import os
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from torchvision.datasets import ImageFolder
except Exception:
    ImageFolder = None

src = SimpleNamespace()


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_epoch_log(out_dir: str, msg: str) -> None:
    with open(os.path.join(out_dir, "epoch.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def cosine_lr(base_lr: float, step: int, total_steps: int) -> float:
    if total_steps <= 0:
        return base_lr
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))


def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask(path: str) -> Image.Image:
    return Image.open(path).convert("L")


def pil_resize(img: Image.Image, size: int, is_mask: bool = False) -> Image.Image:
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return img.resize((size, size), resample=resample)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    arr = arr.transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr)


def normalize_img(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype).view(3, 1, 1)
    return (x - mean) / std


def binarize_mask(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def _make_gaussian(size: int, x: float, y: float, sigma: float) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=torch.float32),
        torch.arange(size, dtype=torch.float32),
        indexing="ij",
    )
    heatmap = torch.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return heatmap.unsqueeze(0)


def soft_argmax_2d(logits: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    bsz, _, h, w = logits.shape
    probs = F.softmax((logits * beta).flatten(2), dim=-1).view(bsz, 1, h, w)
    ys = torch.linspace(0, h - 1, h, device=logits.device, dtype=logits.dtype).view(1, 1, h, 1)
    xs = torch.linspace(0, w - 1, w, device=logits.device, dtype=logits.dtype).view(1, 1, 1, w)
    y = (probs * ys).sum(dim=(2, 3))
    x = (probs * xs).sum(dim=(2, 3))
    return torch.cat([x, y], dim=1)


def compute_tracking_metrics(pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> Dict[str, float]:
    diff = pred_xy.float() - gt_xy.float()
    dist = torch.sqrt((diff ** 2).sum(dim=1))
    return {
        "mse": float((diff ** 2).mean().item()),
        "mean_error": float(dist.mean().item()),
        "precision@5px": float((dist <= 5).float().mean().item()),
        "precision@10px": float((dist <= 10).float().mean().item()),
        "precision@20px": float((dist <= 20).float().mean().item()),
        "failure_rate": float((dist > 20).float().mean().item()),
    }


def dice_loss_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * target).sum(dim=(2, 3))
    den = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    return 1.0 - ((num + eps) / den).mean()


def bce_loss_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target)


class EnhancedAugmentation:
    @staticmethod
    def _jitter(img: Image.Image, s: float = 0.15) -> Image.Image:
        if random.random() < 0.8:
            img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-s, s))
        if random.random() < 0.8:
            img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-s, s))
        if random.random() < 0.4:
            img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-s, s))
        if random.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.8)))
        return img

    @staticmethod
    def augment_dr(img: Image.Image, is_training: bool = True) -> Image.Image:
        if not is_training:
            return img
        return EnhancedAugmentation._jitter(img, s=0.18)

    @staticmethod
    def augment_fovea(img: Image.Image, is_training: bool = True) -> Image.Image:
        if not is_training:
            return img
        return EnhancedAugmentation._jitter(img, s=0.12)

    @staticmethod
    def augment_tracking(template: Image.Image, search: Image.Image, gt_xy: torch.Tensor, is_training: bool = True):
        if not is_training:
            return template, search, gt_xy
        template = EnhancedAugmentation._jitter(template, s=0.1)
        search = EnhancedAugmentation._jitter(search, s=0.1)
        if random.random() < 0.3:
            template = template.transpose(Image.FLIP_LEFT_RIGHT)
            search = search.transpose(Image.FLIP_LEFT_RIGHT)
            gt_xy = gt_xy.clone()
            gt_xy[0] = template.size[0] - 1 - gt_xy[0]
        return template, search, gt_xy


def boundary_region(mask: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size < 3:
        return torch.zeros_like(mask)
    pad = kernel_size // 2
    mask = mask.float()
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def edge_weighted_bce_loss_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_kernel_size: int = 5,
    edge_weight: float = 4.0,
) -> torch.Tensor:
    weights = 1.0 + edge_weight * boundary_region(target, boundary_kernel_size)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def edge_aware_seg_loss_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_kernel_size: int = 5,
    edge_weight: float = 4.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_boundary = boundary_region(probs, boundary_kernel_size)
    target_boundary = boundary_region(target, boundary_kernel_size)
    boundary_l1 = F.l1_loss(pred_boundary, target_boundary)
    weighted_bce = edge_weighted_bce_loss_logits(logits, target, boundary_kernel_size, edge_weight)
    return weighted_bce + boundary_l1


class TaskUncertaintyWeighting(nn.Module):
    def __init__(self, task_names: List[str], init_log_var: float = 0.0):
        super().__init__()
        self.task_names = list(task_names)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.tensor(float(init_log_var), dtype=torch.float32))
            for name in self.task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = next(iter(losses.values())).new_tensor(0.0)
        weighted = {}
        for name in self.task_names:
            loss = losses[name]
            log_var = self.log_vars[name].to(device=loss.device, dtype=loss.dtype)
            weighted_loss = torch.exp(-log_var) * loss + log_var
            total = total + weighted_loss
            weighted[name] = weighted_loss
        return total, weighted


class SwiGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.silu(gate)


class DomainLoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        domains: List[str],
        r: int = 8,
        alpha: float = 16.0,
        bias: bool = True,
        use_residual_lora: bool = False,
        residual_alpha: float = 0.1,
        use_dora: bool = True,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.domains = list(domains)
        self.r = max(0, r)
        self.scale = alpha / max(1, self.r) if self.r > 0 else 0.0
        self.use_residual_lora = use_residual_lora
        self.residual_alpha = residual_alpha
        self.use_dora = use_dora and self.r > 0
        if self.r > 0:
            self.A = nn.ParameterDict({d: nn.Parameter(torch.zeros(self.r, in_features)) for d in self.domains})
            self.B = nn.ParameterDict({d: nn.Parameter(torch.zeros(out_features, self.r)) for d in self.domains})
            if self.use_dora:
                base_norm = torch.linalg.vector_norm(self.linear.weight.detach(), dim=1, keepdim=True).clamp_min(1e-6)
                self.M = nn.ParameterDict({d: nn.Parameter(base_norm.clone()) for d in self.domains})
            else:
                self.M = nn.ParameterDict()
            for d in self.domains:
                nn.init.kaiming_uniform_(self.A[d], a=math.sqrt(5))
                nn.init.zeros_(self.B[d])
        else:
            self.A = nn.ParameterDict()
            self.B = nn.ParameterDict()
            self.M = nn.ParameterDict()
        self.active_domain = self.domains[0]

    def set_domain(self, domain: str) -> None:
        if domain in self.domains:
            self.active_domain = domain

    def copy_domain_params(self, src_domain: str, dst_domain: str) -> None:
        if self.r <= 0 or src_domain not in self.A or dst_domain not in self.A:
            return
        self.A[dst_domain].data.copy_(self.A[src_domain].data)
        self.B[dst_domain].data.copy_(self.B[src_domain].data)
        if self.use_dora and src_domain in self.M and dst_domain in self.M:
            self.M[dst_domain].data.copy_(self.M[src_domain].data)

    def _active_domains(self) -> List[str]:
        if not self.use_residual_lora:
            return [self.active_domain]
        idx = self.domains.index(self.active_domain)
        return self.domains[: idx + 1]

    def _delta_weight(self) -> torch.Tensor:
        delta = self.linear.weight.new_zeros(self.linear.weight.shape)
        domains = self._active_domains()
        for idx, domain in enumerate(domains):
            scale = self.scale if idx == len(domains) - 1 else self.scale * self.residual_alpha
            delta = delta + scale * (self.B[domain] @ self.A[domain])
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r <= 0:
            return self.linear(x)
        adapted_weight = self.linear.weight + self._delta_weight()
        if self.use_dora:
            row_norm = torch.linalg.vector_norm(adapted_weight, dim=1, keepdim=True).clamp_min(1e-6)
            adapted_weight = self.M[self.active_domain] * adapted_weight / row_norm
        return F.linear(x, adapted_weight, self.linear.bias)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, domains: List[str], drop: float, **kwargs):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = DomainLoRALinear(dim, hidden * 2, domains, **kwargs)
        self.act = SwiGLU()
        self.fc2 = DomainLoRALinear(hidden, dim, domains, **kwargs)
        self.drop = nn.Dropout(drop)

    def set_domain(self, domain: str) -> None:
        self.fc1.set_domain(domain)
        self.fc2.set_domain(domain)

    def copy_domain_params(self, src_domain: str, dst_domain: str) -> None:
        self.fc1.copy_domain_params(src_domain, dst_domain)
        self.fc2.copy_domain_params(src_domain, dst_domain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, domains: List[str], attn_drop: float = 0.0, proj_drop: float = 0.0, **kwargs):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = DomainLoRALinear(dim, dim * 3, domains, **kwargs)
        self.proj = DomainLoRALinear(dim, dim, domains, **kwargs)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def set_domain(self, domain: str) -> None:
        self.qkv.set_domain(domain)
        self.proj.set_domain(domain)

    def copy_domain_params(self, src_domain: str, dst_domain: str) -> None:
        self.qkv.copy_domain_params(src_domain, dst_domain)
        self.proj.copy_domain_params(src_domain, dst_domain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n, c = x.shape
        qkv = self.qkv(x).reshape(bsz, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(bsz, n, c)
        return self.proj_drop(self.proj(out))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, domains: List[str], drop: float = 0.0, attn_drop: float = 0.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, domains, attn_drop=attn_drop, proj_drop=drop, **kwargs)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, domains, drop=drop, **kwargs)

    def set_domain(self, domain: str) -> None:
        self.attn.set_domain(domain)
        self.mlp.set_domain(domain)

    def copy_domain_params(self, src_domain: str, dst_domain: str) -> None:
        self.attn.copy_domain_params(src_domain, dst_domain)
        self.mlp.copy_domain_params(src_domain, dst_domain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        domains: List[str],
        lora_ranks: List[int],
        use_residual_lora: bool,
        residual_alpha: float,
        use_dora: bool,
    ):
        super().__init__()
        if len(lora_ranks) != depth:
            raise ValueError(f"Expected {depth} LoRA ranks, got {len(lora_ranks)}")
        self.domains = list(domains)
        self.lora_ranks = list(lora_ranks)
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch.num_patches, embed_dim))
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList()
        for rank in self.lora_ranks:
            kwargs = dict(
                r=rank,
                alpha=max(1, rank * 2),
                use_residual_lora=use_residual_lora,
                residual_alpha=residual_alpha,
                use_dora=use_dora,
            )
            self.blocks.append(Block(embed_dim, heads, mlp_ratio, self.domains, drop=drop, attn_drop=attn_drop, **kwargs))
        self.norm = nn.LayerNorm(embed_dim)
        self.active_domain = self.domains[0]
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def set_domain(self, domain: str) -> None:
        self.active_domain = domain
        for block in self.blocks:
            block.set_domain(domain)

    def copy_domain_params(self, src_domain: str, dst_domain: str) -> None:
        for block in self.blocks:
            block.copy_domain_params(src_domain, dst_domain)

    def forward_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = x.size(0)
        x = self.patch(x)
        cls = self.cls.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.drop(x + self.pos[:, : x.size(1)])
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls, _ = self.forward_tokens(x)
        return cls


def build_encoder(cfg) -> ViTEncoder:
    return ViTEncoder(
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        heads=cfg.heads,
        mlp_ratio=cfg.mlp_ratio,
        drop=cfg.drop,
        attn_drop=cfg.attn_drop,
        domains=["dr", "pre", "intra", "surg"],
        lora_ranks=cfg.lora_layer_ranks,
        use_residual_lora=cfg.use_residual_lora,
        residual_alpha=cfg.residual_alpha,
        use_dora=cfg.use_dora,
    )


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class DRHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ModelEMA:
    def __init__(self, module: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(module).eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        msd = module.state_dict()
        for key, value in self.module.state_dict().items():
            if key in msd:
                src_value = msd[key].detach()
                if value.dtype.is_floating_point:
                    value.mul_(self.decay).add_(src_value, alpha=1.0 - self.decay)
                else:
                    value.copy_(src_value)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.module.state_dict()


def save_ckpt(path: str, payload: dict) -> None:
    torch.save(payload, path)


def resize_pos_embed(pos_embed: torch.Tensor, new_grid: int) -> torch.Tensor:
    cls = pos_embed[:, :1]
    tok = pos_embed[:, 1:]
    old_grid = int(math.sqrt(tok.size(1)))
    tok = tok.transpose(1, 2).reshape(1, tok.size(-1), old_grid, old_grid)
    tok = F.interpolate(tok, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
    tok = tok.flatten(2).transpose(1, 2)
    return torch.cat([cls, tok], dim=1)


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefixes: List[str]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        out[new_key] = value
    return out


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
    filtered, resized, skipped = {}, [], []
    for key, value in state_dict.items():
        if key not in target_state:
            continue
        target_value = target_state[key]
        if value.shape == target_value.shape:
            filtered[key] = value
            continue
        if value.ndim == 3 and (key == "pos" or key.endswith(".pos")):
            new_grid = _infer_pos_resize_grid(model, key)
            if new_grid is not None:
                try:
                    resized_value = resize_pos_embed(value, new_grid)
                    if resized_value.shape == target_value.shape:
                        filtered[key] = resized_value
                        resized.append(f"{key}: {tuple(value.shape)} -> {tuple(resized_value.shape)}")
                        continue
                except Exception as exc:
                    skipped.append(f"{key}: resize failed ({exc})")
                    continue
        skipped.append(f"{key}: ckpt {tuple(value.shape)} != model {tuple(target_value.shape)}")
    return filtered, resized, skipped


def load_ckpt(path: str, model: nn.Module, optim: Optional[torch.optim.Optimizer] = None, scaler=None, ema: Optional[ModelEMA] = None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    compatible, resized, skipped = _filter_compatible_state_dict(model, state_dict)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"[CKPT] Loaded {len(compatible)} tensors from {path}")
    if resized:
        print(f"[CKPT] Resized position embeddings ({len(resized)}):")
        for item in resized[:10]:
            print(f"  {item}")
    if skipped:
        print(f"[CKPT] Skipped incompatible tensors ({len(skipped)}):")
        for item in skipped[:10]:
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
    if ema is not None and ckpt.get("ema") is not None:
        try:
            ema.module.load_state_dict(ckpt["ema"], strict=False)
        except Exception:
            pass
    return ckpt


def load_pretrained_encoder(encoder: ViTEncoder, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = _strip_prefix(state_dict, ["enc.", "encoder."])
    compatible, resized, skipped = _filter_compatible_state_dict(encoder, state_dict)
    encoder.load_state_dict(compatible, strict=False)
    print(f"[PRETRAIN] Loaded {len(compatible)} tensors from {ckpt_path}")
    if resized:
        print(f"[PRETRAIN] Resized keys: {len(resized)}")
    if skipped:
        print(f"[PRETRAIN] Skipped incompatible tensors: {len(skipped)}")


class DRDataset(Dataset):
    def __init__(self, csv_path: str, img_size: int, augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        self.items: List[Tuple[str, int]] = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.items.append((row["image_path"], int(row["label"])))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = pil_resize(load_image_rgb(path), self.img_size, False)
        img = EnhancedAugmentation.augment_dr(img, self.augment)
        return normalize_img(pil_to_tensor(img)), torch.tensor(label, dtype=torch.long)


class DRImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int, indices: Optional[List[int]] = None, augment: bool = True, label_map: Optional[Dict[str, int]] = None):
        if ImageFolder is None:
            raise ImportError("torchvision is required for --dr_imagefolder_dir")
        self.base = ImageFolder(root=root_dir)
        self.img_size = img_size
        self.augment = augment
        self.indices = list(indices) if indices is not None else list(range(len(self.base.samples)))
        self.label_map = dict(label_map) if label_map is not None else {name: idx for name, idx in self.base.class_to_idx.items()}
        self.items = []
        for sample_idx in self.indices:
            path, orig_label = self.base.samples[sample_idx]
            class_name = self.base.classes[orig_label]
            self.items.append((path, self.label_map[class_name]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = pil_resize(load_image_rgb(path), self.img_size, False)
        img = EnhancedAugmentation.augment_dr(img, self.augment)
        return normalize_img(pil_to_tensor(img)), torch.tensor(label, dtype=torch.long)


def build_dr_datasets(cfg) -> Tuple[Dataset, Dataset]:
    if cfg.dr_imagefolder_dir:
        if ImageFolder is None:
            raise ImportError("torchvision is required for --dr_imagefolder_dir")
        base_ds = ImageFolder(root=cfg.dr_imagefolder_dir)
        if set(base_ds.classes) == {"No_DR", "DR"}:
            label_map = {"No_DR": 0, "DR": 1}
        else:
            label_map = {name: idx for idx, name in enumerate(sorted(base_ds.classes))}
        by_label: Dict[int, List[int]] = {}
        for i, (_path, label) in enumerate(base_ds.samples):
            mapped = label_map[base_ds.classes[label]]
            by_label.setdefault(mapped, []).append(i)
        train_indices, val_indices = [], []
        for _, indices in sorted(by_label.items()):
            indices = indices.copy()
            random.shuffle(indices)
            if len(indices) == 1:
                train_split, val_split = indices, indices
            else:
                n_val = max(1, int(round(len(indices) * cfg.dr_val_ratio)))
                n_val = min(n_val, len(indices) - 1)
                val_split = indices[:n_val]
                train_split = indices[n_val:]
            train_indices.extend(train_split)
            val_indices.extend(val_split)
        return (
            DRImageFolderDataset(cfg.dr_imagefolder_dir, cfg.img_size, train_indices, augment=True, label_map=label_map),
            DRImageFolderDataset(cfg.dr_imagefolder_dir, cfg.img_size, val_indices, augment=False, label_map=label_map),
        )
    if not cfg.dr_train_csv or not cfg.dr_val_csv:
        raise ValueError("Need either --dr_imagefolder_dir or both --dr_train_csv and --dr_val_csv")
    return DRDataset(cfg.dr_train_csv, cfg.img_size, augment=True), DRDataset(cfg.dr_val_csv, cfg.img_size, augment=False)


def _stack_masks(vessel: Image.Image, od: Image.Image, size: int) -> torch.Tensor:
    vessel = pil_resize(vessel, size, True)
    od = pil_resize(od, size, True)
    return torch.cat([binarize_mask(vessel), binarize_mask(od)], dim=0)


class FOVEAPairedDataset(Dataset):
    def __init__(self, pairs: List[dict], img_size: int, augment: bool = True):
        self.pairs = list(pairs)
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.pairs[idx]
        pre_img = pil_resize(load_image_rgb(item["pre_img"]), self.img_size, False)
        intra_img = pil_resize(load_image_rgb(item["intra_img"]), self.img_size, False)
        if self.augment:
            pre_img = EnhancedAugmentation.augment_fovea(pre_img, True)
            intra_img = EnhancedAugmentation.augment_fovea(intra_img, True)
        pre_y = _stack_masks(load_mask(item["pre_vessel"]), load_mask(item["pre_od"]), self.img_size)
        intra_y = _stack_masks(load_mask(item["intra_vessel"]), load_mask(item["intra_od"]), self.img_size)
        return {
            "pre_x": normalize_img(pil_to_tensor(pre_img)),
            "intra_x": normalize_img(pil_to_tensor(intra_img)),
            "pre_y": pre_y,
            "intra_y": intra_y,
        }


class FOVEAPairedDatasetV2(Dataset):
    """
    FOVEA V2 数据集：纯跨域对比对齐，不加载分割掩码。
    
    支持同步数据增强：pre 和 intra 图像使用相同的随机增强参数（如水平翻转）。
    """
    def __init__(self, pairs: List[dict], img_size: int, augment: bool = True):
        self.pairs = list(pairs)
        self.img_size = img_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.pairs[idx]
        patient_id = item.get("patient_id", f"unknown_{idx}")
        
        # 加载图像
        pre_img = pil_resize(load_image_rgb(item["pre_img"]), self.img_size, False)
        intra_img = pil_resize(load_image_rgb(item["intra_img"]), self.img_size, False)
        
        if self.augment:
            # 同步随机水平翻转
            if random.random() < 0.5:
                pre_img = pre_img.transpose(Image.FLIP_LEFT_RIGHT)
                intra_img = intra_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 同步随机垂直翻转
            if random.random() < 0.3:
                pre_img = pre_img.transpose(Image.FLIP_TOP_BOTTOM)
                intra_img = intra_img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # 独立颜色抖动（术前术中图像颜色差异大，使用独立抖动）
            pre_img = EnhancedAugmentation.augment_fovea(pre_img, True)
            intra_img = EnhancedAugmentation.augment_fovea(intra_img, True)
        
        return {
            "pre_x": normalize_img(pil_to_tensor(pre_img)),
            "intra_x": normalize_img(pil_to_tensor(intra_img)),
            "patient_id": patient_id,
        }


def _split_fovea_by_patient(pairs: List[dict], val_ratio: float) -> Tuple[List[dict], List[dict]]:
    by_patient: Dict[str, List[dict]] = {}
    for item in pairs:
        pid = item.get("patient_id")
        if pid is None:
            pid = os.path.basename(item["pre_img"]).split("_")[0]
        by_patient.setdefault(pid, []).append(item)
    patient_ids = list(by_patient.keys())
    random.shuffle(patient_ids)
    if len(patient_ids) == 1:
        return by_patient[patient_ids[0]], by_patient[patient_ids[0]]
    n_val = max(1, int(round(len(patient_ids) * val_ratio)))
    n_val = min(n_val, len(patient_ids) - 1)
    val_ids = set(patient_ids[:n_val])
    train_pairs, val_pairs = [], []
    for pid, items in by_patient.items():
        (val_pairs if pid in val_ids else train_pairs).extend(items)
    return train_pairs, val_pairs


def _find_col(row: dict, names: List[str]) -> Optional[str]:
    for name in names:
        if name in row:
            return name
    return None


class RMITTrackingDataset(Dataset):
    def __init__(self, manifest_json: str, img_size: int, is_training: bool, max_pairs_per_seq: Optional[int], protocol: str, test_seq: Optional[str], val_ratio: float):
        self.img_size = img_size
        self.is_training = is_training
        self.items: List[dict] = []
        with open(manifest_json, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        sequences = manifest["sequences"] if isinstance(manifest, dict) and "sequences" in manifest else manifest
        if protocol == "leave_one_out" and test_seq:
            selected = [seq for seq in sequences if (seq["name"] != test_seq) == is_training]
        else:
            seqs = list(sequences)
            random.shuffle(seqs)
            n_val = max(1, int(round(len(seqs) * val_ratio))) if len(seqs) > 1 else 1
            val_names = {seq["name"] for seq in seqs[:n_val]}
            selected = [seq for seq in sequences if (seq["name"] not in val_names) == is_training]
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

    def _load_sequence(self, frames_dir: str, ann_csv: str) -> List[dict]:
        records = []
        with open(ann_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_key = _find_col(row, ["frame", "frame_id", "image", "image_name", "filename", "file"])
                x_key = _find_col(row, ["x", "tip_x", "cx"])
                y_key = _find_col(row, ["y", "tip_y", "cy"])
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
        template_raw = load_image_rgb(item["template_path"])
        search_raw = load_image_rgb(item["search_path"])
        gt_xy = item["search_xy"].clone()
        scale_x = self.img_size / max(search_raw.size[0], 1)
        scale_y = self.img_size / max(search_raw.size[1], 1)
        gt_xy = torch.tensor([gt_xy[0] * scale_x, gt_xy[1] * scale_y], dtype=torch.float32)
        template = pil_resize(template_raw, self.img_size, False)
        search = pil_resize(search_raw, self.img_size, False)
        if self.is_training:
            template, search, gt_xy = EnhancedAugmentation.augment_tracking(template, search, gt_xy, True)
        gt_hm = _make_gaussian(self.img_size, gt_xy[0].item(), gt_xy[1].item(), sigma=max(2.0, self.img_size / 64.0))
        return {
            "template": normalize_img(pil_to_tensor(template)),
            "search": normalize_img(pil_to_tensor(search)),
            "gt_xy": gt_xy,
            "gt_hm": gt_hm,
        }


@torch.no_grad()
def eval_dr(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        model["enc"].set_domain("dr")
        logits = model["head"](model["enc"](x))
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return {"acc": correct / max(1, total)}


def parse_int_list(spec: str, depth: int) -> List[int]:
    values = sorted(set(int(x.strip()) for x in spec.split(",") if x.strip()))
    values = [v for v in values if 0 <= v < depth]
    if not values:
        raise ValueError(f"Invalid feature layers '{spec}' for depth={depth}")
    return values


def resolve_layer_weights(num_layers: int, spec: Optional[str]) -> List[float]:
    if spec:
        values = [float(x.strip()) for x in spec.split(",") if x.strip()]
        if len(values) != num_layers:
            raise ValueError(f"align_layer_weights requires {num_layers} values, got {len(values)}")
        return values
    if num_layers == 1:
        return [1.0]
    return [(i + 1) / num_layers for i in range(num_layers)]


def resolve_lora_layer_ranks(
    depth: int,
    base_rank: int,
    strategy: str,
    min_rank: int = 0,
    spec: Optional[str] = None,
) -> List[int]:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if spec:
        values = [max(0, int(x.strip())) for x in spec.split(",") if x.strip()]
        if len(values) != depth:
            raise ValueError(f"lora_rank_spec requires {depth} values, got {len(values)}")
        return values
    max_rank = max(0, int(base_rank))
    if max_rank == 0:
        return [0] * depth
    min_rank = max(1, min_rank) if min_rank > 0 else max(1, max_rank // 2)
    min_rank = min(min_rank, max_rank)
    if strategy == "uniform":
        return [max_rank] * depth
    if strategy == "linear":
        if depth == 1:
            return [max_rank]
        return [
            int(round(min_rank + (max_rank - min_rank) * idx / (depth - 1)))
            for idx in range(depth)
        ]
    if strategy == "late_heavy":
        if depth == 1:
            return [max_rank]
        values = []
        for idx in range(depth):
            progress = idx / (depth - 1)
            scaled = progress ** 2
            values.append(int(round(min_rank + (max_rank - min_rank) * scaled)))
        return values
    raise ValueError(f"Unsupported lora_rank_strategy: {strategy}")


class LayerwiseContrastiveHead(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, proj_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, proj_dim),
            )
            for _ in range(num_layers)
        ])

    def forward(self, cls_tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        return [F.normalize(projector(token), dim=-1) for projector, token in zip(self.projectors, cls_tokens)]


def multilayer_info_nce(
    z1_list: List[torch.Tensor],
    z2_list: List[torch.Tensor],
    temperature: float,
    layer_weights: List[float],
) -> torch.Tensor:
    total = z1_list[0].new_tensor(0.0)
    total_weight = sum(layer_weights)
    for weight, z1, z2 in zip(layer_weights, z1_list, z2_list):
        logits = (z1 @ z2.t()) / temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        total = total + weight * loss
    return total / max(total_weight, 1e-6)


def compute_retrieval_accuracy(
    z_pre_list: List[torch.Tensor],
    z_intra_list: List[torch.Tensor]
) -> float:
    """
    计算跨域检索准确率（Cross-Domain Retrieval Accuracy）。
    
    对于 batch 内的每个 pre embedding，在所有 intra embeddings 中找最相似的，
    如果找到的是对应的那个（同一病人），就算正确。
    
    Args:
        z_pre_list: list of [B, D] tensors from pre-operative images
        z_intra_list: list of [B, D] tensors from intra-operative images
    
    Returns:
        平均检索准确率 (0.0 - 1.0)
    """
    # 拼接所有层的 embedding
    all_pre = torch.cat(z_pre_list, dim=1)   # [B, D*num_layers]
    all_intra = torch.cat(z_intra_list, dim=1)
    # 已经在 LayerwiseContrastiveHead 中归一化过了，这里再归一化一次确保
    all_pre = F.normalize(all_pre, dim=1)
    all_intra = F.normalize(all_intra, dim=1)
    
    # 计算相似度矩阵
    sim = torch.mm(all_pre, all_intra.t())  # [B, B]
    labels = torch.arange(sim.size(0), device=sim.device)
    
    # pre→intra accuracy
    p2i_acc = (sim.argmax(dim=1) == labels).float().mean().item()
    # intra→pre accuracy  
    i2p_acc = (sim.argmax(dim=0) == labels).float().mean().item()
    
    return (p2i_acc + i2p_acc) / 2


class MultiScaleTokenFuser(nn.Module):
    def __init__(self, embed_dim: int, out_dim: int, num_levels: int):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(embed_dim, out_dim, kernel_size=1) for _ in range(num_levels)])
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim * num_levels, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, tokens_list: List[torch.Tensor], grid: int) -> torch.Tensor:
        feats = []
        for tokens, conv in zip(tokens_list, self.lateral):
            bsz, _, channels = tokens.shape
            feat = tokens.transpose(1, 2).reshape(bsz, channels, grid, grid)
            feats.append(conv(feat))
        return self.refine(torch.cat(feats, dim=1))


class SegDecoderV2(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, feature_layers: List[int], out_ch: int = 2, fpn_out_dim: int = 256):
        super().__init__()
        self.img_size = img_size
        self.grid = img_size // patch_size
        self.fuser = MultiScaleTokenFuser(embed_dim, fpn_out_dim, len(feature_layers))
        self.head = nn.Sequential(
            nn.Conv2d(fpn_out_dim, fpn_out_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(fpn_out_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_ch, kernel_size=1),
        )

    def forward(self, tokens_list: List[torch.Tensor]) -> torch.Tensor:
        feat = self.fuser(tokens_list, self.grid)
        logits = self.head(feat)
        return F.interpolate(logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)


class UncertaintyHeatmapHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),
        )
        self.heatmap_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.log_var_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        nn.init.normal_(self.heatmap_head.weight, std=0.001)
        nn.init.zeros_(self.heatmap_head.bias)
        nn.init.normal_(self.log_var_head.weight, std=0.001)
        nn.init.constant_(self.log_var_head.bias, -2.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared(x)
        heatmap_logits = self.heatmap_head(feat)
        log_variance = torch.clamp(self.log_var_head(feat), min=-6.0, max=3.0)
        uncertainty = torch.exp(log_variance)
        return {
            "heatmap_logits": heatmap_logits,
            "log_variance": log_variance,
            "uncertainty": uncertainty,
        }

    @staticmethod
    def heteroscedastic_loss(pred: torch.Tensor, target: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        precision = torch.exp(-log_variance)
        return (precision * (pred - target) ** 2 + log_variance).mean()


def encoder_forward_selected(
    encoder: ViTEncoder,
    x: torch.Tensor,
    domain: str,
    layer_ids: List[int],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    encoder.set_domain(domain)
    bsz = x.size(0)
    hidden = encoder.patch(x)
    cls_token = encoder.cls.expand(bsz, -1, -1)
    hidden = torch.cat([cls_token, hidden], dim=1)
    hidden = hidden + encoder.pos[:, : hidden.size(1)]
    hidden = encoder.drop(hidden)
    cls_map: Dict[int, torch.Tensor] = {}
    patch_map: Dict[int, torch.Tensor] = {}
    target = set(layer_ids)
    for idx, block in enumerate(encoder.blocks):
        hidden = block(hidden)
        if idx in target:
            normalized = encoder.norm(hidden)
            cls_map[idx] = normalized[:, 0]
            patch_map[idx] = normalized[:, 1:]
    return [cls_map[idx] for idx in layer_ids], [patch_map[idx] for idx in layer_ids]


class OneStreamTipTrackerV2(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        encoder: ViTEncoder,
        embed_dim: int,
        feature_layers: List[int],
        use_anatomy_anchor: bool = True,
        use_uncertainty_head: bool = True,
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
        self.fuser = MultiScaleTokenFuser(embed_dim, fpn_out_dim, len(feature_layers))
        if use_anatomy_anchor:
            self.anatomy_attention = nn.Sequential(
                nn.Conv2d(fpn_out_dim + 2, fpn_out_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(fpn_out_dim, fpn_out_dim, kernel_size=1),
            )
        if use_uncertainty_head:
            self.hm_head = UncertaintyHeatmapHead(fpn_out_dim)
        else:
            self.hm_head = nn.Sequential(
                nn.Conv2d(fpn_out_dim, 256, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(256, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 1, kernel_size=1),
            )

    def forward(self, template: torch.Tensor, search: torch.Tensor, anatomy_mask: Optional[torch.Tensor] = None) -> Dict[str, Optional[torch.Tensor]]:
        self.encoder.set_domain("surg")
        bsz = template.size(0)
        template_tokens = self.encoder.patch(template)
        search_tokens = self.encoder.patch(search)
        cls_token = self.encoder.cls.expand(bsz, -1, -1)
        hidden = torch.cat([cls_token, template_tokens, search_tokens], dim=1)
        hidden = hidden + self.pos_ts[:, : hidden.size(1)]
        hidden = self.encoder.drop(hidden)
        selected: Dict[int, torch.Tensor] = {}
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
        if self.use_uncertainty_head:
            out = self.hm_head(feat)
            return {
                "heatmap_logits": F.interpolate(out["heatmap_logits"], size=(self.img_size, self.img_size), mode="bilinear", align_corners=False),
                "log_variance": F.interpolate(out["log_variance"], size=(self.img_size, self.img_size), mode="bilinear", align_corners=False),
                "uncertainty": F.interpolate(out["uncertainty"], size=(self.img_size, self.img_size), mode="bilinear", align_corners=False),
            }
        heatmap_logits = self.hm_head(feat)
        return {
            "heatmap_logits": F.interpolate(heatmap_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False),
            "log_variance": None,
            "uncertainty": None,
        }


@torch.no_grad()
def eval_fovea_seg_v2(model: nn.Module, loader: DataLoader, device: torch.device, feature_layers: List[int]) -> Dict[str, float]:
    model.eval()
    dices = []
    for batch in loader:
        pre_x = batch["pre_x"].to(device)
        pre_y = batch["pre_y"].to(device)
        intra_x = batch["intra_x"].to(device)
        intra_y = batch["intra_y"].to(device)
        enc = model["enc"]
        seg = model["seg"]
        _, pre_tok = encoder_forward_selected(enc, pre_x, "pre", feature_layers)
        _, intra_tok = encoder_forward_selected(enc, intra_x, "intra", feature_layers)
        pre_logits = seg(pre_tok)
        intra_logits = seg(intra_tok)
        for logits, target in [(pre_logits, pre_y), (intra_logits, intra_y)]:
            probs = torch.sigmoid(logits)
            num = 2 * (probs * target).sum(dim=(2, 3))
            den = (probs + target).sum(dim=(2, 3)) + 1e-6
            dice = ((num + 1e-6) / den).mean(dim=0)
            dices.append(dice.cpu())
    if not dices:
        return {"dice_vessel": 0.0, "dice_od": 0.0, "dice_mean": 0.0}
    dices_tensor = torch.stack(dices, dim=0).mean(dim=0)
    return {
        "dice_vessel": float(dices_tensor[0]),
        "dice_od": float(dices_tensor[1]),
        "dice_mean": float(dices_tensor.mean()),
    }


@torch.no_grad()
def eval_rmit_v2(model: nn.Module, loader: DataLoader, device: torch.device, feature_layers: List[int], use_anatomy_anchor: bool) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []
    for batch in loader:
        template = batch["template"].to(device)
        search = batch["search"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        anatomy_mask = None
        if use_anatomy_anchor:
            _, search_tok = encoder_forward_selected(model["enc"], search, "surg", feature_layers)
            logits = model["seg"](search_tok)
            anatomy_mask = torch.sigmoid(logits)
        tracker_out = model["tracker"](template, search, anatomy_mask=anatomy_mask)
        pred_xy = soft_argmax_2d(tracker_out["heatmap_logits"])
        preds.append(pred_xy.cpu())
        gts.append(gt_xy.cpu())
    if not preds:
        return {"mse": 0.0, "mean_error": 0.0, "precision@5px": 0.0, "precision@10px": 0.0, "precision@20px": 0.0, "failure_rate": 0.0}
    return compute_tracking_metrics(torch.cat(preds, dim=0), torch.cat(gts, dim=0))


src.seed_all = seed_all
src.ensure_dir = ensure_dir
src.append_epoch_log = append_epoch_log
src.cosine_lr = cosine_lr
src.load_image_rgb = load_image_rgb
src.load_mask = load_mask
src.pil_resize = pil_resize
src.pil_to_tensor = pil_to_tensor
src.normalize_img = normalize_img
src.soft_argmax_2d = soft_argmax_2d
src.compute_tracking_metrics = compute_tracking_metrics
src.dice_loss_logits = dice_loss_logits
src.bce_loss_logits = bce_loss_logits
src.EnhancedAugmentation = EnhancedAugmentation
src.PatchEmbed = PatchEmbed
src.DRHead = DRHead
src.ModelEMA = ModelEMA
src.save_ckpt = save_ckpt
src.load_ckpt = load_ckpt
src.load_pretrained_encoder = load_pretrained_encoder
src.DRDataset = DRDataset
src.DRImageFolderDataset = DRImageFolderDataset
src.build_dr_datasets = build_dr_datasets
src.FOVEAPairedDataset = FOVEAPairedDataset
src.FOVEAPairedDatasetV2 = FOVEAPairedDatasetV2
src._split_fovea_by_patient = _split_fovea_by_patient
src.RMITTrackingDataset = RMITTrackingDataset
src.eval_dr = eval_dr
src.parse_int_list = parse_int_list
src.resolve_layer_weights = resolve_layer_weights
src.resolve_lora_layer_ranks = resolve_lora_layer_ranks
src.LayerwiseContrastiveHead = LayerwiseContrastiveHead
src.multilayer_info_nce = multilayer_info_nce
src.SegDecoderV2 = SegDecoderV2
src.UncertaintyHeatmapHead = UncertaintyHeatmapHead
src.TaskUncertaintyWeighting = TaskUncertaintyWeighting
src.encoder_forward_selected = encoder_forward_selected
src.OneStreamTipTrackerV2 = OneStreamTipTrackerV2
src.eval_fovea_seg_v2 = eval_fovea_seg_v2
src.eval_rmit_v2 = eval_rmit_v2
src.np = np
src.WeightedRandomSampler = WeightedRandomSampler
src.tqdm = tqdm


def get_lora_and_head_params(model: nn.Module, verbose: bool = True) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        is_adapter = ("A." in name) or ("B." in name) or ("M." in name)
        is_head = not name.startswith("enc.")
        param.requires_grad = bool(is_adapter or is_head)
        if param.requires_grad:
            params.append(param)
            trainable += 1
        else:
            frozen += 1
    if verbose:
        print(f"[PEFT] Frozen params={frozen}, Trainable params={trainable}")
    return params


def verify_lora_frozen(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if name.startswith("enc.") and ("A." not in name and "B." not in name and "M." not in name) and param.requires_grad:
            raise RuntimeError(f"Encoder param should be frozen under PEFT: {name}")
    print("✅ PEFT freezing verification passed")


@dataclass
class TrainCfgV2:
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
    lora_rank_strategy: str
    lora_rank_min: int
    lora_rank_spec: Optional[str]
    lora_layer_ranks: List[int]
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
    dr_imagefolder_dir: Optional[str]
    dr_train_csv: Optional[str]
    dr_val_csv: Optional[str]
    dr_val_ratio: float
    dr_num_classes: int
    fovea_pairs_json: Optional[str]
    fovea_val_ratio: float
    lambda_align: float
    lambda_seg: float
    rmit_manifest_json: Optional[str]
    rmit_protocol: str
    rmit_test_seq: Optional[str]
    rmit_val_ratio: float
    max_pairs_per_seq: Optional[int]
    use_anatomy_anchor: bool
    eval_metrics: bool
    feature_layers: List[int] = field(default_factory=lambda: [2, 5, 7])
    align_layer_weights: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0])
    contrast_temperature: float = 0.07
    proj_dim: int = 256
    fpn_out_dim: int = 256
    use_uncertainty_head: bool = True
    lambda_hm: float = 1.0
    lambda_xy: float = 0.25
    use_dora: bool = True
    use_task_uncertainty: bool = True
    lambda_edge: float = 1.0
    edge_weight: float = 4.0
    edge_kernel_size: int = 5


def train_stage_dr(cfg: TrainCfgV2, device: torch.device) -> None:
    src.ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)
    enc = build_encoder(cfg).to(device)
    head = src.DRHead(cfg.embed_dim, cfg.dr_num_classes).to(device)
    model = nn.ModuleDict({"enc": enc, "head": head})
    if cfg.pretrained_encoder_ckpt:
        src.load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)
    if cfg.init_ckpt:
        src.load_ckpt(cfg.init_ckpt, model, optim=None)
    train_ds, val_ds = src.build_dr_datasets(cfg)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Empty DR dataset split: train={len(train_ds)} val={len(val_ds)}")
    train_labels = [label for _, label in train_ds.items]
    counts = src.np.bincount(train_labels, minlength=cfg.dr_num_classes).astype(src.np.float32)
    weights = 1.0 / src.np.maximum(counts, 1.0)
    sample_weights = [weights[label] for label in train_labels]
    sampler = src.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    params = get_lora_and_head_params(model, verbose=False) if cfg.freeze_backbone else list(model.parameters())
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = src.ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    best = -1.0
    global_step = 0
    total_steps = cfg.epochs * max(1, len(train_ld)) // max(1, cfg.grad_accum)
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        correct, total = 0, 0
        optim.zero_grad(set_to_none=True)
        iterator = enumerate(train_ld)
        if src.tqdm is not None:
            iterator = src.tqdm(iterator, total=len(train_ld), desc=f"DR {epoch+1}/{cfg.epochs}", leave=False)
        for it, (x, y) in iterator:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            lr_now = src.cosine_lr(cfg.lr, global_step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                enc.set_domain("dr")
                logits = head(enc(x))
                loss = F.cross_entropy(logits, y, weight=class_weight) / max(1, cfg.grad_accum)
            scaler.scale(loss).backward()
            running += loss.item() * max(1, cfg.grad_accum)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
            if (it + 1) % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                global_step += 1
        if len(train_ld) % cfg.grad_accum != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1
        metrics = src.eval_dr(model, val_ld, device)
        writer.add_scalar("dr/train_loss", running / len(train_ld), epoch)
        writer.add_scalar("dr/train_acc", correct / max(1, total), epoch)
        writer.add_scalar("dr/val_acc", metrics["acc"], epoch)
        msg = f"[DR] epoch {epoch+1}/{cfg.epochs} train_loss={running/len(train_ld):.4f} train_acc={correct/max(1,total):.4f} val_acc={metrics['acc']:.4f}"
        print(msg)
        src.append_epoch_log(cfg.out_dir, msg)
        metric = metrics["acc"]
        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "ema": ema.state_dict() if ema is not None else None,
            "extra": {"stage": "dr_dora_swiglu"},
        }
        src.save_ckpt(f"{cfg.out_dir}/last.pt", payload)
        if metric > best:
            best = metric
            payload["best_metric"] = best
            src.save_ckpt(f"{cfg.out_dir}/best.pt", payload)
    writer.close()


def train_stage_fovea_v2(cfg: TrainCfgV2, device: torch.device) -> None:
    """
    FOVEA V2 阶段：纯跨域对比对齐训练（无分割任务）。
    
    核心目标：通过对比学习将术前(pre)和术中(intra)图像的特征对齐到同一空间，
    为后续 RMIT 阶段的跨域跟踪奠定基础。
    
    训练流程：
    1. 对每对 (pre_img, intra_img)，分别用对应的 domain adapter 提取多层特征
    2. 通过 LayerwiseContrastiveHead 投影到对比空间
    3. 使用双向 InfoNCE loss 拉近同一病人的 pre-intra 特征，推远不同病人的特征
    4. 验证时计算 val_loss（对比损失）
    """
    src.ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)
    
    # 1. 构建模型
    enc = build_encoder(cfg).to(device)
    # 纯对比学习，只需要 encoder 和 contrastive head
    con = src.LayerwiseContrastiveHead(
        cfg.embed_dim, 
        len(cfg.feature_layers), 
        proj_dim=getattr(cfg, 'fovea_embed_dim', cfg.proj_dim),
        hidden_dim=getattr(cfg, 'fovea_embed_dim', cfg.proj_dim),
    ).to(device)
    
    model_items = {"enc": enc, "con": con}
    model = nn.ModuleDict(model_items)
    
    # 2. 加载预训练权重
    if cfg.pretrained_encoder_ckpt:
        src.load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)
    if cfg.init_ckpt:
        print(f"[FOVEA-V2-Align] Loading init ckpt: {cfg.init_ckpt}")
        src.load_ckpt(cfg.init_ckpt, model, optim=None)
    
    # 3. 加载数据
    with open(cfg.fovea_pairs_json, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    
    train_pairs, val_pairs = src._split_fovea_by_patient(pairs, cfg.fovea_val_ratio)
    if len(train_pairs) == 0 and len(val_pairs) > 1:
        train_pairs = val_pairs[1:]
        val_pairs = val_pairs[:1]
    if len(val_pairs) == 0 and len(train_pairs) > 1:
        val_pairs = train_pairs[:1]
        train_pairs = train_pairs[1:]
    
    # 使用 V2 数据集（纯对比，无分割掩码）
    train_ds = src.FOVEAPairedDatasetV2(train_pairs, cfg.img_size, augment=True)
    val_ds = src.FOVEAPairedDatasetV2(val_pairs, cfg.img_size, augment=False)
    
    train_ld = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True, 
        drop_last=True,  # 对比学习需要 drop_last 避免 batch=1 的情况
        num_workers=cfg.num_workers, 
        pin_memory=True
    )
    val_ld = DataLoader(
        val_ds, 
        batch_size=cfg.batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=cfg.num_workers, 
        pin_memory=True
    )
    
    print(f"[FOVEA-V2-Align] Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
    print(f"[FOVEA-V2-Align] Feature layers: {cfg.feature_layers}")
    print(f"[FOVEA-V2-Align] Temperature: {cfg.contrast_temperature}")
    
    # 4. 设置优化器
    if cfg.freeze_backbone:
        params = get_lora_and_head_params(model, verbose=False)
        verify_lora_frozen(model)
        optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))
    ema = src.ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    
    # 5. 训练参数
    best_val_loss = float('inf')
    patience_counter = 0
    patience = getattr(cfg, 'patience', 15)
    global_step = 0
    total_steps = cfg.epochs * max(1, len(train_ld)) // cfg.grad_accum
    
    # 6. 训练循环
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        optim.zero_grad(set_to_none=True)
        
        iterator = enumerate(train_ld)
        if src.tqdm is not None:
            iterator = src.tqdm(iterator, total=len(train_ld), desc=f"FOVEA-Align {epoch+1}/{cfg.epochs}", leave=False)
        
        for it, batch in iterator:
            pre_x = batch["pre_x"].to(device, non_blocking=True)
            intra_x = batch["intra_x"].to(device, non_blocking=True)
            
            # 学习率调度
            lr_now = src.cosine_lr(cfg.lr, global_step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now
            
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                # 提取多层特征（使用不同 domain 的 adapter）
                pre_cls_list, _ = src.encoder_forward_selected(enc, pre_x, "pre", cfg.feature_layers)
                intra_cls_list, _ = src.encoder_forward_selected(enc, intra_x, "intra", cfg.feature_layers)
                
                # 投影到对比空间
                z_pre = con(pre_cls_list)
                z_intra = con(intra_cls_list)
                
                # 双向 InfoNCE loss
                temperature = getattr(cfg, 'fovea_temperature', cfg.contrast_temperature)
                align_loss = src.multilayer_info_nce(z_pre, z_intra, temperature, cfg.align_layer_weights)
                
                loss = align_loss / max(1, cfg.grad_accum)
            
            scaler.scale(loss).backward()
            running_loss += align_loss.item()
            
            if (it + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                
                if ema is not None:
                    ema.update(model)
                
                if global_step % 20 == 0:
                    writer.add_scalar("fovea_align/train_loss", align_loss.item(), global_step)
                    writer.add_scalar("fovea_align/lr", lr_now, global_step)
                
                global_step += 1
        
        # 处理剩余梯度
        if len(train_ld) % cfg.grad_accum != 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1
        
        # 7. 验证
        val_loss, val_retrieval_acc = _eval_fovea_contrast_v2(model, val_ld, device, cfg)
        
        ema_val_loss = None
        ema_val_retrieval_acc = None
        if ema is not None:
            ema_val_loss, ema_val_retrieval_acc = _eval_fovea_contrast_v2(ema.module, val_ld, device, cfg)
        
        # 日志
        avg_train_loss = running_loss / max(1, len(train_ld))
        writer.add_scalar("fovea_align/val_loss", val_loss, epoch)
        writer.add_scalar("fovea_align/val_retrieval_acc", val_retrieval_acc, epoch)
        if ema_val_loss is not None:
            writer.add_scalar("fovea_align/ema_val_loss", ema_val_loss, epoch)
            writer.add_scalar("fovea_align/ema_val_retrieval_acc", ema_val_retrieval_acc, epoch)
        
        msg = f"[FOVEA-Align] epoch {epoch+1}/{cfg.epochs} train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_retrieval_acc={val_retrieval_acc:.1%}"
        if ema_val_loss is not None:
            msg += f" ema_val_loss={ema_val_loss:.4f} ema_val_retrieval_acc={ema_val_retrieval_acc:.1%}"
        print(msg)
        src.append_epoch_log(cfg.out_dir, msg)
        
        # 8. Early stopping & checkpoint
        metric = ema_val_loss if ema_val_loss is not None else val_loss
        
        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best_val_loss,
            "ema": ema.state_dict() if ema is not None else None,
            "extra": {"stage": "fovea_v2_pure_contrast_align"},
        }
        src.save_ckpt(f"{cfg.out_dir}/last.pt", payload)
        
        if metric < best_val_loss:
            best_val_loss = metric
            patience_counter = 0
            payload["best_metric"] = best_val_loss
            src.save_ckpt(f"{cfg.out_dir}/best.pt", payload)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"[FOVEA-Align] Early stopping at epoch {epoch+1}")
            break
    
    writer.close()


@torch.no_grad()
def _eval_fovea_contrast_v2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg,
) -> Tuple[float, float]:
    """
    FOVEA V2 验证：计算对比损失和检索准确率。
    
    Returns:
        (val_loss, val_retrieval_acc)
    """
    model.eval()
    enc = model["enc"]
    con = model["con"]
    
    total_loss = 0.0
    total_retrieval_acc = 0.0
    num_batches = 0
    
    for batch in loader:
        pre_x = batch["pre_x"].to(device)
        intra_x = batch["intra_x"].to(device)
        
        # 提取特征
        pre_cls_list, _ = src.encoder_forward_selected(enc, pre_x, "pre", cfg.feature_layers)
        intra_cls_list, _ = src.encoder_forward_selected(enc, intra_x, "intra", cfg.feature_layers)
        
        # 投影 + 对比损失
        z_pre = con(pre_cls_list)
        z_intra = con(intra_cls_list)
        
        temperature = getattr(cfg, 'fovea_temperature', cfg.contrast_temperature)
        align_loss = src.multilayer_info_nce(z_pre, z_intra, temperature, cfg.align_layer_weights)
        
        total_loss += align_loss.item()
        
        # 计算检索准确率
        retrieval_acc = compute_retrieval_accuracy(z_pre, z_intra)
        total_retrieval_acc += retrieval_acc
        num_batches += 1
    
    return total_loss / max(1, num_batches), total_retrieval_acc / max(1, num_batches)


def train_stage_rmit_v2(cfg: TrainCfgV2, device: torch.device) -> None:
    src.ensure_dir(cfg.out_dir)
    writer = SummaryWriter(cfg.out_dir)
    enc = build_encoder(cfg).to(device)
    seg = src.SegDecoderV2(cfg.img_size, cfg.patch_size, cfg.embed_dim, cfg.feature_layers, out_ch=2, fpn_out_dim=cfg.fpn_out_dim).to(device)
    tracker = src.OneStreamTipTrackerV2(
        cfg.img_size,
        cfg.patch_size,
        enc,
        cfg.embed_dim,
        cfg.feature_layers,
        use_anatomy_anchor=cfg.use_anatomy_anchor,
        use_uncertainty_head=cfg.use_uncertainty_head,
        fpn_out_dim=cfg.fpn_out_dim,
    ).to(device)
    task_weight = src.TaskUncertaintyWeighting(["hm", "xy"]).to(device) if cfg.use_task_uncertainty else None
    model_items = {"enc": enc, "seg": seg, "tracker": tracker}
    if task_weight is not None:
        model_items["task_weight"] = task_weight
    model = nn.ModuleDict(model_items)
    if cfg.pretrained_encoder_ckpt:
        src.load_pretrained_encoder(enc, cfg.pretrained_encoder_ckpt)
    if cfg.init_ckpt:
        print(f"[RMIT-V2] Loading init ckpt: {cfg.init_ckpt}")
        src.load_ckpt(cfg.init_ckpt, model, optim=None)
        try:
            enc.copy_domain_params(src_domain="intra", dst_domain="surg")
        except Exception:
            pass
    train_ds = src.RMITTrackingDataset(cfg.rmit_manifest_json, cfg.img_size, is_training=True, max_pairs_per_seq=cfg.max_pairs_per_seq, protocol=cfg.rmit_protocol, test_seq=cfg.rmit_test_seq, val_ratio=cfg.rmit_val_ratio)
    val_ds = src.RMITTrackingDataset(cfg.rmit_manifest_json, cfg.img_size, is_training=False, max_pairs_per_seq=None, protocol=cfg.rmit_protocol, test_seq=cfg.rmit_test_seq, val_ratio=cfg.rmit_val_ratio)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    params: List[nn.Parameter] = []
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
    ema = src.ModelEMA(model, decay=cfg.ema_decay) if cfg.use_ema else None
    best = 1e9
    global_step = 0
    total_steps = cfg.epochs * max(1, len(train_ld)) // cfg.grad_accum
    for epoch in range(cfg.epochs):
        model.train()
        if cfg.use_anatomy_anchor:
            seg.eval()
        running = 0.0
        optim.zero_grad(set_to_none=True)
        iterator = enumerate(train_ld)
        if src.tqdm is not None:
            iterator = src.tqdm(iterator, total=len(train_ld), desc=f"RMIT-V2 {epoch+1}/{cfg.epochs}", leave=False)
        for it, batch in iterator:
            template = batch["template"].to(device, non_blocking=True)
            search = batch["search"].to(device, non_blocking=True)
            gt_hm = batch["gt_hm"].to(device, non_blocking=True)
            gt_xy = batch["gt_xy"].to(device, non_blocking=True)
            lr_now = src.cosine_lr(cfg.lr, global_step, total_steps)
            for pg in optim.param_groups:
                pg["lr"] = lr_now
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                anatomy_mask = None
                if cfg.use_anatomy_anchor:
                    with torch.no_grad():
                        _, search_tokens = src.encoder_forward_selected(enc, search, "surg", cfg.feature_layers)
                        anatomy_mask = torch.sigmoid(seg(search_tokens)).detach()
                tracker_out = tracker(template, search, anatomy_mask=anatomy_mask)
                pred_hm = tracker_out["heatmap_logits"]
                if cfg.use_uncertainty_head and tracker_out["log_variance"] is not None:
                    loss_hm = src.UncertaintyHeatmapHead.heteroscedastic_loss(torch.sigmoid(pred_hm), gt_hm, tracker_out["log_variance"])
                else:
                    loss_hm = F.mse_loss(torch.sigmoid(pred_hm), gt_hm)
                pred_xy = src.soft_argmax_2d(pred_hm)
                loss_xy = F.smooth_l1_loss(pred_xy, gt_xy)
                if task_weight is not None:
                    total_loss, _ = task_weight({"hm": loss_hm, "xy": loss_xy})
                    loss = total_loss / max(1, cfg.grad_accum)
                else:
                    loss = (cfg.lambda_hm * loss_hm + cfg.lambda_xy * loss_xy) / max(1, cfg.grad_accum)
            scaler.scale(loss).backward()
            running += loss.item() * max(1, cfg.grad_accum)
            if (it + 1) % cfg.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if global_step % 20 == 0:
                    writer.add_scalar("rmit_v2/train_loss", loss.item() * max(1, cfg.grad_accum), global_step)
                    writer.add_scalar("rmit_v2/loss_hm", loss_hm.item(), global_step)
                    writer.add_scalar("rmit_v2/loss_xy", loss_xy.item(), global_step)
                    if task_weight is not None:
                        writer.add_scalar("rmit_v2/log_var_hm", task_weight.log_vars["hm"].item(), global_step)
                        writer.add_scalar("rmit_v2/log_var_xy", task_weight.log_vars["xy"].item(), global_step)
                    writer.add_scalar("rmit_v2/lr", lr_now, global_step)
                global_step += 1
        if len(train_ld) % cfg.grad_accum != 0:
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            global_step += 1
        metrics = src.eval_rmit_v2(model, val_ld, device, cfg.feature_layers, cfg.use_anatomy_anchor)
        ema_metrics = src.eval_rmit_v2(ema.module, val_ld, device, cfg.feature_layers, cfg.use_anatomy_anchor) if ema is not None else None
        writer.add_scalar("rmit_v2/val_mean_error", metrics["mean_error"], epoch)
        if cfg.eval_metrics:
            writer.add_scalar("rmit_v2/val_precision@10px", metrics["precision@10px"], epoch)
        msg = f"[RMIT-V2] epoch {epoch+1}/{cfg.epochs} train_loss={running/len(train_ld):.4f} val_mean_error={metrics['mean_error']:.3f}"
        if cfg.eval_metrics:
            msg += f" p@10={metrics['precision@10px']:.3f}"
        if ema_metrics is not None:
            msg += f" ema_val_mean_error={ema_metrics['mean_error']:.3f}"
        print(msg)
        src.append_epoch_log(cfg.out_dir, msg)
        metric = ema_metrics["mean_error"] if ema_metrics is not None else metrics["mean_error"]
        payload = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "ema": ema.state_dict() if ema is not None else None,
            "extra": {"stage": "rmit_v2_dora"},
        }
        src.save_ckpt(f"{cfg.out_dir}/last.pt", payload)
        if metric < best:
            best = metric
            payload["best_metric"] = best
            src.save_ckpt(f"{cfg.out_dir}/best.pt", payload)
    writer.close()


def parse_args() -> TrainCfgV2:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True, choices=["dr", "fovea", "rmit"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--pretrained_encoder_ckpt", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--attn_drop", type=float, default=0.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_rank_strategy", type=str, default="late_heavy", choices=["uniform", "linear", "late_heavy"])
    parser.add_argument("--lora_rank_min", type=int, default=0)
    parser.add_argument("--lora_rank_spec", type=str, default=None)
    parser.add_argument("--use_residual_lora", action="store_true")
    parser.add_argument("--residual_alpha", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--dr_imagefolder_dir", type=str, default=None)
    parser.add_argument("--dr_train_csv", type=str, default=None)
    parser.add_argument("--dr_val_csv", type=str, default=None)
    parser.add_argument("--dr_val_ratio", type=float, default=0.2)
    parser.add_argument("--dr_num_classes", type=int, default=2)
    parser.add_argument("--fovea_pairs_json", type=str, default=None)
    parser.add_argument("--fovea_val_ratio", type=float, default=0.2)
    parser.add_argument("--lambda_align", type=float, default=0.5)
    parser.add_argument("--lambda_seg", type=float, default=2.0)
    parser.add_argument("--lambda_hm", type=float, default=1.0)
    parser.add_argument("--lambda_xy", type=float, default=0.25)
    parser.add_argument("--rmit_manifest_json", type=str, default=None)
    parser.add_argument("--rmit_protocol", type=str, default="leave_one_out", choices=["leave_one_out", "random"])
    parser.add_argument("--rmit_test_seq", type=str, default=None)
    parser.add_argument("--rmit_val_ratio", type=float, default=0.2)
    parser.add_argument("--max_pairs_per_seq", type=int, default=None)
    parser.add_argument("--use_anatomy_anchor", action="store_true")
    parser.add_argument("--eval_metrics", action="store_true")
    parser.add_argument("--feature_layers", type=str, default="2,5,7")
    parser.add_argument("--align_layer_weights", type=str, default=None)
    parser.add_argument("--contrast_temperature", type=float, default=0.07)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--fpn_out_dim", type=int, default=256)
    parser.add_argument("--use_uncertainty_head", action="store_true")
    parser.add_argument("--disable_dora", action="store_true")
    parser.add_argument("--disable_task_uncertainty", action="store_true")
    parser.add_argument("--lambda_edge", type=float, default=1.0)
    parser.add_argument("--edge_weight", type=float, default=4.0)
    parser.add_argument("--edge_kernel_size", type=int, default=5)
    args = parser.parse_args()
    feature_layers = src.parse_int_list(args.feature_layers, args.depth)
    align_layer_weights = src.resolve_layer_weights(len(feature_layers), args.align_layer_weights)
    lora_layer_ranks = src.resolve_lora_layer_ranks(
        depth=args.depth,
        base_rank=args.lora_r,
        strategy=args.lora_rank_strategy,
        min_rank=args.lora_rank_min,
        spec=args.lora_rank_spec,
    )
    return TrainCfgV2(
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
        lambda_align=args.lambda_align,
        lambda_seg=args.lambda_seg,
        rmit_manifest_json=args.rmit_manifest_json,
        rmit_protocol=args.rmit_protocol,
        rmit_test_seq=args.rmit_test_seq,
        rmit_val_ratio=args.rmit_val_ratio,
        max_pairs_per_seq=args.max_pairs_per_seq,
        use_anatomy_anchor=args.use_anatomy_anchor,
        eval_metrics=args.eval_metrics,
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
    )


def main() -> None:
    cfg = parse_args()
    src.seed_all(cfg.seed)
    src.ensure_dir(cfg.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage: {cfg.stage}")
    print(f"LoRA layer ranks: {cfg.lora_layer_ranks}")
    if cfg.stage == "dr":
        train_stage_dr(cfg, device)
    elif cfg.stage == "fovea":
        if not cfg.fovea_pairs_json:
            raise ValueError("Need --fovea_pairs_json")
        train_stage_fovea_v2(cfg, device)
    elif cfg.stage == "rmit":
        if not cfg.rmit_manifest_json:
            raise ValueError("Need --rmit_manifest_json")
        train_stage_rmit_v2(cfg, device)
    else:
        raise ValueError(cfg.stage)


if __name__ == "__main__":
    main()
