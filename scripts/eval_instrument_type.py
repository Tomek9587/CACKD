"""Per-instrument-type error breakdown for Cataract 30-fold LOSO.

Evaluates Baseline and CAC-KD on each test fold, groups errors by instrument type.
Outputs a summary table.
"""
import os, sys, json, csv
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dr_11 import HeatmapKeypointNet, RMITHeatmapDataset, resolve_loso_split
from dr_cackd import CACKeypointNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512
MANIFEST = "data/manifests/cataract_rmit_manifest.json"


def load_instrument_map(manifest_path, test_name):
    """Load ann.csv for the test sequence, return {frame_basename: instrument_type}."""
    import json as _json
    manifest = _json.load(open(manifest_path))
    for seq in manifest["sequences"]:
        if seq["name"] == test_name:
            ann_csv = seq["ann_csv"]
            mapping = {}
            with open(ann_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame = row.get("frame", row.get("Frame", ""))
                    cls = row.get("class", row.get("Class", "Unknown"))
                    # frame is like "case5013_03.png", store basename
                    base = os.path.basename(frame)
                    mapping[base] = cls
            return mapping
    return {}


@torch.no_grad()
def eval_model(model, loader, img_size):
    """Evaluate model, return list of (frame_path, per_kp_error[4])."""
    model.eval()
    results = []
    for batch in loader:
        imgs = batch["img"].to(DEVICE)
        gt_xy = batch["gt_xy"].to(DEVICE)
        paths = batch["path"] if isinstance(batch["path"], list) else [batch["path"]]
        
        if isinstance(model, CACKeypointNet):
            pred = model(imgs, confidence=None, is_aux=None)
            pred_xy = pred["xy"]
        else:
            pred = model(imgs)
            pred_xy = pred.get("xy", pred.get("coarse_xy"))
        
        pred_px = pred_xy * (img_size - 1)
        gt_px = gt_xy * (img_size - 1)
        err = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1))  # [B, K]
        
        for i in range(err.size(0)):
            results.append((paths[i], err[i].cpu().numpy()))
    return results


def main():
    # Collect errors by instrument type
    # {instrument_type: {"baseline": [errors], "cackd": [errors]}}
    by_type = defaultdict(lambda: {"baseline": [], "cackd": []})
    
    for fold in range(30):
        # Resolve test sequence
        train_names, val_names, test_name = resolve_loso_split(MANIFEST, fold)
        
        # Load instrument type mapping
        inst_map = load_instrument_map(MANIFEST, test_name)
        if not inst_map:
            print(f"[WARN] No instrument map for fold {fold} ({test_name})")
            continue
        
        # Create test dataset
        val_ds = RMITHeatmapDataset(
            MANIFEST, IMG_SIZE, val_names, is_training=False,
            sigma=2.0, augment=False,
        )
        from torch.utils.data import DataLoader
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        
        # --- Baseline ---
        bl_path = f"logs/v15_cat_effv2_f{fold}/best.pt"
        if os.path.exists(bl_path):
            bl_model = HeatmapKeypointNet(
                img_size=IMG_SIZE, pretrained=False, backbone="efficientnet_v2_s",
                decoder_stride=4, decoder_type="heatmap_offset", use_offset=True,
            ).to(DEVICE)
            bl_ckpt = torch.load(bl_path, map_location=DEVICE)
            bl_state = bl_ckpt["model"] if "model" in bl_ckpt else bl_ckpt
            bl_model.load_state_dict(bl_state)
            
            bl_results = eval_model(bl_model, val_loader, IMG_SIZE)
            for path, errs in bl_results:
                fname = os.path.basename(path)
                inst = inst_map.get(fname, "Unknown")
                by_type[inst]["baseline"].extend(errs.tolist())
            del bl_model
            torch.cuda.empty_cache()
        
        # --- CAC-KD Fixed ---
        ck_path = f"logs/cackd_cat_fixed_f{fold}/best.pt"
        if os.path.exists(ck_path):
            ck_model = CACKeypointNet(
                img_size=IMG_SIZE, pretrained=False, backbone="efficientnet_v2_s",
                decoder_stride=4, decoder_type="heatmap_offset", use_offset=True,
            ).to(DEVICE)
            ck_ckpt = torch.load(ck_path, map_location=DEVICE)
            ck_state = ck_ckpt["model"] if "model" in ck_ckpt else ck_ckpt
            # Handle base. prefix
            if any(k.startswith("base.") for k in ck_state):
                ck_state = {k.replace("base.", "", 1): v for k, v in ck_state.items() if k.startswith("base.")}
                base_state = {f"base.{k}": v for k, v in ck_state.items()}
            else:
                base_state = ck_state
            ck_model.load_state_dict(base_state, strict=False)
            
            ck_results = eval_model(ck_model, val_loader, IMG_SIZE)
            for path, errs in ck_results:
                fname = os.path.basename(path)
                inst = inst_map.get(fname, "Unknown")
                by_type[inst]["cackd"].extend(errs.tolist())
            del ck_model
            torch.cuda.empty_cache()
        
        print(f"Fold {fold} ({test_name}): {len(val_ds)} frames, {len(inst_map)} instrument entries")
    
    # Print summary
    print("\n" + "=" * 80)
    print(f"{'Instrument Type':<35} {'N':>5} {'Base':>8} {'CAC-KD':>8} {'Delta':>8} {'Improv':>8}")
    print("-" * 80)
    
    all_base = []
    all_cackd = []
    
    for inst in sorted(by_type.keys()):
        base_errs = np.array(by_type[inst]["baseline"])
        cackd_errs = np.array(by_type[inst]["cackd"])
        n = len(base_errs)
        base_mean = np.mean(base_errs)
        cackd_mean = np.mean(cackd_errs)
        delta = cackd_mean - base_mean
        improv = (base_mean - cackd_mean) / base_mean * 100 if base_mean > 0 else 0
        
        print(f"{inst:<35} {n:>5} {base_mean:>8.2f} {cackd_mean:>8.2f} {delta:>+8.2f} {improv:>+7.1f}%")
        
        all_base.extend(base_errs.tolist())
        all_cackd.extend(cackd_errs.tolist())
    
    print("-" * 80)
    total_base = np.mean(all_base)
    total_cackd = np.mean(all_cackd)
    total_improv = (total_base - total_cackd) / total_base * 100
    print(f"{'ALL':<35} {len(all_base):>5} {total_base:>8.2f} {total_cackd:>8.2f} {total_cackd-total_base:>+8.2f} {total_improv:>+7.1f}%")
    
    # Save as JSON for later use
    output = {}
    for inst in sorted(by_type.keys()):
        output[inst] = {
            "n": len(by_type[inst]["baseline"]),
            "baseline_mean": float(np.mean(by_type[inst]["baseline"])),
            "cackd_mean": float(np.mean(by_type[inst]["cackd"])),
            "delta": float(np.mean(by_type[inst]["cackd"]) - np.mean(by_type[inst]["baseline"])),
        }
    json.dump(output, open("logs/instrument_type_breakdown.json", "w"), indent=2)
    print(f"\nSaved to logs/instrument_type_breakdown.json")


if __name__ == "__main__":
    main()
