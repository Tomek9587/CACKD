"""Evaluate CAC-KD on half-split with modules OFF (domain-adaptive switch verification).

Tests three configurations:
1. Baseline on half-split (ground truth reference)
2. CAC-KD Stage2 with modules OFF (confidence=None, is_aux=None → CGFA passthrough, UMGR uniform)
3. CAC-KD Stage2 with modules FULLY OFF (use xy_pre_refine instead of UMGR-refined xy)
"""
import os, sys, json, torch, numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from dr_11 import RMITHeatmapDataset, load_rmit_manifest_sequences, soft_argmax_2d
from dr_cackd import CACKeypointNet
from tqdm import tqdm


def eval_cackd(model, loader, device, img_size, use_refine=True):
    """Evaluate CAC-KD model. use_refine=False skips UMGR (switch fully OFF)."""
    model.eval()
    all_kp = [[] for _ in range(4)]
    pck_20 = pck_50 = total = 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        imgs = batch["img"].to(device)
        gt_xy = batch["gt_xy"].to(device)
        # Switch OFF: no confidence signal → modules bypass
        pred = model(imgs, confidence=None, is_aux=None)
        
        if use_refine:
            pred_px = pred["xy"] * (img_size - 1)
        else:
            # Fully OFF: use xy before UMGR refinement
            pred_px = pred["xy_pre_refine"] * (img_size - 1)
        
        gt_px = gt_xy * (img_size - 1)
        err = torch.sqrt(((pred_px - gt_px) ** 2).sum(dim=-1))
        for k in range(4):
            all_kp[k].extend(err[:, k].detach().cpu().numpy().tolist())
        pck_20 += (err < 20).float().sum().item()
        pck_50 += (err < 50).float().sum().item()
        total += err.numel()

    all_flat = [e for kp_errs in all_kp for e in kp_errs]
    return {
        "mean_error": float(np.mean(all_flat)),
        "kp_errors": [float(np.mean(e)) for e in all_kp],
        "pck_20": float(pck_20 / total), "pck_50": float(pck_50 / total),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 512
    manifest = "data/manifests/rmit_manifest.json"
    
    # Half-split setup
    seqs = load_rmit_manifest_sequences(manifest)
    train_names = val_names = {s["name"] for s in seqs}
    
    val_ds = RMITHeatmapDataset(
        manifest, img_size, val_names, is_training=False,
        sigma=2.0, augment=False, half_split="second",
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    results = {}
    
    # === Test 1: CAC-KD Stage2 (CDT backbone) with modules OFF on half-split ===
    print("\n=== Test 1: CAC-KD Stage2 (CDT backbone), modules OFF (conf=None) ===")
    model = CACKeypointNet(
        img_size=img_size, pretrained=True, backbone="efficientnet_v2_s",
        decoder_stride=4, decoder_type="heatmap_offset", use_offset=True,
    ).to(device)
    
    # Load Stage2 checkpoint (CDT backbone + trained CGFA+UMGR)
    ckpt = torch.load("logs/cac_s2_half/best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded CAC-KD Stage2 checkpoint (epoch={ckpt['epoch']})")
    
    # Evaluate with switch OFF: confidence=None, is_aux=None
    metrics_off = eval_cackd(model, val_loader, device, img_size, use_refine=True)
    print(f"  Switch OFF (UMGR active with uniform conf): "
          f"mean_err={metrics_off['mean_error']:.2f}px "
          f"PCK@20={metrics_off['pck_20']:.3f} PCK@50={metrics_off['pck_50']:.3f}")
    results["cackd_s2_switch_off"] = metrics_off
    
    # === Test 2: CAC-KD Stage2 with modules FULLY OFF (skip UMGR) ===
    print("\n=== Test 2: CAC-KD Stage2, modules FULLY OFF (use xy_pre_refine) ===")
    metrics_full_off = eval_cackd(model, val_loader, device, img_size, use_refine=False)
    print(f"  Switch FULLY OFF (skip UMGR): "
          f"mean_err={metrics_full_off['mean_error']:.2f}px "
          f"PCK@20={metrics_full_off['pck_20']:.3f} PCK@50={metrics_full_off['pck_50']:.3f}")
    results["cackd_s2_fully_off"] = metrics_full_off
    
    # === Test 3: Also test each LOSO fold with modules OFF ===
    for fold in [0, 1, 2]:
        print(f"\n=== Test 3.{fold}: CAC-KD Stage2 fold{fold}, modules OFF ===")
        ckpt_path = f"logs/cac_s2_f{fold}/best.pt"
        if not os.path.exists(ckpt_path):
            print(f"  Skip: {ckpt_path} not found")
            continue
        
        model2 = CACKeypointNet(
            img_size=img_size, pretrained=True, backbone="efficientnet_v2_s",
            decoder_stride=4, decoder_type="heatmap_offset", use_offset=True,
        ).to(device)
        ckpt2 = torch.load(ckpt_path, map_location=device)
        model2.load_state_dict(ckpt2["model"])
        
        # LOSO val dataset for this fold
        from dr_11 import resolve_loso_split
        train_names_f, val_names_f, test_name = resolve_loso_split(manifest, fold)
        val_ds_f = RMITHeatmapDataset(
            manifest, img_size, val_names_f, is_training=False,
            sigma=2.0, augment=False,
        )
        val_loader_f = DataLoader(val_ds_f, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        
        m_off = eval_cackd(model2, val_loader_f, device, img_size, use_refine=True)
        m_full = eval_cackd(model2, val_loader_f, device, img_size, use_refine=False)
        print(f"  Switch OFF (uniform conf): mean_err={m_off['mean_error']:.2f}px "
              f"PCK@20={m_off['pck_20']:.3f}")
        print(f"  Fully OFF (skip UMGR): mean_err={m_full['mean_error']:.2f}px "
              f"PCK@20={m_full['pck_20']:.3f}")
        results[f"cackd_s2_f{fold}_off"] = m_off
        results[f"cackd_s2_f{fold}_fully_off"] = m_full
    
    # Save results
    out_path = "logs/cackd_switch_off_eval.json"
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Summary comparison
    print("\n=== SUMMARY ===")
    print(f"Baseline half-split: 3.76px / PCK@20=0.984 (known)")
    print(f"CDT half-split: 15.66px (known, with aux)")
    print(f"CAC-KD Stage2 half-split (switch OFF): {metrics_off['mean_error']:.2f}px / PCK@20={metrics_off['pck_20']:.3f}")
    print(f"CAC-KD Stage2 half-split (fully OFF): {metrics_full_off['mean_error']:.2f}px / PCK@20={metrics_full_off['pck_20']:.3f}")


if __name__ == "__main__":
    main()
