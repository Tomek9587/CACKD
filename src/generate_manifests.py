import os
import glob
import json
import random
import argparse

# Paths
ROOT_DIR = "/root/autodl-tmp/longfei"
ODIR_DIR = os.path.join(ROOT_DIR, "ODIR-5K_Testing_Images")
FOVEA_DIR = os.path.join(ROOT_DIR, "FOVEA")
RMIT_DIR = os.path.join(ROOT_DIR, "RMIT")
MANIFEST_DIR = os.path.join(ROOT_DIR, "manifests")

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def _find_existing_image_path(root_dir: str, filename: str, cache: dict) -> str:
    if not filename:
        return ""
    if filename in cache:
        return cache[filename]

    # 1) direct join
    p = os.path.join(root_dir, filename)
    if os.path.exists(p):
        cache[filename] = p
        return p

    # 2) recursive search (first hit)
    hits = glob.glob(os.path.join(root_dir, "**", filename), recursive=True)
    p2 = hits[0] if hits else ""
    cache[filename] = p2
    return p2


def _extract_odir_labels_8(row: list) -> list:
    cands = []
    for i in range(0, len(row) - 8 + 1):
        w = row[i:i + 8]
        if all(str(x).strip() in ("0", "1") for x in w):
            cands.append([int(str(x).strip()) for x in w])
    return cands[-1] if cands else []


def generate_dr_manifest():
    print("\n--- Generating DR Manifest (ODIR) ---")
    if not os.path.exists(ODIR_DIR):
        print(f"Warning: {ODIR_DIR} not found. Skipping DR manifest.")
        return

    odir_csv = os.path.join(ODIR_DIR, "ODIR-5K.csv")
    if not os.path.exists(odir_csv):
        print(f"Warning: {odir_csv} not found. Skipping DR manifest.")
        return

    import csv

    img_cache = {}
    data = []
    pos = 0

    with open(odir_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "id":
                continue
            if not row[0].strip().isdigit():
                continue
            if len(row) < 6:
                continue

            left_fn = row[3].strip()
            right_fn = row[4].strip()

            labels8 = _extract_odir_labels_8(row)
            if not labels8:
                continue
            d_flag = labels8[1]
            label = 1 if d_flag == 1 else 0
            pos += label

            left_path = _find_existing_image_path(ODIR_DIR, left_fn, img_cache)
            right_path = _find_existing_image_path(ODIR_DIR, right_fn, img_cache)

            if left_path:
                data.append((left_path, label))
            if right_path:
                data.append((right_path, label))

    print(f"Loaded {len(data)} eye-images from {odir_csv}")
    print(f"Positive samples (label=1): {pos} (patient-rows counted, not per-eye)")
    if len(data) == 0:
        print("No valid samples found for DR. Please check filenames under ODIR directory.")
        return

    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    ensure_dir(MANIFEST_DIR)
    train_csv = os.path.join(MANIFEST_DIR, "dr_train.csv")
    val_csv = os.path.join(MANIFEST_DIR, "dr_val.csv")

    with open(train_csv, "w", encoding="utf-8", newline="") as f:
        f.write("image_path,label\n")
        for p, l in train_data:
            f.write(f"{p},{l}\n")

    with open(val_csv, "w", encoding="utf-8", newline="") as f:
        f.write("image_path,label\n")
        for p, l in val_data:
            f.write(f"{p},{l}\n")

    print(f"Saved {train_csv} ({len(train_data)} samples)")
    print(f"Saved {val_csv} ({len(val_data)} samples)")
    print("DR 标签说明：这里默认二分类 label=1 表示 D(糖网/糖尿病相关)=1，否则为 0。")

def generate_fovea_manifest():
    print(f"\n--- Generating FOVEA Manifest ---")
    if not os.path.exists(FOVEA_DIR):
        print(f"Warning: {FOVEA_DIR} not found. Skipping FOVEA manifest.")
        return

    # FOVEA structure assumption:
    # We need to pair pre_img, intra_img, pre_vessel, pre_od, intra_vessel, intra_od
    # Based on metadata.json presence, let's scan all files first
    files = sorted(glob.glob(os.path.join(FOVEA_DIR, "**", "*"), recursive=True))
    images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))]
    
    print(f"Found {len(images)} images in {FOVEA_DIR}")
    
    # Heuristic grouping by filename
    # Expecting filenames to contain IDs and types (pre/intra/mask)
    # Example heuristic: ID_pre.png, ID_intra.png
    
    # Group by potential ID (first part of filename)
    grouped = {}
    for img in images:
        basename = os.path.basename(img)
        name_no_ext = os.path.splitext(basename)[0]
        # Heuristic: split by '_' and take first part as ID?
        # Adjust this based on actual filenames
        parts = name_no_ext.split('_')
        pid = parts[0] # Simplest assumption
        
        if pid not in grouped:
            grouped[pid] = {}
        
        # Categorize
        if "pre" in name_no_ext and "vessel" in name_no_ext:
            grouped[pid]["pre_vessel"] = img
        elif "pre" in name_no_ext and "od" in name_no_ext:
            grouped[pid]["pre_od"] = img
        elif "pre" in name_no_ext:
            grouped[pid]["pre_img"] = img
        elif "intra" in name_no_ext and "vessel" in name_no_ext:
            grouped[pid]["intra_vessel"] = img
        elif "intra" in name_no_ext and "od" in name_no_ext:
            grouped[pid]["intra_od"] = img
        elif "intra" in name_no_ext:
            grouped[pid]["intra_img"] = img

    # Filter complete pairs
    pairs = []
    required_keys = ["pre_img", "intra_img", "pre_vessel", "pre_od", "intra_vessel", "intra_od"]
    
    for pid, items in grouped.items():
        if all(k in items for k in required_keys):
            item = {"patient_id": pid}
            item.update(items)
            pairs.append(item)
    
    print(f"Found {len(pairs)} complete pairs out of {len(grouped)} potential IDs.")
    
    if len(pairs) == 0:
        print("Warning: No complete pairs found. Check filename patterns.")
        # Fallback: create dummy if at least some images found? No, dangerous.
        return

    out_json = os.path.join(MANIFEST_DIR, "fovea_pairs.json")
    ensure_dir(MANIFEST_DIR)
    with open(out_json, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {out_json}")

def generate_rmit_manifest():
    print(f"\n--- Generating RMIT Manifest ---")
    if not os.path.exists(RMIT_DIR):
        print(f"Warning: {RMIT_DIR} not found. Skipping RMIT manifest.")
        return
        
    # Check if rmit_manifest.json already exists in RMIT_DIR
    existing_manifest = os.path.join(RMIT_DIR, "rmit_manifest.json")
    if os.path.exists(existing_manifest):
        print(f"RMIT manifest already exists at {existing_manifest}. Using it.")
        # Copy to MANIFEST_DIR for consistency
        ensure_dir(MANIFEST_DIR)
        import shutil
        dst = os.path.join(MANIFEST_DIR, "rmit_manifest.json")
        shutil.copy2(existing_manifest, dst)
        print(f"Copied to {dst}")
        return

    # If not, try to run build_rmit_manifest.py logic or assume structure
    # Since we saw build_rmit_manifest.py exists, we should probably run it manually or trust the user run it.
    # Here we just look for sequences in RMIT_DIR
    
    seq_dirs = sorted(glob.glob(os.path.join(RMIT_DIR, "seq*")))
    sequences = []
    for d in seq_dirs:
        seq_name = os.path.basename(d)
        ann_csv = os.path.join(d, "ann.csv")
        if os.path.exists(ann_csv):
            sequences.append({
                "name": seq_name,
                "frames_dir": os.path.abspath(d),
                "ann_csv": os.path.abspath(ann_csv)
            })
    
    if sequences:
        out_json = os.path.join(MANIFEST_DIR, "rmit_manifest.json")
        ensure_dir(MANIFEST_DIR)
        with open(out_json, "w") as f:
            json.dump({"sequences": sequences}, f, indent=2)
        print(f"Saved {out_json} with {len(sequences)} sequences.")
    else:
        print("No RMIT sequences with ann.csv found.")

if __name__ == "__main__":
    generate_dr_manifest()
    generate_fovea_manifest()
    generate_rmit_manifest()
    raise SystemExit(0)
