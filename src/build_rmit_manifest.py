import os
import json
import csv
import argparse
import shutil

def list_image_files(seq_dir):
    if not os.path.isdir(seq_dir):
        return {}, 0
    files = []
    for fn in os.listdir(seq_dir):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(fn)
    files.sort()
    frame_map = {}
    pad_len = 0
    for fn in files:
        stem = os.path.splitext(fn)[0]
        if stem.isdigit():
            pad_len = max(pad_len, len(stem))
            frame_map[stem] = fn
    return frame_map, pad_len

def parse_annotation_file(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            frame_id = parts[0]
            nums = []
            ok = True
            for p in parts[1:]:
                try:
                    nums.append(float(p))
                except ValueError:
                    ok = False
                    break
            if not ok:
                continue
            if len(nums) % 2 != 0:
                continue
            pairs = []
            for i in range(0, len(nums), 2):
                pairs.append((nums[i], nums[i + 1]))
            items.append((frame_id, pairs))
    return items

def build_manifest(base_dir, out_dir, seqs, manifest_path):
    print(f"Organizing dataset into: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    sequences = []
    
    for s in seqs:
        seq_name = f"seq{s}"
        ann_src = os.path.join(base_dir, "package", "annotation_source", f"test{s}.txt")
        src_seq_dir = os.path.join(base_dir, "separate", seq_name)
        
        if not os.path.isfile(ann_src) or not os.path.isdir(src_seq_dir):
            print(f"Skipping {seq_name}: source not found at {src_seq_dir} or {ann_src}")
            continue

        # Target directory for this sequence
        target_seq_dir = os.path.join(out_dir, seq_name)
        os.makedirs(target_seq_dir, exist_ok=True)
        
        print(f"Processing {seq_name}...")
        
        # 1. List and Copy Images
        frame_map, pad_len = list_image_files(src_seq_dir)
        # Copy all image files to the new location
        for fname in os.listdir(src_seq_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                src_file = os.path.join(src_seq_dir, fname)
                dst_file = os.path.join(target_seq_dir, fname)
                # Only copy if destination doesn't exist or size differs (simple check)
                if not os.path.exists(dst_file) or os.path.getsize(src_file) != os.path.getsize(dst_file):
                    shutil.copy2(src_file, dst_file)
        
        # 2. Generate Annotation CSV (V6: 4 keypoints)
        ann_csv = os.path.join(target_seq_dir, "ann.csv")
        rows = 0
        with open(ann_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # V6: 4 keypoints — LeftTip(0), RightTip(1), Junction(2), Midpoint(3)
            writer.writerow(["frame", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3"])
            
            for frame_id, pairs in parse_annotation_file(ann_src):
                if len(pairs) < 4:
                    continue
                
                # Swap XY for all points: raw (y, x) -> (x, y)
                coords = []
                valid = True
                for k in range(4):
                    raw_pt = pairs[k]
                    x, y = raw_pt[1], raw_pt[0]
                    if x < 0 or y < 0:
                        valid = False
                        break
                    coords.extend([x, y])
                
                if not valid:
                    continue
                
                # Find matching filename
                key = frame_id.zfill(pad_len) if pad_len > 0 else frame_id
                if key in frame_map:
                    frame_file = frame_map[key]
                elif frame_id in frame_map:
                    frame_file = frame_map[frame_id]
                else:
                    continue
                
                row = [frame_file] + [f"{c:.6f}" for c in coords]
                writer.writerow(row)
                rows += 1

        print(f"  - Copied images and generated ann.csv with {rows} frames.")

        if rows == 0:
            print(f"  - Warning: No valid annotations found for {seq_name}")
            continue

        sequences.append({
            "name": seq_name,
            "frames_dir": os.path.abspath(target_seq_dir),
            "ann_csv": os.path.abspath(ann_csv)
        })

    # 3. Save Manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"sequences": sequences}, f, indent=2, ensure_ascii=False)
    print(f"Manifest saved to {manifest_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default="./MICCAI14_Yeqing-master/MICCAI14_Yeqing-master/RetinalData")
    # Output to the requested RMIT directory
    p.add_argument("--out_dir", type=str, default="/root/autodl-tmp/longfei/surgical_tracking/data/RMIT", help="Output directory for reorganized RMIT dataset")
    p.add_argument("--manifest_path", type=str, default="/root/autodl-tmp/longfei/surgical_tracking/data/RMIT/rmit_manifest.json", help="Path to save the manifest JSON")
    
    args = p.parse_args()

    # seqs 1, 2, 3 are available in RetinalData
    seqs = [1, 2, 3]

    build_manifest(args.base_dir, args.out_dir, seqs, args.manifest_path)

if __name__ == "__main__":
    main()