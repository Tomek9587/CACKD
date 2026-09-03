import json
import os
import shutil
import numpy as np
from PIL import Image, ImageDraw
from collections import defaultdict, Counter
from tqdm import tqdm
import cv2

"""
Extract pseudo keypoints from cataract segmentation masks.
Each instrument mask is converted to 4 keypoints along the principal axis:
    kp0: tip (far end along major axis)
    kp1: 1/3 from tip
    kp2: 2/3 from tip
    kp3: end (other end)
The output structure mirrors RMIT: each case has frames/ and ann.csv.
"""

ROOT = "/root/autodl-tmp/segmentation_cataract_1k/segmentation_cataract_1k"
OUT_ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking/data/cataract_keypoints"
VIS_DIR = os.path.join(OUT_ROOT, "visualizations")

# Instrument classes we care about (ignore anatomical structures)
INSTRUMENT_CLASSES = {
    "Phacoemulsification Tip",
    "Spatula",
    "Irrigation-Aspiration",
    "Gauge",
    "Capsulorhexis Forceps",
    "Capsulorhexis Cystotome",
    "Lens Injector",
    "Katena Forceps",
    "Slit Knife",
    "Incision Knife",
}


def polygon_to_keypoints(points):
    """
    points: (N, 2) array of polygon coordinates
    Returns: (4, 2) keypoints along the principal axis from tip to end.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return None

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, -1]
    if np.dot(major, [1, 0]) < 0:
        major = -major  # prefer pointing roughly right

    proj = centered @ major
    t_min, t_max = float(proj.min()), float(proj.max())
    ts = np.linspace(t_min, t_max, 6)[1:-1]  # 4 interior params? no, 6 points -> 4 are 1,2,3,4
    # Actually we want 4 keypoints: tip, 1/3, 2/3, end
    ts = np.array([t_max, t_max - (t_max - t_min) * 0.25,
                   t_min + (t_max - t_min) * 0.25, t_min])

    keypoints = []
    for t in ts:
        pt = mean + t * major
        keypoints.append(pt)
    return np.array(keypoints)


def process_case(case_name, vis_every=0):
    coco_path = os.path.join(ROOT, "Annotations/Coco-Annotations", case_name, "annotations/instances.json")
    if not os.path.exists(coco_path):
        return None

    with open(coco_path) as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco["categories"]}
    img_dict = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    case_out_dir = os.path.join(OUT_ROOT, "frames", case_name)
    os.makedirs(case_out_dir, exist_ok=True)

    rows = []
    vis_count = 0
    for img_id, anns in anns_by_img.items():
        img_info = img_dict[img_id]
        src_img_path = os.path.join(ROOT, "Annotations/Images-and-Supervisely-Annotations",
                                     case_name, "img", img_info["file_name"])
        if not os.path.exists(src_img_path):
            continue

        # Copy / resize image to RMIT size (640x480) for easier integration
        img_pil = Image.open(src_img_path).convert("RGB")
        W, H = img_pil.size  # 1024x768

        # Collect all instrument keypoints for this frame
        # We will only keep the largest instrument if multiple; or keep all if they are separate frames
        # RMIT has exactly one instrument per frame with 4 keypoints. So we merge to one instrument per frame:
        # pick the instrument with largest mask area.
        best_ann = None
        best_area = 0
        for ann in anns:
            cat_name = cats[ann["category_id"]]
            if cat_name not in INSTRUMENT_CLASSES:
                continue
            if ann["area"] > best_area:
                best_area = ann["area"]
                best_ann = (ann, cat_name)

        if best_ann is None:
            continue

        ann, cat_name = best_ann
        seg = ann["segmentation"]
        if isinstance(seg, dict) or not seg:
            continue
        poly = np.array(seg[0]).reshape(-1, 2)
        kpts = polygon_to_keypoints(poly)
        if kpts is None:
            continue

        # Scale keypoints to 640x480 to match RMIT resolution
        scale_x = 640.0 / W
        scale_y = 480.0 / H
        kpts[:, 0] *= scale_x
        kpts[:, 1] *= scale_y
        kpts = np.clip(kpts, 0, 9999)

        dst_img_name = img_info["file_name"]
        dst_img_path = os.path.join(case_out_dir, dst_img_name)
        img_pil.resize((640, 480), Image.BILINEAR).save(dst_img_path)

        rows.append({
            "frame": dst_img_name,
            "x0": kpts[0, 0], "y0": kpts[0, 1],
            "x1": kpts[1, 0], "y1": kpts[1, 1],
            "x2": kpts[2, 0], "y2": kpts[2, 1],
            "x3": kpts[3, 0], "y3": kpts[3, 1],
            "class": cat_name,
        })

        # Visualization
        if vis_every > 0 and vis_count < vis_every:
            vis_count += 1
            img_viz = img_pil.resize((640, 480), Image.BILINEAR).convert("RGBA")
            overlay = Image.new("RGBA", img_viz.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            # scale polygon
            poly_scaled = poly.copy()
            poly_scaled[:, 0] *= scale_x
            poly_scaled[:, 1] *= scale_y
            draw.polygon([tuple(p) for p in poly_scaled], fill=(255, 0, 0, 60), outline=(255, 0, 0, 200))
            colors = ["red", "green", "blue", "yellow"]
            for i, (x, y) in enumerate(kpts):
                draw.ellipse([x-6, y-6, x+6, y+6], fill=colors[i], outline="white")
                draw.text((x+10, y-10), f"kp{i}", fill=colors[i])
            img_viz = Image.alpha_composite(img_viz, overlay).convert("RGB")
            vis_path = os.path.join(VIS_DIR, f"{case_name}_{dst_img_name}")
            img_viz.save(vis_path)

    if not rows:
        return None

    ann_csv = os.path.join(OUT_ROOT, "frames", case_name, "ann.csv")
    import csv
    with open(ann_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "x0", "y0", "x1", "y1", "x2", "y2", "x3", "y3", "class"])
        writer.writeheader()
        writer.writerows(rows)

    return {
        "case": case_name,
        "frames": len(rows),
        "class": Counter([r["class"] for r in rows]),
    }


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

    coco_root = os.path.join(ROOT, "Annotations/Coco-Annotations")
    cases = sorted([d for d in os.listdir(coco_root) if d.startswith("case_")])
    print(f"Found {len(cases)} cases to process")

    stats = []
    for case in tqdm(cases, desc="Processing cataract cases"):
        s = process_case(case, vis_every=2)
        if s:
            stats.append(s)

    # Summary
    total_frames = sum(s["frames"] for s in stats)
    all_classes = Counter()
    for s in stats:
        all_classes.update(s["class"])

    print(f"\nTotal cases processed: {len(stats)}")
    print(f"Total frames with keypoints: {total_frames}")
    print("Class distribution:")
    for cls, n in all_classes.most_common():
        print(f"  {cls}: {n}")

    # Write manifest
    manifest = {
        "dataset": "cataract_keypoints",
        "source": "segmentation_cataract_1k",
        "cases": [{"name": s["case"], "frames_dir": os.path.join(OUT_ROOT, "frames", s["case"]),
                   "ann_csv": os.path.join(OUT_ROOT, "frames", s["case"], "ann.csv")} for s in stats]
    }
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {os.path.join(OUT_ROOT, 'manifest.json')}")
