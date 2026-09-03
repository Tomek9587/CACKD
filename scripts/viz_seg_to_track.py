#!/usr/bin/env python3
"""
Visualize dataset conversion: cataract segmentation masks -> tracking keypoints.

Shows 4 progressive stages per instrument:
  1. Original image (cropped to instrument region)
  2. + Segmentation mask overlay (original COCO annotation)
  3. + Principal axis (PCA major axis)
  4. + 4 tracking keypoints (tip -> 1/3 -> 2/3 -> end)

Output: analysis/figs/dataset/seg_to_track_conversion.png + individual panels
"""

import json
import os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, FancyBboxPatch
from matplotlib.lines import Line2D

# ============================================================
# Paths
# ============================================================
SEG_ROOT = "/root/autodl-tmp/segmentation_cataract_1k/segmentation_cataract_1k"
COCO_ROOT = os.path.join(SEG_ROOT, "Annotations/Coco-Annotations")
IMG_ROOT = os.path.join(SEG_ROOT, "Annotations/Images-and-Supervisely-Annotations")
PROJECT_ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking"
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/figs/dataset")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Config
# ============================================================
INSTRUMENT_CLASSES = {
    "Phacoemulsification Tip", "Spatula", "Irrigation-Aspiration",
    "Gauge", "Capsulorhexis Forceps", "Capsulorhexis Cystotome",
    "Lens Injector", "Katena Forceps", "Slit Knife", "Incision Knife",
}

# Diverse instrument shapes for visualization
TARGET_INSTRUMENTS = [
    "Phacoemulsification Tip",   # thick, common (518 frames)
    "Capsulorhexis Forceps",      # forceps shape (108 frames)
    "Slit Knife",                 # thin elongated (20 frames)
]

# Visual style
MASK_FILL  = (0.0, 0.75, 0.55, 0.28)
MASK_EDGE  = (0.0, 0.75, 0.55, 1.0)
AXIS_COLOR = (1.0, 0.82, 0.0, 1.0)
KP_COLORS  = [(0.95, 0.30, 0.30), (1.0, 0.62, 0.0),
              (0.0, 0.78, 0.85), (0.55, 0.35, 0.95)]
KP_NAMES   = ["kp$_0$ (tip)", "kp$_1$ (1/3)", "kp$_2$ (2/3)", "kp$_3$ (end)"]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.8,
})


# ============================================================
# Core: PCA-based keypoint extraction (mirrors extract_cataract_keypoints.py)
# ============================================================
def polygon_to_keypoints_and_axis(points):
    """
    Extract 4 keypoints along the principal axis from a segmentation polygon.
    Returns: keypoints (4,2), centroid (2,), major_axis (2,), t_min, t_max
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] < 3:
        return None

    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # PCA: major axis = eigenvector with largest eigenvalue
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, -1]
    if np.dot(major, [1, 0]) < 0:
        major = -major  # prefer pointing right

    # Project polygon points onto major axis
    proj = centered @ major
    t_min, t_max = float(proj.min()), float(proj.max())

    # 4 keypoints: tip, 1/3, 2/3, end
    ts = np.array([
        t_max,                              # tip
        t_max - (t_max - t_min) * 0.25,     # 1/3 from tip
        t_min + (t_max - t_min) * 0.25,     # 2/3 from tip
        t_min,                               # end
    ])
    keypoints = np.array([centroid + t * major for t in ts])
    return keypoints, centroid, major, t_min, t_max


# ============================================================
# Frame selection
# ============================================================
def select_frames():
    """Find one representative frame per target instrument type."""
    selected = {}
    candidates = {t: [] for t in TARGET_INSTRUMENTS}

    cases = sorted([d for d in os.listdir(COCO_ROOT) if d.startswith("case_")])
    for case in cases:
        coco_path = os.path.join(COCO_ROOT, case, "annotations/instances.json")
        if not os.path.exists(coco_path):
            continue
        with open(coco_path) as f:
            coco = json.load(f)

        cats = {c["id"]: c["name"] for c in coco["categories"]}
        img_dict = {img["id"]: img for img in coco["images"]}

        anns_by_img = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        for img_id, anns in anns_by_img.items():
            img_info = img_dict.get(img_id)
            if img_info is None:
                continue

            # Pick largest instrument annotation (same logic as extraction script)
            best_ann, best_name, best_area = None, None, 0
            for ann in anns:
                cat_name = cats.get(ann["category_id"], "")
                if cat_name not in INSTRUMENT_CLASSES:
                    continue
                area = ann.get("area", 0)
                if area > best_area:
                    best_ann, best_name, best_area = ann, cat_name, area

            if best_ann is None or best_name not in TARGET_INSTRUMENTS:
                continue

            seg = best_ann["segmentation"]
            if isinstance(seg, dict) or not seg:
                continue
            if best_area < 3000 or best_area > 80000:
                continue

            poly = np.array(seg[0]).reshape(-1, 2)
            if poly.shape[0] < 3:
                continue

            img_path = os.path.join(IMG_ROOT, case, "img", img_info["file_name"])
            if not os.path.exists(img_path):
                continue

            candidates[best_name].append({
                "case": case,
                "img_path": img_path,
                "polygon": poly,
                "area": best_area,
                "instrument": best_name,
            })

    # Select median-area frame per instrument (avoids extremes)
    for inst, frames in candidates.items():
        if not frames:
            continue
        frames.sort(key=lambda x: x["area"])
        selected[inst] = frames[len(frames) // 2]

    return selected


# ============================================================
# Crop helper
# ============================================================
def crop_to_region(img_shape, polygon, keypoints, margin_ratio=0.35):
    """Compute crop bounds around polygon + keypoints with margin."""
    all_pts = np.vstack([polygon, keypoints])
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    w, h = x_max - x_min, y_max - y_min
    mx = max(w * margin_ratio, 50)
    my = max(h * margin_ratio, 50)
    cx1 = max(0, int(x_min - mx))
    cy1 = max(0, int(y_min - my))
    cx2 = min(img_shape[1], int(x_max + mx))
    cy2 = min(img_shape[0], int(y_max + my))
    return cx1, cy1, cx2, cy2


# ============================================================
# Adaptive keypoint label placement
# ============================================================
def add_kp_label(ax, x, y, i, color, img_width, img_height, font_size=8):
    """Place keypoint label adaptively to avoid clipping at image edges."""
    # Horizontal direction: keep label inside image horizontally
    if x > img_width * 0.75:
        dx = -12
        ha = 'right'
    elif x < img_width * 0.25:
        dx = 12
        ha = 'left'
    else:
        dx = 10 if i % 2 == 0 else -10
        ha = 'left' if dx > 0 else 'right'

    # Vertical direction: push label away from top/bottom edges
    if y > img_height * 0.75:
        dy = 10
        va = 'top'
    elif y < img_height * 0.25:
        dy = -10
        va = 'bottom'
    else:
        dy = 10 if i % 2 == 0 else -10
        va = 'top' if dy > 0 else 'bottom'

    ax.annotate(f'kp$_{{{i}}}$', (x, y), textcoords="offset points",
                xytext=(dx, dy), fontsize=font_size, color=color,
                fontweight='bold', zorder=7, ha=ha, va=va, clip_on=False,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none',
                          pad=0.5, boxstyle='round,pad=0.2'))


# ============================================================
# Main visualization
# ============================================================
def main():
    print("Selecting representative frames...")
    selected = select_frames()
    instruments = [t for t in TARGET_INSTRUMENTS if t in selected]

    if not instruments:
        print("ERROR: No suitable frames found!")
        return

    n = len(instruments)
    print(f"Found {n} instrument types: {instruments}")

    col_titles = [
        "Original Image",
        "+ Segmentation Mask",
        "+ Principal Axis (PCA)",
        "+ Tracking Keypoints",
    ]

    fig, axes = plt.subplots(n, 4, figsize=(18, 4.8 * n + 1.2))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, instrument in enumerate(instruments):
        info = selected[instrument]
        img = np.array(Image.open(info["img_path"]).convert("RGB"))
        poly = info["polygon"]
        result = polygon_to_keypoints_and_axis(poly)
        if result is None:
            continue
        kpts, centroid, major, t_min, t_max = result

        cx1, cy1, cx2, cy2 = crop_to_region(img.shape, poly, kpts)

        for col in range(4):
            ax = axes[row, col]
            ax.imshow(img[cy1:cy2, cx1:cx2])
            ax.set_xticks([])
            ax.set_yticks([])

            # Thin border
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color('#888888')

            def tc(pts):
                return np.asarray(pts) - np.array([cx1, cy1])

            # Stage 2+: segmentation mask
            if col >= 1:
                poly_c = tc(poly)
                patch = MplPolygon(poly_c, closed=True,
                                   facecolor=MASK_FILL, edgecolor=MASK_EDGE,
                                   linewidth=1.5, zorder=3)
                ax.add_patch(patch)

            # Stage 3+: principal axis
            if col >= 2:
                a_start = tc(centroid + t_min * major)
                a_end   = tc(centroid + t_max * major)
                ax.plot([a_start[0], a_end[0]], [a_start[1], a_end[1]],
                        color=AXIS_COLOR, linewidth=2.5, linestyle='--',
                        zorder=4, solid_capstyle='round')
                # Arrow at tip end
                ax.annotate('', xy=a_end, xytext=tc(centroid + (t_max - (t_max-t_min)*0.12) * major),
                            arrowprops=dict(arrowstyle='->', color=AXIS_COLOR, lw=2),
                            zorder=5)
                c = tc(centroid)
                ax.plot(c[0], c[1], '+', color=AXIS_COLOR, markersize=7,
                        markeredgewidth=2, zorder=5)

            # Stage 4: keypoints
            if col >= 3:
                kc = tc(kpts)
                h, w = img[cy1:cy2, cx1:cx2].shape[:2]
                # Skeleton line connecting keypoints
                ax.plot(kc[:, 0], kc[:, 1], '-', color=(1, 1, 1, 0.5),
                        linewidth=1.2, zorder=5)
                for i, (x, y) in enumerate(kc):
                    ax.plot(x, y, 'o', color=KP_COLORS[i], markersize=9,
                            markeredgecolor='white', markeredgewidth=1.3, zorder=6)
                    add_kp_label(ax, x, y, i, KP_COLORS[i], w, h, font_size=8)

            # Titles
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12, fontweight='bold', pad=10)

            # Row label (instrument name)
            if col == 0:
                ax.set_ylabel(instrument, fontsize=10, fontweight='bold',
                              rotation=0, labelpad=15, va='center', ha='right')

    # Legend at bottom
    legend_elements = [
        MplPolygon([[0, 0], [1, 0], [1, 1], [0, 1]],
                   facecolor=MASK_FILL, edgecolor=MASK_EDGE, linewidth=1.5,
                   label='Segmentation mask (COCO)'),
        Line2D([0], [0], color=AXIS_COLOR, linewidth=2.5, linestyle='--',
               marker='+', markersize=7, markeredgewidth=2,
               label='Principal axis (PCA)'),
    ]
    for i in range(4):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=KP_COLORS[i],
                   markersize=9, markeredgecolor='white', markeredgewidth=1.2,
                   label=KP_NAMES[i])
        )

    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
               fontsize=9, frameon=True, fancybox=True, edgecolor='#cccccc',
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Dataset Conversion: Segmentation Masks $\\rightarrow$ Tracking Keypoints",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = os.path.join(OUT_DIR, "seg_to_track_conversion.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.12,
                facecolor='white')
    plt.close()
    print(f"Saved: {out_path}")

    # ---- Individual high-res panels ----
    for instrument in instruments:
        info = selected[instrument]
        img = np.array(Image.open(info["img_path"]).convert("RGB"))
        poly = info["polygon"]
        result = polygon_to_keypoints_and_axis(poly)
        if result is None:
            continue
        kpts, centroid, major, t_min, t_max = result
        cx1, cy1, cx2, cy2 = crop_to_region(img.shape, poly, kpts, margin_ratio=0.4)

        fig2, ax2 = plt.subplots(1, 1, figsize=(6.5, 5))
        ax2.imshow(img[cy1:cy2, cx1:cx2])
        ax2.set_xticks([])
        ax2.set_yticks([])

        def tc(pts):
            return np.asarray(pts) - np.array([cx1, cy1])

        poly_c = tc(poly)
        ax2.add_patch(MplPolygon(poly_c, closed=True,
                                facecolor=MASK_FILL, edgecolor=MASK_EDGE,
                                linewidth=1.8, zorder=3))

        kc = tc(kpts)
        h, w = img[cy1:cy2, cx1:cx2].shape[:2]
        ax2.plot(kc[:, 0], kc[:, 1], '-', color=(1, 1, 1, 0.5), linewidth=1.2, zorder=5)
        for i, (x, y) in enumerate(kc):
            ax2.plot(x, y, 'o', color=KP_COLORS[i], markersize=11,
                     markeredgecolor='white', markeredgewidth=1.5, zorder=6)
            add_kp_label(ax2, x, y, i, KP_COLORS[i], w, h, font_size=9)

        ax2.set_title(f"{instrument}", fontsize=11, fontweight='bold')
        plt.tight_layout()
        safe = instrument.replace(' ', '_').lower()
        out2 = os.path.join(OUT_DIR, f"conversion_{safe}.png")
        plt.savefig(out2, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {out2}")


if __name__ == '__main__':
    main()
