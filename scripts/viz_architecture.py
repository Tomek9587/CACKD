#!/usr/bin/env python3
"""
Architecture and pipeline diagrams for the paper.

Generates:
  1. CDT pipeline flowchart
  2. CAC-KD network architecture diagram

Output: analysis/figs/architecture/
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PROJECT_ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking"
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/figs/architecture")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'figure.dpi': 300,
})


# ============================================================
# Helper: draw rounded box + arrow
# ============================================================
def box(ax, x, y, w, h, text, color='#E8F4FD', edge='#2B7BB9', text_color='#1a1a1a',
        fontsize=10, fontweight='normal'):
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.02",
                         facecolor=color, edgecolor=edge, linewidth=1.5, zorder=2)
    ax.add_patch(patch)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=fontweight, zorder=3)
    return patch


def arrow(ax, x1, y1, x2, y2, color='#555555', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle="arc3,rad=0"),
                zorder=1)


# ============================================================
# 1. CDT pipeline flowchart
# ============================================================
def plot_cdt_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Title
    ax.text(7, 6.5, 'Cross-Dataset Transfer (CDT) Pipeline',
            fontsize=16, ha='center', fontweight='bold', color='#1a1a1a')

    # Column 1: Target domain
    box(ax, 2.2, 5.0, 2.8, 0.9, 'Target Domain\n(RMIT real labels)', color='#D6EAF8', edge='#2B7BB9', fontweight='bold')
    box(ax, 2.2, 3.2, 2.8, 0.9, 'Train Baseline\nV11 heatmap_offset', color='#AED6F1', edge='#2B7BB9')

    # Column 2: Auxiliary domain
    box(ax, 7.0, 5.0, 2.8, 0.9, 'Auxiliary Domain\n(cataract pseudo labels)', color='#FADBD8', edge='#C0392B', fontweight='bold')
    box(ax, 7.0, 3.2, 2.8, 0.9, 'Predict Keypoints\nwith baseline model', color='#F5B7B1', edge='#C0392B')
    box(ax, 7.0, 1.5, 2.8, 0.9, 'Compute Confidence\nconf = 1 - err/scale', color='#F5B7B1', edge='#C0392B')

    # Column 3: Joint training
    box(ax, 11.5, 3.2, 2.8, 1.2, 'Confidence-Weighted\nJoint Training', color='#D5F5E3', edge='#1E8449', fontweight='bold', fontsize=11)
    box(ax, 11.5, 1.5, 2.8, 0.9, 'Final CDT Model', color='#ABEBC6', edge='#1E8449', fontweight='bold')

    # Arrows
    arrow(ax, 2.2, 4.5, 2.2, 3.7)   # target -> train baseline
    arrow(ax, 2.2, 2.7, 2.2, 1.8, color='#2B7BB9')  # baseline -> joint (target conf=1.0)
    arrow(ax, 2.2, 1.5, 9.0, 1.5, color='#2B7BB9')  # target conf=1.0 -> joint
    arrow(ax, 7.0, 4.5, 7.0, 3.7)   # aux -> predict
    arrow(ax, 7.0, 2.7, 7.0, 2.0)   # predict -> confidence
    arrow(ax, 8.4, 1.5, 10.0, 1.5, color='#C0392B')  # confidence -> joint
    arrow(ax, 11.5, 2.6, 11.5, 2.0)  # joint -> final
    arrow(ax, 3.7, 3.2, 5.5, 3.2, color='#2B7BB9')  # baseline -> predict aux

    # Confidence annotations
    ax.text(2.2, 1.0, 'conf = 1.0', fontsize=9, color='#2B7BB9', ha='center', style='italic')
    ax.text(7.0, 0.9, 'conf = max(0, 1 - err/scale)', fontsize=9, color='#C0392B', ha='center', style='italic')
    ax.text(7.0, 0.4, 'scale = 3 × median(error)', fontsize=9, color='#C0392B', ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cdt_pipeline.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/cdt_pipeline.png")


# ============================================================
# 2. CAC-KD architecture diagram
# ============================================================
def plot_cackd_architecture():
    fig, ax = plt.subplots(figsize=(13, 8.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(-3.2, 8.5)
    ax.axis('off')

    # Title
    ax.text(6.5, 8.1, 'CAC-KD Network Architecture',
            fontsize=16, ha='center', fontweight='bold', color='#1a1a1a')
    ax.text(6.5, 7.7, 'Confidence-Aware Cross-Domain Keypoint Detection',
            fontsize=11, ha='center', color='#555555', style='italic')

    # Color scheme
    c_input = '#F2F3F4'
    c_backbone = '#D6EAF8'
    c_module = '#FADBD8'
    c_head = '#E8DAEF'
    c_out = '#D5F5E3'
    c_edge = '#555555'

    # Input image
    box(ax, 6.5, 6.5, 2.2, 0.8, 'Input Image\n(512 × 512)', color=c_input, edge=c_edge, fontsize=9)
    arrow(ax, 6.5, 6.1, 6.5, 5.6)

    # Encoder
    box(ax, 6.5, 5.2, 3.2, 0.8, 'EfficientNetV2-S Encoder\nImageNet pretrained', color=c_backbone, edge='#2B7BB9', fontweight='bold')
    arrow(ax, 6.5, 4.8, 6.5, 4.3)

    # Decoder
    box(ax, 6.5, 3.9, 3.0, 0.8, 'Decoder\n[d1, d2, d3] features', color=c_backbone, edge='#2B7BB9')
    arrow(ax, 6.5, 3.5, 6.5, 3.0)

    # CGFA
    box(ax, 6.5, 2.55, 3.2, 0.8, 'CGFA\nConfidence-Gated Feature Adapter\n~320 params', color=c_module, edge='#C0392B', fontweight='bold')
    # Offline confidence arrow to CGFA
    arrow(ax, 2.5, 2.55, 4.9, 2.55, color='#C0392B')
    ax.text(2.5, 2.9, 'Offline confidence\n(from CDT)', fontsize=9, color='#C0392B', ha='center')
    arrow(ax, 6.5, 2.15, 6.5, 1.7)

    # Heads (split)
    box(ax, 4.0, 1.05, 2.4, 0.8, 'Heatmap Head\n(4 channels)', color=c_head, edge='#7D3C98')
    box(ax, 9.0, 1.05, 2.4, 0.8, 'Offset Head\n(4 × 2 channels)', color=c_head, edge='#7D3C98')
    arrow(ax, 5.0, 2.15, 5.0, 1.5)  # to heatmap
    arrow(ax, 8.0, 2.15, 8.0, 1.5)  # to offset

    # Coarse xy
    box(ax, 4.0, 0.05, 2.2, 0.5, 'Coarse xy', color=c_head, edge='#7D3C98', fontsize=9)
    # Offset correction
    box(ax, 9.0, 0.05, 2.2, 0.5, 'Offset Δxy', color=c_head, edge='#7D3C98', fontsize=9)

    # Arrows to UMGR
    arrow(ax, 5.1, 0.05, 5.8, 0.05, color='#7D3C98')
    arrow(ax, 8.0, 0.05, 7.3, 0.05, color='#7D3C98')

    # UMGR (bottom center)
    box(ax, 6.5, -1.3, 3.4, 0.8, 'UMGR\nUncertainty-Modulated Geometry Refinement\n~664 params', color=c_module, edge='#C0392B', fontweight='bold')
    # Confidence arrow to UMGR
    arrow(ax, 2.5, -1.3, 4.7, -1.3, color='#C0392B')
    ax.text(2.5, -0.95, 'Offline confidence\n(from CDT)', fontsize=9, color='#C0392B', ha='center')

    # Output
    arrow(ax, 6.5, -1.7, 6.5, -1.8)
    box(ax, 6.5, -2.2, 3.0, 0.8, 'Refined 4 Keypoints\n(x, y) normalised', color=c_out, edge='#1E8449', fontweight='bold')

    # Side note
    ax.text(12.0, 4.5, 'Key modules:', fontsize=11, fontweight='bold', color='#1a1a1a', ha='right')
    ax.text(12.0, 4.0, '• CGFA: aux-only feature gating', fontsize=9, color='#555555', ha='right')
    ax.text(12.0, 3.6, '• UMGR: geometry refinement', fontsize=9, color='#555555', ha='right')
    ax.text(12.0, 3.2, '• Offline confidence from CDT', fontsize=9, color='#555555', ha='right')
    ax.text(12.0, 2.8, '• Total new params: ~1K', fontsize=9, color='#555555', ha='right')

    # Legend box
    ax.text(0.5, 4.5, 'Inference mode:', fontsize=11, fontweight='bold', color='#1a1a1a')
    ax.text(0.5, 4.0, 'Target domain: CGFA passes\nfeatures unchanged', fontsize=9, color='#555555')
    ax.text(0.5, 3.35, 'Auxiliary domain: CGFA gates\nfeatures by confidence', fontsize=9, color='#555555')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'cackd_architecture.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/cackd_architecture.png")


if __name__ == '__main__':
    plot_cdt_pipeline()
    plot_cackd_architecture()
    print("\nAll architecture diagrams generated.")
