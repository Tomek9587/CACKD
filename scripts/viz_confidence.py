#!/usr/bin/env python3
"""
Confidence distribution visualization for CAC-KD (cataract 30-fold).

Generates:
  1. Per-keypoint confidence histograms (2x2, with high/mid/low segments)
  2. Per-fold mean confidence grouped bar chart
  3. Violin plot: confidence distribution per keypoint (aggregated)

Output: analysis/figs/confidence/
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking"
CONF_DIR = os.path.join(PROJECT_ROOT, "logs")
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/figs/confidence")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

KP_COLORS = [
    (0.95, 0.30, 0.30),  # kp0 tip - red
    (1.00, 0.62, 0.00),  # kp1     - orange
    (0.00, 0.78, 0.85),  # kp2     - cyan
    (0.55, 0.35, 0.95),  # kp3 end - purple
]
KP_NAMES = [
    "kp$_0$ (tip)",
    "kp$_1$ (1/3)",
    "kp$_2$ (2/3)",
    "kp$_3$ (end)",
]

# Anomalous folds (from PAPER_RESULTS_SUMMARY.md Section 6.3)
ANOMALOUS_FOLDS = [9, 11, 14, 23, 25]


# ============================================================
# Load all confidence caches
# ============================================================
def load_all_confidences():
    """Load cackd_cat_conf_f{0..29}.json -> {fold: {img_path: [c0,c1,c2,c3]}}"""
    all_confs = {}
    for f in range(30):
        path = os.path.join(CONF_DIR, f"cackd_cat_conf_f{f}.json")
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping fold {f}")
            continue
        with open(path) as fh:
            all_confs[f] = json.load(fh)
    return all_confs


# ============================================================
# 1. Per-keypoint confidence histogram
# ============================================================
def plot_confidence_histogram(all_confs):
    # Aggregate all confidence values per keypoint
    all_vals = [[] for _ in range(4)]
    for fold, confs in all_confs.items():
        for img_path, kp_confs in confs.items():
            for i in range(4):
                all_vals[i].append(kp_confs[i])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    bins = np.linspace(0, 1, 31)

    for i, ax in enumerate(axes.flat):
        vals = np.array(all_vals[i])
        ax.hist(vals, bins=bins, color=KP_COLORS[i], alpha=0.72,
                edgecolor='white', linewidth=0.4, zorder=3)

        # Segment boundaries
        ax.axvline(0.33, color='#888888', linestyle='--', linewidth=0.9, alpha=0.6)
        ax.axvline(0.66, color='#888888', linestyle='--', linewidth=0.9, alpha=0.6)

        # Segment stats
        high = (vals > 0.66).mean() * 100
        mid  = ((vals > 0.33) & (vals <= 0.66)).mean() * 100
        low  = (vals <= 0.33).mean() * 100

        # Shade low-confidence region lightly
        ax.axvspan(0, 0.33, alpha=0.06, color='#C0504D', zorder=0)
        ax.axvspan(0.66, 1.0, alpha=0.06, color='#70AD47', zorder=0)

        stat_text = (f"High (>0.66):  {high:5.1f}%\n"
                     f"Mid  (0.33-0.66): {mid:5.1f}%\n"
                     f"Low  (<0.33):  {low:5.1f}%")
        ax.text(0.97, 0.95, stat_text, transform=ax.transAxes,
                fontsize=9, va='top', ha='right', family='monospace',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc',
                          boxstyle='round,pad=0.4'))

        mean_v = vals.mean()
        median_v = np.median(vals)
        ax.axvline(mean_v, color=KP_COLORS[i], linestyle='-', linewidth=1.5, alpha=0.8)
        ax.text(mean_v + 0.01, ax.get_ylim()[1] * 0.5,
                f'$\\mu$={mean_v:.3f}', fontsize=8, color=KP_COLORS[i],
                fontweight='bold', rotation=90, va='center')

        ax.set_title(KP_NAMES[i], fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('Confidence Score', fontsize=10)
        ax.set_ylabel('Frame Count', fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis='y', alpha=0.2, zorder=1)

    n_frames = len(all_vals[0])
    fig.suptitle(f"Confidence Distribution — CAC-KD Cataract 30-Fold ({n_frames:,} frames across 30 folds)",
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUT_DIR, 'confidence_histogram.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/confidence_histogram.png")


# ============================================================
# 2. Per-fold mean confidence grouped bar
# ============================================================
def plot_confidence_per_fold(all_confs):
    folds = sorted(all_confs.keys())
    mean_confs = np.zeros((len(folds), 4))
    for i, f in enumerate(folds):
        vals = np.array(list(all_confs[f].values()))
        mean_confs[i] = vals.mean(axis=0)

    fig, ax = plt.subplots(figsize=(17, 5.5))
    x = np.arange(len(folds))
    w = 0.2

    for i in range(4):
        bars = ax.bar(x + (i - 1.5) * w, mean_confs[:, i], w,
                       label=KP_NAMES[i], color=KP_COLORS[i],
                       edgecolor='white', linewidth=0.4, zorder=3)

    # Highlight anomalous folds
    for f_idx in ANOMALOUS_FOLDS:
        if f_idx in folds:
            pos = folds.index(f_idx)
            ax.annotate(f'f{f_idx}', (pos, max(mean_confs[pos]) + 0.02),
                        textcoords="offset points", xytext=(0, 5),
                        fontsize=7, color='#C0504D', ha='center', fontweight='bold')

    # Overall mean line
    overall_mean = mean_confs.mean()
    ax.axhline(overall_mean, color='#555555', linestyle='--', linewidth=1, alpha=0.6)
    ax.text(0.5, overall_mean + 0.06, f'Overall mean = {overall_mean:.3f}',
            fontsize=9, color='#555555', ha='left', va='bottom',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Mean Confidence per Fold per Keypoint (CAC-KD Cataract)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(f) for f in folds], fontsize=8)
    ax.set_ylim(0, max(mean_confs.max() * 1.15, 1.0))
    ax.legend(fontsize=9, loc='upper right', ncol=2, framealpha=0.9)
    ax.grid(axis='y', alpha=0.2, zorder=1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'confidence_per_fold.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/confidence_per_fold.png")


# ============================================================
# 3. Violin plot: confidence distribution per keypoint (aggregated)
# ============================================================
def plot_confidence_violin(all_confs):
    all_vals = [[] for _ in range(4)]
    for fold, confs in all_confs.items():
        for img_path, kp_confs in confs.items():
            for i in range(4):
                all_vals[i].append(kp_confs[i])

    data = [np.array(all_vals[i]) for i in range(4)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    parts = ax.violinplot(data, positions=range(4), showmeans=True,
                          showmedians=True, widths=0.7)

    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(KP_COLORS[i])
        body.set_alpha(0.6)
        body.set_edgecolor(KP_COLORS[i])
        body.set_linewidth(1.5)

    if 'cmeans' in parts:
        parts['cmeans'].set_color('#333333')
        parts['cmeans'].set_linewidth(1.5)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('#666666')
        parts['cmedians'].set_linestyle('--')
        parts['cmedians'].set_linewidth(1)
    for key in ('cmaxes', 'cmins', 'cbars'):
        if key in parts:
            parts[key].set_color('#888888')

    # Annotate means (offset well above mean/median lines)
    for i, vals in enumerate(data):
        mean_v = vals.mean()
        y_pos = min(mean_v + 0.13, 0.97)
        ax.text(i, y_pos, f'$\\mu$={mean_v:.3f}',
                ha='center', fontsize=9, fontweight='bold', color=KP_COLORS[i],
                zorder=8,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))

    ax.set_xticks(range(4))
    ax.set_xticklabels(KP_NAMES, fontsize=11)
    ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.02, 1.08)
    ax.set_title('Confidence Distribution per Keypoint (All 30 Folds Aggregated)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.axhline(0.33, color='#C0504D', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(0.66, color='#70AD47', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.grid(axis='y', alpha=0.2)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color='#333333', linewidth=1.5, label='Mean'),
        Line2D([0], [0], color='#666666', linewidth=1, linestyle='--', label='Median'),
    ]
    ax.legend(handles=legend, fontsize=9, loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'confidence_violin.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/confidence_violin.png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print("Loading confidence caches...")
    all_confs = load_all_confidences()
    print(f"Loaded {len(all_confs)} folds")

    plot_confidence_histogram(all_confs)
    plot_confidence_per_fold(all_confs)
    plot_confidence_violin(all_confs)
    print("\nAll confidence plots generated.")
