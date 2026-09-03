#!/usr/bin/env python3
"""
Quantitative results visualization for the paper.

Generates:
  1. 30-fold cataract LOSO: Baseline vs CAC-KD grouped bar + delta
  2. Ablation bar chart (RMIT LOSO: Baseline / Naive / CDT / CAC-KD)
  3. CDT confidence scale sweep (line plot)

Output: analysis/figs/quantitative/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROJECT_ROOT = "/root/autodl-tmp/root/autodl-tmp/longfei/surgical_tracking"
OUT_DIR = os.path.join(PROJECT_ROOT, "analysis/figs/quantitative")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})

# ============================================================
# Data (from PAPER_RESULTS_SUMMARY.md)
# ============================================================
BASELINE_30F = np.array([
    17.33, 10.08, 14.01, 10.88, 27.02, 13.71, 21.14, 6.53, 8.75, 9.87,
    11.04, 28.08, 8.62, 8.33, 12.73, 10.63, 6.73, 13.48, 11.29, 7.71,
    8.94, 6.52, 10.52, 6.87, 13.58, 7.20, 6.02, 10.77, 5.71, 12.06,
])
CACKD_30F = np.array([
    6.77, 5.77, 4.50, 2.63, 8.28, 6.59, 4.17, 6.13, 7.74, 14.98,
    6.66, 37.10, 6.22, 6.54, 25.52, 3.89, 4.10, 5.79, 5.35, 5.41,
    6.23, 5.41, 5.80, 7.23, 11.82, 7.74, 3.71, 6.17, 3.91, 5.44,
])

# RMIT LOSO ablation (Section 5.1)
ABLATION_NAMES = ['Baseline', 'Naive aux', 'CDT', 'CAC-KD']
ABLATION_ERR   = [10.72, 8.71, 8.98, 9.89]
ABLATION_PCK20 = [0.819, 0.949, 0.901, 0.879]

# CDT scale sweep (Section 5.3, fold0)
CDT_SCALES  = [1.0, 2.0, 3.0, 5.0]
CDT_S_ERR   = [10.03, 8.98, 8.99, 9.84]
CDT_S_PCK20 = [0.866, 0.953, 0.991, 0.916]

# Backbone / pretrain comparison (Section 4.2, cataract fold 0/10/20, V11 code)
BB_NAMES  = ['ResNet18\n+ImageNet', 'ConvNeXt-T\n+ImageNet', 'EffV2-S\n+ImageNet', 'ResNet50\n+PeskaVLP', 'CAFormer-S18\n+SurgeNetXL']
BB_ERR    = [15.84, 12.40, 12.44, 15.21, 10.74]
BB_PCK20  = [0.828, 0.865, 0.891, 0.855, 0.901]
BB_FOLDS  = {  # per-fold errors for display
    'ResNet18':  [25.42, 12.13, 9.96],
    'ConvNeXt-T':[15.58, 13.97, 7.66],
    'EffV2-S':   [17.33, 11.04, 8.94],
    'PeskaVLP':  [22.40, 12.99, 10.24],
}

# Colors
C_BASE    = '#5B9BD5'   # blue
C_CACKD   = '#ED7D31'   # orange
C_IMPROVE = '#70AD47'   # green
C_WORSEN  = '#C0504D'   # red
C_MEAN    = '#7F7F7F'   # gray


# ============================================================
# 1. 30-fold comparison
# ============================================================
def plot_30fold_comparison():
    n = len(BASELINE_30F)
    x = np.arange(n)
    w = 0.38

    delta = BASELINE_30F - CACKD_30F  # positive = improvement
    mean_b, mean_c = BASELINE_30F.mean(), CACKD_30F.mean()
    anomalous = np.where(delta < 0)[0]  # worsened folds

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 10),
        gridspec_kw={'height_ratios': [3, 1.3], 'hspace': 0.12}
    )

    # --- Top: grouped bars ---
    bars_b = ax1.bar(x - w/2, BASELINE_30F, w, label='Baseline',
                      color=C_BASE, edgecolor='white', linewidth=0.4, zorder=3)
    bars_c = ax1.bar(x + w/2, CACKD_30F, w, label='CAC-KD',
                      color=C_CACKD, edgecolor='white', linewidth=0.4, zorder=3)

    # Highlight anomalous folds
    for i in anomalous:
        bars_c[i].set_color(C_WORSEN)
        bars_c[i].set_hatch('//')

    # Mean lines
    ax1.axhline(mean_b, color=C_BASE, linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
    ax1.axhline(mean_c, color=C_CACKD, linestyle='--', linewidth=1.2, alpha=0.7, zorder=2)
    ax1.text(n - 0.5, mean_b + 0.5, f'Baseline mean = {mean_b:.2f}px',
             fontsize=8, color=C_BASE, ha='right', va='bottom')
    ax1.text(n - 0.5, mean_c + 0.5, f'CAC-KD mean = {mean_c:.2f}px',
             fontsize=8, color=C_CACKD, ha='right', va='bottom')

    ax1.set_ylabel('Mean Error (px)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([])
    ax1.set_xlim(-0.8, n - 0.2)
    ax1.set_ylim(0, 42)
    ax1.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax1.grid(axis='y', alpha=0.25, zorder=1)
    ax1.set_title('Cataract 30-Fold LOSO: Baseline vs CAC-KD',
                  fontsize=14, fontweight='bold', pad=12)

    # --- Bottom: delta ---
    colors = [C_IMPROVE if d > 0 else C_WORSEN for d in delta]
    ax2.bar(x, delta, 0.7, color=colors, edgecolor='white', linewidth=0.4, zorder=3)
    ax2.axhline(0, color='#333333', linewidth=0.8, zorder=2)

    # Annotate anomalous folds
    for i in anomalous:
        ax2.annotate(f'f{i}', (i, delta[i]), textcoords="offset points",
                     xytext=(0, -14), fontsize=7, color=C_WORSEN, ha='center',
                     fontweight='bold')

    ax2.set_ylabel('$\\Delta$ Error (px)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(i) for i in range(n)], fontsize=8)
    ax2.set_xlim(-0.8, n - 0.2)

    legend_el = [
        Patch(facecolor=C_IMPROVE, label='Improved (25 folds)'),
        Patch(facecolor=C_WORSEN, hatch='//', label='Worsened (5 folds)'),
    ]
    ax2.legend(handles=legend_el, fontsize=9, loc='upper right', framealpha=0.9)
    ax2.grid(axis='y', alpha=0.25, zorder=1)

    plt.savefig(os.path.join(OUT_DIR, '30fold_comparison.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/30fold_comparison.png")


# ============================================================
# 2. Ablation bar chart
# ============================================================
def plot_ablation():
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(ABLATION_NAMES))
    w = 0.55

    colors = [C_BASE, '#A5A5A5', C_CACKD, '#FFC000']
    bars = ax1.bar(x, ABLATION_ERR, w, color=colors, edgecolor='white',
                   linewidth=0.8, zorder=3)

    # Value labels
    for bar, val in zip(bars, ABLATION_ERR):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.15, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Mean Error (px)', fontsize=12, fontweight='bold', color='#333333')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ABLATION_NAMES, fontsize=11)
    ax1.set_ylim(0, 13)
    ax1.grid(axis='y', alpha=0.25, zorder=1)

    # PCK@20 on twin axis
    ax2 = ax1.twinx()
    ax2.plot(x, ABLATION_PCK20, 'D-', color='#7030A0', linewidth=2,
             markersize=8, markerfacecolor='white', markeredgewidth=2, zorder=5)
    for xi, val in zip(x, ABLATION_PCK20):
        ax2.text(xi, val + 0.01, f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9, color='#7030A0')
    ax2.set_ylabel('PCK@20', fontsize=12, fontweight='bold', color='#7030A0')
    ax2.set_ylim(0.75, 1.02)
    ax2.tick_params(axis='y', colors='#7030A0')

    # Baseline reference line
    ax1.axhline(ABLATION_ERR[0], color=C_BASE, linestyle=':', linewidth=1, alpha=0.5)

    # Figure-level title avoids clipping
    fig.text(0.5, 0.96, 'CDT Ablation Study (RMIT 3-Fold LOSO)',
             ha='center', fontsize=14, fontweight='bold', color='#1a1a1a')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(OUT_DIR, 'ablation_rmit.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/ablation_rmit.png")


# ============================================================
# 3. CDT scale sweep
# ============================================================
def plot_cdt_scale():
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(CDT_SCALES))
    labels = [f'{s}$\\times$' for s in CDT_SCALES]

    # Error line
    ax1.plot(x, CDT_S_ERR, 'o-', color=C_CACKD, linewidth=2.5,
             markersize=10, markerfacecolor='white', markeredgewidth=2.5, zorder=5)
    for xi, val in zip(x, CDT_S_ERR):
        ax1.text(xi, val + 0.15, f'{val:.2f}', ha='center', va='bottom',
                 fontsize=10, fontweight='bold', color=C_CACKD)

    ax1.set_xlabel('Confidence Scale Factor ($\\times$ median error)',
                   fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean Error (px)', fontsize=12, fontweight='bold', color=C_CACKD)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(8, 11)
    ax1.tick_params(axis='y', colors=C_CACKD)
    ax1.grid(axis='y', alpha=0.25, zorder=1)

    # Highlight optimal (3x)
    best_idx = CDT_S_ERR.index(min(CDT_S_ERR))
    ax1.axvspan(best_idx - 0.35, best_idx + 0.35, alpha=0.08, color=C_CACKD, zorder=0)
    ax1.annotate('Default', xy=(best_idx, CDT_S_ERR[best_idx]),
                 xytext=(best_idx + 0.5, CDT_S_ERR[best_idx] - 0.5),
                 fontsize=9, color=C_CACKD, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C_CACKD, lw=1.2))

    # PCK@20 on twin axis
    ax2 = ax1.twinx()
    ax2.plot(x, CDT_S_PCK20, 's--', color='#7030A0', linewidth=1.8,
             markersize=8, markerfacecolor='white', markeredgewidth=2, zorder=4, alpha=0.8)
    ax2.set_ylabel('PCK@20', fontsize=12, fontweight='bold', color='#7030A0')
    ax2.set_ylim(0.84, 1.01)
    ax2.tick_params(axis='y', colors='#7030A0')

    # Legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color=C_CACKD, marker='o', markersize=8,
               markerfacecolor='white', markeredgewidth=2, label='Error (px)'),
        Line2D([0], [0], color='#7030A0', marker='s', markersize=7,
               markerfacecolor='white', markeredgewidth=2, linestyle='--',
               label='PCK@20', alpha=0.8),
    ]
    ax1.legend(handles=legend, fontsize=9, loc='center right')

    # Figure-level title avoids clipping
    fig.text(0.5, 0.96, 'CDT Confidence Scale Ablation (RMIT Fold 0)',
             ha='center', fontsize=14, fontweight='bold', color='#1a1a1a')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(OUT_DIR, 'cdt_scale_sweep.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/cdt_scale_sweep.png")


# ============================================================
# 4. Backbone / pretrain comparison (Section 4.2)
# ============================================================
def plot_backbone_comparison():
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(BB_NAMES))
    w = 0.55

    colors = ['#A5A5A5', '#5B9BD5', '#ED7D31', '#70AD47', '#FFC000']
    bars = ax1.bar(x, BB_ERR, w, color=colors, edgecolor='white',
                   linewidth=0.8, zorder=3)

    # Highlight best (SurgeNetXL, index 4)
    best_idx = BB_ERR.index(min(BB_ERR))
    bars[best_idx].set_edgecolor('#333333')
    bars[best_idx].set_linewidth(2)

    # Value labels
    for bar, val in zip(bars, BB_ERR):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.25, f'{val:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Mean Error (px)', fontsize=12, fontweight='bold', color='#333333')
    ax1.set_xticks(x)
    ax1.set_xticklabels(BB_NAMES, fontsize=10)
    ax1.set_ylim(0, 19)
    ax1.grid(axis='y', alpha=0.25, zorder=1)

    # PCK@20 on twin axis
    ax2 = ax1.twinx()
    ax2.plot(x, BB_PCK20, 'D-', color='#7030A0', linewidth=2,
             markersize=8, markerfacecolor='white', markeredgewidth=2, zorder=5)
    for xi, val in zip(x, BB_PCK20):
        ax2.text(xi, val + 0.008, f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9, color='#7030A0')
    ax2.set_ylabel('PCK@20', fontsize=12, fontweight='bold', color='#7030A0')
    ax2.set_ylim(0.78, 0.95)
    ax2.tick_params(axis='y', colors='#7030A0')

    # Figure-level title
    fig.text(0.5, 0.96, 'Backbone & Pretrain Comparison (Cataract, Fold 0/10/20)',
             ha='center', fontsize=14, fontweight='bold', color='#1a1a1a')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(os.path.join(OUT_DIR, 'backbone_comparison.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {OUT_DIR}/backbone_comparison.png")


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    plot_30fold_comparison()
    plot_ablation()
    plot_cdt_scale()
    plot_backbone_comparison()
    print("\nAll quantitative plots generated.")
