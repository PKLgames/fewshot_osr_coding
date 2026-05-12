#!/usr/bin/env python3
"""
Generate Way-Shot Surface Plot: way × shot × AUROC, 3D surface.

Reads: experiment/episodic_exp/wayshot/results.json
Output: essay_metirial/wayshot_surface.png

Shows Fusion V2 AUROC dominating at all way-shot settings, with OOD head
family (osr23/clust_v2/ext_v2) surfaces overlaid for comparison.
"""

import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/coding/experiment/episodic_exp'

METHOD_ORDER = ['anti_prototype', 'feature_mahalanobis', 'ood_head_osr23',
                'ood_head_extended_v2', 'ood_head_cluster_v2', 'ood_head_fusion_v2']
METHOD_SHORT = ['Anti-Prot', 'Mahal', 'OSR23', 'Ext V2', 'Clust V2', 'Fusion V2']
METHOD_COLORS = ['#E53935', '#FB8C00', '#FDD835', '#43A047', '#1E88E5', '#8E24AA']


def load_json(path):
    with open(path) as f:
        return json.load(f)


def generate_surface(save_path):
    wayshot = load_json(os.path.join(DATA_DIR, 'wayshot', 'results.json'))

    settings_ordered = ['5w1s', '5w5s', '5w10s', '6w1s', '6w5s', '6w10s']
    ways = np.array([5, 5, 5, 6, 6, 6])
    shots = np.array([1, 5, 10, 1, 5, 10])

    fig = plt.figure(figsize=(16, 7))

    # ── (a) 3D surface: Fusion V2 AUROC on TAU19 ──
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # Create mesh for smooth surface
    way_grid = np.array([5, 6])
    shot_grid = np.array([1, 5, 10])
    W, S = np.meshgrid(way_grid, shot_grid)

    for ds, ds_label, ax_idx in [('TAU19', 'TAU19', 1), ('TAU22', 'TAU22', 2)]:
        pass  # We'll plot both datasets on separate subplots

    # TAU19 surface
    ds = 'TAU19'
    fusion_vals_t19 = np.array([wayshot[s]['osr']['ood_head_fusion_v2']['auroc'] * 100 for s in settings_ordered])
    Z_t19 = fusion_vals_t19.reshape(2, 3).T  # shots × ways

    surf1 = ax.plot_surface(W, S, Z_t19, cmap=cm.viridis, alpha=0.85, edgecolor='k', linewidth=0.3)

    # Scatter actual data points
    for i, s in enumerate(settings_ordered):
        ax.scatter([ways[i]], [shots[i]], [fusion_vals_t19[i]],
                   c='black', s=40, marker='o', zorder=10)
        ax.text(ways[i], shots[i], fusion_vals_t19[i] + 0.5,
                f'{fusion_vals_t19[i]:.1f}', fontsize=7, ha='center', fontweight='bold')

    ax.set_xlabel('N-way', fontsize=10, labelpad=10)
    ax.set_ylabel('K-shot', fontsize=10, labelpad=10)
    ax.set_zlabel('AUROC (%)', fontsize=10, labelpad=8)
    ax.set_title('(a) Way-Shot Surface — Fusion V2, TAU19', fontsize=10, fontweight='bold')
    ax.view_init(elev=25, azim=-45)
    ax.set_xticks([5, 6])
    ax.set_yticks([1, 5, 10])
    fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, label='AUROC (%)')

    # ── (b) 2D comparison: all 6 methods across way-shot (TAU19) ──
    ax = fig.add_subplot(1, 2, 2)
    x = np.arange(len(settings_ordered))

    for j, method in enumerate(METHOD_ORDER):
        vals = [wayshot[s]['osr'][method]['auroc'] * 100 for s in settings_ordered]
        ls = '-' if method == 'ood_head_fusion_v2' else '--'
        lw = 2.5 if method == 'ood_head_fusion_v2' else 1.2
        alpha = 1.0 if method == 'ood_head_fusion_v2' else 0.5
        ax.plot(x, vals, 'o-', color=METHOD_COLORS[j], linestyle=ls,
                linewidth=lw, markersize=5 if ls == '--' else 7,
                alpha=alpha, label=METHOD_SHORT[j])

    ax.set_xticks(x)
    ax.set_xticklabels(settings_ordered, fontsize=8.5)
    ax.set_ylabel('AUROC (%)', fontsize=10)
    ax.set_title('(b) All Methods — Way-Shot AUROC (TAU19)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6.5, ncol=3)
    ax.grid(True, alpha=0.3)

    # Diminishing returns annotation
    ax.annotate('Diminishing returns\nplateau beyond K=5',
                xy=(2, fusion_vals_t19[2]), xytext=(3.5, fusion_vals_t19[2] + 1.5),
                fontsize=7.5, fontweight='bold', color='#555',
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.2),
                bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.7))

    # Performance floor annotation (anti_prototype)
    ap_vals = [wayshot[s]['osr']['anti_prototype']['auroc'] * 100 for s in settings_ordered]
    ax.annotate('Anti-Prototype floor',
                xy=(0, ap_vals[0]), xytext=(-0.3, ap_vals[0] - 8),
                fontsize=7, color='#C62828',
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.0))

    fig.suptitle('Way-Shot Scaling: Fusion V2 Dominance & Diminishing Returns',
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {os.path.basename(save_path)}")

    # Print summary
    print("\n  Fusion V2 AUROC across way-shot settings:")
    for s in settings_ordered:
        t19 = wayshot[s]['osr']['ood_head_fusion_v2']['auroc'] * 100
        t22_ws = wayshot[s]['osr']['ood_head_fusion_v2']['auroc'] * 100  # Same data source
        print(f"  {s}: {t19:.1f}%")

    print("\n  AUROC gain from K=1 to K=10 (TAU19):")
    for w in [5, 6]:
        v1 = wayshot[f'{w}w1s']['osr']['ood_head_fusion_v2']['auroc'] * 100
        v10 = wayshot[f'{w}w10s']['osr']['ood_head_fusion_v2']['auroc'] * 100
        print(f"  {w}-way: {v1:.1f}% → {v10:.1f}%  (Δ = {v10-v1:+.1f} pp)")


if __name__ == '__main__':
    generate_surface(os.path.join(OUT_DIR, 'wayshot_surface.png'))
    print("\nInsertion point in IV.G1 (after Way-Shot Key Findings):")
    print("  → wayshot_surface.png")
