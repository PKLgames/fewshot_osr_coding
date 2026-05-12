#!/usr/bin/env python3
"""
Generate Multi-Dimensional Comparison Chart: 6 configs × 5 dimensions.

Reads: experiment/episodic_exp/ablation/results.json + complexity/results.json
Output: essay_metirial/radar_multidimensional.png

Layout: (a) grouped bar chart (5 metrics × 6 configs), (b) normalized heatmap.
"""

import os, json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/coding/experiment/episodic_exp'

CONFIG_ORDER = ['full_model', 'wo_flow_transform', 'wo_ood_head',
                'wo_reciprocal', 'wo_threshold', 'only_distance_head']
CONFIG_LABELS = ['Full\nModel', 'w/o\nFlow', 'w/o\nOOD Head', 'w/o\nReciprocal',
                 'w/o\nThreshold', 'Only\nDistance']
CONFIG_COLORS = ['#1565C0', '#00897B', '#EF6C00', '#7B1FA2', '#C62828', '#78909C']

COMPLEXITY_NAME_MAP = {
    'Full_Model': 'full_model', 'wo_Flow_Transform': 'wo_flow_transform',
    'wo_OOD_Head': 'wo_ood_head', 'wo_Reciprocal': 'wo_reciprocal',
    'wo_Threshold': 'wo_threshold', 'Baseline': 'only_distance_head',
}

DIMENSIONS = ['ACC (%)', 'AUROC (%)', 'OSR Score (%)', 'AIT (ms)', 'Params']


def load_json(path):
    with open(path) as f:
        return json.load(f)


def generate_radar(save_path):
    ablation = load_json(os.path.join(DATA_DIR, 'ablation', 'results.json'))
    complexity = load_json(os.path.join(DATA_DIR, 'complexity', 'results.json'))
    cmap_data = {COMPLEXITY_NAME_MAP[e['name']]: e for e in complexity}

    ds = 'TAU19'
    acc_raw   = [ablation[ds][c]['base_5shot']['mean_acc'] * 100 for c in CONFIG_ORDER]
    auroc_raw = [ablation[ds][c]['osr']['ood_head_fusion_v2']['auroc'] * 100 for c in CONFIG_ORDER]
    osr_raw   = [ablation[ds][c]['osr']['ood_head_fusion_v2']['osr_score'] * 100 for c in CONFIG_ORDER]
    ait_raw   = [cmap_data[c]['ait_mean_ms'] for c in CONFIG_ORDER]
    params_raw = [cmap_data[c]['trainable_params'] for c in CONFIG_ORDER]

    # Normalize each dimension [0, 1]; invert AIT and Params (lower better)
    def norm(v, inv=False):
        a = np.array(v, dtype=float)
        rng = a.max() - a.min()
        if rng == 0: return np.ones_like(a) * 0.5
        n = (a - a.min()) / rng
        return 1 - n if inv else n

    heatmap_data = np.array([
        norm(acc_raw),
        norm(auroc_raw),
        norm(osr_raw),
        norm(ait_raw, inv=True),
        norm(params_raw, inv=True),
    ])

    fig = plt.figure(figsize=(16, 7))

    # ── (a) Grouped bar: 5 dimension rows × 6 configs ──
    ax = fig.add_subplot(1, 2, 1)
    x = np.arange(len(CONFIG_ORDER))
    n_dims = len(DIMENSIONS)
    w = 0.8 / n_dims

    raw_data = [acc_raw, auroc_raw, osr_raw, ait_raw, params_raw]
    bar_colors = ['#42A5F5', '#FF7043', '#66BB6A', '#AB47BC', '#78909C']

    # Normalize display: each dimension scaled so max = 1.0
    for i, (vals, dim_name) in enumerate(zip(raw_data, DIMENSIONS)):
        vmax = max(vals)
        normed = [v / vmax for v in vals]
        offset = (i - n_dims / 2 + 0.5) * w
        bars = ax.bar(x + offset, normed, w, label=dim_name, color=bar_colors[i], alpha=0.85, edgecolor='white', linewidth=0.3)
        # Label best value
        best_idx = np.argmax(vals)
        ax.text(x[best_idx] + offset, normed[best_idx] + 0.03, f'{vals[best_idx]:.1f}' if i < 4 else f'{vals[best_idx]:.0f}',
                ha='center', fontsize=5.5, fontweight='bold', color=bar_colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_LABELS, fontsize=7.5, rotation=30, ha='right')
    ax.set_ylabel('Normalized Score (max = 1.0)', fontsize=9)
    ax.set_title('(a) 5-Dimensional Comparison (TAU19, Fusion V2)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6.5, ncol=3, loc='upper right')
    ax.set_ylim(0, 1.25)

    # ── (b) Heatmap ──
    ax = fig.add_subplot(1, 2, 2)
    cmap = LinearSegmentedColormap.from_list('heat', ['#FFFDE7', '#FFD54F', '#FF8F00', '#E65100', '#BF360C'])
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # Annotate cells
    raw_labels = [
        [f'{v:.1f}' for v in acc_raw],
        [f'{v:.1f}' for v in auroc_raw],
        [f'{v:.1f}' for v in osr_raw],
        [f'{v:.2f}' for v in ait_raw],
        [f'{v:.0f}' for v in params_raw],
    ]
    for i in range(n_dims):
        for j in range(len(CONFIG_ORDER)):
            color = 'white' if heatmap_data[i, j] < 0.5 else 'black'
            ax.text(j, i, raw_labels[i][j], ha='center', va='center', fontsize=7.5, fontweight='bold', color=color)

    # Mark global optimum (highest AUROC = row 1)
    best_auroc_col = np.argmax(auroc_raw)
    ax.add_patch(plt.Rectangle((best_auroc_col - 0.5, 1 - 0.5), 1, 1,
                                fill=False, edgecolor='#00E676', linewidth=3))
    ax.text(best_auroc_col, 0.6, '★ BEST', ha='center', fontsize=6.5, color='#00E676', fontweight='bold')

    ax.set_xticks(range(len(CONFIG_ORDER)))
    ax.set_xticklabels([l.replace('\n', ' ') for l in CONFIG_LABELS], fontsize=7.5, rotation=30, ha='right')
    ax.set_yticks(range(n_dims))
    ax.set_yticklabels(DIMENSIONS, fontsize=8)
    plt.colorbar(im, ax=ax, label='Normalized score (1.0 = best)')
    ax.set_title('(b) Normalized Heatmap (TAU19, Fusion V2)', fontsize=10, fontweight='bold')

    fig.suptitle('Multi-Dimensional Model Configuration Comparison (TAU19)',
                 fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {os.path.basename(save_path)}")

    # Print raw data
    print(f"\n  Raw metrics (TAU19, Fusion V2):")
    print(f"  {'Config':<22s} {'ACC':>7s} {'AUROC':>7s} {'OSR':>7s} {'AIT(ms)':>8s} {'Params':>8s}")
    for i, cfg in enumerate(CONFIG_ORDER):
        print(f"  {cfg:<22s} {acc_raw[i]:6.2f}% {auroc_raw[i]:6.1f}% {osr_raw[i]:6.1f}% {ait_raw[i]:7.2f}  {params_raw[i]:>7,}")


if __name__ == '__main__':
    generate_radar(os.path.join(OUT_DIR, 'radar_multidimensional.png'))
    print("\nInsertion point in IV.D (after TABLE 5 Key Findings):")
    print("  → radar_multidimensional.png")
