#!/usr/bin/env python3
"""
Generate Fig 5: Combined AUROC comparison across all experiments.

Reads real data from experiment/episodic_exp/ JSON files.
Output: fig5_auroc_curves.png
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/coding/experiment/episodic_exp'

METHOD_ORDER = ['anti_prototype', 'feature_mahalanobis', 'ood_head_osr23',
                'ood_head_extended_v2', 'ood_head_cluster_v2', 'ood_head_fusion_v2']
METHOD_SHORT = ['Anti-Prot', 'Mahal', 'OSR23', 'Ext V2', 'Clust V2', 'Fusion\nV2']
METHOD_COLORS = ['#E53935', '#FB8C00', '#FDD835', '#43A047', '#1E88E5', '#8E24AA']
FUSION_IDX = 5


def load_json(path):
    with open(path) as f:
        return json.load(f)


def generate_fig5(save_path):
    ablation = load_json(os.path.join(DATA_DIR, 'ablation', 'results.json'))
    cross = load_json(os.path.join(DATA_DIR, 'cross_domain', 'results.json'))
    wayshot = load_json(os.path.join(DATA_DIR, 'wayshot', 'results.json'))
    openness = load_json(os.path.join(DATA_DIR, 'openness', 'results.json'))

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)

    # ── (a) Within-domain AUROC: TAU22 vs TAU19 (full_model) ──
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(METHOD_ORDER))
    w = 0.35
    t22_vals = [ablation['TAU22']['full_model']['osr'][m]['auroc'] * 100 for m in METHOD_ORDER]
    t19_vals = [ablation['TAU19']['full_model']['osr'][m]['auroc'] * 100 for m in METHOD_ORDER]
    ax.bar(x - w/2, t22_vals, w, label='TAU22', color='#42A5F5', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, t19_vals, w, label='TAU19', color='#EF5350', alpha=0.85, edgecolor='white')
    for i in range(len(METHOD_ORDER)):
        ax.text(i - w/2, t22_vals[i] + 0.8, f'{t22_vals[i]:.1f}', ha='center', fontsize=6, fontweight='bold')
        ax.text(i + w/2, t19_vals[i] + 0.8, f'{t19_vals[i]:.1f}', ha='center', fontsize=6, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_SHORT, fontsize=7.5)
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(a) Within-Domain AUROC (6-way 5-shot, full_model)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(30, 105)

    # ── (b) Cross-domain AUROC ──
    ax = fig.add_subplot(gs[0, 1])
    t22_19 = [cross['TAU22->TAU19']['osr'][m]['auroc'] * 100 for m in METHOD_ORDER]
    t19_22 = [cross['TAU19->TAU22']['osr'][m]['auroc'] * 100 for m in METHOD_ORDER]
    ax.bar(x - w/2, t22_19, w, label='TAU22→TAU19', color='#66BB6A', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, t19_22, w, label='TAU19→TAU22', color='#FFA726', alpha=0.85, edgecolor='white')
    for i in range(len(METHOD_ORDER)):
        ax.text(i - w/2, t22_19[i] + 0.8, f'{t22_19[i]:.1f}', ha='center', fontsize=6, fontweight='bold')
        ax.text(i + w/2, t19_22[i] + 0.8, f'{t19_22[i]:.1f}', ha='center', fontsize=6, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_SHORT, fontsize=7.5)
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(b) Cross-Domain AUROC', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(30, 105)

    # ── (c) Openness robustness ──
    ax = fig.add_subplot(gs[0, 2])
    open2 = [openness['open2']['osr'][m]['auroc'] * 100 for m in METHOD_ORDER]
    open4 = [openness['open4']['osr'][m]['auroc'] * 100 for m in METHOD_ORDER]
    ax.bar(x - w/2, open2, w, label='Open=2', color='#7E57C2', alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, open4, w, label='Open=4', color='#FF7043', alpha=0.85, edgecolor='white')
    for i in range(len(METHOD_ORDER)):
        ax.text(i - w/2, open2[i] + 0.8, f'{open2[i]:.1f}', ha='center', fontsize=6, fontweight='bold')
        ax.text(i + w/2, open4[i] + 0.8, f'{open4[i]:.1f}', ha='center', fontsize=6, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_SHORT, fontsize=7.5)
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(c) Openness Robustness (TAU22)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(30, 105)

    # ── (d) Way-shot AUROC surface (Fusion V2 only) ──
    ax = fig.add_subplot(gs[1, 0])
    ways = [5, 6]
    shots = [1, 5, 10]
    settings_ordered = ['5w1s', '5w5s', '5w10s', '6w1s', '6w5s', '6w10s']
    fusion_aurocs = [wayshot[s]['osr']['ood_head_fusion_v2']['auroc'] * 100 for s in settings_ordered]
    x_pos = np.arange(len(settings_ordered))
    bars = ax.bar(x_pos, fusion_aurocs, color=['#BBDEFB', '#64B5F6', '#1E88E5',
                                                '#FFCDD2', '#E57373', '#E53935'],
                  edgecolor='white', alpha=0.85)
    for i, (s, v) in enumerate(zip(settings_ordered, fusion_aurocs)):
        ax.text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=7, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(settings_ordered, fontsize=8)
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(d) Way-Shot Scaling (Fusion V2, TAU19)', fontsize=10, fontweight='bold')
    ax.set_ylim(85, 102)

    # ── (e) Ablation AUROC range per method (TAU19, 6 configs) ──
    ax = fig.add_subplot(gs[1, 1])
    configs = ['full_model', 'wo_flow_transform', 'wo_ood_head', 'wo_reciprocal',
               'wo_threshold', 'only_distance_head']
    config_labels = ['Full', 'w/o Flow', 'w/o OOD', 'w/o Recip', 'w/o Thresh', 'Only Dist']
    for j, method in enumerate(METHOD_ORDER):
        vals = [ablation['TAU19'][c]['osr'][method]['auroc'] * 100 for c in configs]
        ax.plot(range(len(configs)), vals, 'o-', color=METHOD_COLORS[j],
                label=METHOD_SHORT[j].replace('\n', ' '), markersize=6, linewidth=1.5)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(config_labels, fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(e) Ablation Stability (TAU19, 6 configs)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, ncol=3)
    ax.set_ylim(45, 105)
    ax.grid(True, alpha=0.3)

    # ── (f) Method dominance summary ──
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Count wins across all conditions
    all_auroc_data = []
    for ds in ['TAU22', 'TAU19']:
        for cfg in configs:
            all_auroc_data.append([ablation[ds][cfg]['osr'][m]['auroc'] for m in METHOD_ORDER])
    # Cross-domain
    for direction in ['TAU22->TAU19', 'TAU19->TAU22']:
        all_auroc_data.append([cross[direction]['osr'][m]['auroc'] for m in METHOD_ORDER])
    # Openness
    for o in ['open2', 'open4']:
        all_auroc_data.append([openness[o]['osr'][m]['auroc'] for m in METHOD_ORDER])
    # Way-shot
    for s in settings_ordered:
        all_auroc_data.append([wayshot[s]['osr'][m]['auroc'] for m in METHOD_ORDER])

    all_auroc_data = np.array(all_auroc_data)
    n_conditions = all_auroc_data.shape[0]
    wins = [(all_auroc_data[:, i] == all_auroc_data.max(axis=1)).sum() for i in range(len(METHOD_ORDER))]
    win_pct = [w / n_conditions * 100 for w in wins]

    y_pos = np.arange(len(METHOD_SHORT))
    bars = ax.barh(y_pos, win_pct, color=METHOD_COLORS, alpha=0.85, edgecolor='black', linewidth=0.5)
    for i, (pct, w_count) in enumerate(zip(win_pct, wins)):
        ax.text(pct + 1, i, f'{int(w_count)}/{n_conditions} ({pct:.0f}%)',
                fontsize=7.5, va='center', fontweight='bold')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('\n', ' ') for m in METHOD_SHORT], fontsize=8)
    ax.set_xlabel('% Conditions Best (%)', fontsize=9)
    ax.set_title(f'(f) Method Dominance ({n_conditions} conditions)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 115)

    # Global title
    fig.suptitle('Fig 5. Comprehensive AUROC Comparison Across All Experiments',
                 fontsize=13, fontweight='bold', y=0.99)

    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {os.path.basename(save_path)}")

    # Print summary
    print(f"\n  Conditions evaluated: {n_conditions}")
    for i, method in enumerate(METHOD_SHORT):
        m = method.replace('\n', ' ')
        print(f"  {m:12s}: wins={int(wins[i]):2d}/{n_conditions}  ({win_pct[i]:.0f}%)")


if __name__ == '__main__':
    generate_fig5(os.path.join(OUT_DIR, 'fig5_auroc_curves.png'))
    print("\nInsertion point:")
    print("  IV.B / Appendix → [Insert Fig 5 about here.] fig5_auroc_curves.png")
