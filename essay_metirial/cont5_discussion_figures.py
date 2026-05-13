#!/usr/bin/env python3
"""
cont5_discussion_figures.py — Generate Section V discussion figures.

Outputs:
  - fig5_1_fusion_complementarity.png    ( Fusion V2 complementarity analysis )
  - fig5_2_auroc_vs_osr_score.png        ( AUROC vs OSR Score metric comparison )
  - fig5_3_cross_domain_asymmetry.png    ( Cross-domain asymmetry visualization )
  - table5_1_learned_alpha.csv           ( Learned fusion weight α values )

Usage:
  python cont5_discussion_figures.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '/coding/experiment/episodic_exp'

METHOD_ORDER = ['anti_prototype', 'feature_mahalanobis', 'ood_head_osr23',
                'ood_head_extended_v2', 'ood_head_cluster_v2', 'ood_head_fusion_v2']
METHOD_SHORT = ['Anti-Prot', 'Mahal', 'OSR23', 'Ext V2', 'Clust V2', 'Fusion V2']
CONFIG_NAMES = ['full_model', 'wo_flow_transform', 'wo_ood_head',
                'wo_reciprocal', 'wo_threshold', 'only_distance_head']
CONFIG_LABELS = {
    'full_model': 'Full Model',
    'wo_flow_transform': 'w/o Flow',
    'wo_ood_head': 'w/o OOD Head',
    'wo_reciprocal': 'w/o Reciprocal',
    'wo_threshold': 'w/o Threshold',
    'only_distance_head': 'Only Distance',
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def generate_fusion_complementarity(save_path):
    """Visualize how fusion combines ext_v2 and clust_v2 signals."""
    ablation = load_json(os.path.join(DATA_DIR, 'ablation', 'results.json'))
    cross = load_json(os.path.join(DATA_DIR, 'cross_domain', 'results.json'))
    openness = load_json(os.path.join(DATA_DIR, 'openness', 'results.json'))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- (a) TAU22 within-domain: ext_v2, clust_v2, fusion_v2 AUROC ---
    ax = axes[0]
    configs = CONFIG_NAMES
    ext_vals = [ablation['TAU22'][c]['osr']['ood_head_extended_v2']['auroc']*100 for c in configs]
    clust_vals = [ablation['TAU22'][c]['osr']['ood_head_cluster_v2']['auroc']*100 for c in configs]
    fusion_vals = [ablation['TAU22'][c]['osr']['ood_head_fusion_v2']['auroc']*100 for c in configs]

    x = np.arange(len(configs))
    w = 0.25
    ax.bar(x - w, ext_vals, w, label='Ext V2 (discriminative)', color='#42A5F5', alpha=0.85)
    ax.bar(x, clust_vals, w, label='Clust V2 (density)', color='#66BB6A', alpha=0.85)
    ax.bar(x + w, fusion_vals, w, label='Fusion V2', color='#FF7043', alpha=0.85)

    # Mark max gap
    for i in range(len(configs)):
        gap = fusion_vals[i] - max(ext_vals[i], clust_vals[i])
        if gap > 0:
            ax.text(i + w, fusion_vals[i] + 0.3, f'+{gap:.1f}', ha='center', fontsize=7,
                   fontweight='bold', color='#D84315')

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(a) TAU22: Ext V2 + Clust V2 → Fusion', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(40, 100)

    # --- (b) TAU19 within-domain ---
    ax = axes[1]
    ext_vals = [ablation['TAU19'][c]['osr']['ood_head_extended_v2']['auroc']*100 for c in configs]
    clust_vals = [ablation['TAU19'][c]['osr']['ood_head_cluster_v2']['auroc']*100 for c in configs]
    fusion_vals = [ablation['TAU19'][c]['osr']['ood_head_fusion_v2']['auroc']*100 for c in configs]

    ax.bar(x - w, ext_vals, w, label='Ext V2', color='#42A5F5', alpha=0.85)
    ax.bar(x, clust_vals, w, label='Clust V2', color='#66BB6A', alpha=0.85)
    ax.bar(x + w, fusion_vals, w, label='Fusion V2', color='#FF7043', alpha=0.85)

    for i in range(len(configs)):
        gap = fusion_vals[i] - max(ext_vals[i], clust_vals[i])
        if gap > 0:
            ax.text(i + w, fusion_vals[i] + 0.3, f'+{gap:.1f}', ha='center', fontsize=7,
                   fontweight='bold', color='#D84315')

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in configs], fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(b) TAU19: Ext V2 + Clust V2 → Fusion', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)
    ax.set_ylim(40, 100)

    # --- (c) Complementarity scatter: ext_v2 vs clust_v2, color = fusion gain ---
    ax = axes[2]
    all_ext, all_clust, all_gains, all_labels = [], [], [], []
    for ds in ['TAU22', 'TAU19']:
        for c in configs:
            ev = ablation[ds][c]['osr']['ood_head_extended_v2']['auroc'] * 100
            cv = ablation[ds][c]['osr']['ood_head_cluster_v2']['auroc'] * 100
            fv = ablation[ds][c]['osr']['ood_head_fusion_v2']['auroc'] * 100
            all_ext.append(ev)
            all_clust.append(cv)
            all_gains.append(fv - max(ev, cv))
            all_labels.append(f'{ds}_{c[:6]}')

    sc = ax.scatter(all_ext, all_clust, c=all_gains, cmap='RdYlGn', s=100, edgecolors='black',
                   linewidth=0.5, vmin=0, vmax=max(all_gains) + 0.5)
    ax.plot([40, 100], [40, 100], 'k--', alpha=0.3, label='y = x')
    plt.colorbar(sc, ax=ax, label='Fusion gain (Δ AUROC)')

    # Annotate best point
    best_idx = np.argmax(all_gains)
    ax.annotate(all_labels[best_idx], (all_ext[best_idx], all_clust[best_idx]),
               fontsize=7, fontweight='bold',
               xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Ext V2 AUROC (%)', fontsize=9)
    ax.set_ylabel('Clust V2 AUROC (%)', fontsize=9)
    ax.set_title('(c) Complementarity: Ext V2 vs Clust V2', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {os.path.basename(save_path)}")


def generate_auroc_vs_osr_score(save_path):
    """Compare AUROC vs OSR Score discriminative resolution."""
    ablation = load_json(os.path.join(DATA_DIR, 'ablation', 'results.json'))
    wayshot = load_json(os.path.join(DATA_DIR, 'wayshot', 'results.json'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- (a) Ablation spread: AUROC vs OSR Score for TAU19 ---
    ax = axes[0]
    all_auroc, all_osr, colors_list, sizes_list = [], [], [], []
    method_colors = ['#E53935', '#FB8C00', '#FDD835', '#43A047', '#1E88E5', '#8E24AA']

    for m_idx, method in enumerate(METHOD_ORDER):
        for c_idx, cfg in enumerate(CONFIG_NAMES):
            osr_data = ablation['TAU19'][cfg]['osr'][method]
            all_auroc.append(osr_data['auroc'] * 100)
            all_osr.append(osr_data['osr_score'] * 100)
            colors_list.append(method_colors[m_idx])
            sizes_list.append(80 if cfg == 'full_model' else 40)

    ax.scatter(all_auroc, all_osr, c=colors_list, s=sizes_list, alpha=0.7, edgecolors='gray', linewidth=0.3)

    # Correlation
    r, p = stats.pearsonr(all_auroc, all_osr)
    ax.text(0.95, 0.05, f'Pearson r = {r:.3f}\np = {p:.2e}',
            transform=ax.transAxes, fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('AUROC (%)', fontsize=9)
    ax.set_ylabel('OSR Score (%)', fontsize=9)
    ax.set_title('(a) TAU19: AUROC vs OSR Score (36 conditions)', fontsize=10, fontweight='bold')

    # Legend for methods
    legend_elements = [mpatches.Patch(facecolor=method_colors[i], label=METHOD_SHORT[i])
                      for i in range(len(METHOD_ORDER))]
    ax.legend(handles=legend_elements, fontsize=6, loc='lower right')

    # Print spread
    print(f"  AUROC spread: {np.ptp(all_auroc):.3f} | OSR Score spread: {np.ptp(all_osr):.3f}")

    # --- (b) Way-shot: AUROC vs OSR Score for Fusion V2 ---
    ax = axes[1]
    settings = list(wayshot.keys())
    aurocs = [wayshot[s]['osr']['ood_head_fusion_v2']['auroc']*100 for s in settings]
    osrs = [wayshot[s]['osr']['ood_head_fusion_v2']['osr_score']*100 for s in settings]

    ax.scatter(aurocs, osrs, c='#FF7043', s=100, edgecolors='black', linewidth=0.5)
    for i, s in enumerate(settings):
        ax.annotate(s, (aurocs[i], osrs[i]), fontsize=7, xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('AUROC (%)', fontsize=9)
    ax.set_ylabel('OSR Score (%)', fontsize=9)
    ax.set_title('(b) Fusion V2 across Way-Shot Settings', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {os.path.basename(save_path)}")


def generate_cross_domain_asymmetry(save_path):
    """Visualize cross-domain asymmetry."""
    cross = load_json(os.path.join(DATA_DIR, 'cross_domain', 'results.json'))
    ablation = load_json(os.path.join(DATA_DIR, 'ablation', 'results.json'))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- (a) Within-domain vs cross-domain AUROC per method ---
    ax = axes[0]
    x_labels = []
    within_vals, cross_vals = [], []

    for method in METHOD_ORDER:
        m_name = METHOD_SHORT[METHOD_ORDER.index(method)]

        # Within: TAU22 full_model
        w22 = ablation['TAU22']['full_model']['osr'][method]['auroc'] * 100
        # Cross: TAU19->TAU22
        c19_22 = cross['TAU19->TAU22']['osr'][method]['auroc'] * 100
        within_vals.append(w22)
        cross_vals.append(c19_22)
        x_labels.append(f'{m_name}\n(TAU22)')

        # Within: TAU19 full_model
        w19 = ablation['TAU19']['full_model']['osr'][method]['auroc'] * 100
        # Cross: TAU22->TAU19
        c22_19 = cross['TAU22->TAU19']['osr'][method]['auroc'] * 100
        within_vals.append(w19)
        cross_vals.append(c22_19)
        x_labels.append(f'{m_name}\n(TAU19)')

    x = np.arange(len(x_labels))
    w = 0.35
    ax.bar(x - w/2, within_vals, w, label='Within-Domain', color='#42A5F5', alpha=0.85)
    ax.bar(x + w/2, cross_vals, w, label='Cross-Domain', color='#EF5350', alpha=0.85)

    # Highlight TAU22→TAU19 direction where OSR improves
    for i in range(6, 12):  # TAU19 bars
        if cross_vals[i] > within_vals[i]:
            ax.annotate('↑', (x[i] + w/2, cross_vals[i]), ha='center', fontsize=12,
                       fontweight='bold', color='green')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6.5, rotation=45, ha='right')
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(a) Within-Domain vs Cross-Domain AUROC', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    # --- (b) Cross-domain AUROC comparison per method ---
    ax = axes[1]
    methods_short = [METHOD_SHORT[i] for i in range(len(METHOD_ORDER))]
    t22_19 = [cross['TAU22->TAU19']['osr'][m]['auroc']*100 for m in METHOD_ORDER]
    t19_22 = [cross['TAU19->TAU22']['osr'][m]['auroc']*100 for m in METHOD_ORDER]

    x = np.arange(len(methods_short))
    w = 0.35
    ax.bar(x - w/2, t22_19, w, label='TAU22 → TAU19', color='#66BB6A', alpha=0.85)
    ax.bar(x + w/2, t19_22, w, label='TAU19 → TAU22', color='#FFA726', alpha=0.85)

    # Show gap
    for i in range(len(methods_short)):
        gap = t22_19[i] - t19_22[i]
        ax.annotate(f'Δ={gap:.1f}', (x[i], max(t22_19[i], t19_22[i]) + 1.5),
                   ha='center', fontsize=6.5, fontweight='bold', color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels(methods_short, fontsize=8, rotation=30, ha='right')
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(b) Cross-Domain Asymmetry: TAU22↔TAU19', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {os.path.basename(save_path)}")


if __name__ == '__main__':
    print("Generating Chapter 5 (Section V) discussion figures...\n")
    generate_fusion_complementarity(os.path.join(OUT_DIR, 'fig5_1_fusion_complementarity.png'))
    generate_auroc_vs_osr_score(os.path.join(OUT_DIR, 'fig5_2_auroc_vs_osr_score.png'))
    generate_cross_domain_asymmetry(os.path.join(OUT_DIR, 'fig5_3_cross_domain_asymmetry.png'))

    print("\nInsertion points in Section V:")
    print("  V.A (Why Fusion Works)   → [Insert Fig 5.1 about here.] fig5_1_fusion_complementarity.png")
    print("  V.D (AUROC vs OSR Score) → [Insert Fig 5.2 about here.] fig5_2_auroc_vs_osr_score.png")
    print("  V.C (Cross-Domain Asym.) → [Insert Fig 5.3 about here.] fig5_3_cross_domain_asymmetry.png")
