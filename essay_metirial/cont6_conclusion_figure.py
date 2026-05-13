#!/usr/bin/env python3
"""
cont6_conclusion_figure.py — Generate Section VI conclusion summary figure.

Outputs:
  - fig6_1_results_summary.png  ( Radar/summary chart of all experiments )

Usage:
  python cont6_conclusion_figure.py
"""

import os
import sys
import json
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


def load_json(path):
    with open(path) as f:
        return json.load(f)


def generate_summary(save_path):
    """Generate a comprehensive results summary figure for the conclusion."""
    ablation = load_json(os.path.join(DATA_DIR, 'ablation', 'results.json'))
    cross = load_json(os.path.join(DATA_DIR, 'cross_domain', 'results.json'))
    wayshot = load_json(os.path.join(DATA_DIR, 'wayshot', 'results.json'))
    openness = load_json(os.path.join(DATA_DIR, 'openness', 'results.json'))
    complexity = load_json(os.path.join(DATA_DIR, 'complexity', 'results.json'))

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    method_colors = ['#E53935', '#FB8C00', '#FDD835', '#43A047', '#1E88E5', '#8E24AA']

    # --- (a) AUROC: All methods across all experiments ---
    ax = fig.add_subplot(gs[0, :2])
    experiments = ['TAU22\nFull', 'TAU19\nFull', 'TAU22\nCross→', 'TAU19\nCross→',
                   'Open2', 'Open4', '5w1s', '6w10s']
    data_matrix = np.zeros((len(METHOD_ORDER), len(experiments)))

    for j, method in enumerate(METHOD_ORDER):
        data_matrix[j, 0] = ablation['TAU22']['full_model']['osr'][method]['auroc'] * 100
        data_matrix[j, 1] = ablation['TAU19']['full_model']['osr'][method]['auroc'] * 100
        data_matrix[j, 2] = cross['TAU19->TAU22']['osr'][method]['auroc'] * 100
        data_matrix[j, 3] = cross['TAU22->TAU19']['osr'][method]['auroc'] * 100
        data_matrix[j, 4] = openness['open2']['osr'][method]['auroc'] * 100
        data_matrix[j, 5] = openness['open4']['osr'][method]['auroc'] * 100
        data_matrix[j, 6] = wayshot['5w1s']['osr'][method]['auroc'] * 100
        data_matrix[j, 7] = wayshot['6w10s']['osr'][method]['auroc'] * 100

    x = np.arange(len(experiments))
    n_methods = len(METHOD_ORDER)
    w = 0.8 / n_methods

    for j in range(n_methods):
        offset = (j - n_methods/2 + 0.5) * w
        ax.bar(x + offset, data_matrix[j], w, label=METHOD_SHORT[j],
               color=method_colors[j], alpha=0.85, edgecolor='white', linewidth=0.3)

    # Mark Fusion V2 as best
    for i in range(len(experiments)):
        best_val = data_matrix[:, i].max()
        ax.text(i, best_val + 1.5, f'{best_val:.1f}', ha='center', fontsize=6.5,
               fontweight='bold', color='#333')

    ax.set_xticks(x)
    ax.set_xticklabels(experiments, fontsize=7.5)
    ax.set_ylabel('AUROC (%)', fontsize=9)
    ax.set_title('(a) AUROC Across All Experiments (6 methods × 8 conditions)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=6.5, ncol=6, loc='upper right')
    ax.set_ylim(30, 105)

    # --- (b) Key numbers box ---
    ax = fig.add_subplot(gs[0, 2])
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    key_numbers = [
        ('Fusion V2 Best AUROC', 'TAU19 wo_ood_head', '0.985'),
        ('Fusion V2 Best OSR Score', 'TAU19 wo_ood_head', '93.31%'),
        ('Friedman Test', 'χ²=49.29', 'p=1.94×10⁻⁹'),
        ('Cross-Domain Gap', 'TAU22→TAU19 vs TAU19→TAU22', 'Δ0.099 AUROC'),
        ('Speedup (w/o Flow)', '4.16ms → 0.65ms', '6.5×'),
        ('Param Reduction', '40.1K → 14.8K', '−63%'),
    ]

    for i, (title, context, value) in enumerate(key_numbers):
        y = 9 - i * 1.4
        # Background box
        rect = mpatches.FancyBboxPatch((0.3, y-0.5), 9.4, 1.2,
                                        boxstyle="round,pad=0.08",
                                        facecolor='#F5F5F5' if i % 2 == 0 else '#E8EAF6',
                                        edgecolor='#90A4AE', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(0.6, y+0.35, title, fontsize=8, fontweight='bold', color='#1A237E')
        ax.text(0.6, y-0.15, context, fontsize=7, color='#555')
        ax.text(9.0, y+0.1, value, fontsize=9, fontweight='bold', color='#C62828', ha='right')

    ax.set_title('(b) Key Results Summary', fontsize=10, fontweight='bold')

    # --- (c) Efficiency trade-off ---
    ax = fig.add_subplot(gs[1, :2])
    configs = ['Full\nModel', 'w/o\nFlow', 'w/o\nOOD', 'w/o\nRecip', 'w/o\nThresh', 'Only\nDist']
    params = [c['trainable_params'] for c in complexity]
    aits = [c['ait_mean_ms'] for c in complexity]
    auroc_vals = [ablation['TAU19'][cfg]['osr']['ood_head_fusion_v2']['auroc']*100 for cfg in
                  ['full_model', 'wo_flow_transform', 'wo_ood_head', 'wo_reciprocal',
                   'wo_threshold', 'only_distance_head']]

    ax2 = ax.twinx()
    bars = ax.bar(np.arange(len(configs)) - 0.15, aits, 0.25, color='#FF7043', alpha=0.7, label='AIT (ms)')
    bars2 = ax.bar(np.arange(len(configs)) + 0.15, auroc_vals, 0.25, color='#42A5F5', alpha=0.7, label='AUROC (%)')

    # Annotate best efficiency
    best_eff_idx = 1  # wo_flow
    ax.annotate(f'Best efficiency:\n{auroc_vals[best_eff_idx]:.1f}% AUROC\n{aits[best_eff_idx]:.2f} ms',
               (best_eff_idx, aits[best_eff_idx] + 0.8),
               fontsize=7, ha='center', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels(configs, fontsize=8)
    ax.set_ylabel('AIT (ms/query)', fontsize=9, color='#BF360C')
    ax2.set_ylabel('AUROC (%)', fontsize=9, color='#1565C0')
    ax.set_title('(c) Accuracy-Efficiency Trade-off (TAU19, Fusion V2)',
                 fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)

    # --- (d) Method ranking consistency ---
    ax = fig.add_subplot(gs[1, 2])
    # Rank methods across conditions
    all_conditions = []
    for ds in ['TAU22', 'TAU19']:
        for cfg in ['full_model', 'wo_flow_transform', 'wo_ood_head', 'wo_reciprocal',
                     'wo_threshold', 'only_distance_head']:
            vals = [ablation[ds][cfg]['osr'][m]['auroc'] for m in METHOD_ORDER]
            all_conditions.append(vals)

    all_conditions = np.array(all_conditions)
    # Count how many times each method is #1
    n_conditions = all_conditions.shape[0]
    wins = [(all_conditions[:, i] == all_conditions.max(axis=1)).sum() for i in range(len(METHOD_ORDER))]
    win_pct = [w/n_conditions*100 for w in wins]

    bars = ax.barh(np.arange(len(METHOD_SHORT)), win_pct, color=method_colors, alpha=0.85,
                   edgecolor='black', linewidth=0.5)
    for i, (pct, w_count) in enumerate(zip(win_pct, wins)):
        ax.text(pct + 1, i, f'{int(w_count)}/{n_conditions}', fontsize=7, va='center',
               fontweight='bold', color='#333')

    ax.set_yticks(np.arange(len(METHOD_SHORT)))
    ax.set_yticklabels(METHOD_SHORT, fontsize=8)
    ax.set_xlabel('% of conditions where method is best (%)', fontsize=8)
    ax.set_title('(d) Method Dominance\n(12 conditions)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 110)

    # --- (e) Future directions ---
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    directions = [
        ('1. Edge Deployment', 'Test lightweight config (12.7K params, 0.64 ms) on embedded hardware'),
        ('2. Cross-Domain Validation', 'Extend to speech [31], musical instruments [30], bioacoustics'),
        ('3. Foundation Models', 'Compare with CLAP [15], AST [16], PaSST [28] — does wo_ood_head generalize?'),
        ('4. Theoretical Analysis', 'Why does AUROC saturate with K while ACC grows? Formalize prototype variance → OSR score'),
    ]

    for i, (title, desc) in enumerate(directions):
        y = 3.3 - i * 0.85
        circle = plt.Circle((0.5, y+0.15), 0.2, color='#1565C0', alpha=0.8)
        ax.add_patch(circle)
        ax.text(1.0, y+0.3, title, fontsize=9, fontweight='bold', color='#0D47A1')
        ax.text(1.0, y-0.05, desc, fontsize=7.5, color='#555')

    ax.set_title('(e) Future Work Directions', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {os.path.basename(save_path)}")


if __name__ == '__main__':
    print("Generating Chapter 6 (Section VI) conclusion summary figure...\n")
    generate_summary(os.path.join(OUT_DIR, 'fig6_1_results_summary.png'))
    print("\nInsertion point in Section VI:")
    print("  After concluding paragraph → [Insert Fig 6.1 about here.] fig6_1_results_summary.png")
