#!/usr/bin/env python3
"""
cont3_method_figures.py — Generate Section III figures and tables.

Outputs:
  - fig3_1_foac_architecture.png   ( Fig 1: FOAC episodic training architecture )
  - table3_1_model_configs.csv      ( TABLE 1: Model configuration variants )

Usage:
  python cont3_method_figures.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_architecture_diagram(save_path):
    """Draw FOAC episodic training architecture diagram (Fig 1)."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_aspect('equal')

    # Color palette
    c_input    = '#E8F5E9'  # light green
    c_feat     = '#E3F2FD'  # light blue
    c_flow     = '#FFF3E0'  # light orange
    c_cls      = '#F3E5F5'  # light purple
    c_osr      = '#FFEBEE'  # light red
    c_decision = '#D7CCC8'  # light brown
    c_osr_inset= '#FCE4EC'  # lighter pink

    # === Input ===
    box_input = mpatches.FancyBboxPatch((0.3, 3.8), 2.0, 1.2,
                                         boxstyle="round,pad=0.1",
                                         facecolor=c_input, edgecolor='#2E7D32', linewidth=2)
    ax.add_patch(box_input)
    ax.text(1.3, 4.4, 'Audio Input\n(log-mel spectrogram)', ha='center', va='center',
            fontsize=8, fontweight='bold', color='#1B5E20')

    # Arrow input -> backbone
    ax.annotate('', xy=(2.6, 4.4), xytext=(2.3, 4.4),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    # === YAMNet Backbone ===
    box_backbone = mpatches.FancyBboxPatch((2.6, 3.6), 2.2, 1.6,
                                            boxstyle="round,pad=0.1",
                                            facecolor=c_feat, edgecolor='#1565C0', linewidth=2)
    ax.add_patch(box_backbone)
    ax.text(3.7, 4.6, 'Frozen YAMNet\nBackbone [11]', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#0D47A1')
    ax.text(3.7, 4.0, '1024-dim\nembedding z', ha='center', va='center',
            fontsize=7, fontstyle='italic', color='#1976D2')

    # Arrow backbone -> flow
    ax.annotate('', xy=(5.1, 4.4), xytext=(4.8, 4.4),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    # === Flow Transform (dashed) ===
    box_flow = mpatches.FancyBboxPatch((5.1, 3.8), 1.6, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor=c_flow, edgecolor='#E65100', linewidth=2,
                                        linestyle='--')
    ax.add_patch(box_flow)
    ax.text(5.9, 4.4, 'Flow\nTransform\n[17][18]', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#BF360C')
    ax.text(5.9, 3.95, '~25K params', ha='center', va='center',
            fontsize=6.5, fontstyle='italic', color='#E65100')

    # Arrow flow -> split
    ax.annotate('', xy=(7.0, 4.4), xytext=(6.7, 4.4),
                arrowprops=dict(arrowstyle='->', color='#555', lw=2))

    # === Embedding h label ===
    ax.text(7.15, 5.1, 'embedding h', ha='center', va='center',
            fontsize=7, fontstyle='italic', color='#333')

    # === Split into two branches ===
    # Upper branch -> Classification
    ax.annotate('', xy=(7.3, 5.3), xytext=(7.0, 4.9),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5))

    # Classification branch
    box_cls = mpatches.FancyBboxPatch((7.3, 5.3), 3.0, 2.2,
                                       boxstyle="round,pad=0.1",
                                       facecolor=c_cls, edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(box_cls)
    ax.text(8.8, 7.1, 'Classification Branch', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#4A148C')
    ax.text(8.8, 6.5, 'N class prototypes\nvia support-set averaging\nSoftmax over Euclidean dist.',
            ha='center', va='center', fontsize=7, color='#6A1B9A')
    ax.text(8.8, 5.5, 'y_hat ∈ Δ^N', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#4A148C')

    # Lower branch -> OSR
    ax.annotate('', xy=(7.3, 3.2), xytext=(7.0, 3.9),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))

    # OSR Branch (main box)
    box_osr = mpatches.FancyBboxPatch((7.3, 1.5), 4.5, 2.0,
                                       boxstyle="round,pad=0.1",
                                       facecolor=c_osr, edgecolor='#C62828', linewidth=2)
    ax.add_patch(box_osr)
    ax.text(9.55, 3.2, 'OSR Detection Branch', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#B71C1C')

    # OSR Inset box
    box_inset = mpatches.FancyBboxPatch((7.6, 1.7), 3.9, 1.1,
                                         boxstyle="round,pad=0.1",
                                         facecolor=c_osr_inset, edgecolor='#E57373', linewidth=1.2)
    ax.add_patch(box_inset)
    ax.text(9.55, 2.5, 'One of 6 heads:', ha='center', va='center',
            fontsize=7, fontweight='bold', color='#555')
    ax.text(9.55, 2.15, 'anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2',
            ha='center', va='center', fontsize=6.2, color='#777')

    # s_osr output
    ax.annotate('', xy=(9.55, 1.2), xytext=(9.55, 1.5),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5))
    ax.text(9.55, 0.95, 's_osr ∈ ℝ', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#B71C1C')

    # Output arrow from cls
    ax.annotate('', xy=(9.55, 5.3), xytext=(9.55, 5.5),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.5))
    ax.text(9.55, 5.05, 'y_hat', ha='center', va='center',
            fontsize=7.5, fontweight='bold', color='#4A148C')

    # === Decision Box ===
    box_decision = mpatches.FancyBboxPatch((11.2, 3.5), 2.4, 1.8,
                                            boxstyle="round,pad=0.1",
                                            facecolor=c_decision, edgecolor='#4E342E', linewidth=2)
    ax.add_patch(box_decision)
    ax.text(12.4, 4.8, 'Decision Rule', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#3E2723')
    ax.text(12.4, 4.2, 'if s_osr < τ:\n  → REJECT (unknown)\nelse:\n  → argmax(y_hat)',
            ha='center', va='center', fontsize=7, color='#4E342E', family='monospace')

    # Arrows into decision
    ax.annotate('', xy=(11.2, 4.8), xytext=(10.3, 6.4),
                arrowprops=dict(arrowstyle='->', color='#7B1FA2', lw=1.2,
                                connectionstyle='arc3,rad=0.3'))
    ax.annotate('', xy=(11.2, 3.8), xytext=(9.55, 1.2),
                arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2,
                                connectionstyle='arc3,rad=-0.3'))

    # === Learning signals (left side, red arrows going into branches) ===
    ax.text(0.3, 8.0, 'Training Loss Signals', fontsize=8, fontweight='bold', color='#333')
    loss_signals = [
        'L_cls: Classification CE',
        'L_osr: Open-set BCE',
        'L_rec: Reciprocal attention [9]',
        'L_sep: Feature separation [24]',
    ]
    for i, sig in enumerate(loss_signals):
        ax.text(0.3, 7.6 - i*0.3, sig, fontsize=6.5, color='#555', family='monospace')

    # === Episode legend ===
    ax.text(0.3, 1.0, 'Episode Input:', fontsize=7, fontweight='bold', color='#333')
    ax.text(0.3, 0.7, 'S = {(x_i, y_i)}, i=1..N×K', fontsize=6.5, color='#555', family='monospace')
    ax.text(0.3, 0.4, 'Q = known ∪ unknown queries', fontsize=6.5, color='#555', family='monospace')

    # === Title ===
    ax.set_title('Fig 1. FOAC Episodic Training Architecture', fontsize=12, fontweight='bold',
                 loc='center', pad=15)
    ax.text(0.02, -0.02, '(Dashed box: optional Flow Transform; OSR inset: 6 alternative detection heads)',
            transform=ax.transAxes, fontsize=7, fontstyle='italic', color='#888')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Architecture diagram saved: {save_path}")


def generate_config_table(save_path):
    """Generate TABLE 1: Model configuration variants."""
    headers = ['Configuration', 'Flow Transform', 'OOD Head Training',
               'Reciprocal Loss', 'Threshold Adaptation', 'Description']
    rows = [
        ['full_model',       '✓', '✓', '✓', '✓', 'All components active'],
        ['wo_flow_transform', '', '✓', '✓', '✓', 'No normalizing flow [17][18]'],
        ['wo_ood_head',      '✓', '',  '✓', '✓', 'No OSR-specific BCE loss'],
        ['wo_reciprocal',    '✓', '✓', '',  '✓', 'No reciprocal attention [9]'],
        ['wo_threshold',     '✓', '✓', '✓', '',  'Fixed percentile threshold'],
        ['only_distance_head','', '',  '',  '',  'Bare Mahalanobis detector [8]'],
    ]

    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
    ax.axis('off')

    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor('#37474F')
        cell.set_text_props(weight='bold', color='white', fontsize=9)

    # Style body
    for i in range(len(rows)):
        color = '#F5F5F5' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[i+1, j].set_facecolor(color)

    # Adjust column widths
    for (i, j), cell in table.get_celld().items():
        if j == 0:
            cell.set_width(0.18)
        elif j == len(headers) - 1:
            cell.set_width(0.32)
        else:
            cell.set_width(0.12)

    ax.set_title('TABLE 1. Model Configuration Variants for Ablation Experiments',
                 fontsize=10, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Config table saved: {save_path}")

    # Also save CSV
    csv_path = save_path.replace('.png', '.csv')
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"[OK] Config CSV saved: {csv_path}")


if __name__ == '__main__':
    draw_architecture_diagram(os.path.join(OUT_DIR, 'fig3_1_foac_architecture.png'))
    generate_config_table(os.path.join(OUT_DIR, 'table3_1_model_configs.png'))
    print("\n=== Chapter 3 figures done. Insertion points:")
    print("  [Insert Fig 1 about here.] → fig3_1_foac_architecture.png")
    print("  [Insert TABLE 1 about here.] → table3_1_model_configs.png")
