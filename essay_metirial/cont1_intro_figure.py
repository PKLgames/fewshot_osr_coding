#!/usr/bin/env python3
"""
cont1_intro_figure.py — Generate Section I conceptual illustration.

Outputs:
  - fig1_intro_foac_problem.png  ( FOAC problem definition & three core challenges )

Usage:
  python cont1_intro_figure.py
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_foac_problem_diagram(save_path):
    """Draw conceptual FOAC problem illustration."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (ax0, ax1), (ax2, ax3) = axes

    np.random.seed(42)

    # ── Panel (a): FOAC Problem Definition ──
    ax0.set_xlim(-3, 3)
    ax0.set_ylim(-3, 3)
    ax0.set_aspect('equal')
    ax0.set_title('(a) FOAC Task: Classify Known + Reject Unknown', fontsize=10, fontweight='bold')

    # Known class clusters
    known_colors = ['#2196F3', '#4CAF50', '#FF9800']
    known_centers = [(-1.5, 1.0), (1.5, 1.0), (0, -0.8)]
    for i, (cx, cy) in enumerate(known_centers):
        pts = np.random.randn(60, 2) * 0.22 + [cx, cy]
        ax0.scatter(pts[:, 0], pts[:, 1], c=known_colors[i], s=10, alpha=0.6, edgecolors='none')
        ax0.scatter([cx], [cy], c=known_colors[i], s=80, marker='*', edgecolors='black', linewidth=1,
                   label=f'Known Class {i+1} (N={i+1})')
        # Support samples
        support_pts = np.random.randn(5, 2) * 0.15 + [cx, cy]
        ax0.scatter(support_pts[:, 0], support_pts[:, 1], c='black', s=25, marker='s', alpha=0.8)

    # Unknown samples
    unknown_pts = np.concatenate([
        np.random.randn(20, 2) * 0.6 + [-2.5, -2],
        np.random.randn(20, 2) * 0.6 + [2.5, -2],
        np.random.randn(20, 2) * 0.5 + [0, 2.5],
    ])
    ax0.scatter(unknown_pts[:, 0], unknown_pts[:, 1], c='gray', marker='x', s=20, alpha=0.7,
               label='Unknown Class (reject)')

    # Support sample legend
    ax0.scatter([], [], c='black', s=25, marker='s', label='Support Samples (K-shot)')
    ax0.legend(fontsize=6.5, loc='upper right', ncol=2)
    ax0.set_xticks([])
    ax0.set_yticks([])

    # ── Panel (b): Challenge 1 – Prototype Instability ──
    ax1.set_title('(b) Challenge 1: Prototype Instability (K=1 vs K=10)',
                  fontsize=10, fontweight='bold')
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')

    # True distribution
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(1.0*np.cos(theta), 1.0*np.sin(theta), 'b--', alpha=0.3, label='True class manifold')
    true_center = np.array([0.0, 0.0])

    # K=1 prototypes (high variance)
    for _ in range(8):
        proto = true_center + np.random.randn(2) * 0.8
        ax1.scatter([proto[0]], [proto[1]], c='red', s=40, marker='^', alpha=0.5, edgecolors='none')

    # K=10 prototypes (low variance)
    for _ in range(8):
        samples = true_center + np.random.randn(10, 2) * 0.8
        proto = samples.mean(axis=0)
        ax1.scatter([proto[0]], [proto[1]], c='green', s=40, marker='o', alpha=0.5, edgecolors='none')

    ax1.scatter([], [], c='red', s=40, marker='^', label='K=1 prototypes (high var)')
    ax1.scatter([], [], c='green', s=40, marker='o', label='K=10 prototypes (low var)')
    ax1.scatter([0], [0], c='blue', s=60, marker='*', label='True class center')
    ax1.legend(fontsize=7, loc='upper right')

    # ── Panel (c): Challenge 2 – Open-Set Boundary ──
    ax2.set_title('(c) Challenge 2: Open-Set vs Closed-Set Decision',
                  fontsize=10, fontweight='bold')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')

    # Known cluster
    p_known = np.random.randn(150, 2) * 0.5 + [0, 0]
    ax2.scatter(p_known[:, 0], p_known[:, 1], c='steelblue', s=6, alpha=0.5, label='Known samples')

    # Unknown
    p_unknown = np.random.randn(80, 2) * 0.8 + [2.2, 1.5]
    p_unknown2 = np.random.randn(80, 2) * 0.6 + [-2, -1.8]
    ax2.scatter(p_unknown[:, 0], p_unknown[:, 1], c='tomato', s=8, alpha=0.5, marker='x',
               label='Unknown samples')
    ax2.scatter(p_unknown2[:, 0], p_unknown2[:, 1], c='tomato', s=8, alpha=0.5, marker='x')

    # Closed-set boundary (softmax always maps to known)
    circle = plt.Circle((0, 0), 1.5, fill=False, color='gray', linestyle=':', linewidth=2,
                        label='Closed-set: all → known')
    ax2.add_patch(circle)

    # OSR boundary
    circle2 = plt.Circle((0, 0), 1.0, fill=False, color='green', linestyle='-', linewidth=2,
                         label='Open-set: reject outside')
    ax2.add_patch(circle2)

    # Arrow showing rejection
    ax2.annotate('REJECT', xy=(2.2, 1.5), xytext=(1.0, 2.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=8, color='red', fontweight='bold')

    ax2.legend(fontsize=7, loc='lower left')

    # ── Panel (d): Challenge 3 – Domain Shift ──
    ax3.set_title('(d) Challenge 3: Cross-Domain Generalization',
                  fontsize=10, fontweight='bold')
    ax3.set_xlim(-4, 4)
    ax3.set_ylim(-4, 4)
    ax3.set_aspect('equal')

    # Source domain
    src_centers = [(-2, 1), (0, 1.5), (2, 1)]
    src_colors = ['#2196F3', '#4CAF50', '#FF9800']
    for i, (cx, cy) in enumerate(src_centers):
        pts = np.random.randn(50, 2) * 0.3 + [cx, cy]
        ax3.scatter(pts[:, 0], pts[:, 1], c=src_colors[i], s=8, alpha=0.5)

    # Draw ellipse around source
    src_ellipse = mpatches.Ellipse((0, 1), 5.0, 1.8, fill=False, edgecolor='blue',
                                     linewidth=1.5, linestyle='--')
    ax3.add_patch(src_ellipse)
    ax3.text(-2.5, 2.2, 'Source Domain\n(e.g. TAU22)', fontsize=7, color='blue', ha='center')

    # Target domain (shifted)
    tgt_centers = [(-1.5, -1.5), (0.5, -1), (2, -2)]
    for i, (cx, cy) in enumerate(tgt_centers):
        pts = np.random.randn(50, 2) * 0.45 + [cx, cy]
        ax3.scatter(pts[:, 0], pts[:, 1], c=src_colors[i], s=8, alpha=0.3)

    tgt_ellipse = mpatches.Ellipse((0, -1.5), 5.0, 2.0, fill=False, edgecolor='red',
                                     linewidth=1.5, linestyle='--')
    ax3.add_patch(tgt_ellipse)
    ax3.text(-2.5, -2.8, 'Target Domain\n(e.g. TAU19)', fontsize=7, color='red', ha='center')

    # Domain shift arrow
    ax3.annotate('', xy=(0.5, -0.3), xytext=(0.5, 0.8),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2,
                               connectionstyle='arc3,rad=0'))
    ax3.text(1.3, 0.3, 'Domain\nShift', fontsize=7, color='purple', fontweight='bold')

    ax3.set_xticks([])
    ax3.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] FOAC problem diagram saved: {save_path}")


if __name__ == '__main__':
    draw_foac_problem_diagram(os.path.join(OUT_DIR, 'fig1_intro_foac_problem.png'))
    print("\nInsertion point in Section I:")
    print("  After paragraph on three core difficulties → [Insert Fig 0 about here.] fig1_intro_foac_problem.png")
