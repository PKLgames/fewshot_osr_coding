#!/usr/bin/env python3
"""
Generate Fig 4 from real t-SNE images (episodic YAMNet checkpoints).

Composes 4 real t-SNE images into a single 2×2 figure:
  Top:    TAU22 full_model | TAU22 baseline
  Bottom: TAU19 full_model | TAU19 baseline

Replaces the synthetic fig4_tsne.png.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TSNE_DIR = '/coding/experiment/episodic_exp/tsne'


def compose_fig4(save_path):
    images = {
        (0, 0): ('TAU22_full_model.png', 'TAU22 — Full Model'),
        (0, 1): ('TAU22_baseline.png', 'TAU22 — Baseline (Dist. Only)'),
        (1, 0): ('TAU19_full_model.png', 'TAU19 — Full Model'),
        (1, 1): ('TAU19_baseline.png', 'TAU19 — Baseline (Dist. Only)'),
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for (row, col), (filename, title) in images.items():
        img_path = os.path.join(TSNE_DIR, filename)
        img = mpimg.imread(img_path)
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    fig.suptitle('Fig 4. Real t-SNE Feature Space Visualization (YAMNet Backbone, 2000 samples)',
                 fontsize=13, fontweight='bold', y=0.99)

    # Annotation explaining what to look for
    fig.text(0.5, 0.01,
             'Known classes shown as colored clusters; unknown class samples (gray ×) at periphery. '
             'Full Model (left) vs. Baseline distance-head only (right). '
             'Darker unknown-class separation in Full Model indicates stronger open-set boundary.',
             ha='center', fontsize=8, fontstyle='italic', color='#555')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] {os.path.basename(save_path)} — composite from real episodic t-SNE images")


if __name__ == '__main__':
    compose_fig4(os.path.join(OUT_DIR, 'fig4_tsne.png'))
    print("\nInsertion point in Section IV.G3:")
    print("  [Insert Fig 4 about here.] → fig4_tsne.png (real YAMNet t-SNE features)")
