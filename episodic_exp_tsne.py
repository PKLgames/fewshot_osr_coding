#!/usr/bin/env python3
"""
episodic_exp_tsne.py — t-SNE Feature Visualization for Episodic Trainer

Generates t-SNE plots of base vs novel class features in the
EpisodicFlowClassifier's feature space. Can compare full model vs baseline.

Usage:
  cd /coding
  python episodic_exp_tsne.py
  python episodic_exp_tsne.py --compare_ablation
"""

import os
import json
import argparse
import random
import numpy as np
import torch

from episodic_trainer import (
    FeatureExtractor, FeatureCache, EpisodicFlowClassifier, TAUDataset,
)


CLASS_NAMES = {
    0: 'Airport', 1: 'Shopping Mall', 2: 'Metro Station', 3: 'Street Pedestrian',
    4: 'Public Square', 5: 'Street Traffic', 6: 'Tram', 7: 'Bus', 8: 'Metro', 9: 'Park',
}


def load_trained_classifier(experiment_dir, device='cuda', feature_dim=64,
                             use_flow_transform=False):
    """Load trained EpisodicFlowClassifier from experiment directory."""
    final_path = os.path.join(experiment_dir, 'fewshot_final.pth')
    if not os.path.exists(final_path):
        return None

    ckpt = torch.load(final_path, map_location=device, weights_only=False)
    classifier = EpisodicFlowClassifier(
        input_dim=feature_dim, condition_dim=feature_dim,
        use_flow_transform=use_flow_transform,
    ).to(device)
    classifier.load_state_dict(ckpt['flow_state_dict'])
    classifier.eval()
    return classifier


def extract_projected_features(classifier, cache, classes, device='cuda',
                                 max_per_class=200):
    """Extract features projected through the classifier's feature_adapter."""
    features, labels = [], []

    with torch.no_grad():
        for c in classes:
            class_feats = cache.get_class_features(c)
            if len(class_feats) == 0:
                continue
            # Subsample if too many
            if len(class_feats) > max_per_class:
                idx = np.random.choice(len(class_feats), max_per_class, replace=False)
                class_feats = class_feats[idx]
            # Project through feature adapter
            projected = classifier.project(class_feats.to(device))
            features.append(projected.cpu().numpy())
            labels.extend([c] * len(class_feats))

    return np.concatenate(features), np.array(labels)


def plot_tsne(features, labels, base_classes, unknown_classes, save_path, title=''):
    """Generate t-SNE plot."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn/matplotlib not available")
        return

    # Subsample
    max_samples = 3000
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    print(f"  Computing t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Base classes (circles)
    cmap_base = plt.cm.Set2(np.linspace(0, 1, max(len(base_classes), 1)))
    for i, c in enumerate(sorted(set(labels) & set(base_classes))):
        mask = labels == c
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[cmap_base[i]], marker='o', alpha=0.6, s=15,
                   label=f'{CLASS_NAMES.get(c, c)} (known)')

    # Unknown classes (triangles)
    cmap_unk = plt.cm.Set1(np.linspace(0, 1, max(len(unknown_classes), 1)))
    for i, c in enumerate(sorted(set(labels) & set(unknown_classes))):
        mask = labels == c
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[cmap_unk[i]], marker='^', alpha=0.6, s=25,
                   label=f'{CLASS_NAMES.get(c, c)} (unknown)')

    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.set_title(title)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Episodic t-SNE Visualization')
    parser.add_argument('--experiment_dir', type=str,
                        default='experiment/yamnet_fewshot_osr22_relabel')
    parser.add_argument('--dataset', choices=['TAU22', 'TAU19'], default='TAU22')
    parser.add_argument('--compare_ablation', action='store_true',
                        help='Compare full model vs baseline (no extras)')
    cl_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_classes = [0, 1, 2, 3, 4, 5]
    unknown_classes = [6, 7, 8, 9]

    # Load features (reuse cache if available)
    backbone_path = os.path.join(cl_args.experiment_dir, 'base_feature_extractor.pth')
    if not os.path.exists(backbone_path):
        print(f"Backbone not found: {backbone_path}")
        return

    feature_extractor = FeatureExtractor(pretrained_path=backbone_path)
    feature_extractor = feature_extractor.to(device).freeze()

    # Load dataset
    if cl_args.dataset == 'TAU22':
        from utils.TAU22 import TAUDataset
    else:
        from utils.TAU19 import TAUDataset

    train_dataset = TAUDataset(split='train')
    test_dataset = TAUDataset(split='test')

    train_cache = FeatureCache()
    train_cache.extract_and_cache(feature_extractor, train_dataset, 'train',
                                   device, batch_size=64)
    test_cache = FeatureCache()
    test_cache.extract_and_cache(feature_extractor, test_dataset, 'test',
                                  device, batch_size=64)

    save_dir = os.path.join(cl_args.experiment_dir, 'tsne')
    os.makedirs(save_dir, exist_ok=True)

    # --- Full model ---
    print("\n=== Full Model t-SNE ===")
    classifier = load_trained_classifier(cl_args.experiment_dir, device)
    if classifier:
        train_feats, train_labels = extract_projected_features(
            classifier, train_cache, base_classes, device)
        test_feats, test_labels = extract_projected_features(
            classifier, test_cache, unknown_classes, device)

        all_feats = np.concatenate([train_feats, test_feats])
        all_labels = np.concatenate([train_labels, test_labels])

        plot_tsne(all_feats, all_labels, base_classes, unknown_classes,
                  os.path.join(save_dir, f'{cl_args.dataset}_full_model.png'),
                  title=f'{cl_args.dataset} — Full Model (Episodic Trainer)')

    # --- Baseline comparison ---
    if cl_args.compare_ablation:
        print("\n=== Baseline (distance head only) t-SNE ===")
        # Train a minimal classifier just for projection
        baseline = EpisodicFlowClassifier(
            input_dim=64, condition_dim=64,
            use_flow_transform=False, use_ood_head=False,
            use_reciprocal=False, use_threshold=False,
        ).to(device)

        # Copy feature_adapter weights from full model if available
        if classifier:
            baseline.feature_adapter.load_state_dict(
                classifier.feature_adapter.state_dict())
        baseline.eval()

        train_feats, train_labels = extract_projected_features(
            baseline, train_cache, base_classes, device)
        test_feats, test_labels = extract_projected_features(
            baseline, test_cache, unknown_classes, device)

        all_feats = np.concatenate([train_feats, test_feats])
        all_labels = np.concatenate([train_labels, test_labels])

        plot_tsne(all_feats, all_labels, base_classes, unknown_classes,
                  os.path.join(save_dir, f'{cl_args.dataset}_baseline.png'),
                  title=f'{cl_args.dataset} — Baseline (Distance Head Only)')


if __name__ == '__main__':
    main()
