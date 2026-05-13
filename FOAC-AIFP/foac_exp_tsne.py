#!/usr/bin/env python3
"""
foac_exp_tsne.py — t-SNE Feature Visualization for FOAC-AIFP

Generates t-SNE plots showing feature distributions of known vs unknown
classes. Can compare full model vs baseline (no NPM) to visualize
the effect of open-set prototype generation.

Usage:
  cd /coding/FOAC-AIFP
  python foac_exp_tsne.py --config tau22_aligned.yml
  python foac_exp_tsne.py --config tau19_aligned.yml --compare_ablation
"""

import os
import sys
import argparse
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import yaml
from tqdm import tqdm

from trainers import trainer
from models.Network import My_Net
from datasets import dataloaders


CLASS_NAMES_TAU = {
    0: 'Airport', 1: 'Shopping Mall', 2: 'Metro Station', 3: 'Street Pedestrian',
    4: 'Public Square', 5: 'Street Traffic', 6: 'Tram', 7: 'Bus', 8: 'Metro', 9: 'Park'
}


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = cfg['train']
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    base_parser = trainer.train_parser()
    sys.argv = saved_argv
    merged = vars(base_parser)
    merged.update(cfg)
    args = argparse.Namespace(**merged)
    for k, v in vars(args).items():
        if isinstance(v, dict):
            setattr(args, k, argparse.Namespace(**v))
    return args


def extract_features(model, dataloader, args, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Extract features from dataloader, grouped by class.

    FOAC dataloader returns raw audio tensors of shape (1, N, 16000).
    model.encode() expects (N, 16000) — 2D: batch × time (no channel dim;
    the internal Spectrogram adds it via input[:, None, :]).
    """
    model.eval()
    features = []
    labels = []
    splits = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Extracting features'):
            support_data, support_label, query_data, query_label, \
                suppopen_data, suppopen_label, openset_data, openset_label, \
                supp_idx, open_idx = data

            # Support features (known/base classes)
            # (1, N, 16000) → (N, 16000)
            s_data = support_data.float().squeeze(0).to(device)
            s_label = support_label.squeeze(0).cpu().numpy()
            s_feat, _ = model.encode(s_data)
            s_feat = s_feat.cpu().numpy()

            for feat, label in zip(s_feat, s_label):
                features.append(feat.flatten())
                labels.append(int(label))
                splits.append('known')

            # Open-set features (unknown/novel classes)
            o_data = openset_data.float().squeeze(0).to(device)
            o_label = openset_label.squeeze(0).cpu().numpy()
            if o_data.numel() == 0:
                continue
            o_feat, _ = model.encode(o_data)
            o_feat = o_feat.cpu().numpy()

            for feat, label in zip(o_feat, o_label):
                features.append(feat.flatten())
                labels.append(int(label))
                splits.append('unknown')

    return np.array(features), np.array(labels), np.array(splits)


def plot_tsne(features, labels, splits, class_names, base_classes,
              unknown_classes, save_path, title='t-SNE Visualization'):
    """Generate t-SNE plot."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn/matplotlib not available")
        return

    # Subsample for speed
    max_samples = 5000
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
        splits = splits[idx]

    print(f"Computing t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))

    # Use different colormaps for known and unknown
    known_mask = splits == 'known'
    unknown_mask = splits == 'unknown'

    # Plot known classes
    unique_known = sorted(set(labels[known_mask]))
    cmap_known = plt.cm.Set2(np.linspace(0, 1, max(len(unique_known), 1)))
    for i, c in enumerate(unique_known):
        mask = (labels == c) & known_mask
        name = class_names.get(c, f'Class {c}')
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[cmap_known[i]], marker='o', alpha=0.6, s=15,
                   label=f'{name} (known)')

    # Plot unknown classes
    unique_unknown = sorted(set(labels[unknown_mask]))
    cmap_unknown = plt.cm.Set1(np.linspace(0, 1, max(len(unique_unknown), 1)))
    for i, c in enumerate(unique_unknown):
        mask = (labels == c) & unknown_mask
        name = class_names.get(c, f'Class {c}')
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[cmap_unknown[i]], marker='^', alpha=0.6, s=25,
                   label=f'{name} (unknown)')

    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.set_title(title)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='FOAC t-SNE Visualization')
    parser.add_argument('--config', type=str, default='tau22_aligned.yml')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint path (default: auto-detect)')
    parser.add_argument('--compare_ablation', action='store_true',
                        help='Generate comparison plots: full model vs baseline')
    cl_args = parser.parse_args()

    args = load_config(cl_args.config)
    dataset = args.dataset
    base_classes = list(range(args.train_classes))  # 0-5
    unknown_classes = list(range(args.train_classes, 10))  # 6-9
    class_names = CLASS_NAMES_TAU

    save_dir = os.path.join(args.save_folder, 'tsne')
    os.makedirs(save_dir, exist_ok=True)

    # --- Full model ---
    print("\n=== Full Model t-SNE ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = My_Net(args=args, mode='train')
    model = model.to(device)

    ckpt_path = cl_args.ckpt
    if ckpt_path is None:
        for suffix in ['max_auroc', 'max_osr', 'max_acc']:
            ckpt_path = os.path.join(args.save_folder, f'model_{dataset}_{suffix}.pth')
            if os.path.exists(ckpt_path):
                break

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        return

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)

    # weight_base / weight_base_open are dynamically registered in init_representation(),
    # not in __init__. Pre-create them so load_state_dict can find them.
    if 'weight_base' in state_dict:
        model.weight_base = nn.Parameter(state_dict['weight_base'].to(device))
    if 'weight_base_open' in state_dict:
        model.weight_base_open = nn.Parameter(state_dict['weight_base_open'].to(device))

    model.load_state_dict(state_dict, strict=False)

    test_loader = dataloaders.meta_test_dataloader(args)
    features, labels, splits = extract_features(model, test_loader, args)

    plot_tsne(features, labels, splits, class_names, base_classes,
              unknown_classes,
              os.path.join(save_dir, f'{dataset}_full_model.png'),
              title=f'{dataset} — Full Model (CIAM+PAM+NPM)')

    # --- Baseline (no NPM) comparison ---
    if cl_args.compare_ablation:
        print("\n=== Baseline (no NPM) t-SNE ===")
        model_baseline = My_Net(args=args, mode='train', use_npm=False)
        model_baseline = model_baseline.to(device)

        # Try to load baseline ablation checkpoint
        baseline_ckpt = os.path.join(
            args.save_folder + '_ablation/wo_NPM',
            f'model_{dataset}_max_acc.pth')
        if os.path.exists(baseline_ckpt):
            state = torch.load(baseline_ckpt, map_location=device)
            if 'weight_base' in state:
                model_baseline.weight_base = nn.Parameter(state['weight_base'].to(device))
            if 'weight_base_open' in state:
                model_baseline.weight_base_open = nn.Parameter(state['weight_base_open'].to(device))
            model_baseline.load_state_dict(state, strict=False)
        else:
            print(f"  No baseline checkpoint found, using same weights (feature comparison only)")
            if 'weight_base' in state_dict:
                model_baseline.weight_base = nn.Parameter(state_dict['weight_base'].to(device))
            if 'weight_base_open' in state_dict:
                model_baseline.weight_base_open = nn.Parameter(state_dict['weight_base_open'].to(device))
            model_baseline.load_state_dict(state_dict, strict=False)

        features_bl, labels_bl, splits_bl = extract_features(
            model_baseline, test_loader, args)

        plot_tsne(features_bl, labels_bl, splits_bl, class_names, base_classes,
                  unknown_classes,
                  os.path.join(save_dir, f'{dataset}_wo_NPM.png'),
                  title=f'{dataset} — Without NPM (Baseline)')

        # Side-by-side comparison
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

            for ax, feat, lbl, spl, ttl in [
                (ax1, features, labels, splits, 'Full Model'),
                (ax2, features_bl, labels_bl, splits_bl, 'w/o NPM (Baseline)')
            ]:
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                emb = tsne.fit_transform(feat)

                known_mask = spl == 'known'
                unknown_mask = spl == 'unknown'
                for c in sorted(set(lbl[known_mask])):
                    mask = (lbl == c) & known_mask
                    ax.scatter(emb[mask, 0], emb[mask, 1], marker='o', alpha=0.5, s=12,
                               label=f'{class_names.get(c, c)} (known)')
                for c in sorted(set(lbl[unknown_mask])):
                    mask = (lbl == c) & unknown_mask
                    ax.scatter(emb[mask, 0], emb[mask, 1], marker='^', alpha=0.5, s=20,
                               label=f'{class_names.get(c, c)} (unknown)')
                ax.set_title(ttl)
                ax.legend(fontsize=6, ncol=2)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{dataset}_comparison.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Comparison plot saved")
        except Exception as e:
            print(f"Comparison plot failed: {e}")


if __name__ == '__main__':
    main()
