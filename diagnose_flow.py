#!/usr/local/miniconda3/envs/py312/bin/python
"""
Flow Latent Space Diagnostic Tool

Diagnoses whether the conditional normalizing flow has sufficient capacity
to map known-class features to standard Gaussian z-space, and whether
inter-class separation is adequate for OSR.

Usage:
  python diagnose_flow.py

Checks:
  1. Z-space Gaussianity:     Are z_correct samples ~ N(0, I)?
  2. Inter-class separation:  Are z_wrong far from z_correct?
  3. Known vs Unknown gap:    Is there a clear density gap for OSR?
  4. Flow expressiveness:     Is the flow capacity sufficient?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import random
import sys
from typing import Dict, List
from scipy import stats as scipy_stats

torch.backends.cudnn.benchmark = True

from episodic_trainer import (
    FeatureExtractor, FeatureCache, EpisodicFlowClassifier
)
from utils.TAU22 import TAUDataset


def load_trained_model(experiment_dir='experiment/yamnet_fewshot_osr4',
                       device='cuda'):
    """Load trained flow classifier and feature caches."""
    base_pretrained = os.path.join(experiment_dir, 'base_feature_extractor.pth')

    feature_extractor = FeatureExtractor(pretrained_path=base_pretrained)
    feature_extractor = feature_extractor.to(device).eval()

    # Load flow classifier
    ckpt = torch.load(os.path.join(experiment_dir, 'fewshot_final.pth'),
                       map_location=device, weights_only=False)
    flow_config = ckpt['config']

    # Match the training config
    flow_classifier = EpisodicFlowClassifier(
        input_dim=flow_config['input_dim'],
        condition_dim=flow_config['condition_dim'],
        num_coupling_layers=8,
        hidden_dims=[256, 256],
        use_projection=True,
        s_clamp_max=3.0
    ).to(device)
    flow_classifier.load_state_dict(ckpt['flow_state_dict'])
    flow_classifier.eval()

    # Load caches
    train_cache = FeatureCache()
    train_dataset = TAUDataset(split='train')
    train_cache.extract_and_cache(feature_extractor, train_dataset, 'train', device)

    test_cache = FeatureCache()
    test_dataset = TAUDataset(split='test')
    test_cache.extract_and_cache(feature_extractor, test_dataset, 'test', device)

    return flow_classifier, train_cache, test_cache


def compute_z_for_class(flow_classifier, features, prototype, device):
    """Map features to z-space conditioned on a single prototype."""
    flow_classifier.eval()
    with torch.no_grad():
        adapted_feats, adapted_proto = flow_classifier.adapt_features(
            features.to(device), prototype.unsqueeze(0).to(device))
        proj_feats = flow_classifier.project(adapted_feats)
        proj_proto = flow_classifier.project(adapted_proto).squeeze(0)

        condition = proj_proto.unsqueeze(0).expand(proj_feats.size(0), -1)
        z, _ = flow_classifier.flow.forward(proj_feats, condition, compute_jacobian=False)
    return z.cpu()


def compute_log_likelihoods(flow_classifier, features, prototypes, device):
    """Compute flow log-likelihood for features under each prototype."""
    flow_classifier.eval()
    with torch.no_grad():
        adapted_feats, adapted_protos = flow_classifier.adapt_features(
            features.to(device), prototypes.to(device))
        proj_feats = flow_classifier.project(adapted_feats)
        proj_protos = flow_classifier.project(adapted_protos)

        B = proj_feats.size(0)
        N = proj_protos.size(0)
        log_probs = torch.zeros(B, N)

        for c in range(N):
            condition = proj_protos[c].unsqueeze(0).expand(B, -1)
            lp = flow_classifier.flow.log_prob(proj_feats, condition)
            log_probs[:, c] = lp.cpu()
    return log_probs


# ============================================================
# Diagnosis 1: Gaussianity of z_correct
# ============================================================
def diagnose_gaussianity(z_samples: torch.Tensor, label: str):
    """Test if z_samples follow N(0, I)."""
    z_np = z_samples.numpy()
    D = z_np.shape[1]
    n_samples = z_np.shape[0]

    print(f"\n{'='*60}")
    print(f"  Gaussianity: {label}")
    print(f"  Samples={n_samples}, Dims={D}")
    print(f"{'='*60}")

    dim_means = z_np.mean(axis=0)
    dim_stds = z_np.std(axis=0)
    dim_kurtosis = scipy_stats.kurtosis(z_np, axis=0, fisher=True)
    dim_skewness = scipy_stats.skew(z_np, axis=0)

    print(f"\n  Per-dimension (ideal: mean=0, std=1, skew=0, kurt=0):")
    print(f"    Mean:  avg={dim_means.mean():.4f}  max|.|={np.abs(dim_means).max():.4f}")
    print(f"    Std:   avg={dim_stds.mean():.4f}  range=[{dim_stds.min():.4f}, {dim_stds.max():.4f}]")
    print(f"    Skew:  avg|.||={np.abs(dim_skewness).mean():.4f}  max|.||={np.abs(dim_skewness).max():.4f}")
    print(f"    Kurt:  avg|.||={np.abs(dim_kurtosis).mean():.4f}  max|.||={np.abs(dim_kurtosis).max():.4f}")

    # Shapiro-Wilk per dim
    n_test = min(n_samples, 5000)
    test_dims = random.sample(range(D), min(10, D))
    sw_pvalues = []
    for d in test_dims:
        _, p = scipy_stats.shapiro(z_np[:n_test, d])
        sw_pvalues.append(p)
    avg_p = np.mean(sw_pvalues)
    n_pass = sum(1 for p in sw_pvalues if p > 0.05)
    print(f"\n  Shapiro-Wilk (10 dims, p>0.05=Gaussian): avg_p={avg_p:.4f} pass={n_pass}/10")
    if avg_p < 0.01:
        print(f"    *** NOT Gaussian (p={avg_p:.4e}) ***")
    elif avg_p < 0.05:
        print(f"    WARNING: borderline (p={avg_p:.4f})")
    else:
        print(f"    OK: approximately Gaussian")

    # ||z||^2 vs Chi-squared(D)
    z_norm_sq = (z_np ** 2).sum(axis=1)
    mean_ratio = z_norm_sq.mean() / D
    var_ratio = z_norm_sq.var() / (2 * D)
    ks_stat, ks_p = scipy_stats.kstest(z_norm_sq, scipy_stats.chi2(df=D).cdf)
    print(f"\n  ||z||^2 vs chi2({D}):")
    print(f"    mean_ratio={mean_ratio:.3f} (ideal=1.0)  var_ratio={var_ratio:.3f} (ideal=1.0)")
    print(f"    KS: stat={ks_stat:.4f} p={ks_p:.4e}")
    if ks_p < 0.01:
        print(f"    *** ||z||^2 NOT chi2 distributed ***")

    return {'mean_ratio': mean_ratio, 'std_avg': dim_stds.mean(),
            'sw_avg_p': avg_p, 'ks_p': ks_p}


# ============================================================
# Diagnosis 2: Inter-class separation in z-space
# ============================================================
def diagnose_separation(z_by_class: Dict[int, torch.Tensor], label: str):
    """Check inter-class separation in z-space."""
    classes = sorted(z_by_class.keys())
    print(f"\n{'='*60}")
    print(f"  Inter-class Separation: {label}")
    print(f"  Classes: {classes}")
    print(f"{'='*60}")

    centroids = {c: z_by_class[c].mean(dim=0) for c in classes}

    # Pairwise centroid distances
    print(f"\n  Pairwise centroid distances:")
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            dist = (centroids[c1] - centroids[c2]).norm().item()
            print(f"    {c1} <-> {c2}: {dist:.3f}")

    # Intra-class spread
    print(f"\n  Intra-class spread (avg ||z - centroid||):")
    for c in classes:
        spread = (z_by_class[c] - centroids[c]).norm(dim=1).mean().item()
        print(f"    Class {c}: {spread:.3f}")

    # Inter/Intra ratio
    intra_dists, inter_dists = [], []
    for c in classes:
        z = z_by_class[c][:100]
        n = len(z)
        for i in range(min(50, n)):
            for j in range(i+1, min(i+10, n)):
                intra_dists.append((z[i] - z[j]).norm().item())

    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            z1 = z_by_class[c1][:50]
            z2 = z_by_class[c2][:50]
            for a in range(0, len(z1), 5):
                for b in range(0, len(z2), 5):
                    inter_dists.append((z1[a] - z2[b]).norm().item())

    avg_intra = np.mean(intra_dists)
    avg_inter = np.mean(inter_dists)
    ratio = avg_inter / (avg_intra + 1e-8)
    print(f"\n  Inter/Intra ratio: {ratio:.3f}")
    print(f"    Intra avg: {avg_intra:.3f}  Inter avg: {avg_inter:.3f}")
    if ratio < 1.5:
        print(f"    *** POOR: classes overlap (ratio < 1.5) ***")
    elif ratio < 3.0:
        print(f"    MODERATE separation")
    else:
        print(f"    GOOD separation")

    return {'inter_intra_ratio': ratio, 'avg_intra': avg_intra, 'avg_inter': avg_inter}


# ============================================================
# Diagnosis 3: Known vs Unknown score gap
# ============================================================
def diagnose_osr_discrimination(flow_classifier, train_cache, test_cache,
                                base_classes, unknown_classes, device):
    """Check known vs unknown score separation (OSR capability)."""
    print(f"\n{'='*60}")
    print(f"  OSR: Known vs Unknown Score Gap")
    print(f"{'='*60}")

    # Prototypes from base classes
    prototypes = []
    for c in base_classes:
        feats = train_cache.get_class_features(c)
        prototypes.append(feats[:50].mean(dim=0))
    prototypes = torch.stack(prototypes).to(device)

    # Score known
    known_scores = []
    for c in base_classes:
        feats = train_cache.get_class_features(c)
        lp = compute_log_likelihoods(flow_classifier, feats, prototypes, device)
        known_scores.append(lp.max(dim=1).values)
    known_scores = torch.cat(known_scores).numpy()

    # Score unknown
    unknown_scores = []
    for c in unknown_classes:
        feats = test_cache.get_class_features(c)
        if len(feats) > 0:
            lp = compute_log_likelihoods(flow_classifier, feats, prototypes, device)
            unknown_scores.append(lp.max(dim=1).values)
    unknown_scores = torch.cat(unknown_scores).numpy()

    separation = known_scores.mean() - unknown_scores.mean()
    total_range = max(known_scores.max(), unknown_scores.max()) - min(known_scores.min(), unknown_scores.min())
    overlap = max(0, min(known_scores.max(), unknown_scores.max()) - max(known_scores.min(), unknown_scores.min()))
    overlap_pct = overlap / (total_range + 1e-8) * 100

    print(f"\n  Flow max log-likelihood:")
    print(f"    Known:   mean={known_scores.mean():.2f} std={known_scores.std():.2f}")
    print(f"    Unknown: mean={unknown_scores.mean():.2f} std={unknown_scores.std():.2f}")
    print(f"    Separation: {separation:.2f}")
    print(f"    Distribution overlap: {overlap_pct:.1f}%")

    # AUC-ROC
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(known_scores)), np.zeros(len(unknown_scores))])
    scores = np.concatenate([known_scores, unknown_scores])
    auc = roc_auc_score(labels, scores)
    print(f"    AUC-ROC: {auc:.4f}")
    if auc < 0.6:
        print(f"    *** POOR: flow cannot discriminate known/unknown ***")
    elif auc < 0.8:
        print(f"    MODERATE")
    else:
        print(f"    GOOD")

    return {'auc': auc, 'separation': separation, 'overlap_pct': overlap_pct}


# ============================================================
# Diagnosis 4: Flow capacity
# ============================================================
def diagnose_flow_capacity(flow_classifier, device):
    """Analyze flow architecture capacity."""
    print(f"\n{'='*60}")
    print(f"  Flow Capacity Analysis")
    print(f"{'='*60}")

    flow = flow_classifier.flow
    n_layers = len(flow.coupling_layers)
    D = flow.input_dim
    total_params = sum(p.numel() for p in flow.parameters())

    # Get hidden dims from first layer
    first_layer = flow.coupling_layers[0]
    hidden_info = []
    for layer in first_layer.s1_net:
        if isinstance(layer, nn.Linear):
            hidden_info.append(f"{layer.in_features}→{layer.out_features}")

    max_scale = np.exp(flow.s_clamp_max) ** n_layers

    print(f"    Coupling layers: {n_layers}")
    print(f"    Input dim: {D}")
    print(f"    Hidden: {hidden_info}")
    print(f"    s_clamp_max: {flow.s_clamp_max}")
    print(f"    Total params: {total_params:,}")
    print(f"    Max cumulative scale: {max_scale:.0f}")

    issues = []
    if n_layers < 6:
        issues.append(f"Too few layers ({n_layers} < 6)")
    if total_params < 100000:
        issues.append(f"Too few params ({total_params:,} < 100K)")

    if issues:
        print(f"\n  Issues: {issues}")
    else:
        print(f"\n  Capacity appears adequate.")

    return {'n_layers': n_layers, 'total_params': total_params,
            'max_scale': max_scale, 'issues': issues}


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment_dir = 'experiment/yamnet_fewshot_osr4'

    base_classes = [0, 1, 2, 3, 4, 5]
    unknown_classes = [6, 7, 8, 9]

    print("Loading model and caches...")
    flow_classifier, train_cache, test_cache = load_trained_model(experiment_dir, device)

    # Compute prototypes
    prototypes = []
    for c in base_classes:
        feats = train_cache.get_class_features(c)
        prototypes.append(feats[:50].mean(dim=0))
    prototypes = torch.stack(prototypes)

    results = {}

    # 1. Flow capacity
    results['capacity'] = diagnose_flow_capacity(flow_classifier, device)

    # 2. Z-space Gaussianity for known classes
    print("\nComputing z-space for known classes...")
    z_correct = {}
    for i, c in enumerate(base_classes):
        feats = train_cache.get_class_features(c)
        z_correct[c] = compute_z_for_class(flow_classifier, feats, prototypes[i], device)

    all_z_known = torch.cat(list(z_correct.values()), dim=0)
    results['gaussianity_known'] = diagnose_gaussianity(all_z_known, "Known (z under correct proto)")

    # 3. Z-space for unknown classes
    print("\nComputing z-space for unknown classes...")
    z_unknown = {}
    for c in unknown_classes:
        feats = test_cache.get_class_features(c)
        if len(feats) > 0:
            dists = torch.cdist(feats[:500], prototypes)
            nearest_proto = dists.argmin(dim=1)
            z_parts = []
            for pi in range(len(base_classes)):
                mask = nearest_proto == pi
                if mask.any():
                    z_parts.append(compute_z_for_class(
                        flow_classifier, feats[:500][mask], prototypes[pi], device))
            if z_parts:
                z_unknown[c] = torch.cat(z_parts, dim=0)

    if z_unknown:
        all_z_unknown = torch.cat(list(z_unknown.values()), dim=0)
        results['gaussianity_unknown'] = diagnose_gaussianity(
            all_z_unknown, "Unknown (z under nearest base proto)")

    # 4. Inter-class separation
    results['separation'] = diagnose_separation(z_correct, "Known classes")

    # 5. OSR discrimination
    results['osr'] = diagnose_osr_discrimination(
        flow_classifier, train_cache, test_cache,
        base_classes, unknown_classes, device)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")

    gk = results.get('gaussianity_known', {})
    print(f"\n  Known z-space Gaussianity:")
    print(f"    ||z||^2/D ratio: {gk.get('mean_ratio', 'N/A'):.3f} (ideal=1.0)")
    print(f"    z_std avg:       {gk.get('std_avg', 'N/A'):.3f} (ideal=1.0)")
    print(f"    Shapiro-Wilk p:  {gk.get('sw_avg_p', 'N/A'):.4f} (ideal>0.05)")

    sep = results.get('separation', {})
    print(f"\n  Inter-class Separation:")
    print(f"    Inter/Intra ratio: {sep.get('inter_intra_ratio', 'N/A'):.3f} (want>2)")

    osr = results.get('osr', {})
    print(f"\n  OSR Discrimination:")
    print(f"    AUC-ROC:     {osr.get('auc', 'N/A'):.4f}")
    print(f"    Separation:  {osr.get('separation', 'N/A'):.2f}")
    print(f"    Overlap:     {osr.get('overlap_pct', 'N/A'):.1f}%")

    # Save
    save_path = os.path.join(experiment_dir, 'flow_diagnosis.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {save_path}")


if __name__ == "__main__":
    main()
