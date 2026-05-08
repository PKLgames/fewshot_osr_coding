#!/usr/bin/env python3
"""
episodic_exp_complexity.py — Complexity Analysis for Episodic Trainer

Measures MACs, parameter count, and inference time for
EpisodicFlowClassifier and its ablation variants.

Usage:
  cd /coding
  python episodic_exp_complexity.py
  python episodic_exp_complexity.py --all_ablation
"""

import os
import json
import argparse
import time

import torch
import torch.nn as nn
import numpy as np

from episodic_trainer import (
    EpisodicFlowClassifier, FeatureExtractor, EpisodeSampler, FeatureCache,
    TAUDataset,
)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_macs_thop(model, input_dim=64, device='cuda'):
    """Estimate MACs using thop."""
    try:
        from thop import profile
        dummy_feat = torch.randn(1, input_dim).to(device)
        dummy_proto = torch.randn(5, input_dim).to(device)
        macs, params = profile(model, inputs=(dummy_feat, dummy_proto), verbose=False)
        return macs
    except ImportError:
        return None


def measure_inference_time(classifier, input_dim=64, N_way=6, device='cuda',
                           n_runs=200, warmup=50):
    """Measure average inference time per episode."""
    classifier.eval()

    # Create dummy episode data
    support_feats = torch.randn(N_way * 5, input_dim).to(device)
    query_feats = torch.randn(N_way * 15, input_dim).to(device)
    prototypes = torch.randn(N_way, input_dim).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = classifier.classify(query_feats, prototypes)
    torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = classifier.classify(query_feats, prototypes)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'median_ms': float(np.median(times) * 1000),
    }


def analyze_config(name, use_flow=True, use_ood=True, use_recip=True,
                   use_thresh=True, input_dim=64, device='cuda'):
    """Analyze one model configuration."""
    print(f"\n  Analyzing: {name}")

    model = EpisodicFlowClassifier(
        input_dim=input_dim, condition_dim=input_dim,
        use_flow_transform=use_flow, use_ood_head=use_ood,
        use_reciprocal=use_recip, use_threshold=use_thresh,
    ).to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    macs = measure_macs_thop(model, input_dim, device)
    ait = measure_inference_time(model, input_dim, device=device)

    result = {
        'name': name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'macs': macs,
        'ait_mean_ms': ait['mean_ms'],
        'ait_std_ms': ait['std_ms'],
        'ait_median_ms': ait['median_ms'],
    }

    print(f"    Params: {total_params:,}")
    print(f"    MACs:   {macs:,}" if macs else "    MACs:   N/A (thop not installed)")
    print(f"    AIT:    {ait['mean_ms']:.2f} +/- {ait['std_ms']:.2f} ms")

    return result


def main():
    parser = argparse.ArgumentParser(description='Episodic Complexity Analysis')
    parser.add_argument('--all_ablation', action='store_true')
    cl_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    # Full model
    all_results.append(analyze_config('Full_Model', device=device))

    if cl_args.all_ablation:
        configs = [
            ('wo_Flow_Transform', False, True,  True,  True),
            ('wo_OOD_Head',       True,  False, True,  True),
            ('wo_Reciprocal',     True,  True,  False, True),
            ('wo_Threshold',      True,  True,  True,  False),
            ('Baseline',          False, False, False, False),
        ]
        for name, flow, ood, recip, thresh in configs:
            all_results.append(analyze_config(
                name, flow, ood, recip, thresh, device=device))

    # Save
    save_path = 'episodic_exp_complexity_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    # Print table
    print(f"\n{'='*70}")
    print("COMPLEXITY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Model':<20s} {'Params':>12s} {'MACs':>14s} {'AIT(ms)':>12s}")
    for r in all_results:
        p = f"{r['total_params']:,}"
        m = f"{r['macs']:,.0f}" if r['macs'] else 'N/A'
        t = f"{r['ait_mean_ms']:.2f}"
        print(f"  {r['name']:<20s} {p:>12s} {m:>14s} {t:>12s}")


if __name__ == '__main__':
    main()
