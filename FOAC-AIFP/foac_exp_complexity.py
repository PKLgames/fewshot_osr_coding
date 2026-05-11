#!/usr/bin/env python3
"""
foac_exp_complexity.py — Model Complexity Analysis

Computes MACs (FLOPs), parameter count, and average inference time
for FOAC-AIFP model and comparison configurations.

Usage:
  cd /coding/FOAC-AIFP
  python foac_exp_complexity.py --config tau22_aligned.yml
  python foac_exp_complexity.py --config tau22_aligned.yml --all_ablation
"""

import os
import sys
import json
import argparse
import time
from functools import partial

import torch
import torch.nn as nn
import numpy as np
import yaml

from trainers import trainer
from models.Network import My_Net


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


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_macs(model, args, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Estimate MACs using thop if available, otherwise use manual estimation.
    """
    try:
        from thop import profile, clever_format
        # Create dummy input matching the model's expected shape
        # Episode format: support, query, open_support, open_query
        n_ways = args.n_ways
        n_shots = args.n_shots
        n_queries = args.n_queries
        n_open = args.n_open_ways
        seq_len = 52  # temporal frames from mel-spectrogram

        # Dummy waveform input (batch, time) — will be converted to mel-spec internally
        # Actually we need to feed through encode() which expects raw audio
        # Use a simpler approach: profile just the encoder + task modules
        dummy_audio = torch.randn(1, 16000).to(device)  # 1 second at 16kHz

        # Profile the encoder part
        macs_encoder, params_encoder = profile(
            model.encoder, inputs=(dummy_audio.unsqueeze(0).repeat(1, 3, 1, 1),),
            verbose=False
        )
        return macs_encoder, params_encoder
    except ImportError:
        print("thop not installed. Install with: pip install thop")
        return None, None


def estimate_macs_manual(model, args):
    """
    Manual MACs estimation by summing layer-wise operations.
    Fallback when thop is not available.
    """
    total_macs = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # MACs for Conv2d = K^2 * C_in * C_out * H_out * W_out
            # Approximate with input size
            out_h, out_w = 1, 1  # unknown exact size, use minimum
            total_macs += (module.kernel_size[0] * module.kernel_size[1] *
                          module.in_channels * module.out_channels)
        elif isinstance(module, nn.Linear):
            total_macs += module.in_features * module.out_features
    return total_macs


def measure_inference_time(model, args, device=None, n_runs=100, warmup=20):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Measure average inference time (AIT) per episode.
    """
    model.eval()

    n_ways = args.n_ways
    n_shots = args.n_shots
    n_queries = args.n_queries
    n_open = args.n_open_ways

    # Create dummy episode data matching training format
    # Each sample: (batch=1, channels, height, width)
    # After mel-spec: (1, 1, 128, 52) -> repeated to (1, 3, 128, 52)
    audio_len = 16000  # 1 second

    support_data = torch.randn(1, n_ways * n_shots, 1, audio_len).to(device)
    query_data = torch.randn(1, n_ways * n_queries, 1, audio_len).to(device)
    suppopen_data = torch.randn(1, n_open * n_shots, 1, audio_len).to(device)
    openset_data = torch.randn(1, n_open * n_queries, 1, audio_len).to(device)

    support_label = torch.arange(n_ways).repeat_interleave(n_shots).unsqueeze(0).to(device)
    query_label = torch.arange(n_ways).repeat_interleave(n_queries).unsqueeze(0).to(device)
    suppopen_label = torch.arange(n_open).repeat_interleave(n_shots).unsqueeze(0).to(device)
    openset_label = torch.arange(n_open).repeat_interleave(n_queries).unsqueeze(0).to(device)
    supp_idx = torch.arange(n_ways).unsqueeze(0).to(device)
    open_idx = torch.arange(n_open).unsqueeze(0).to(device)

    the_img = [support_data.squeeze(), query_data.squeeze(),
               suppopen_data.squeeze(), openset_data.squeeze()]
    the_label = tuple(x.squeeze() for x in (support_label, query_label,
                                             suppopen_label, openset_label))

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(the_img, the_label, supp_idx, open_idx, test=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(the_img, the_label, supp_idx, open_idx, test=True)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'median_ms': float(np.median(times) * 1000),
        'n_runs': n_runs,
    }


def analyze_model(args, name, use_ciam=True, use_pam=True, use_npm=True):
    """Run full complexity analysis on one model configuration."""
    print(f"\n  Analyzing: {name}")

    model = My_Net(args=args, mode='train',
                   use_ciam=use_ciam, use_pam=use_pam, use_npm=use_npm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load pretrained weights if available
    if os.path.exists(args.pretrained_model_path):
        ckpt = torch.load(args.pretrained_model_path, weights_only=False)
        state_dict = ckpt.get('feature_params', ckpt.get('params', ckpt))
        model.load_state_dict(state_dict, strict=False)
        model.init_representation(ckpt)

    model.eval()

    # Parameters
    total_params, trainable_params = count_parameters(model)

    # MACs
    macs, _ = measure_macs(model, args)
    if macs is None:
        macs = estimate_macs_manual(model, args)

    # Inference time
    ait = measure_inference_time(model, args)

    result = {
        'name': name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'macs': macs,
        'ait_mean_ms': ait['mean_ms'],
        'ait_std_ms': ait['std_ms'],
        'ait_median_ms': ait['median_ms'],
    }

    print(f"    Params: {total_params:,} total, {trainable_params:,} trainable")
    print(f"    MACs: {macs:,}" if macs else "    MACs: N/A")
    print(f"    AIT: {ait['mean_ms']:.2f} +/- {ait['std_ms']:.2f} ms")

    return result


def main():
    parser = argparse.ArgumentParser(description='FOAC Complexity Analysis')
    parser.add_argument('--config', type=str, default='tau22_aligned.yml')
    parser.add_argument('--all_ablation', action='store_true',
                        help='Also measure all ablation configurations')
    cl_args = parser.parse_args()

    args = load_config(cl_args.config)
    all_results = []

    # Full model
    all_results.append(analyze_model(args, 'Full_Model'))

    if cl_args.all_ablation:
        configs = [
            ('wo_CIAM', False, True, True),
            ('wo_PAM', True, False, True),
            ('wo_NPM', True, True, False),
            ('Baseline', False, False, False),
        ]
        for name, ciam, pam, npm in configs:
            all_results.append(analyze_model(args, name, ciam, pam, npm))

    # Save results
    results_path = 'foac_exp_complexity_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComplexity results saved to {results_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPLEXITY COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Model':<20s} {'Params':>12s} {'MACs':>14s} {'AIT(ms)':>12s}")
    for r in all_results:
        p = f"{r['total_params']:,}"
        m = f"{r['macs']:,.0f}" if r['macs'] else 'N/A'
        t = f"{r['ait_mean_ms']:.2f}"
        print(f"  {r['name']:<20s} {p:>12s} {m:>14s} {t:>12s}")


if __name__ == '__main__':
    main()
