#!/usr/bin/env python3
"""
episodic_exp_openness.py — Openness Sweep for Episodic Trainer

Tests AUROC/OSR performance with varying numbers of unknown classes
(n_open_ways = 1, 2, 3, 4). Fixed N-way and K-shot.

Usage:
  cd /coding
  python episodic_exp_openness.py --dataset TAU22
  python episodic_exp_openness.py --quick
"""

import os
import json
import argparse
import random
import numpy as np
import torch

from episodic_trainer import (
    FeatureExtractor, FeatureCache, EpisodicFlowClassifier,
    EpisodicTrainer, FewShotEvaluator, OSRCalibrator, TAUDataset,
)


OPEN_WAY_VALUES = [1, 2, 3, 4]


def find_pretrained():
    for p in ['experiment/yamnet_fewshot_osr22/base_feature_extractor.pth',
              'experiment/yamnet_fewshot_osr13/base_feature_extractor.pth']:
        if os.path.exists(p):
            return p
    return None


def run_openness(train_cache, calib_cache, test_cache, feature_extractor,
                 base_classes, all_unknown, n_open, N_way=6, K_shot=5,
                 num_episodes=3000, device='cuda'):
    """Run with a subset of unknown classes."""
    unknown_classes = all_unknown[:n_open]
    experiment_dir = f'experiment/episodic_openness/open{n_open}'

    print(f"\n  n_open={n_open}, unknown={unknown_classes}")

    flow_classifier = EpisodicFlowClassifier(
        input_dim=64, condition_dim=64, use_flow_transform=False)

    ood_features = {}
    for c in unknown_classes:
        feats = test_cache.get_class_features(c)
        if len(feats) > 0:
            ood_features[c] = feats

    trainer = EpisodicTrainer(
        feature_extractor=feature_extractor,
        flow_classifier=flow_classifier,
        train_cache=train_cache, calib_cache=calib_cache, test_cache=test_cache,
        base_classes=base_classes, unknown_classes=unknown_classes,
        ood_features_by_class=ood_features,
        N_way=N_way, K_shot=K_shot, Q_query=15,
        lr=5e-5, gradient_accum_steps=1, device=device,
    )

    trainer.train(
        num_episodes=num_episodes, eval_every=500,
        num_val_episodes=10, save_dir=experiment_dir, resume=True)

    # Evaluate OSR with all methods
    osr_results = {}
    for method in ['anti_prototype', 'feature_mahalanobis']:
        try:
            calibrator = OSRCalibrator(
                trainer.flow_classifier, calib_cache,
                base_classes, unknown_classes, device)
            calibrator.calibrate(target_fpr=0.05, K_shot=50, method=method)

            test_calibrator = OSRCalibrator(
                trainer.flow_classifier, test_cache,
                base_classes, unknown_classes, device)
            osr_results[method] = test_calibrator.evaluate_osr(
                K_shot=50, num_rounds=10,
                method=method, recalibrate_per_round=True)
        except Exception as e:
            print(f"    OSR {method} failed: {e}")

    return {'osr': osr_results, 'n_open': n_open, 'unknown_classes': unknown_classes}


def main():
    parser = argparse.ArgumentParser(description='Episodic Openness Sweep')
    parser.add_argument('--dataset', choices=['TAU22', 'TAU19'], default='TAU22')
    parser.add_argument('--open_ways', type=int, nargs='*', default=OPEN_WAY_VALUES)
    parser.add_argument('--num_episodes', type=int, default=3000)
    parser.add_argument('--quick', action='store_true')
    cl_args = parser.parse_args()

    if cl_args.quick:
        cl_args.open_ways = [2, 4]
        cl_args.num_episodes = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = find_pretrained()
    if not pretrained_path:
        print("No pretrained backbone found.")
        return

    feature_extractor = FeatureExtractor(pretrained_path=pretrained_path)
    feature_extractor = feature_extractor.to(device).freeze()

    # Extract features
    print("Extracting features...")
    train_dataset = TAUDataset(split='train')
    calib_dataset = TAUDataset(split='calib')
    test_dataset = TAUDataset(split='test')

    train_cache = FeatureCache()
    train_cache.extract_and_cache(feature_extractor, train_dataset, 'train', device, batch_size=64)
    calib_cache = FeatureCache()
    calib_cache.extract_and_cache(feature_extractor, calib_dataset, 'calib', device, batch_size=64)
    test_cache = FeatureCache()
    test_cache.extract_and_cache(feature_extractor, test_dataset, 'test', device, batch_size=64)

    base_classes = [0, 1, 2, 3, 4, 5]
    all_unknown = [6, 7, 8, 9]

    all_results = {}
    for n_open in cl_args.open_ways:
        r = run_openness(
            train_cache, calib_cache, test_cache, feature_extractor,
            base_classes, all_unknown, n_open,
            num_episodes=cl_args.num_episodes, device=device)
        if r:
            all_results[f'open{n_open}'] = r

    # Save
    save_path = 'episodic_exp_openness_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    # Print table
    print(f"\n{'='*60}")
    print("OPENNESS SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"  {'OpenWays':>10s} {'TNR':>8s} {'TPR':>8s} {'OSR':>8s}")
    for key, r in sorted(all_results.items()):
        osr = r.get('osr', {})
        best = max(osr.values(), key=lambda x: x.get('osr_score', 0)) if osr else {}
        print(f"  {key:>10s} "
              f"{best.get('known_tnr', 0):>7.2%} "
              f"{best.get('unknown_tpr', 0):>7.2%} "
              f"{best.get('osr_score', 0):>7.2%}")


if __name__ == '__main__':
    main()
