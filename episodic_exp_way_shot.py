#!/usr/bin/env python3
"""
episodic_exp_way_shot.py — Way-Shot Parameter Sweep for Episodic Trainer

Tests performance across N-way and K-shot combinations.
Uses pre-extracted features (Phase 1) from a trained backbone,
then sweeps Phase 2 (episodic training) and Phase 3 (evaluation).

Usage:
  cd /coding
  python episodic_exp_way_shot.py --dataset TAU22
  python episodic_exp_way_shot.py --quick   # debug mode
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch

from episodic_trainer import (
    FeatureExtractor, FeatureCache, EpisodicFlowClassifier,
    EpisodicTrainer, FewShotEvaluator, OSRCalibrator,
)
from utils.TAU22 import TAUDataset as TAU22Dataset
from utils.TAU19 import TAUDataset as TAU19Dataset


WAY_VALUES = [5, 6]
SHOT_VALUES = [1, 5, 10]


def find_pretrained():
    candidates = [
        'experiment/yamnet_fewshot_osr22/base_feature_extractor.pth',
        'experiment/yamnet_fewshot_osr13/base_feature_extractor.pth',
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_features(feature_extractor, device, dataset_name='TAU22'):
    """Extract and cache features for train/calib/test."""
    TAUDataset = TAU19Dataset if dataset_name == 'TAU19' else TAU22Dataset
    train_dataset = TAUDataset(split='train')
    calib_dataset = TAUDataset(split='calib')
    test_dataset = TAUDataset(split='test')

    train_cache = FeatureCache(dataset_name=dataset_name.lower())
    train_cache.extract_and_cache(feature_extractor, train_dataset, 'train', device, batch_size=64)

    calib_cache = FeatureCache(dataset_name=dataset_name.lower())
    calib_cache.extract_and_cache(feature_extractor, calib_dataset, 'calib', device, batch_size=64)

    test_cache = FeatureCache(dataset_name=dataset_name.lower())
    test_cache.extract_and_cache(feature_extractor, test_dataset, 'test', device, batch_size=64)

    return train_cache, calib_cache, test_cache


ALL_OSR_METHODS = ['anti_prototype', 'feature_mahalanobis', 'ood_head_osr23',
                    'ood_head_extended_v2', 'ood_head_cluster_v2', 'ood_head_fusion_v2']


def run_way_shot(train_cache, calib_cache, test_cache, feature_extractor,
                 base_classes, unknown_classes, n_way, k_shot,
                 num_episodes=3000, device='cuda', output_root='experiment/episodic_exp',
                 osr_methods=None):
    """Run episodic training + evaluation for a specific N-way K-shot."""
    # Adjust Q_query based on available samples per class
    min_class_samples = min(
        len(train_cache.get_class_features(c)) for c in base_classes)
    q_query = min(15, min_class_samples - k_shot)
    if q_query < 5:
        print(f"    SKIP: not enough samples for {n_way}w{k_shot}s (need {k_shot + 5}, have {min_class_samples})")
        return None

    experiment_dir = f'{output_root}/wayshot/way{n_way}_shot{k_shot}'

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
        N_way=n_way, K_shot=k_shot, Q_query=q_query,
        lr=2e-4, gradient_accum_steps=1, device=device,
    )

    trainer.train(
        num_episodes=num_episodes, eval_every=500,
        num_val_episodes=10, save_dir=experiment_dir, resume=True)

    # Evaluate
    evaluator = FewShotEvaluator(trainer.flow_classifier, device)
    result = {}
    for k in [k_shot]:  # Evaluate at the training shot count
        try:
            res = evaluator.evaluate(train_cache, base_classes,
                                     N_way=n_way, K_shot=k)
            result['acc'] = res['mean_acc']
            result['ci95'] = res['ci95']
        except Exception as e:
            print(f"    Eval failed: {e}")
            result['acc'] = 0
            result['ci95'] = 0

    # OSR evaluation
    if osr_methods is None:
        osr_methods = ALL_OSR_METHODS
    osr_results = {}
    for method in osr_methods:
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
    result['osr'] = osr_results

    return result


def main():
    parser = argparse.ArgumentParser(description='Episodic Way-Shot Sweep')
    parser.add_argument('--dataset', choices=['TAU22', 'TAU19'], default='TAU22')
    parser.add_argument('--ways', type=int, nargs='*', default=WAY_VALUES)
    parser.add_argument('--shots', type=int, nargs='*', default=SHOT_VALUES)
    parser.add_argument('--num_episodes', type=int, default=3000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--osr_methods', type=str, default='all',
                        help='Comma-separated OSR methods, or "all" for all 6')
    parser.add_argument('--output_dir', type=str, default='experiment/episodic_exp',
                        help='Root output directory for all results')
    cl_args = parser.parse_args()

    if cl_args.osr_methods == 'all':
        osr_methods = ALL_OSR_METHODS
    else:
        osr_methods = [m.strip() for m in cl_args.osr_methods.split(',')]

    if cl_args.quick:
        cl_args.ways = [5, 6]
        cl_args.shots = [1, 5]
        cl_args.num_episodes = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load backbone
    pretrained_path = find_pretrained()
    if not pretrained_path:
        print("No pretrained backbone found. Run episodic_trainer.py first.")
        return

    print(f"Using backbone: {pretrained_path}")
    feature_extractor = FeatureExtractor(pretrained_path=pretrained_path)
    feature_extractor = feature_extractor.to(device).freeze()

    # Extract features (once, reused for all configs)
    print("Extracting features...")
    train_cache, calib_cache, test_cache = get_features(feature_extractor, device,
                                                         dataset_name=cl_args.dataset)

    base_classes = [0, 1, 2, 3, 4, 5]
    unknown_classes = [6, 7, 8, 9]

    all_results = {}
    for n_way in cl_args.ways:
        for k_shot in cl_args.shots:
            key = f"{n_way}w{k_shot}s"
            print(f"\n--- {key} ---")
            result = run_way_shot(
                train_cache, calib_cache, test_cache, feature_extractor,
                base_classes, unknown_classes, n_way, k_shot,
                num_episodes=cl_args.num_episodes, device=device,
                output_root=cl_args.output_dir, osr_methods=osr_methods)
            if result:
                all_results[key] = result

    # Save
    os.makedirs(os.path.join(cl_args.output_dir, 'wayshot'), exist_ok=True)
    save_path = os.path.join(cl_args.output_dir, 'wayshot', 'results.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    # Print table
    print(f"\n{'='*60}")
    print("WAY-SHOT SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Config':<10s} {'Acc':>7s} {'CI95':>7s} {'AP':>7s} {'Mah':>7s} {'O23':>7s} {'ExtV2':>7s} {'CluV2':>7s} {'FusV2':>7s}")
    for key, r in sorted(all_results.items()):
        acc = r.get('acc', 0)
        ci = r.get('ci95', 0)
        osr = r.get('osr', {})
        osr_ap  = osr.get('anti_prototype', {}).get('osr_score', 0)
        osr_mah = osr.get('feature_mahalanobis', {}).get('osr_score', 0)
        osr_o23 = osr.get('ood_head_osr23', {}).get('osr_score', 0)
        osr_ext = osr.get('ood_head_extended_v2', {}).get('osr_score', 0)
        osr_clu = osr.get('ood_head_cluster_v2', {}).get('osr_score', 0)
        osr_fv2 = osr.get('ood_head_fusion_v2', {}).get('osr_score', 0)
        print(f"  {key:<10s} {acc:>6.2%} {ci:>6.2%} {osr_ap:>6.2%} {osr_mah:>6.2%} {osr_o23:>6.2%} {osr_ext:>6.2%} {osr_clu:>6.2%} {osr_fv2:>6.2%}")


if __name__ == '__main__':
    main()
