#!/usr/bin/env python3
"""
episodic_exp_way_shot.py — Way-Shot Parameter Sweep for Episodic Trainer

Tests performance across N-way and K-shot combinations.
Uses pre-extracted features (Phase 1) from a trained backbone,
then sweeps Phase 2 (episodic training) and Phase 3 (evaluation).

Usage:
  cd /coding
  python episodic_exp_way_shot.py --dataset TAU22
  python episodic_exp_way_shot.py --dataset DCASE18 --fold all
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
from utils.DCASE18 import DCASE18Dataset


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


def get_features(feature_extractor, device, dataset_name='TAU22', fold=1):
    """Extract and cache features for train/calib/test."""
    if dataset_name == 'DCASE18':
        from functools import partial
        TAUDataset = partial(DCASE18Dataset, fold=fold)
        cache_name = f'dcase18_fold{fold}'
    elif dataset_name == 'TAU19':
        TAUDataset = TAU19Dataset
        cache_name = dataset_name.lower()
    else:
        TAUDataset = TAU22Dataset
        cache_name = dataset_name.lower()

    train_dataset = TAUDataset(split='train')
    calib_dataset = TAUDataset(split='calib')
    test_dataset = TAUDataset(split='test')

    train_cache = FeatureCache(dataset_name=cache_name)
    train_cache.extract_and_cache(feature_extractor, train_dataset, 'train', device, batch_size=64)

    calib_cache = FeatureCache(dataset_name=cache_name)
    calib_cache.extract_and_cache(feature_extractor, calib_dataset, 'calib', device, batch_size=64)

    test_cache = FeatureCache(dataset_name=cache_name)
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
    parser.add_argument('--dataset', choices=['TAU22', 'TAU19', 'DCASE18'], default='TAU22')
    parser.add_argument('--ways', type=int, nargs='*', default=WAY_VALUES)
    parser.add_argument('--shots', type=int, nargs='*', default=SHOT_VALUES)
    parser.add_argument('--num_episodes', type=int, default=3000)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--osr_methods', type=str, default='all',
                        help='Comma-separated OSR methods, or "all" for all 6')
    parser.add_argument('--output_dir', type=str, default='experiment/episodic_exp',
                        help='Root output directory for all results')
    parser.add_argument('--fold', type=str, default='1',
                        help='Fold for DCASE18: 1-4 or "all" for 4-fold CV (default: 1)')
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

    if cl_args.dataset == 'DCASE18':
        base_classes = [0, 1, 2, 3, 4]
        unknown_classes = [5, 6, 7, 8]
        folds = [1, 2, 3, 4] if cl_args.fold == 'all' else [int(cl_args.fold)]
    else:
        base_classes = [0, 1, 2, 3, 4, 5]
        unknown_classes = [6, 7, 8, 9]
        folds = [None]

    all_results = {}
    for n_way in cl_args.ways:
        for k_shot in cl_args.shots:
            for fold in folds:
                # Extract features (per-fold for DCASE18)
                print(f"Extracting features" + (f" (fold={fold})" if fold is not None else "") + "...")
                train_cache, calib_cache, test_cache = get_features(
                    feature_extractor, device,
                    dataset_name=cl_args.dataset,
                    fold=(fold if fold is not None else 1))

                key = f"{n_way}w{k_shot}s"
                fold_tag = f'_f{fold}' if fold is not None else ''
                print(f"\n--- {key}{fold_tag} ---")
                result = run_way_shot(
                    train_cache, calib_cache, test_cache, feature_extractor,
                    base_classes, unknown_classes, n_way, k_shot,
                    num_episodes=cl_args.num_episodes, device=device,
                    output_root=os.path.join(cl_args.output_dir,
                        f'wayshot_fold{fold}' if fold is not None else ''),
                    osr_methods=osr_methods)
                if result:
                    all_results[f"{key}{fold_tag}"] = result

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
    if cl_args.dataset == 'DCASE18' and cl_args.fold == 'all':
        print(f"  {'Config':<10s} {'Acc':>12s} {'AP':>8s} {'Mah':>8s} {'O23':>8s} {'ExtV2':>8s} {'CluV2':>8s} {'FusV2':>8s}")
        way_shot_combos = sorted(set(k.rsplit('_f', 1)[0] for k in all_results.keys()))
        for ws in way_shot_combos:
            fold_vals = [all_results.get(f'{ws}_f{f}', {}) for f in [1,2,3,4]]
            fold_vals = [v for v in fold_vals if v]
            if not fold_vals:
                continue
            accs = [v.get('acc', 0) for v in fold_vals]
            acc_m, acc_s = np.mean(accs), np.std(accs)
            row = f"  {ws:<10s} {acc_m:>6.2%}±{acc_s:<4.2%}"
            osr_keys_short = ['anti_prototype', 'feature_mahalanobis', 'ood_head_osr23',
                             'ood_head_extended_v2', 'ood_head_cluster_v2', 'ood_head_fusion_v2']
            for ok in osr_keys_short:
                vals = [v.get('osr', {}).get(ok, {}).get('osr_score', 0) for v in fold_vals]
                row += f" {np.mean(vals):>7.2%}"
            print(row)
    else:
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
