#!/usr/bin/env python3
"""
episodic_exp_openness.py — Openness Sweep for Episodic Trainer

Tests AUROC/OSR performance with varying numbers of unknown classes
(n_open_ways = 1, 2, 3, 4). Fixed N-way and K-shot.

Usage:
  cd /coding
  python episodic_exp_openness.py --dataset TAU22
  python episodic_exp_openness.py --dataset DCASE18 --fold all
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
    EpisodicTrainer, FewShotEvaluator, OSRCalibrator,
)
from utils.TAU22 import TAUDataset as TAU22Dataset
from utils.TAU19 import TAUDataset as TAU19Dataset
from utils.DCASE18 import DCASE18Dataset


OPEN_WAY_VALUES = [2, 4]
ALL_OSR_METHODS = ['anti_prototype', 'feature_mahalanobis', 'ood_head_osr23',
                    'ood_head_extended_v2', 'ood_head_cluster_v2', 'ood_head_fusion_v2']


def find_pretrained():
    for p in ['experiment/yamnet_fewshot_osr22/base_feature_extractor.pth',
              'experiment/yamnet_fewshot_osr13/base_feature_extractor.pth']:
        if os.path.exists(p):
            return p
    return None


def run_openness(train_cache, calib_cache, test_cache, feature_extractor,
                 base_classes, all_unknown, n_open, N_way=6, K_shot=5,
                 num_episodes=3000, device='cuda', output_root='experiment/episodic_exp',
                 osr_methods=None):
    """Run with a subset of unknown classes."""
    unknown_classes = all_unknown[:n_open]
    experiment_dir = f'{output_root}/openness/open{n_open}'

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
        lr=2e-4, gradient_accum_steps=1, device=device,
    )

    trainer.train(
        num_episodes=num_episodes, eval_every=500,
        num_val_episodes=10, save_dir=experiment_dir, resume=True)

    # Evaluate OSR with all methods
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

    return {'osr': osr_results, 'n_open': n_open, 'unknown_classes': unknown_classes}


def main():
    parser = argparse.ArgumentParser(description='Episodic Openness Sweep')
    parser.add_argument('--dataset', choices=['TAU22', 'TAU19', 'DCASE18'], default='TAU22')
    parser.add_argument('--open_ways', type=int, nargs='*', default=OPEN_WAY_VALUES)
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
        cl_args.open_ways = [2, 4]
        cl_args.num_episodes = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = find_pretrained()
    if not pretrained_path:
        print("No pretrained backbone found.")
        return

    feature_extractor = FeatureExtractor(pretrained_path=pretrained_path)
    feature_extractor = feature_extractor.to(device).freeze()

    if cl_args.dataset == 'DCASE18':
        from functools import partial
        base_classes = [0, 1, 2, 3, 4]
        all_unknown = [5, 6, 7, 8]
        n_way = 4
        folds = [1, 2, 3, 4] if cl_args.fold == 'all' else [int(cl_args.fold)]
    else:
        base_classes = [0, 1, 2, 3, 4, 5]
        all_unknown = [6, 7, 8, 9]
        n_way = 6
        folds = [None]

    all_results = {}
    for n_open in cl_args.open_ways:
        for fold in folds:
            # Extract features (per-fold for DCASE18)
            if cl_args.dataset == 'DCASE18':
                TAUDataset = partial(DCASE18Dataset, fold=fold)
                cache_name = f'dcase18_fold{fold}'
            else:
                TAUDataset = TAU19Dataset if cl_args.dataset == 'TAU19' else TAU22Dataset
                cache_name = cl_args.dataset.lower()

            print(f"Extracting features" + (f" (fold={fold})" if fold is not None else "") + "...")
            train_dataset = TAUDataset(split='train')
            calib_dataset = TAUDataset(split='calib')
            test_dataset = TAUDataset(split='test')

            train_cache = FeatureCache(dataset_name=cache_name)
            train_cache.extract_and_cache(feature_extractor, train_dataset, 'train', device, batch_size=64)
            calib_cache = FeatureCache(dataset_name=cache_name)
            calib_cache.extract_and_cache(feature_extractor, calib_dataset, 'calib', device, batch_size=64)
            test_cache = FeatureCache(dataset_name=cache_name)
            test_cache.extract_and_cache(feature_extractor, test_dataset, 'test', device, batch_size=64)

            fold_tag = f'_f{fold}' if fold is not None else ''
            r = run_openness(
                train_cache, calib_cache, test_cache, feature_extractor,
                base_classes, all_unknown, n_open, N_way=n_way,
                num_episodes=cl_args.num_episodes, device=device,
                output_root=os.path.join(cl_args.output_dir,
                    f'openness_fold{fold}' if fold is not None else ''),
                osr_methods=osr_methods)
            if r:
                all_results[f'open{n_open}{fold_tag}'] = r

    # Save
    os.makedirs(os.path.join(cl_args.output_dir, 'openness'), exist_ok=True)
    save_path = os.path.join(cl_args.output_dir, 'openness', 'results.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    # Print table
    print(f"\n{'='*60}")
    print("OPENNESS SWEEP SUMMARY")
    print(f"{'='*60}")
    if cl_args.dataset == 'DCASE18' and cl_args.fold == 'all':
        print(f"  {'OpenWays':>10s} {'AP':>10s} {'Mah':>10s} {'O23':>10s} {'ExtV2':>10s} {'CluV2':>10s} {'FusV2':>10s}")
        osr_keys = ['anti_prototype', 'feature_mahalanobis', 'ood_head_osr23',
                   'ood_head_extended_v2', 'ood_head_cluster_v2', 'ood_head_fusion_v2']
        open_keys = sorted(set(k.rsplit('_f', 1)[0] for k in all_results.keys()))
        for ok in open_keys:
            fold_vals = [all_results.get(f'{ok}_f{f}', {}) for f in [1,2,3,4]]
            fold_vals = [v for v in fold_vals if v]
            if not fold_vals:
                continue
            row = f"  {ok:>10s}"
            for osk in osr_keys:
                vals = [v.get('osr', {}).get(osk, {}).get('osr_score', 0) for v in fold_vals]
                row += f" {np.mean(vals):>9.2%}"
            print(row)
    else:
        print(f"  {'OpenWays':>10s} {'AP':>7s} {'Mah':>7s} {'O23':>7s} {'ExtV2':>7s} {'CluV2':>7s} {'FusV2':>7s}")
        for key, r in sorted(all_results.items()):
            osr = r.get('osr', {})
            osr_ap  = osr.get('anti_prototype', {}).get('osr_score', 0)
            osr_mah = osr.get('feature_mahalanobis', {}).get('osr_score', 0)
            osr_o23 = osr.get('ood_head_osr23', {}).get('osr_score', 0)
            osr_ext = osr.get('ood_head_extended_v2', {}).get('osr_score', 0)
            osr_clu = osr.get('ood_head_cluster_v2', {}).get('osr_score', 0)
            osr_fv2 = osr.get('ood_head_fusion_v2', {}).get('osr_score', 0)
            print(f"  {key:>10s} {osr_ap:>6.2%} {osr_mah:>6.2%} {osr_o23:>6.2%} {osr_ext:>6.2%} {osr_clu:>6.2%} {osr_fv2:>6.2%}")


if __name__ == '__main__':
    main()
