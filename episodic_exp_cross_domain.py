#!/usr/bin/env python3
"""
episodic_exp_cross_domain.py — Cross-Domain Evaluation for Episodic Trainer

Train on one dataset (TAU22), extract features from another (TAU19),
evaluate few-shot + OSR performance under domain shift.

The key insight: the backbone (YAMNet) is shared, so we can extract
features from any dataset and evaluate the trained classifier on them.

Usage:
  cd /coding
  python episodic_exp_cross_domain.py --source TAU22 --target TAU19
  python episodic_exp_cross_domain.py --all
"""

import os
import json
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

from episodic_trainer import (
    FeatureExtractor, FeatureCache, EpisodicFlowClassifier,
    EpisodeSampler, FewShotEvaluator, OSRCalibrator,
)
from utils.TAU22 import TAUDataset as TAU22Dataset
from utils.TAU19 import TAUDataset as TAU19Dataset


def get_dataset_cls(name):
    return TAU22Dataset if name == 'TAU22' else TAU19Dataset


def find_best_checkpoint(experiment_dir):
    """Find the best saved model in an experiment directory."""
    final_path = os.path.join(experiment_dir, 'fewshot_final.pth')
    if os.path.exists(final_path):
        return final_path
    # Look for any checkpoint
    for f in sorted(os.listdir(experiment_dir)):
        if f.endswith('.pth') and 'checkpoint' in f:
            return os.path.join(experiment_dir, f)
    return None


def cross_domain_eval(source_name, target_name, base_classes, unknown_classes,
                      N_way=6, K_shot=5, feature_dim=64, device='cuda'):
    """
    1. Load classifier trained on source
    2. Extract features from target using same backbone
    3. Evaluate few-shot + OSR on target
    """
    print(f"\n{'='*60}")
    print(f"Cross-Domain: {source_name} -> {target_name}")
    print(f"{'='*60}")

    # --- Load trained classifier from source ---
    source_dirs = [
        f'experiment/yamnet_fewshot_osr22_relabel',
        f'experiment/yamnet_fewshot_osr22',
        f'experiment/yamnet_fewshot_osr13',
    ]
    ckpt_path = None
    for d in source_dirs:
        p = os.path.join(d, 'fewshot_final.pth')
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path is None:
        print(f"  No trained model found. Checked: {source_dirs}")
        return None

    print(f"  Loading classifier from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    flow_classifier = EpisodicFlowClassifier(
        input_dim=feature_dim, condition_dim=feature_dim,
        use_flow_transform=ckpt.get('config', {}).get('use_flow_transform', False),
    ).to(device)
    flow_classifier.load_state_dict(ckpt['flow_state_dict'])
    flow_classifier.eval()

    # --- Load backbone (same for both datasets) ---
    backbone_path = None
    for d in source_dirs:
        p = os.path.join(d, 'base_feature_extractor.pth')
        if os.path.exists(p):
            backbone_path = p
            break
    if backbone_path is None:
        print("  No backbone found.")
        return None

    feature_extractor = FeatureExtractor(pretrained_path=backbone_path)
    feature_extractor = feature_extractor.to(device).freeze()

    # --- Extract features from target dataset ---
    TargetDataset = get_dataset_cls(target_name)
    print(f"  Extracting features from {target_name}...")

    target_train = TargetDataset(split='train')
    target_calib = TargetDataset(split='calib')
    target_test = TargetDataset(split='test')

    train_cache = FeatureCache()
    train_cache.extract_and_cache(feature_extractor, target_train, 'train', device, batch_size=64)

    calib_cache = FeatureCache()
    calib_cache.extract_and_cache(feature_extractor, target_calib, 'calib', device, batch_size=64)

    test_cache = FeatureCache()
    test_cache.extract_and_cache(feature_extractor, target_test, 'test', device, batch_size=64)

    # --- Few-shot evaluation on target ---
    evaluator = FewShotEvaluator(flow_classifier, device)
    fewshot_results = {}

    for k in [1, 5]:
        try:
            res = evaluator.evaluate(train_cache, base_classes, N_way=N_way, K_shot=k)
            fewshot_results[f'base_{k}shot'] = res
            print(f"  Base {k}-shot: Acc={res['mean_acc']:.2%} +/- {res['ci95']:.2%}")
        except Exception as e:
            print(f"  Base {k}-shot eval failed: {e}")

    for k in [1, 5]:
        n_novel = min(len(unknown_classes), N_way)
        try:
            res = evaluator.evaluate(test_cache, unknown_classes, N_way=n_novel, K_shot=k)
            fewshot_results[f'novel_{k}shot'] = res
            print(f"  Novel {k}-shot: Acc={res['mean_acc']:.2%} +/- {res['ci95']:.2%}")
        except Exception as e:
            print(f"  Novel {k}-shot eval failed: {e}")

    # --- OSR evaluation on target ---
    osr_results = {}
    for method in ['anti_prototype', 'feature_mahalanobis']:
        try:
            calibrator = OSRCalibrator(
                flow_classifier, calib_cache,
                base_classes, unknown_classes, device)
            calibrator.calibrate(target_fpr=0.05, K_shot=50, method=method)

            test_calibrator = OSRCalibrator(
                flow_classifier, test_cache,
                base_classes, unknown_classes, device)
            osr_results[method] = test_calibrator.evaluate_osr(
                K_shot=50, num_rounds=10,
                method=method, recalibrate_per_round=True)
        except Exception as e:
            print(f"  OSR {method} failed: {e}")

    return {
        'source': source_name,
        'target': target_name,
        'fewshot': fewshot_results,
        'osr': osr_results,
    }


def main():
    parser = argparse.ArgumentParser(description='Episodic Cross-Domain Evaluation')
    parser.add_argument('--source', choices=['TAU22', 'TAU19'], default=None)
    parser.add_argument('--target', choices=['TAU22', 'TAU19'], default=None)
    parser.add_argument('--all', action='store_true')
    cl_args = parser.parse_args()

    if cl_args.all:
        pairs = [('TAU22', 'TAU19'), ('TAU19', 'TAU22')]
    elif cl_args.source and cl_args.target:
        pairs = [(cl_args.source, cl_args.target)]
    else:
        print("Specify --source and --target, or --all")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_classes = [0, 1, 2, 3, 4, 5]
    unknown_classes = [6, 7, 8, 9]

    all_results = {}
    for source, target in pairs:
        key = f"{source}->{target}"
        r = cross_domain_eval(source, target, base_classes, unknown_classes,
                              device=device)
        if r:
            all_results[key] = r

    # Save
    save_path = 'episodic_exp_cross_domain_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Direction':<15s} {'Base5s':>8s} {'Novel5s':>8s} {'OSR':>8s}")
    for key, r in all_results.items():
        fs = r.get('fewshot', {})
        base5 = fs.get('base_5shot', {}).get('mean_acc', 0)
        novel5 = fs.get('novel_5shot', {}).get('mean_acc', 0)
        osr = r.get('osr', {})
        best_osr = max(osr.values(), key=lambda x: x.get('osr_score', 0)) if osr else {}
        print(f"  {key:<15s} {base5:>7.2%} {novel5:>7.2%} "
              f"{best_osr.get('osr_score', 0):>7.2%}")


if __name__ == '__main__':
    main()
