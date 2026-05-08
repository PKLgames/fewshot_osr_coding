#!/usr/bin/env python3
"""
episodic_exp_ablation.py — Ablation Experiment for Episodic Trainer

Toggles core components of EpisodicFlowClassifier to measure each component's
contribution: feature_adapter, flow_transform, ood_head, reciprocal_points,
osr_threshold.

Usage:
  cd /coding
  python episodic_exp_ablation.py --dataset TAU22
  python episodic_exp_ablation.py --dataset TAU19
  python episodic_exp_ablation.py --all
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
from datetime import datetime

# Import everything from episodic_trainer (or tau19 variant)
def get_trainer_module(dataset):
    if dataset == 'TAU19':
        import importlib
        spec = importlib.util.spec_from_file_location(
            'episodic_trainer_tau19', 'episodic_trainer_tau19.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    else:
        import importlib
        spec = importlib.util.spec_from_file_location(
            'episodic_trainer', 'episodic_trainer.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


# Ablation configurations
# (name, use_flow_transform, use_ood_head, use_reciprocal, use_threshold)
ABLATION_CONFIGS = [
    ("full_model",           True,  True,  True,  True),
    ("wo_flow_transform",    False, True,  True,  True),
    ("wo_ood_head",          True,  False, True,  True),
    ("wo_reciprocal",        True,  True,  False, True),
    ("wo_threshold",         True,  True,  True,  False),
    ("only_distance_head",   False, False, False, False),
]


def run_ablation(mod, config_name, use_flow, use_ood, use_recip, use_thresh,
                 base_classes, unknown_classes, backbone_choice='yamnet',
                 N_way=6, K_shot=5, Q_query=15, num_episodes=3000,
                 feature_dim=64):
    """Run one ablation configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_dir = f'experiment/episodic_ablation/{config_name}'

    print(f"\n{'='*60}")
    print(f"Ablation: {config_name}")
    print(f"  flow={use_flow}, ood={use_ood}, recip={use_recip}, thresh={use_thresh}")
    print(f"{'='*60}")

    # --- Phase 0: Feature extractor (reuse existing) ---
    # Find existing pretrained backbone
    pretrained_candidates = [
        'experiment/yamnet_fewshot_osr22/base_feature_extractor.pth',
        'experiment/yamnet_fewshot_osr13/base_feature_extractor.pth',
    ]
    pretrained_path = None
    for p in pretrained_candidates:
        if os.path.exists(p):
            pretrained_path = p
            break

    if pretrained_path is None:
        print("  ERROR: No pretrained backbone found. Run episodic_trainer.py first.")
        return None

    if backbone_choice == 'yamnet':
        feature_extractor = mod.FeatureExtractor(pretrained_path=pretrained_path)
    elif backbone_choice == 'panns_cnn14':
        feature_extractor = mod.PANNsFeatureExtractor(pretrained_path=pretrained_path)
    else:
        feature_extractor = mod.DistilASTFeatureExtractor(pretrained_path=pretrained_path)
    feature_extractor = feature_extractor.to(device).freeze()

    # --- Phase 1: Feature extraction ---
    TAUDataset = mod.TAUDataset
    train_dataset = TAUDataset(split='train')
    calib_dataset = TAUDataset(split='calib')
    test_dataset = TAUDataset(split='test')

    train_cache = mod.FeatureCache()
    train_cache.extract_and_cache(feature_extractor, train_dataset, 'train', device, batch_size=64)
    calib_cache = mod.FeatureCache()
    calib_cache.extract_and_cache(feature_extractor, calib_dataset, 'calib', device, batch_size=64)
    test_cache = mod.FeatureCache()
    test_cache.extract_and_cache(feature_extractor, test_dataset, 'test', device, batch_size=64)

    # --- Phase 2: Episodic training ---
    flow_classifier = mod.EpisodicFlowClassifier(
        input_dim=feature_dim, condition_dim=feature_dim,
        use_flow_transform=use_flow,
        use_ood_head=use_ood,
        use_reciprocal=use_recip,
        use_threshold=use_thresh,
    )

    ood_features = {}
    for c in unknown_classes:
        feats = test_cache.get_class_features(c)
        if len(feats) > 0:
            ood_features[c] = feats

    trainer = mod.EpisodicTrainer(
        feature_extractor=feature_extractor,
        flow_classifier=flow_classifier,
        train_cache=train_cache,
        calib_cache=calib_cache,
        test_cache=test_cache,
        base_classes=base_classes,
        unknown_classes=unknown_classes,
        ood_features_by_class=ood_features,
        N_way=N_way, K_shot=K_shot, Q_query=Q_query,
        lr=5e-5, gradient_accum_steps=1,
        device=device,
    )

    trainer.train(
        num_episodes=num_episodes,
        eval_every=500,
        num_val_episodes=10,
        save_dir=experiment_dir,
        resume=True,
    )

    # --- Phase 3: Evaluation ---
    evaluator = mod.FewShotEvaluator(trainer.flow_classifier, device)

    results = {}
    for k in [1, 5, 10]:
        # Base class eval
        try:
            base_res = evaluator.evaluate(train_cache, base_classes, N_way=N_way, K_shot=k)
            results[f'base_{k}shot'] = base_res
        except Exception as e:
            print(f"  base {k}-shot eval failed: {e}")

    # OSR evaluation
    osr_results = {}
    for method in ['anti_prototype', 'feature_mahalanobis']:
        try:
            calibrator = mod.OSRCalibrator(
                trainer.flow_classifier, calib_cache,
                base_classes, unknown_classes, device)
            calibrator.calibrate(target_fpr=0.05, K_shot=50, method=method)

            test_calibrator = mod.OSRCalibrator(
                trainer.flow_classifier, test_cache,
                base_classes, unknown_classes, device)
            osr_results[method] = test_calibrator.evaluate_osr(
                K_shot=50, num_rounds=10,
                method=method, recalibrate_per_round=True)
        except Exception as e:
            print(f"  OSR {method} failed: {e}")

    results['osr'] = osr_results
    results['config'] = {
        'name': config_name,
        'use_flow': use_flow, 'use_ood': use_ood,
        'use_reciprocal': use_recip, 'use_threshold': use_thresh,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Episodic Trainer Ablation Experiments')
    parser.add_argument('--dataset', choices=['TAU22', 'TAU19', 'all'], default='all')
    parser.add_argument('--configs', type=str, nargs='*', default=None,
                        help='Run only specific configs (e.g. full_model wo_ood_head)')
    parser.add_argument('--num_episodes', type=int, default=3000)
    parser.add_argument('--quick', action='store_true', help='Fewer episodes for debugging')
    cl_args = parser.parse_args()

    if cl_args.quick:
        cl_args.num_episodes = 500

    datasets = ['TAU22', 'TAU19'] if cl_args.dataset == 'all' else [cl_args.dataset]
    all_results = {}

    for dataset in datasets:
        mod = get_trainer_module(dataset)
        base_classes = [0, 1, 2, 3, 4, 5]
        unknown_classes = [6, 7, 8, 9]

        all_results[dataset] = {}
        for name, flow, ood, recip, thresh in ABLATION_CONFIGS:
            if cl_args.configs and name not in cl_args.configs:
                continue
            r = run_ablation(mod, name, flow, ood, recip, thresh,
                             base_classes, unknown_classes,
                             num_episodes=cl_args.num_episodes)
            if r:
                all_results[dataset][name] = r

    # Save
    save_path = 'episodic_exp_ablation_results.json'
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")
    for dataset in all_results:
        print(f"\n  Dataset: {dataset}")
        print(f"  {'Config':<25s} {'BaseAcc':>8s} {'TNR':>8s} {'TPR':>8s} {'OSR':>8s}")
        for name, r in all_results[dataset].items():
            acc = r.get('base_5shot', {}).get('mean_acc', 0)
            osr = r.get('osr', {})
            # Get best OSR method
            best_osr = max(osr.values(), key=lambda x: x.get('osr_score', 0)) if osr else {}
            print(f"  {name:<25s} {acc:>7.2%} "
                  f"{best_osr.get('known_tnr', 0):>7.2%} "
                  f"{best_osr.get('unknown_tpr', 0):>7.2%} "
                  f"{best_osr.get('osr_score', 0):>7.2%}")


if __name__ == '__main__':
    main()
