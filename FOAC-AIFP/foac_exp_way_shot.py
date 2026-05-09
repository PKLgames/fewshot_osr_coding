#!/usr/bin/env python3
"""
foac_exp_way_shot.py — Way-Shot Parameter Sweep

Tests model performance across different N-way and K-shot combinations.
Generates curves showing how Acc/AUROC/OSR change with N and K.

Usage:
  cd /coding/FOAC-AIFP
  python foac_exp_way_shot.py --dataset TAU22 --config tau22_aligned.yml
  python foac_exp_way_shot.py --dataset TAU19 --config tau19_aligned.yml
"""

import os
import sys
import json
import argparse
from functools import partial
from datetime import datetime

import torch
import numpy as np
import yaml

from trainers import trainer, C2_Net_train
from models.Network import My_Net
from datasets import dataloaders


# Sweep configurations
WAY_VALUES = [2, 3, 4, 5, 6]
SHOT_VALUES = [1, 3, 5, 7, 10]


def load_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = cfg['train']
    # Prevent train_parser()'s internal parse_args() from consuming sys.argv
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


def run_way_shot_config(args, n_way, k_shot):
    """Train and evaluate with specific N-way K-shot settings."""
    # Override way/shot parameters
    args.train_way = n_way
    args.test_way = n_way
    args.train_shot = k_shot
    args.test_shot = [k_shot]
    args.n_ways = n_way
    args.n_shots = k_shot

    # Adjust open ways to not exceed available classes
    max_open = args.train_classes - n_way
    args.n_open_ways = min(args.n_open_ways, max_open) if max_open > 0 else 1
    args.test_n_open_ways = args.n_open_ways

    save_dir = f"{args.save_folder.rstrip('/')}_wayshot/way{n_way}_shot{k_shot}"
    args.save_folder = save_dir
    args.test = False
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n  N-way={n_way}, K-shot={k_shot} -> {save_dir}")

    try:
        train_loader = dataloaders.meta_train_dataloader(args)
        eval_loader = dataloaders.meta_test_dataloader(args)
    except Exception as e:
        print(f"    SKIP: data loading failed ({e})")
        return None

    train_func = partial(C2_Net_train.default_train, train_loader=train_loader)

    model = My_Net(args=args, mode='train')
    model = model.to('cuda')

    if os.path.exists(args.pretrained_model_path):
        full_params = torch.load(args.pretrained_model_path, weights_only=False)
        state_dict = full_params.get('feature_params', full_params.get('params', full_params))
        model.load_state_dict(state_dict, strict=False)
        model.init_representation(full_params)

    tm = trainer.Train_Manager(args, train_func=train_func)
    tm.train(model, eval_loader)

    # Evaluate best model
    model.eval()
    best_result = None
    ckpt_path = os.path.join(save_dir, f'model_{args.dataset}_max_acc.pth')
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, weights_only=False)
        model.weight_base.data.copy_(state_dict['weight_base'].to('cuda'))
        model.weight_base_open.data.copy_(state_dict['weight_base_open'].to('cuda'))
        model.load_state_dict(state_dict, strict=False)
        result, loss = tm.run_test_fsl(model, eval_loader)
        best_result = {
            'acc': result[0], 'auroc': result[1], 'fscore': result[2],
            'tnr': result[3], 'tpr': result[4], 'osr': result[5],
        }
    else:
        print(f"    WARNING: no checkpoint saved for way={n_way} shot={k_shot}")

    return best_result


def main():
    parser = argparse.ArgumentParser(description='FOAC Way-Shot Sweep')
    parser.add_argument('--dataset', choices=['TAU22', 'TAU19', 'all'], default='all')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--ways', type=int, nargs='*', default=WAY_VALUES)
    parser.add_argument('--shots', type=int, nargs='*', default=SHOT_VALUES)
    parser.add_argument('--quick', action='store_true',
                        help='Only test a few combinations for debugging')
    cl_args = parser.parse_args()

    if cl_args.quick:
        cl_args.ways = [5, 6]
        cl_args.shots = [1, 5]

    datasets = ['TAU22', 'TAU19'] if cl_args.dataset == 'all' else [cl_args.dataset]
    all_results = {}

    for dataset in datasets:
        config_path = cl_args.config or f'{dataset.lower()}_aligned.yml'
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}, skipping {dataset}")
            continue

        print(f"\n{'='*60}")
        print(f"Way-Shot Sweep: {dataset}")
        print(f"  Ways: {cl_args.ways}")
        print(f"  Shots: {cl_args.shots}")
        print(f"{'='*60}")

        all_results[dataset] = {}

        for n_way in cl_args.ways:
            for k_shot in cl_args.shots:
                args = load_config(config_path)
                result = run_way_shot_config(args, n_way, k_shot)
                if result is not None:
                    all_results[dataset][f"{n_way}w{k_shot}s"] = result

    # Save results
    results_path = 'foac_exp_way_shot_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print table
    print(f"\n{'='*80}")
    print("WAY-SHOT SWEEP SUMMARY")
    print(f"{'='*80}")
    for dataset in all_results:
        print(f"\n  Dataset: {dataset}")
        print(f"  {'Config':<10s} {'ACC':>8s} {'AUROC':>8s} {'OSR':>8s} {'F1':>8s}")
        for key, r in sorted(all_results[dataset].items()):
            acc = r.get('acc', [0, 0])
            auroc = r.get('auroc', [0, 0])
            osr = r.get('osr', [0, 0])
            f1 = r.get('fscore', [0, 0])
            print(f"  {key:<10s} {acc[0]:>7.1f} {auroc[0]:>7.1f} {osr[0]:>7.1f} {f1[0]:>7.1f}")


if __name__ == '__main__':
    main()
