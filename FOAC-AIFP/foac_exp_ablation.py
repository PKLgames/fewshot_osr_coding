#!/usr/bin/env python3
"""
foac_exp_ablation.py — Ablation Experiment Runner

Systematically toggles core components (CIAM, PAM, NPM) to validate
each component's contribution. Runs on TAU22, TAU19, and DCASE18.

Usage:
  cd /coding/FOAC-AIFP
  python foac_exp_ablation.py --dataset TAU22 --config tau22_aligned.yml
  python foac_exp_ablation.py --dataset TAU19 --config tau19_aligned.yml
  python foac_exp_ablation.py --dataset DCASE18 --fold all   # 4-fold CV
  python foac_exp_ablation.py --dataset DCASE18 --fold 1     # single fold
  python foac_exp_ablation.py --all   # run TAU22 + TAU19
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


# 6 ablation configurations: (name, use_ciam, use_pam, use_npm)
ABLATION_CONFIGS = [
    ("full_model",            True,  True,  True),
    ("wo_CIAM",               False, True,  True),
    ("wo_PAM",                True,  False, True),
    ("wo_NPM",                True,  True,  False),
    ("only_NPM",              False, False, True),
    ("baseline_no_modules",   False, False, False),
]


def load_config(config_path):
    """Load YAML config and merge with argparse defaults."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = cfg['train']
    # Create base args from parser defaults
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    base_parser = trainer.train_parser()
    sys.argv = saved_argv
    merged = vars(base_parser)
    merged.update(cfg)
    args = argparse.Namespace(**merged)
    # Convert nested dicts to namespaces
    for k, v in vars(args).items():
        if isinstance(v, dict):
            setattr(args, k, argparse.Namespace(**v))
    return args


def run_single_ablation(args, config_name, use_ciam, use_pam, use_npm, test_only=False):
    """Run one ablation configuration and return results."""
    # Override save folder for this ablation
    base_save = args.save_folder.rstrip('/')
    args.save_folder = f"{base_save}_ablation/{config_name}"
    os.makedirs(args.save_folder, exist_ok=True)

    # Force test=False for training
    args.test = False

    print(f"\n{'='*60}")
    print(f"Ablation: {config_name}")
    print(f"  CIAM={use_ciam}, PAM={use_pam}, NPM={use_npm}")
    print(f"  Save to: {args.save_folder}")
    print(f"{'='*60}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Data loaders
    train_loader = dataloaders.meta_train_dataloader(args)
    eval_loader = dataloaders.meta_test_dataloader(args)
    train_func = partial(C2_Net_train.default_train, train_loader=train_loader)

    # Model with ablation flags
    model = My_Net(args=args, mode='train',
                   use_ciam=use_ciam, use_pam=use_pam, use_npm=use_npm)
    model = model.to(device)

    # Load pretrained backbone
    if os.path.exists(args.pretrained_model_path):
        full_params = torch.load(args.pretrained_model_path, weights_only=False)
        state_dict = full_params.get('feature_params', full_params.get('params', full_params))
        # Filter fc.* from pretrained checkpoint: pretrain n_ways ≠ downstream n_ways
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
        model.init_representation(full_params)

    tm = trainer.Train_Manager(args, train_func=train_func)
    if not test_only:
        tm.train(model, eval_loader)
    else:
        print("  --test_only: skipping training, evaluating existing checkpoints")

    # Test
    model.eval()
    results = {}
    ckpt_names = [f'model_{args.dataset}_max_acc.pth', f'model_{args.dataset}_max_osr.pth',
                  f'model_{args.dataset}_max_auroc.pth', f'model_{args.dataset}_max_fscore.pth']
    test_log_path = os.path.join(args.save_folder, f'{args.dataset}test.log')
    with open(test_log_path, 'w') as f:
        for ckpt_name in ckpt_names:
            ckpt_path = os.path.join(args.save_folder, ckpt_name)
            if not os.path.exists(ckpt_path):
                continue
            state_dict = torch.load(ckpt_path, weights_only=False)
            model.weight_base.data.copy_(state_dict['weight_base'].to(device))
            model.weight_base_open.data.copy_(state_dict['weight_base_open'].to(device))
            model.load_state_dict(state_dict, strict=False)
            result, loss = tm.run_test_fsl(model, eval_loader)
            acc, auroc, fscore, tnr, tpr, osr_score = result
            f.write(f"{'='*50}\n{ckpt_name}\n")
            f.write(f"ACC:       {acc[0]:.3f} +/- {acc[1]:.3f}\n")
            f.write(f"AUROC:     {auroc[0]:.3f} +/- {auroc[1]:.3f}\n")
            f.write(f"F-score:   {fscore[0]:.3f} +/- {fscore[1]:.3f}\n")
            f.write(f"TNR:       {tnr[0]:.3f} +/- {tnr[1]:.3f}\n")
            f.write(f"TPR@TNR95: {tpr[0]:.3f} +/- {tpr[1]:.3f}\n")
            f.write(f"OSR Score: {osr_score[0]:.3f} +/- {osr_score[1]:.3f}\n\n")
            results[ckpt_name] = {
                'acc': acc, 'auroc': auroc, 'fscore': fscore,
                'tnr': tnr, 'tpr': tpr, 'osr': osr_score,
                'loss': loss
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='FOAC Ablation Experiments')
    parser.add_argument('--dataset', choices=['TAU22', 'TAU19', 'DCASE18', 'all'], default='all')
    parser.add_argument('--config', type=str, default=None,
                        help='Override config path (e.g. tau22_aligned.yml)')
    parser.add_argument('--configs', type=str, nargs='*', default=None,
                        help='Run only specific ablation configs (e.g. full_model wo_NPM)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer test runs (50 instead of 200)')
    parser.add_argument('--test_only', action='store_true',
                        help='Skip training, only evaluate existing checkpoints')
    parser.add_argument('--fold', type=str, default='1',
                        help='Fold for DCASE18: 1-4 or "all" for 4-fold CV (default: 1)')
    args = parser.parse_args()

    all_results = {}

    datasets = []
    if args.dataset == 'all':
        datasets = ['TAU22', 'TAU19']
    else:
        datasets = [args.dataset]

    for dataset in datasets:
        config_path = args.config
        if config_path is None:
            if dataset == 'DCASE18':
                config_path = 'dcase18_aligned.yml'
            else:
                config_path = f'{dataset.lower()}_aligned.yml'
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}, skipping {dataset}")
            continue

        # Determine folds to run
        if dataset == 'DCASE18':
            folds = [1, 2, 3, 4] if args.fold == 'all' else [int(args.fold)]
        else:
            folds = [None]  # TAU datasets have no fold

        all_results[dataset] = {}

        for fold in folds:
            fold_suffix = f'_fold{fold}' if fold is not None else ''
            fold_key = f'fold{fold}' if fold is not None else 'default'

            for name, use_ciam, use_pam, use_npm in ABLATION_CONFIGS:
                if args.configs and name not in args.configs:
                    continue
                # Reload config fresh each run
                run_cfg = load_config(config_path)
                if fold is not None:
                    run_cfg.fold = fold
                    run_cfg.save_folder = run_cfg.save_folder.rstrip('/') + fold_suffix
                    # Update pretrained_model_path to match fold-specific save_folder
                    run_cfg.pretrained_model_path = run_cfg.pretrained_model_path.replace(
                        'dcase18_aligned/', f'dcase18_aligned_fold{fold}/')
                results = run_single_ablation(run_cfg, name, use_ciam, use_pam, use_npm, test_only=args.test_only)
                result_key = f"{name}{fold_suffix}" if fold is not None else name
                all_results[dataset][result_key] = {
                    'use_ciam': use_ciam, 'use_pam': use_pam, 'use_npm': use_npm,
                    'fold': fold,
                    'results': {k: {kk: (vv[0] if isinstance(vv, tuple) else vv)
                                    for kk, vv in v.items()}
                                for k, v in results.items()}
                }

    # Save summary
    summary_path = 'foac_exp_ablation_results.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAblation results saved to {summary_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY")
    print(f"{'='*80}")
    for dataset in all_results:
        print(f"\n  Dataset: {dataset}")
        if dataset == 'DCASE18':
            # Print per-fold then mean±std across folds
            print(f"  {'Config':<25s} {'ACC':>8s} {'AUROC':>8s} {'OSR':>8s} {'F1':>8s}")
            config_names = [c[0] for c in ABLATION_CONFIGS]
            for name in config_names:
                fold_results = []
                for fold in [1, 2, 3, 4]:
                    rk = f'{name}_fold{fold}'
                    data = all_results[dataset].get(rk, {})
                    key = f'model_{dataset}_max_acc.pth'
                    r = data.get('results', {}).get(key, {})
                    if not r:
                        key = f'model_{dataset}_max_osr.pth'
                        r = data.get('results', {}).get(key, {})
                    fold_results.append(r)
                def _mean_std(vals, k):
                    vs = [v.get(k, 0) for v in fold_results]
                    vs = [x[0] if isinstance(x, (list, tuple)) else x for x in vs]
                    return np.mean(vs), np.std(vs)
                if any(fold_results):
                    acc_m, acc_s = _mean_std(fold_results, 'acc')
                    auroc_m, auroc_s = _mean_std(fold_results, 'auroc')
                    osr_m, osr_s = _mean_std(fold_results, 'osr')
                    f1_m, f1_s = _mean_std(fold_results, 'fscore')
                    print(f"  {name:<25s} {acc_m:>6.1f}±{acc_s:<4.1f} {auroc_m:>6.1f}±{auroc_s:<4.1f} {osr_m:>6.1f}±{osr_s:<4.1f} {f1_m:>6.1f}±{f1_s:<4.1f}")
        else:
            print(f"  {'Config':<25s} {'ACC':>8s} {'AUROC':>8s} {'OSR':>8s} {'F1':>8s}")
            for name, data in all_results[dataset].items():
                key = f'model_{dataset}_max_acc.pth'
                r = data['results'].get(key, {})
                if not r:
                    key = f'model_{dataset}_max_osr.pth'
                    r = data['results'].get(key, {})
                def _v(d, k):
                    val = d.get(k, 0)
                    return val[0] if isinstance(val, (list, tuple)) else val
                print(f"  {name:<25s} {_v(r,'acc'):>7.1f} {_v(r,'auroc'):>7.1f} {_v(r,'osr'):>7.1f} {_v(r,'fscore'):>7.1f}")


if __name__ == '__main__':
    main()
