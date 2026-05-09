#!/usr/bin/env python3
"""
foac_exp_cross_domain.py — Cross-Domain (Generalizability) Evaluation

Train on one dataset (e.g. TAU22), test on another (e.g. TAU19),
to evaluate model generalization under domain shift.

Protocol:
  1. Load model pretrained on source domain
  2. Extract features from target domain data
  3. Run episodic evaluation (few-shot + OSR) on target domain

Usage:
  cd /coding/FOAC-AIFP
  python foac_exp_cross_domain.py --source TAU22 --target TAU19
  python foac_exp_cross_domain.py --source TAU19 --target TAU22
  python foac_exp_cross_domain.py --all   # both directions
"""

import os
import sys
import json
import argparse
from functools import partial
import numpy as np
import scipy.stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from tqdm import tqdm
from sklearn import metrics as sk_metrics

from trainers import trainer, C2_Net_train
from models.Network import My_Net
from datasets import dataloaders


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


def extract_features_per_class(model, dataloader, device='cuda'):
    """Extract features grouped by class from a dataloader."""
    model.eval()
    features_by_class = {}

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Extracting features'):
            support_data, support_label, query_data, query_label, \
                suppopen_data, suppopen_label, openset_data, openset_label, \
                supp_idx, open_idx = data

            # Extract support features (known classes)
            support_data = support_data.float().squeeze().to(device)
            support_label = support_label.squeeze().numpy()
            support_feat, _ = model.encode(support_data)
            support_feat = support_feat.cpu().numpy()

            for feat, label in zip(support_feat, support_label):
                label = int(label)
                if label not in features_by_class:
                    features_by_class[label] = []
                features_by_class[label].append(feat)

            # Extract open-set features (unknown classes)
            openset_data = openset_data.float().squeeze().to(device)
            openset_label_np = openset_label.squeeze().numpy()
            if openset_data.dim() == 3:
                openset_feat, _ = model.encode(openset_data)
            else:
                openset_feat, _ = model.encode(openset_data.unsqueeze(0))
            openset_feat = openset_feat.cpu().numpy()

            for feat, label in zip(openset_feat, openset_label_np):
                label = int(label)
                if label not in features_by_class:
                    features_by_class[label] = []
                features_by_class[label].append(feat)

    # Convert to numpy arrays
    for c in features_by_class:
        features_by_class[c] = np.array(features_by_class[c])

    return features_by_class


def cross_domain_eval(model, target_loader, args, device='cuda'):
    """
    Evaluate model trained on source domain using target domain data.
    Uses the model's built-in evaluation protocol.
    """
    model.eval()
    acc_trace, auroc_trace, osr_trace = [], [], []

    with torch.no_grad():
        for idx, data in enumerate(tqdm(target_loader, desc='Cross-domain eval')):
            support_data, support_label, query_data, query_label, \
                suppopen_data, suppopen_label, openset_data, openset_label, \
                supp_idx, open_idx = data

            support_data = support_data.float().to(device)
            query_data = query_data.float().to(device)
            suppopen_data = suppopen_data.float().to(device)
            openset_data = openset_data.float().to(device)
            support_label = support_label.long()
            query_label = query_label.long()
            openset_label = openset_label.long()

            openset_label_mapped = support_label.squeeze().max().item() + 1 + \
                                   torch.zeros_like(openset_label)
            the_label = tuple(x.squeeze() for x in
                              (support_label, query_label, suppopen_label, openset_label_mapped))
            the_img = tuple(x.squeeze() for x in
                            (support_data, query_data, suppopen_data, openset_data))
            supp_idx_dev = supp_idx.long().to(device)
            open_idx_dev = open_idx.long().to(device)

            try:
                prediction, loss_cls, loss_fake = model(
                    the_img, the_label, supp_idx_dev, open_idx_dev, test=True)
                query_probs, open_probs = prediction

                # Compute metrics
                query_probs_np = query_probs.cpu().numpy()
                open_probs_np = open_probs.cpu().numpy()

                query_label_np = query_label.squeeze().cpu().numpy()
                n_query = query_label_np.shape[0]

                # Accuracy
                query_pred = np.argmax(query_probs_np[:, :args.n_ways], axis=-1)
                acc = sk_metrics.accuracy_score(query_label_np, query_pred)
                acc_trace.append(acc)

                # AUROC
                open_label_binary = np.concatenate([
                    np.ones(n_query), np.zeros(open_probs_np.shape[0])])
                all_scores = np.max(
                    np.concatenate([query_probs_np, open_probs_np], axis=0)[:, :args.n_ways],
                    axis=-1)
                try:
                    auroc = sk_metrics.roc_auc_score(open_label_binary, all_scores)
                except ValueError:
                    auroc = 0.5
                auroc_trace.append(auroc)

            except Exception as e:
                print(f"  Episode {idx} failed: {e}")
                continue

    if not acc_trace:
        return None

    from scipy.stats import t as t_dist
    def ci95(data):
        n = len(data)
        m, se = np.mean(data), scipy.stats.sem(data)
        h = se * t_dist._ppf(0.975, n - 1)
        return float(m), float(h)

    result = {
        'acc': ci95(acc_trace),
        'auroc': ci95(auroc_trace),
        'n_episodes': len(acc_trace),
    }
    return result


def run_cross_domain(source, target):
    """Run one cross-domain experiment."""
    source_config = f'{source.lower()}_aligned.yml'
    target_config = f'{target.lower()}_aligned.yml'

    if not os.path.exists(source_config):
        print(f"Source config not found: {source_config}")
        return None
    if not os.path.exists(target_config):
        print(f"Target config not found: {target_config}")
        return None

    src_args = load_config(source_config)
    tgt_args = load_config(target_config)

    print(f"\n{'='*60}")
    print(f"Cross-Domain: {source} -> {target}")
    print(f"{'='*60}")

    # Load source model (best checkpoint)
    model = My_Net(args=src_args, mode='train')
    model = model.to('cuda')

    # Initialize representation from pretrained backbone first
    if os.path.exists(src_args.pretrained_model_path):
        pretrain_ckpt = torch.load(src_args.pretrained_model_path, weights_only=False)
        state_dict = pretrain_ckpt.get('feature_params', pretrain_ckpt.get('params', pretrain_ckpt))
        model.load_state_dict(state_dict, strict=False)
        model.init_representation(pretrain_ckpt)

    ckpt_path = os.path.join(src_args.save_folder, f'model_{source}_max_acc.pth')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(src_args.save_folder, f'model_{source}_max_osr.pth')
    if not os.path.exists(ckpt_path):
        print(f"  No trained model found at {src_args.save_folder}")
        return None

    print(f"  Loading: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    model.weight_base.data.copy_(state_dict['weight_base'].to('cuda'))
    model.weight_base_open.data.copy_(state_dict['weight_base_open'].to('cuda'))
    model.load_state_dict(state_dict, strict=False)

    # Evaluate on target domain
    # Use target dataset's test loader but source model's way/shot config
    eval_args = tgt_args
    eval_args.train_way = src_args.train_way
    eval_args.test_way = src_args.test_way
    eval_args.n_ways = src_args.n_ways
    eval_args.n_shots = src_args.n_shots

    target_loader = dataloaders.meta_test_dataloader(eval_args)
    result = cross_domain_eval(model, target_loader, src_args)

    if result:
        print(f"\n  {source} -> {target} Results:")
        print(f"    ACC:   {result['acc'][0]:.2f} +/- {result['acc'][1]:.2f}")
        print(f"    AUROC: {result['auroc'][0]:.2f} +/- {result['auroc'][1]:.2f}")

    return result


def main():
    import scipy

    parser = argparse.ArgumentParser(description='FOAC Cross-Domain Evaluation')
    parser.add_argument('--source', choices=['TAU22', 'TAU19'], default=None)
    parser.add_argument('--target', choices=['TAU22', 'TAU19'], default=None)
    parser.add_argument('--all', action='store_true', help='Run both directions')
    cl_args = parser.parse_args()

    if cl_args.all:
        pairs = [('TAU22', 'TAU19'), ('TAU19', 'TAU22')]
    elif cl_args.source and cl_args.target:
        pairs = [(cl_args.source, cl_args.target)]
    else:
        print("Specify --source and --target, or --all")
        return

    all_results = {}
    for source, target in pairs:
        key = f"{source}->{target}"
        result = run_cross_domain(source, target)
        if result:
            all_results[key] = result

    # Save
    results_path = 'foac_exp_cross_domain_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCross-domain results saved to {results_path}")

    # Print table
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Direction':<15s} {'ACC':>10s} {'AUROC':>10s}")
    for key, r in all_results.items():
        print(f"  {key:<15s} {r['acc'][0]:>9.2f} {r['auroc'][0]:>9.2f}")


if __name__ == '__main__':
    main()
