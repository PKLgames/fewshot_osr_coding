#!/usr/bin/env python3
"""
episodic_exp_statistical.py — Statistical Significance Tests

Friedman test + Nemenyi post-hoc for comparing methods across datasets.
Works with results from ablation, way-shot, or other episodic experiments.

Usage:
  cd /coding
  python episodic_exp_statistical.py --results episodic_exp_ablation_results.json
  python episodic_exp_statistical.py --manual
"""

import os
import json
import argparse
import numpy as np
from scipy import stats
from scipy.stats import friedmanchisquare, rankdata


def friedman_nemenyi_test(results_matrix, method_names, dataset_names, alpha=0.05):
    """
    Perform Friedman test + Nemenyi post-hoc.

    Args:
        results_matrix: np.array of shape (n_datasets, n_methods)
        method_names: list of method names
        dataset_names: list of dataset names
    """
    n_datasets, n_methods = results_matrix.shape

    # Rank per dataset (1 = best = highest value)
    ranks_per_dataset = np.zeros_like(results_matrix, dtype=float)
    for i in range(n_datasets):
        ranks_per_dataset[i] = rankdata(-results_matrix[i], method='average')

    avg_ranks = ranks_per_dataset.mean(axis=0)

    print(f"\nFriedman-Nemenyi Test")
    print(f"  Methods: {n_methods}, Datasets: {n_datasets}")

    print(f"\n  Per-dataset ranks:")
    header = f"  {'Dataset':<15s}" + "".join(f"{m:>15s}" for m in method_names)
    print(header)
    for i, ds in enumerate(dataset_names):
        row = f"  {ds:<15s}" + "".join(f"{ranks_per_dataset[i,j]:>15.1f}" for j in range(n_methods))
        print(row)
    print(f"  {'Average':<15s}" + "".join(f"{avg_ranks[j]:>15.2f}" for j in range(n_methods)))

    # Friedman test
    groups = [results_matrix[:, j] for j in range(n_methods)]
    stat, p_value = friedmanchisquare(*groups)

    print(f"\n  Friedman chi-square: {stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant: {'YES' if p_value < alpha else 'NO'}")

    # Nemenyi CD
    k = n_methods
    N = n_datasets
    q_alpha_005 = {
        2: 2.773, 3: 3.315, 4: 3.633, 5: 3.858,
        6: 4.030, 7: 4.170, 8: 4.286, 9: 4.387, 10: 4.474
    }
    q_val = q_alpha_005.get(k, 4.5)
    cd = q_val * np.sqrt(k * (k + 1) / (6.0 * N))

    print(f"\n  Nemenyi CD: {cd:.4f}")

    # Pairwise
    significant_pairs = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = diff > cd
            if sig:
                significant_pairs.append((method_names[i], method_names[j]))
            print(f"    {method_names[i]:<25s} vs {method_names[j]:<25s}: "
                  f"|{diff:.2f}| {'***' if sig else ''}")

    return {
        'friedman_stat': float(stat),
        'friedman_p': float(p_value),
        'significant': bool(p_value < alpha),
        'avg_ranks': {m: float(r) for m, r in zip(method_names, avg_ranks)},
        'critical_difference': float(cd),
        'significant_pairs': significant_pairs,
    }


def plot_cd_diagram(avg_ranks, n_methods, n_datasets, cd, method_names, save_path):
    """Generate CD diagram."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping CD diagram")
        return

    sorted_idx = np.argsort(list(avg_ranks.values()))
    sorted_methods = [method_names[i] for i in sorted_idx]
    sorted_ranks = [list(avg_ranks.values())[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.2), 3))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(0, 2)
    ax.hlines(1.5, 1, n_methods, colors='black', linewidth=1)

    for i, (method, rank) in enumerate(zip(sorted_methods, sorted_ranks)):
        ax.plot(rank, 1.5, 'o', markersize=10, color='steelblue', zorder=5)
        ax.text(rank, 1.7, method, ha='center', va='bottom', fontsize=9, rotation=30)

    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                y = 1.3 - i * 0.08
                ax.hlines(y, sorted_ranks[i], sorted_ranks[j],
                          colors='red', linewidth=2, alpha=0.6)

    ax.set_title(f'CD Diagram (CD={cd:.2f}, N={n_datasets} datasets)')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CD diagram saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Episodic Statistical Tests')
    parser.add_argument('--results', type=str, default=None)
    parser.add_argument('--metric', type=str, default='acc',
                        choices=['acc', 'osr_score', 'auroc'])
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--output_dir', type=str, default='experiment/episodic_exp',
                        help='Root output directory for all results')
    cl_args = parser.parse_args()

    if cl_args.manual:
        # Replace with actual experiment results
        method_names = ['Full Model', 'w/o Flow', 'w/o OOD Head',
                        'w/o Reciprocal', 'w/o Threshold', 'Baseline']
        dataset_names = ['TAU22', 'TAU19']
        results_matrix = np.array([
            [74.0, 72.0, 70.0, 68.0, 71.0, 65.0],
            [43.0, 41.0, 39.0, 37.0, 40.0, 35.0],
        ])
    elif cl_args.results:
        with open(cl_args.results) as f:
            data = json.load(f)

        dataset_names = list(data.keys())
        method_names = list(data[dataset_names[0]].keys())
        metric = cl_args.metric

        results_matrix = []
        for ds in dataset_names:
            row = []
            for method in method_names:
                r = data[ds][method]
                # Try to extract the metric
                val = 0
                if metric == 'acc':
                    val = r.get('base_5shot', {}).get('mean_acc', 0) * 100
                elif metric == 'osr_score':
                    osr = r.get('osr', {})
                    val = max(v.get('osr_score', 0) for v in osr.values()) * 100 if osr else 0
                row.append(val)
            results_matrix.append(row)
        results_matrix = np.array(results_matrix)
    else:
        print("Provide --results JSON or --manual")
        return

    test_results = friedman_nemenyi_test(
        results_matrix, method_names, dataset_names, cl_args.alpha)

    save_dir = os.path.join(cl_args.output_dir, 'statistical')
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'results.json')
    with open(save_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    plot_cd_diagram(
        test_results['avg_ranks'], len(method_names), len(dataset_names),
        test_results['critical_difference'], method_names,
        os.path.join(save_dir, 'cd_diagram.png'))


if __name__ == '__main__':
    main()
