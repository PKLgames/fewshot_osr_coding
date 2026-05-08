#!/usr/bin/env python3
"""
foac_exp_statistical.py — Statistical Significance Tests

Performs Friedman test and Nemenyi post-hoc test on method comparison results.
Generates Critical Difference (CD) diagram.

Input: JSON file with method results per dataset (from other experiments).
Output: p-values, rank table, CD diagram.

Usage:
  cd /coding/FOAC-AIFP
  python foac_exp_statistical.py --results foac_exp_ablation_results.json
  python foac_exp_statistical.py --manual  # input results manually
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
        alpha: significance level

    Returns:
        dict with test results
    """
    n_datasets, n_methods = results_matrix.shape
    print(f"\nFriedman-Nemenyi Test")
    print(f"  Methods: {n_methods}, Datasets: {n_datasets}")
    print(f"  Alpha: {alpha}")

    # --- Step 1: Rank methods per dataset (lower rank = better) ---
    ranks_per_dataset = np.zeros_like(results_matrix, dtype=float)
    for i in range(n_datasets):
        # rankdata: 1 = best (highest value)
        ranks_per_dataset[i] = rankdata(-results_matrix[i], method='average')

    avg_ranks = ranks_per_dataset.mean(axis=0)

    print(f"\n  Per-dataset ranks:")
    header = f"  {'Dataset':<15s}" + "".join(f"{m:>12s}" for m in method_names)
    print(header)
    for i, ds in enumerate(dataset_names):
        row = f"  {ds:<15s}" + "".join(f"{ranks_per_dataset[i,j]:>12.1f}" for j in range(n_methods))
        print(row)
    print(f"  {'Average':<15s}" + "".join(f"{avg_ranks[j]:>12.2f}" for j in range(n_methods)))

    # --- Step 2: Friedman test ---
    # friedmanchisquare expects (n_groups, n_samples) or separate arrays
    groups = [results_matrix[:, j] for j in range(n_methods)]
    stat, p_value = friedmanchisquare(*groups)

    print(f"\n  Friedman chi-square: {stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  Significant: {'YES' if p_value < alpha else 'NO'}")

    # --- Step 3: Nemenyi post-hoc (critical difference) ---
    # CD = q_alpha * sqrt(n_methods * (n_methods + 1) / (6 * n_datasets))
    # q_alpha from Nemenyi table (studentized range statistic / sqrt(2))
    # For small n_methods, use exact critical values
    k = n_methods
    N = n_datasets

    # Approximate q_alpha values for alpha=0.05 (Nemenyi critical values)
    # These are q_alpha / sqrt(2) values
    q_alpha_005 = {
        2: 2.773, 3: 3.315, 4: 3.633, 5: 3.858,
        6: 4.030, 7: 4.170, 8: 4.286, 9: 4.387,
        10: 4.474
    }
    q_val = q_alpha_005.get(k, 4.5)  # fallback
    cd = q_val * np.sqrt(k * (k + 1) / (6.0 * N))

    print(f"\n  Nemenyi Critical Difference (CD): {cd:.4f}")
    print(f"  q_alpha(0.05, k={k}): {q_val}")

    # Pairwise comparisons
    print(f"\n  Pairwise rank differences (significant if > CD={cd:.4f}):")
    significant_pairs = []
    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            diff = abs(avg_ranks[i] - avg_ranks[j])
            sig = "YES" if diff > cd else "no"
            if diff > cd:
                significant_pairs.append((method_names[i], method_names[j]))
            print(f"    {method_names[i]:<20s} vs {method_names[j]:<20s}: "
                  f"|{diff:.2f}| {sig}")

    return {
        'friedman_stat': float(stat),
        'friedman_p': float(p_value),
        'significant': p_value < alpha,
        'avg_ranks': {m: float(r) for m, r in zip(method_names, avg_ranks)},
        'critical_difference': float(cd),
        'significant_pairs': significant_pairs,
        'ranks_per_dataset': ranks_per_dataset.tolist(),
    }


def plot_cd_diagram(avg_ranks, n_methods, n_datasets, cd, method_names, save_path):
    """Generate Critical Difference diagram using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping CD diagram")
        return

    # Sort methods by average rank
    sorted_idx = np.argsort(list(avg_ranks.values()))
    sorted_methods = [method_names[i] for i in sorted_idx]
    sorted_ranks = [list(avg_ranks.values())[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(max(8, n_methods * 1.2), 3))
    ax.set_xlim(0.5, n_methods + 0.5)
    ax.set_ylim(0, 2)

    # Draw axis line
    ax.hlines(1.5, 1, n_methods, colors='black', linewidth=1)

    # Plot method positions
    for i, (method, rank) in enumerate(zip(sorted_methods, sorted_ranks)):
        ax.plot(rank, 1.5, 'o', markersize=10, color='steelblue', zorder=5)
        ax.text(rank, 1.7, method, ha='center', va='bottom', fontsize=9, rotation=30)

    # Connect methods that are NOT significantly different (within CD)
    for i in range(len(sorted_ranks)):
        for j in range(i + 1, len(sorted_ranks)):
            if sorted_ranks[j] - sorted_ranks[i] < cd:
                y = 1.3 - i * 0.08
                ax.hlines(y, sorted_ranks[i], sorted_ranks[j],
                          colors='red', linewidth=2, alpha=0.6)

    ax.set_title(f'Critical Difference Diagram (CD={cd:.2f}, N={n_datasets} datasets)')
    ax.set_xlabel('Average Rank')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"CD diagram saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='FOAC Statistical Significance Tests')
    parser.add_argument('--results', type=str, default=None,
                        help='JSON file with results from ablation/wayshot experiments')
    parser.add_argument('--metric', type=str, default='acc',
                        choices=['acc', 'auroc', 'osr', 'fscore'],
                        help='Metric to use for ranking')
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--manual', action='store_true',
                        help='Use hardcoded example results')
    cl_args = parser.parse_args()

    if cl_args.manual:
        # Example: manual results entry
        # Replace with actual values when available
        method_names = ['Full Model', 'w/o CIAM', 'w/o PAM', 'w/o NPM',
                        'Only NPM', 'Baseline']
        dataset_names = ['TAU22', 'TAU19']
        # results_matrix[i, j] = metric value for dataset i, method j
        # Fill with actual values after running experiments
        results_matrix = np.array([
            [70.0, 65.0, 68.0, 60.0, 62.0, 55.0],  # TAU22
            [100.0, 95.0, 98.0, 90.0, 92.0, 85.0],   # TAU19
        ])
    elif cl_args.results:
        with open(cl_args.results) as f:
            data = json.load(f)

        # Extract results from ablation JSON
        dataset_names = list(data.keys())
        method_names = list(data[dataset_names[0]].keys())
        metric = cl_args.metric

        results_matrix = []
        for ds in dataset_names:
            row = []
            for method in method_names:
                r = data[ds][method]['results']
                # Find best checkpoint
                best_val = 0
                for ckpt, vals in r.items():
                    v = vals.get(metric, [0, 0])
                    if isinstance(v, list):
                        v = v[0]
                    best_val = max(best_val, v)
                row.append(best_val)
            results_matrix.append(row)
        results_matrix = np.array(results_matrix)
    else:
        print("Please provide --results JSON or --manual flag")
        return

    # Run tests
    test_results = friedman_nemenyi_test(
        results_matrix, method_names, dataset_names, cl_args.alpha)

    # Save results
    save_path = 'foac_exp_statistical_results.json'
    with open(save_path, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    print(f"\nStatistical test results saved to {save_path}")

    # Generate CD diagram
    plot_cd_diagram(
        test_results['avg_ranks'],
        len(method_names),
        len(dataset_names),
        test_results['critical_difference'],
        method_names,
        'foac_exp_cd_diagram.png'
    )


if __name__ == '__main__':
    main()
