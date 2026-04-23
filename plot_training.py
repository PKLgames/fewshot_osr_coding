#!/usr/bin/env python3
"""
Training Process & Results Visualization for Few-Shot OSR Pipeline.

Generates comprehensive analysis plots from log files and saved results.
Can be used standalone or called after training.

Usage:
  python plot_training.py                        # from latest log
  python plot_training.py --log path/to.log      # from specific log
"""

import re
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ============================================================
# 1. Log Parsing
# ============================================================

class LogParser:
    """Parse episodic_trainer log files into structured data."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.lines = open(log_path).readlines()

    def parse_phase0a(self) -> Dict:
        """Parse Phase 0a SimCLR contrastive pre-training."""
        epochs = []
        for line in self.lines:
            m = re.match(
                r'Epoch\s+(\d+)/20\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+LR:\s+([\de.+-]+)',
                line)
            if m:
                epochs.append({
                    'epoch': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'lr': float(m.group(3)),
                })
        # Phase 0a is the first 20 epochs
        phase0a = [e for e in epochs[:20]]
        return {'epochs': phase0a}

    def parse_phase0b(self) -> Dict:
        """Parse Phase 0b base class supervised fine-tuning."""
        epochs = []
        for line in self.lines:
            m = re.match(
                r'Epoch\s+(\d+)/60\s+\|\s+Loss:\s+([\d.]+)\s+Acc:\s+([\d.]+)%\s+\|\s+LR:\s+([\de.+-]+)',
                line)
            if m:
                epochs.append({
                    'epoch': int(m.group(1)),
                    'loss': float(m.group(2)),
                    'acc': float(m.group(3)),
                    'lr': float(m.group(4)),
                })
        return {'epochs': epochs}

    def parse_episodic(self) -> Dict:
        """Parse Phase 2 episodic meta-training."""
        episodes = []
        for line in self.lines:
            # Match: Ep   100/15000 | CE: 0.6911 Total: 0.5181 Acc: 95.85% | Val Loss: 2.7951 Acc: 65.60% | LR: 6.00e-05
            m = re.match(
                r'Ep\s+(\d+)/\d+\s+\|\s+CE:\s+([-\d.]+)\s+Total:\s+([-\d.]+)\s+Acc:\s+([\d.]+)%\s+\|\s+'
                r'Val Loss:\s+([-\d.]+)\s+Acc:\s+([\d.]+)%\s+\|\s+LR:\s+([\de.+-]+)',
                line)
            if m:
                episodes.append({
                    'episode': int(m.group(1)),
                    'ce': float(m.group(2)),
                    'total_loss': float(m.group(3)),
                    'train_acc': float(m.group(4)),
                    'val_loss': float(m.group(5)),
                    'val_acc': float(m.group(6)),
                    'lr': float(m.group(7)),
                })
        return {'episodes': episodes}

    def parse_fewshot(self) -> Dict:
        """Parse Phase 3 few-shot evaluation results."""
        results = {'base': {}, 'novel': {}}
        current_section = None
        for line in self.lines:
            if 'Base class few-shot' in line:
                current_section = 'base'
            elif 'Novel class few-shot' in line:
                current_section = 'novel'
            elif current_section:
                m = re.match(
                    r'\s+(\d+)-way\s+(\d+)-shot:\s+Acc\s+=\s+([\d.]+)%\s+\+/-\s+([\d.]+)%\s+\(conf:\s+([\d.]+)%\)',
                    line)
                if m:
                    n_way = int(m.group(1))
                    k_shot = int(m.group(2))
                    results[current_section][k_shot] = {
                        'n_way': n_way,
                        'acc': float(m.group(3)),
                        'ci95': float(m.group(4)),
                        'conf': float(m.group(5)),
                    }
        return results

    def parse_osr(self) -> Dict:
        """Parse Phase 3 OSR calibration and evaluation results."""
        calib = {}
        eval_result = {}
        for line in self.lines:
            # Calibration
            m = re.match(r'\s+Mean separation:\s+([\d.-]+)', line)
            if m:
                calib['mean_separation'] = float(m.group(1))
            m = re.match(r'\s+Known scores:\s+\[([-\d.]+),\s+([-\d.]+)\]\s+mean=([-\d.]+)', line)
            if m:
                calib['known_min'] = float(m.group(1))
                calib['known_max'] = float(m.group(2))
                calib['known_mean'] = float(m.group(3))
            m = re.match(r'\s+Unknown scores:\s+\[([-\d.]+),\s+([-\d.]+)\]\s+mean=([-\d.]+)', line)
            if m:
                calib['unknown_min'] = float(m.group(1))
                calib['unknown_max'] = float(m.group(2))
                calib['unknown_mean'] = float(m.group(3))
            m = re.match(r'\s+Threshold \(tau\):\s+([-\d.]+)', line)
            if m:
                calib['threshold'] = float(m.group(1))
            m = re.match(r'\s+Known accepted \(TNR\):\s+([\d.]+)%', line)
            if m and 'calib' not in calib:
                calib['tnr_calib'] = float(m.group(1))
            m = re.match(r'\s+Unknown detected \(TPR\):\s+([\d.]+)%', line)
            if m and 'tpr_calib' not in calib:
                calib['tpr_calib'] = float(m.group(1))

            # Evaluation
            m = re.match(r'\s+Known accepted \(TNR\):\s+([\d.]+)%', line)
            if m and 'tnr_calib' in calib:
                eval_result['tnr'] = float(m.group(1))
            m = re.match(r'\s+Unknown detected \(TPR\):\s+([\d.]+)%', line)
            if m and 'tpr_calib' in calib:
                eval_result['tpr'] = float(m.group(1))
            m = re.match(r'\s+OSR Score \(mean\):\s+([\d.]+)%', line)
            if m:
                eval_result['osr_score'] = float(m.group(1))

        return {'calib': calib, 'eval': eval_result}

    def parse_config(self) -> Dict:
        """Parse training configuration."""
        config = {}
        for line in self.lines:
            m = re.match(r'\s+YAMNet unfreeze:\s+\[([^\]]+)\]', line)
            if m and 'yamnet_unfreeze' not in config:
                config['yamnet_unfreeze'] = m.group(1).replace("'", "").strip()
            m = re.match(r'\s+Flow classifier params:\s+([\d,]+)', line)
            if m:
                config['flow_params'] = int(m.group(1).replace(',', ''))
            m = re.match(r'\s+OOD ratio:\s+([\d.]+)', line)
            if m:
                config['ood_ratio'] = float(m.group(1))
            m = re.match(r'\s+Proto noise std:\s+([\d.]+)', line)
            if m:
                config['proto_noise'] = float(m.group(1))
        return config


# ============================================================
# 2. Plotting Functions
# ============================================================

# Style config
COLORS = {
    'train': '#2196F3',
    'val': '#FF5722',
    'train_acc': '#4CAF50',
    'val_acc': '#FF9800',
    'ce': '#9C27B0',
    'base': '#2196F3',
    'novel': '#FF5722',
    'known': '#4CAF50',
    'unknown': '#F44336',
    'threshold': '#FF9800',
}
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'legend.framealpha': 0.9,
})


def plot_phase0b(parser: LogParser, save_dir: str):
    """Phase 0b: Base class supervised training curves."""
    data = parser.parse_phase0b()
    if not data['epochs']:
        return
    epochs = data['epochs']
    ep = [e['epoch'] for e in epochs]
    loss = [e['loss'] for e in epochs]
    acc = [e['acc'] for e in epochs]
    lr = [e['lr'] for e in epochs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Phase 0b: Base Class Supervised Fine-tuning', fontsize=14, fontweight='bold')

    axes[0].plot(ep, loss, color=COLORS['train'], linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].axhline(y=loss[-1], color='gray', linestyle='--', alpha=0.5, label=f'Final: {loss[-1]:.4f}')
    axes[0].legend()

    axes[1].plot(ep, acc, color=COLORS['train_acc'], linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].axhline(y=acc[-1], color='gray', linestyle='--', alpha=0.5, label=f'Final: {acc[-1]:.1f}%')
    axes[1].legend()

    axes[2].semilogy(ep, lr, color=COLORS['ce'], linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')

    plt.tight_layout()
    path = os.path.join(save_dir, 'phase0b_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_episodic(parser: LogParser, save_dir: str):
    """Phase 2: Episodic meta-training curves."""
    data = parser.parse_episodic()
    if not data['episodes']:
        return
    eps = data['episodes']
    ep = [e['episode'] for e in eps]
    total_loss = [e['total_loss'] for e in eps]
    ce_loss = [e['ce'] for e in eps]
    train_acc = [e['train_acc'] for e in eps]
    val_acc = [e['val_acc'] for e in eps]
    val_loss = [e['val_loss'] for e in eps]
    lr = [e['lr'] for e in eps]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Phase 2: Episodic Meta-Training', fontsize=14, fontweight='bold')

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, total_loss, color=COLORS['train'], linewidth=1.2, label='Total Loss', alpha=0.9)
    ax1.plot(ep, ce_loss, color=COLORS['ce'], linewidth=1.2, label='CE Loss', alpha=0.9)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()

    # Mark best episode
    best_idx = np.argmax(val_acc)
    if best_idx < len(ep):
        ax1.axvline(x=ep[best_idx], color=COLORS['val'], linestyle='--', alpha=0.5,
                     label=f'Best Ep {ep[best_idx]}')
        ax1.legend()

    # 2. Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep, train_acc, color=COLORS['train_acc'], linewidth=1.5, label='Train Acc')
    ax2.plot(ep, val_acc, color=COLORS['val_acc'], linewidth=1.5, label='Val Acc')
    ax2.fill_between(ep,
                     [a - 2 for a in train_acc], [a + 2 for a in train_acc],
                     color=COLORS['train_acc'], alpha=0.1)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Train vs Val Accuracy')
    ax2.legend()
    # Annotate gap
    if train_acc and val_acc:
        gap = train_acc[-1] - val_acc[-1]
        ax2.annotate(f'Gap: {gap:.1f}%', xy=(ep[-1], val_acc[-1]),
                     xytext=(ep[-1] * 0.6, val_acc[-1] + 5),
                     arrowprops=dict(arrowstyle='->', color='red', alpha=0.6),
                     color='red', fontsize=10)

    # 3. Val loss
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep, val_loss, color=COLORS['val'], linewidth=1.5)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Val Loss')
    ax3.set_title('Validation Loss')
    ax3.axvline(x=ep[best_idx], color=COLORS['val'], linestyle='--', alpha=0.5)

    # 4. Learning rate
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(ep, [l * 1000 for l in lr], color=COLORS['ce'], linewidth=1.5)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('LR (×1e-3)')
    ax4.set_title('Learning Rate Schedule')

    plt.tight_layout()
    path = os.path.join(save_dir, 'phase2_episodic_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_fewshot(parser: LogParser, save_dir: str):
    """Phase 3: Few-shot classification accuracy bar chart."""
    data = parser.parse_fewshot()
    if not data['base'] and not data['novel']:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Phase 3: Few-Shot Classification Accuracy', fontsize=14, fontweight='bold')

    for idx, (section, title) in enumerate([('base', 'Base Classes (Known)'),
                                             ('novel', 'Novel Classes (Unknown)')]):
        ax = axes[idx]
        if section not in data or not data[section]:
            ax.set_title(f'{title}\n(no data)')
            continue

        shots = sorted(data[section].keys())
        accs = [data[section][k]['acc'] for k in shots]
        cis = [data[section][k]['ci95'] for k in shots]
        confs = [data[section][k]['conf'] for k in shots]
        n_way = data[section][shots[0]]['n_way']

        x = np.arange(len(shots))
        width = 0.35

        bars1 = ax.bar(x - width / 2, accs, width, label='Accuracy',
                        color=COLORS['base'] if section == 'base' else COLORS['novel'],
                        alpha=0.85, edgecolor='white')
        ax.errorbar(x - width / 2, accs, yerr=cis, fmt='none', ecolor='black', capsize=4)

        bars2 = ax.bar(x + width / 2, confs, width, label='Confidence',
                        color='#B0BEC5', alpha=0.7, edgecolor='white')

        # Random baseline
        random_acc = 100.0 / n_way
        ax.axhline(y=random_acc, color='red', linestyle='--', alpha=0.5,
                    label=f'Random ({random_acc:.0f}%)')

        ax.set_xlabel('K-Shot')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{title} ({n_way}-way)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{k}-shot' for k in shots])
        ax.legend()
        ax.set_ylim(0, max(max(accs), max(confs)) * 1.15)

        # Value labels
        for bar, val in zip(bars1, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(save_dir, 'phase3_fewshot_accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_osr_summary(parser: LogParser, save_dir: str):
    """Phase 3: OSR score distribution & summary."""
    osr = parser.parse_osr()
    fewshot = parser.parse_fewshot()
    if not osr['calib']:
        return

    calib = osr['calib']
    eval_r = osr['eval']

    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)
    fig.suptitle('Phase 3: Open Set Recognition Analysis', fontsize=14, fontweight='bold')

    # 1. Score distribution (simulated from calibration stats)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'known_mean' in calib and 'unknown_mean' in calib:
        # Generate approximate distributions from calibration stats
        known_min = calib.get('known_min', calib['known_mean'] - 3)
        known_max = calib.get('known_max', calib['known_mean'] + 3)
        unknown_min = calib.get('unknown_min', calib['unknown_mean'] - 3)
        unknown_max = calib.get('unknown_max', calib['unknown_mean'] + 3)

        known_std = max((known_max - known_min) / 6, 0.1)
        unknown_std = max((unknown_max - unknown_min) / 6, 0.1)

        x_range = np.linspace(min(known_min, unknown_min) - 1,
                               max(known_max, unknown_max) + 1, 300)
        from scipy.stats import norm
        known_pdf = norm.pdf(x_range, calib['known_mean'], known_std)
        unknown_pdf = norm.pdf(x_range, calib['unknown_mean'], unknown_std)

        ax1.fill_between(x_range, known_pdf, alpha=0.4, color=COLORS['known'], label='Known')
        ax1.plot(x_range, known_pdf, color=COLORS['known'], linewidth=1.5)
        ax1.fill_between(x_range, unknown_pdf, alpha=0.4, color=COLORS['unknown'], label='Unknown')
        ax1.plot(x_range, unknown_pdf, color=COLORS['unknown'], linewidth=1.5)

        if 'threshold' in calib:
            ax1.axvline(x=calib['threshold'], color=COLORS['threshold'],
                        linestyle='--', linewidth=2, label=f'Threshold: {calib["threshold"]:.2f}')

        ax1.set_xlabel('OSR Score (Mahalanobis)')
        ax1.set_ylabel('Density')
        ax1.set_title('Score Distribution')
        ax1.legend(fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 transform=ax1.transAxes)

    # 2. OSR metrics bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = {}
    if eval_r:
        metrics = {
            'TNR\n(Known Accept)': eval_r.get('tnr', 0),
            'TPR\n(Unknown Detect)': eval_r.get('tpr', 0),
            'OSR Score\n(Mean)': eval_r.get('osr_score', 0),
        }
    elif calib:
        metrics = {
            'TNR\n(Known Accept)': calib.get('tnr_calib', 0),
            'TPR\n(Unknown Detect)': calib.get('tpr_calib', 0),
        }

    if metrics:
        bars = ax2.bar(metrics.keys(), metrics.values(),
                        color=[COLORS['known'], COLORS['unknown'], COLORS['val']][:len(metrics)],
                        alpha=0.85, edgecolor='white', width=0.5)
        for bar, val in zip(bars, metrics.values()):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax2.set_ylabel('Rate (%)')
        ax2.set_title('OSR Metrics (Test)')
        ax2.set_ylim(0, 110)

    # 3. Overall summary radar / comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    config = parser.parse_config()
    summary_lines = [
        f"{'='*30}",
        f"  Configuration",
        f"{'='*30}",
    ]
    if config.get('yamnet_unfreeze'):
        summary_lines.append(f"  YAMNet unfreeze: {config['yamnet_unfreeze'][:30]}...")
    if config.get('flow_params'):
        summary_lines.append(f"  Flow params: {config['flow_params']:,}")
    if config.get('ood_ratio'):
        summary_lines.append(f"  OOD ratio: {config['ood_ratio']}")
    summary_lines.append("")
    summary_lines.append(f"{'='*30}")
    summary_lines.append(f"  Results Summary")
    summary_lines.append(f"{'='*30}")

    for section, label in [('base', 'Base'), ('novel', 'Novel')]:
        if section in fewshot and fewshot[section]:
            summary_lines.append(f"\n  {label}:")
            for k in sorted(fewshot[section].keys()):
                r = fewshot[section][k]
                summary_lines.append(f"    {r['n_way']}w{k}s: {r['acc']:.1f}% +/- {r['ci95']:.1f}%")

    if eval_r:
        summary_lines.append(f"\n  OSR:")
        summary_lines.append(f"    TNR: {eval_r.get('tnr', 0):.1f}%")
        summary_lines.append(f"    TPR: {eval_r.get('tpr', 0):.1f}%")
        summary_lines.append(f"    Score: {eval_r.get('osr_score', 0):.1f}%")
    if calib.get('mean_separation') is not None:
        summary_lines.append(f"    Separation: {calib['mean_separation']:.2f}")

    ax3.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax3.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('Summary', fontsize=12)

    plt.tight_layout()
    path = os.path.join(save_dir, 'phase3_osr_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_comparison(log_paths: Dict[str, str], save_dir: str):
    """Compare results across multiple training runs."""
    parsers = {name: LogParser(path) for name, path in log_paths.items()}
    names = list(parsers.keys())

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Training Run Comparison', fontsize=14, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    # 1. Episodic val accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (name, p) in enumerate(parsers.items()):
        data = p.parse_episodic()
        if data['episodes']:
            eps = [e['episode'] for e in data['episodes']]
            vacc = [e['val_acc'] for e in data['episodes']]
            ax1.plot(eps, vacc, color=colors[i], linewidth=1.5, label=name)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Val Accuracy (%)')
    ax1.set_title('Validation Accuracy')
    ax1.legend()

    # 2. Episodic total loss
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (name, p) in enumerate(parsers.items()):
        data = p.parse_episodic()
        if data['episodes']:
            eps = [e['episode'] for e in data['episodes']]
            tloss = [e['total_loss'] for e in data['episodes']]
            ax2.plot(eps, tloss, color=colors[i], linewidth=1.5, label=name)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Loss')
    ax2.set_title('Total Loss')
    ax2.legend()

    # 3. Few-shot accuracy grouped bar
    ax3 = fig.add_subplot(gs[1, 0])
    shot_keys = [1, 5, 10]
    x = np.arange(len(shot_keys))
    width = 0.8 / max(len(names), 1)

    for i, (name, p) in enumerate(parsers.items()):
        fs = p.parse_fewshot()
        for section in ['base', 'novel']:
            accs = [fs[section].get(k, {}).get('acc', 0) for k in shot_keys]
            style = '-' if section == 'base' else '--'
            marker = 'o' if section == 'base' else 's'
            ax3.plot(x, accs, color=colors[i], linestyle=style, marker=marker,
                     linewidth=1.5, label=f'{name} ({section})')

    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{k}-shot' for k in shot_keys])
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Few-Shot Accuracy')
    ax3.legend(fontsize=8)

    # 4. OSR metrics comparison
    ax4 = fig.add_subplot(gs[1, 1])
    for i, (name, p) in enumerate(parsers.items()):
        osr = p.parse_osr()
        ev = osr.get('eval', {})
        if ev:
            metrics = [ev.get('tnr', 0), ev.get('tpr', 0), ev.get('osr_score', 0)]
            x_pos = np.arange(3) + i * width - width * len(names) / 2
            ax4.bar(x_pos, metrics, width=width, color=colors[i],
                    alpha=0.85, label=name, edgecolor='white')

    ax4.set_xticks(np.arange(3))
    ax4.set_xticklabels(['TNR', 'TPR', 'OSR Score'])
    ax4.set_ylabel('Rate (%)')
    ax4.set_title('OSR Metrics')
    ax4.legend()

    plt.tight_layout()
    path = os.path.join(save_dir, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# 3. Main Plotting Entry Point
# ============================================================

def generate_all_plots(log_path: str, save_dir: str = 'experiment/fewshot_plots'):
    """Generate all analysis plots from a log file."""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    os.makedirs(save_dir, exist_ok=True)
    parser = LogParser(log_path)

    print(f"\nGenerating analysis plots from: {log_path}")
    print(f"Output directory: {save_dir}\n")

    plot_phase0b(parser, save_dir)
    plot_episodic(parser, save_dir)
    plot_fewshot(parser, save_dir)
    plot_osr_summary(parser, save_dir)

    print(f"\nAll plots saved to: {save_dir}/")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default='TAU22_yamnet_osr3step.log',
                    help='Path to training log file')
    ap.add_argument('--save-dir', default='experiment/fewshot_plots',
                    help='Directory to save plots')
    ap.add_argument('--compare', nargs='*', default=[],
                    help='Log files to compare (format: name=path)')
    args = ap.parse_args()

    generate_all_plots(args.log, args.save_dir)

    if args.compare:
        logs = {}
        for item in args.compare:
            if '=' in item:
                name, path = item.split('=', 1)
                logs[name] = path
            else:
                logs[f'Run{len(logs)+1}'] = item
        if len(logs) >= 2:
            plot_comparison(logs, args.save_dir)
