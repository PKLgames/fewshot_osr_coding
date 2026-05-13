"""
第四章图片生成脚本
实验设计与结果分析 — 实验数据可视化

生成/复制图片：
  【图4.1】 phase3_fewshot_accuracy.png   ← 已有图片复制
  【图4.2】 phase2_episodic_curves.png    ← 已有图片复制
  【图4.3】 phase3_osr_analysis.png       ← 已有图片复制
  【图4.4】 全部方法TPR对比柱状图        ← 根据实验数据生成
  【图4.5】 TPR进化历程折线图            ← 根据实验数据生成
  【图4.6】 tsne_features.png            ← 已有图片复制

用法: python essay_metirial/pic4/generate_chapter4_figures.py
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ============ matplotlib 中文支持 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, 'experiment', 'yamnet_fewshot_osr22')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============ 实验数据（来自 osr22a 实验日志） ============

# 全部方法的 TPR/TNR/OSR 结果（10轮均值，TNR=95%）
ALL_METHODS = {
    'Feature\nMahalanobis':    {'tnr': 95.00, 'tpr': 12.54, 'osr': 53.77, 'type': 'single'},
    'Anti-Proto':              {'tnr': 95.00, 'tpr': 18.92, 'osr': 56.96, 'type': 'single'},
    'Geo-Fusion':              {'tnr': 95.00, 'tpr': 14.06, 'osr': 54.53, 'type': 'single'},
    'OOD-Head':                {'tnr': 95.00, 'tpr':  9.01, 'osr': 52.00, 'type': 'single'},
    'OOD-Ext\n(11-dim)':      {'tnr': 95.00, 'tpr': 16.33, 'osr': 55.67, 'type': 'single'},
    'OOD-Ext-V2\n(13-dim)':   {'tnr': 95.00, 'tpr': 21.76, 'osr': 58.38, 'type': 'single'},
    'Cluster\nBoundary':       {'tnr': 95.00, 'tpr': 10.40, 'osr': 52.70, 'type': 'single'},
    'Ensemble\n(Anti+OOD-Ext)': {'tnr': 95.00, 'tpr': 22.91, 'osr': 58.95, 'type': 'ensemble'},
    'Adaptive\nEnsemble':      {'tnr': 95.00, 'tpr': 20.11, 'osr': 57.55, 'type': 'ensemble'},
    'Multi-Proto':             {'tnr': 95.00, 'tpr': 16.10, 'osr': 55.55, 'type': 'single'},
    'Ensemble-Cluster\n(Ours)': {'tnr': 95.00, 'tpr': 24.94, 'osr': 59.97, 'type': 'ours'},
}

# 实验进化历程数据（来自表4.8）
EVOLUTION_DATA = [
    {'label': 'osr18a', 'desc': 'Flow Density\n(Baseline)',        'tpr': 17.62, 'phase': 'flow'},
    {'label': 'osr19a', 'desc': 'Flow-assisted\n(attempts)',       'tpr': 17.69, 'phase': 'flow'},
    {'label': 'osr20a', 'desc': 'Pure Prototype\n(no Flow)',       'tpr': 18.92, 'phase': 'transition'},
    {'label': 'osr20b', 'desc': 'Ensemble\n(Anti+OOD-Ext)',        'tpr': 22.91, 'phase': 'breakthrough'},
    {'label': 'osr21a', 'desc': 'Flow Generator\n(falsified)',     'tpr': 22.29, 'phase': 'flow'},
    {'label': 'osr21b', 'desc': 'Flow Transform\n(falsified)',     'tpr': 23.07, 'phase': 'flow'},
    {'label': 'osr22a', 'desc': 'GMM Cluster\nEnhanced',           'tpr': 24.94, 'phase': 'final'},
]


# ============ 复制已有图片 ============

def copy_existing_figures():
    """复制已有图片到输出目录"""
    existing = {
        'phase3_fewshot_accuracy.png': 'fig4_1_fewshot_accuracy.png',
        'phase2_episodic_curves.png':  'fig4_2_episodic_curves.png',
        'phase3_osr_analysis.png':     'fig4_3_osr_analysis.png',
        'tsne_features.png':           'fig4_6_tsne_features.png',
    }
    for src_name, dst_name in existing.items():
        src = os.path.join(EXPERIMENT_DIR, src_name)
        dst = os.path.join(OUTPUT_DIR, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  复制: {src_name} -> {dst_name}")
        else:
            print(f"  警告: 找不到 {src_name}")


# ============ 图4.4: 全部方法TPR对比柱状图 ============

def plot_fig4_4():
    """【图4.4】全部方法TPR对比柱状图"""
    print("生成 【图4.4】 全部方法TPR对比柱状图 ...")

    # 精选展示的方法（去除 per_class 变体和不太重要的方法）
    display_methods = [
        ('Feature Mahalanobis',   12.54, 'baseline'),
        ('Anti-Prototype',        18.92, 'single'),
        ('OOD-Head (5-dim)',       9.01, 'baseline'),
        ('OOD-Ext (11-dim)',     16.33, 'single'),
        ('OOD-Ext-V2 (13-dim)',  21.76, 'single_highlight'),
        ('Cluster Boundary',     10.40, 'baseline'),
        ('Multi-Proto',          16.10, 'baseline'),
        ('Adaptive Ensemble',    20.11, 'ensemble'),
        ('Ensemble (Anti+Ext)',  22.91, 'ensemble'),
        ('Ensemble-Cluster',     24.94, 'ours'),
    ]

    names = [m[0] for m in display_methods]
    tprs = [m[1] for m in display_methods]
    types = [m[2] for m in display_methods]

    # 颜色方案
    color_map = {
        'baseline':           '#B0BEC5',  # 灰蓝
        'single':             '#64B5F6',  # 蓝色
        'single_highlight':   '#42A5F5',  # 深蓝（GMM增强）
        'ensemble':           '#FFB74D',  # 橙色
        'ours':               '#E53935',  # 红色（本文方法）
    }
    colors = [color_map[t] for t in types]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(names))
    bars = ax.bar(x, tprs, width=0.65, color=colors, edgecolor='white', linewidth=0.8, zorder=3)

    # 标注数值
    for bar, tpr in zip(bars, tprs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{tpr:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 标注关键提升
    # GMM聚类: 16.33 -> 21.76
    idx_oodext = names.index('OOD-Ext (11-dim)')
    idx_oodextv2 = names.index('OOD-Ext-V2 (13-dim)')
    ax.annotate('', xy=(idx_oodextv2, 21.76), xytext=(idx_oodext, 16.33),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, ls='--'))
    ax.text((idx_oodext + idx_oodextv2) / 2, 19.5, '+5.43%\n(GMM Clustering)',
            ha='center', fontsize=8, color='green', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.6))

    # Ensemble: 22.91 -> 24.94
    idx_ens = names.index('Ensemble (Anti+Ext)')
    idx_enscl = names.index('Ensemble-Cluster')
    ax.annotate('', xy=(idx_enscl, 24.94), xytext=(idx_ens, 22.91),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2, ls='--'))
    ax.text((idx_ens + idx_enscl) / 2, 24.2, '+2.03%',
            ha='center', fontsize=8, color='darkred', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('TPR (%) at TNR=95%', fontsize=12)
    ax.set_ylim(0, 30)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['baseline'], label='Baseline methods'),
        Patch(facecolor=color_map['single'], label='Single-signal methods'),
        Patch(facecolor=color_map['single_highlight'], label='GMM-enhanced (single)'),
        Patch(facecolor=color_map['ensemble'], label='Ensemble methods'),
        Patch(facecolor=color_map['ours'], label='Ours (Ensemble-Cluster)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
              framealpha=0.9, ncol=2)

    ax.set_title('TPR Comparison of All Methods (TNR=95%, 10-round average)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'fig4_4_tpr_comparison.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# ============ 图4.5: TPR进化历程折线图 ============

def plot_fig4_5():
    """【图4.5】TPR进化历程折线图"""
    print("生成 【图4.5】 TPR进化历程折线图 ...")

    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [d['label'] for d in EVOLUTION_DATA]
    tprs = [d['tpr'] for d in EVOLUTION_DATA]
    descs = [d['desc'] for d in EVOLUTION_DATA]
    phases = [d['phase'] for d in EVOLUTION_DATA]

    x = np.arange(len(labels))

    # 背景色块标注阶段
    # Flow 阶段: osr18a - osr19a
    ax.axvspan(-0.5, 1.5, alpha=0.08, color='red', label='Flow attempts (falsified)')
    # 过渡: osr20a
    ax.axvspan(1.5, 2.5, alpha=0.08, color='yellow', label='Transition (remove Flow)')
    # 突破: osr20b
    ax.axvspan(2.5, 3.5, alpha=0.08, color='green', label='Ensemble breakthrough')
    # Flow证伪: osr21a-b
    ax.axvspan(3.5, 5.5, alpha=0.08, color='red')
    # 最终: osr22a
    ax.axvspan(5.5, 6.5, alpha=0.15, color='blue', label='GMM clustering (final)')

    # 折线
    ax.plot(x, tprs, 'o-', color='#1565C0', linewidth=2.5, markersize=10,
            markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#1565C0',
            zorder=5)

    # 标注每个点的 TPR 和描述
    for i, (xi, tpr, desc, phase) in enumerate(zip(x, tprs, descs, phases)):
        # TPR 数值标注
        offset_y = 0.8 if i % 2 == 0 else -1.2
        ax.text(xi, tpr + offset_y, f'{tpr:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#1565C0')

        # 描述标注（交替上下）
        desc_y = tpr + 2.0 if i % 2 == 0 else tpr - 2.5
        va = 'bottom' if i % 2 == 0 else 'top'
        ax.text(xi, desc_y, desc, ha='center', va=va,
                fontsize=7.5, color='#555555', style='italic')

    # 标注关键提升
    # osr20a -> osr20b: +5.29 (突破)
    ax.annotate('', xy=(3, 22.91), xytext=(2, 18.92),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(2.5, 21.2, '+5.29', fontsize=10, color='green', fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))

    # osr20b -> osr22a: +2.03 (聚类)
    ax.annotate('', xy=(6, 24.94), xytext=(3, 22.91),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2, ls='--'))
    ax.text(5.0, 24.5, '+2.03', fontsize=10, color='darkblue', fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))

    # osr21 区间标注
    ax.annotate('Flow falsified\n(no improvement)',
                xy=(4.5, 22.7), fontsize=8, color='red', ha='center',
                bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('TPR (%) at TNR=95%', fontsize=12)
    ax.set_ylim(14, 28)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FFCDD2', alpha=0.5, label='Flow attempts (falsified)'),
        Patch(facecolor='#FFF9C4', alpha=0.5, label='Transition (remove Flow)'),
        Patch(facecolor='#C8E6C9', alpha=0.5, label='Ensemble breakthrough'),
        Patch(facecolor='#BBDEFB', alpha=0.7, label='GMM clustering (ours)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

    ax.set_title('TPR Evolution: osr18 → osr22 (Total: +7.32%, Relative: +41.6%)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'fig4_5_tpr_evolution.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# ============ 主函数 ============

def main():
    print("=" * 60)
    print("第四章图片生成脚本")
    print("=" * 60)

    # 1. 复制已有图片
    print("\n--- 复制已有图片 ---")
    copy_existing_figures()

    # 2. 生成需要新绘制的图片
    print("\n--- 生成新图片 ---")
    plot_fig4_4()
    plot_fig4_5()

    print("\n" + "=" * 60)
    print("所有图片处理完毕!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("\n图片清单:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
