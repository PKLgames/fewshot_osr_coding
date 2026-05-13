"""
第三章图片生成脚本
基于原型学习与聚类的OSR算法设计 — 实验数据可视化

生成图片：
  【图3.3】查询依赖反原型构造示意图
  【图3.4】GMM聚类特征示意图
  【图3.5】融合权重搜索曲线
  【图3.6】两种信号的互补性可视化

用法: python essay_metirial/pic3/generate_chapter3_figures.py
"""

import sys
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# 确保项目根目录在 path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# ============ matplotlib 中文支持 ============
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150

# ============ 全局配置 ============
SEED = 42
BASE_CLASSES = [0, 1, 2, 3, 4, 5]
UNKNOWN_CLASSES = [6, 7, 8, 9]
CLASS_NAMES = {
    0: 'Airport', 1: 'Tram', 2: 'Bus', 3: 'Public Square',
    4: 'Shopping Mall', 5: 'Street Pedestrian',
    6: 'Metro Station', 7: 'Street Traffic', 8: 'Metro', 9: 'Park'
}
FEATURE_DIM = 64
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============ 工具函数 ============

def load_features(split='calib'):
    """加载预提取的特征"""
    cache_path = os.path.join(PROJECT_ROOT, 'experiment', 'fewshot_cache', f'{split}_features.pt')
    data = torch.load(cache_path, weights_only=True)
    return data['features'], data['labels']


def get_class_features(features, labels, class_id):
    """获取某个类别的所有特征"""
    mask = labels == class_id
    return features[mask]


def load_model():
    """加载最佳模型（用于 feature adapter 和 distance head）"""
    sys.path.insert(0, PROJECT_ROOT)
    from episodic_trainer import EpisodicFlowClassifier
    ckpt_path = os.path.join(PROJECT_ROOT, 'experiment', 'yamnet_fewshot_osr22', 'fewshot_best.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    model = EpisodicFlowClassifier(
        input_dim=64, condition_dim=64, use_flow_transform=False)
    model.load_state_dict(ckpt['flow_state_dict'], strict=False)
    model.eval()
    return model


def compute_prototypes(train_features, train_labels, K_shot=50, seed=42):
    """计算类原型"""
    rng = random.Random(seed)
    prototypes = []
    for c in BASE_CLASSES:
        feats = get_class_features(train_features, train_labels, c)
        n = min(K_shot, len(feats))
        indices = rng.sample(range(len(feats)), n)
        prototypes.append(feats[indices].mean(dim=0))
    return torch.stack(prototypes)


def apply_model(model, features, prototypes):
    """应用 feature adapter 和 distance head"""
    with torch.no_grad():
        adapted_feats, adapted_protos = model.adapt_features(features, prototypes)
        projected_feats = model.distance_head(adapted_feats)
        projected_protos = model.distance_head(adapted_protos)
    return projected_feats, projected_protos


def fit_pca(train_feats_2d, *other_arrays):
    """PCA 降维到 2D，返回变换后的坐标（使用训练集 fit）"""
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(train_feats_2d)
    results = [pca.transform(train_feats_2d)]
    for arr in other_arrays:
        results.append(pca.transform(arr))
    return results


# ============ 图3.3: 查询依赖反原型构造示意图 ============

def plot_fig3_3(model, calib_features, calib_labels, train_features, train_labels):
    """【图3.3】查询依赖反原型构造示意图"""
    print("生成 【图3.3】 查询依赖反原型构造示意图 ...")

    # 计算原型
    prototypes = compute_prototypes(train_features, train_labels, K_shot=50)
    adapted_feats, adapted_protos = apply_model(model, calib_features, prototypes)

    # 准备数据 — 使用全部 6 个已知类 + 4 个未知类
    known_mask = torch.tensor([c in BASE_CLASSES for c in calib_labels.numpy()])
    unknown_mask = ~known_mask

    known_feats = adapted_feats[known_mask].numpy()
    unknown_feats = adapted_feats[unknown_mask].numpy()
    protos = adapted_protos.numpy()

    # PCA 降维（使用所有数据 fit）
    all_feats = np.vstack([known_feats, unknown_feats, protos])
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(np.vstack([known_feats, unknown_feats]))
    all_2d = pca.transform(all_feats)
    known_2d = all_2d[:len(known_feats)]
    unknown_2d = all_2d[len(known_feats):len(known_feats)+len(unknown_feats)]
    protos_2d = all_2d[len(known_feats)+len(unknown_feats):]

    # 计算中心
    fixed_center = protos_2d.mean(axis=0)  # 固定中心 = 所有原型均值

    # 模拟测试 episode: 随机取一批查询样本（含已知+未知）
    rng = np.random.RandomState(SEED)
    n_query_known = 30
    n_query_unknown = 20
    qk_idx = rng.choice(len(known_feats), n_query_known, replace=False)
    qu_idx = rng.choice(len(unknown_feats), n_query_unknown, replace=False)
    query_known_2d = known_2d[qk_idx]
    query_unknown_2d = unknown_2d[qu_idx]

    query_center_2d = np.vstack([query_known_2d, query_unknown_2d]).mean(axis=0)  # 查询依赖中心

    # 取 2 个已知类（模拟 episode）展示反原型
    episode_classes = [0, 1]
    episode_protos_2d = np.array([protos_2d[c] for c in episode_classes])

    # 固定中心反原型
    fixed_anti_protos_2d = np.array([2 * fixed_center - p for p in episode_protos_2d])

    # 查询依赖反原型
    query_anti_protos_2d = np.array([2 * query_center_2d - p for p in episode_protos_2d])

    # ---- 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 固定中心
    ax = axes[0]
    ax.set_title('Fixed Center (prototypes.mean)', fontsize=13, fontweight='bold')
    # 随机采样一些已知类样本作为背景
    bg_idx = rng.choice(len(known_feats), min(200, len(known_feats)), replace=False)
    ax.scatter(known_2d[bg_idx, 0], known_2d[bg_idx, 1], c='lightblue', alpha=0.3,
               s=15, label='Known samples', zorder=1)
    bg_u_idx = rng.choice(len(unknown_feats), min(100, len(unknown_feats)), replace=False)
    ax.scatter(unknown_2d[bg_u_idx, 0], unknown_2d[bg_u_idx, 1], c='lightcoral', alpha=0.3,
               s=15, label='Unknown samples', zorder=1)

    colors = ['#1f77b4', '#ff7f0e']
    for i, (c, color) in enumerate(zip(episode_classes, colors)):
        name = CLASS_NAMES[c]
        # 原型
        ax.scatter(episode_protos_2d[i, 0], episode_protos_2d[i, 1],
                   marker='*', s=300, c=color, edgecolors='black', linewidths=1.5,
                   zorder=5, label=f'Proto: {name}')
        # 反原型
        ax.scatter(fixed_anti_protos_2d[i, 0], fixed_anti_protos_2d[i, 1],
                   marker='X', s=200, c=color, edgecolors='red', linewidths=1.5,
                   zorder=5, label=f'Anti-proto: {name}')
        # 连线
        ax.plot([episode_protos_2d[i, 0], fixed_anti_protos_2d[i, 0]],
                [episode_protos_2d[i, 1], fixed_anti_protos_2d[i, 1]],
                '--', color=color, alpha=0.5, linewidth=1)

    # 固定中心
    ax.scatter(fixed_center[0], fixed_center[1], marker='D', s=150, c='green',
               edgecolors='black', linewidths=1.5, zorder=6, label='Fixed center')

    ax.legend(fontsize=8, loc='best')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    # 右图: 查询依赖中心
    ax = axes[1]
    ax.set_title('Query-Dependent Center (query.mean)', fontsize=13, fontweight='bold')
    ax.scatter(known_2d[bg_idx, 0], known_2d[bg_idx, 1], c='lightblue', alpha=0.3,
               s=15, label='Known samples', zorder=1)
    ax.scatter(unknown_2d[bg_u_idx, 0], unknown_2d[bg_u_idx, 1], c='lightcoral', alpha=0.3,
               s=15, label='Unknown samples', zorder=1)

    for i, (c, color) in enumerate(zip(episode_classes, colors)):
        name = CLASS_NAMES[c]
        ax.scatter(episode_protos_2d[i, 0], episode_protos_2d[i, 1],
                   marker='*', s=300, c=color, edgecolors='black', linewidths=1.5,
                   zorder=5, label=f'Proto: {name}')
        ax.scatter(query_anti_protos_2d[i, 0], query_anti_protos_2d[i, 1],
                   marker='X', s=200, c=color, edgecolors='red', linewidths=1.5,
                   zorder=5, label=f'Anti-proto: {name}')
        ax.plot([episode_protos_2d[i, 0], query_anti_protos_2d[i, 0]],
                [episode_protos_2d[i, 1], query_anti_protos_2d[i, 1]],
                '--', color=color, alpha=0.5, linewidth=1)

    # 查询依赖中心
    ax.scatter(query_center_2d[0], query_center_2d[1], marker='D', s=150, c='purple',
               edgecolors='black', linewidths=1.5, zorder=6, label='Query-dependent center')

    # 查询样本
    ax.scatter(query_known_2d[:, 0], query_known_2d[:, 1], c='blue', alpha=0.5,
               s=25, marker='o', label='Query (known)', zorder=3)
    ax.scatter(query_unknown_2d[:, 0], query_unknown_2d[:, 1], c='red', alpha=0.5,
               s=25, marker='^', label='Query (unknown)', zorder=3)

    ax.legend(fontsize=8, loc='best')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'fig3_3_anti_prototype.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# ============ 图3.4: GMM聚类特征示意图 ============

def plot_fig3_4(model, calib_features, calib_labels, train_features, train_labels):
    """【图3.4】GMM聚类特征示意图"""
    print("生成 【图3.4】 GMM聚类特征示意图 ...")

    # 计算原型
    prototypes = compute_prototypes(train_features, train_labels, K_shot=50)
    adapted_feats, adapted_protos = apply_model(model, calib_features, prototypes)

    # 拟合 GMM
    known_mask = torch.tensor([c in BASE_CLASSES for c in calib_labels.numpy()])
    known_feats = adapted_feats[known_mask].numpy()
    unknown_feats = adapted_feats[~known_mask].numpy()

    gmm = GaussianMixture(n_components=12, covariance_type='full',
                          max_iter=200, random_state=SEED, reg_covar=1e-4)
    gmm.fit(known_feats)

    # PCA 降维
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(np.vstack([known_feats, unknown_feats]))
    known_2d = pca.transform(known_feats)
    unknown_2d = pca.transform(unknown_feats)
    means_2d = pca.transform(gmm.means_)
    protos_2d = pca.transform(adapted_protos.numpy())

    # 计算 GMM 在 PCA 空间的协方差（近似）
    # 使用 PCA 变换后的协方差
    covs_2d = []
    for k in range(gmm.n_components):
        cov_full = gmm.covariances_[k]  # (64, 64)
        # 变换到 PCA 空间: cov_2d = PCA_matrix @ cov_full @ PCA_matrix.T
        pca_components = pca.components_  # (2, 64)
        cov_2d = pca_components @ cov_full @ pca_components.T  # (2, 2)
        covs_2d.append(cov_2d)

    # 提取聚类特征示例
    sample_known_idx = np.random.RandomState(SEED).choice(len(known_feats), 1)[0]
    sample_unknown_idx = np.random.RandomState(SEED+1).choice(len(unknown_feats), 1)[0]
    sample_known = known_feats[sample_known_idx:sample_known_idx+1]
    sample_unknown = unknown_feats[sample_unknown_idx:sample_unknown_idx+1]

    # 计算聚类特征
    def compute_cluster_features(x):
        log_probs = gmm._estimate_weighted_log_prob(x)
        log_prob_max = log_probs.max(axis=1, keepdims=True)
        log_prob_shifted = log_probs - log_prob_max
        posteriors = np.exp(log_prob_shifted)
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)

        mahal_dists = []
        for k in range(gmm.n_components):
            diff = x - gmm.means_[k]
            prec = gmm.precisions_[k]
            mahal = np.sqrt(np.sum(diff @ prec * diff, axis=1))
            mahal_dists.append(mahal)
        mahal_dists = np.stack(mahal_dists, axis=1)

        cluster_mahal = mahal_dists.min(axis=1)[0]
        cluster_post_max = posteriors.max(axis=1)[0]
        posteriors_clipped = np.clip(posteriors, 1e-10, 1.0)
        cluster_entropy = -(posteriors_clipped * np.log(posteriors_clipped)).sum(axis=1)[0]
        return cluster_mahal, cluster_post_max, cluster_entropy

    kn_mahal, kn_post, kn_ent = compute_cluster_features(sample_known)
    uk_mahal, uk_post, uk_ent = compute_cluster_features(sample_unknown)

    # ---- 绘图 ----
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # 随机采样背景点
    rng = np.random.RandomState(SEED)
    kn_bg_idx = rng.choice(len(known_feats), min(300, len(known_feats)), replace=False)
    uk_bg_idx = rng.choice(len(unknown_feats), min(150, len(unknown_feats)), replace=False)

    ax.scatter(known_2d[kn_bg_idx, 0], known_2d[kn_bg_idx, 1], c='steelblue', alpha=0.2,
               s=10, label='Known samples', zorder=1)
    ax.scatter(unknown_2d[uk_bg_idx, 0], unknown_2d[uk_bg_idx, 1], c='salmon', alpha=0.2,
               s=10, label='Unknown samples', zorder=1)

    # GMM 分量椭圆
    for k in range(gmm.n_components):
        mean = means_2d[k]
        cov = covs_2d[k]
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 0)
            order = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * 2 * np.sqrt(eigenvalues)  # 2-sigma
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                              fill=False, edgecolor='navy', alpha=0.4, linewidth=1, linestyle='--')
            ax.add_patch(ellipse)
            ax.scatter(mean[0], mean[1], marker='+', c='navy', s=50, alpha=0.6, zorder=4)
        except Exception:
            pass

    # 标注已知类原型
    for c in BASE_CLASSES:
        ax.scatter(protos_2d[c, 0], protos_2d[c, 1], marker='*', s=200, c='blue',
                   edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(CLASS_NAMES[c], (protos_2d[c, 0], protos_2d[c, 1]),
                    fontsize=7, ha='left', va='bottom', xytext=(5, 5),
                    textcoords='offset points')

    # 高亮示例样本
    sk_2d = pca.transform(sample_known)[0]
    su_2d = pca.transform(sample_unknown)[0]
    ax.scatter(sk_2d[0], sk_2d[1], marker='o', s=150, c='lime',
               edgecolors='black', linewidths=2, zorder=6, label='Known example')
    ax.scatter(su_2d[0], su_2d[1], marker='^', s=150, c='red',
               edgecolors='black', linewidths=2, zorder=6, label='Unknown example')

    # 标注聚类特征值
    textstr_known = (f"Known example:\n"
                     f"  Mahal dist: {kn_mahal:.2f}\n"
                     f"  Post max:   {kn_post:.3f}\n"
                     f"  Entropy:    {kn_ent:.3f}")
    textstr_unknown = (f"Unknown example:\n"
                       f"  Mahal dist: {uk_mahal:.2f}\n"
                       f"  Post max:   {uk_post:.3f}\n"
                       f"  Entropy:    {uk_ent:.3f}")

    ax.text(0.02, 0.98, textstr_known, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.text(0.02, 0.72, textstr_unknown, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.8))

    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('GMM Cluster Features (12 components, 2-sigma ellipses)',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'fig3_4_gmm_clusters.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# ============ 图3.5: 融合权重搜索曲线 ============

def plot_fig3_5(model, calib_features, calib_labels, train_features, train_labels):
    """【图3.5】融合权重搜索曲线
    使用校准集进行搜索（与实际训练流程一致），展示 Youden J 随 alpha 变化的趋势。
    注: 实际10轮测试集评估中，最优 alpha*=0.60。
    """
    print("生成 【图3.5】 融合权重搜索曲线 ...")

    prototypes = compute_prototypes(train_features, train_labels, K_shot=50)
    adapted_feats, adapted_protos = apply_model(model, calib_features, prototypes)

    known_mask = torch.tensor([c in BASE_CLASSES for c in calib_labels.numpy()])
    known_feats = adapted_feats[known_mask]
    unknown_feats = adapted_feats[~known_mask]

    # 反原型评分（使用全局中心，与原始代码的批处理方式近似）
    all_feats = torch.cat([known_feats, unknown_feats])
    feat_center = all_feats.mean(dim=0, keepdim=True)
    anti_protos = 2 * feat_center - adapted_protos

    dists_proto_kn = torch.cdist(known_feats, adapted_protos, p=2)
    dists_anti_kn = torch.cdist(known_feats, anti_protos, p=2)
    known_anti = -dists_proto_kn.min(dim=1).values + dists_anti_kn.min(dim=1).values

    dists_proto_uk = torch.cdist(unknown_feats, adapted_protos, p=2)
    dists_anti_uk = torch.cdist(unknown_feats, anti_protos, p=2)
    unknown_anti = -dists_proto_uk.min(dim=1).values + dists_anti_uk.min(dim=1).values

    # GMM + OOD Head V2
    gmm = GaussianMixture(n_components=12, covariance_type='full',
                          max_iter=200, random_state=SEED, reg_covar=1e-4)
    gmm.fit(known_feats.numpy())

    def _cluster_feats(x_np, gmm_m):
        log_probs = gmm_m._estimate_weighted_log_prob(x_np)
        log_prob_max = log_probs.max(axis=1, keepdims=True)
        posteriors = np.exp(log_probs - log_prob_max)
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
        mahal_dists = []
        for k in range(gmm_m.n_components):
            diff = x_np - gmm_m.means_[k]
            prec = gmm_m.precisions_[k]
            mahal = np.sqrt(np.sum(diff @ prec * diff, axis=1))
            mahal_dists.append(mahal)
        mahal_dists = np.stack(mahal_dists, axis=1)
        return np.concatenate([
            mahal_dists.min(axis=1, keepdims=True),
            posteriors.max(axis=1, keepdims=True),
            -(np.clip(posteriors, 1e-10, 1.0) * np.log(np.clip(posteriors, 1e-10, 1.0))).sum(axis=1, keepdims=True)
        ], axis=1)

    def _ood_feats_v2(feats_t, protos_t, gmm_m):
        with torch.no_grad():
            fc = protos_t.mean(dim=0, keepdim=True)
            dists = torch.cdist(feats_t, protos_t, p=2)
            sorted_d, _ = dists.sort(dim=1)
            min_d = sorted_d[:, 0:1]
            second_d = sorted_d[:, 1:2]
            log_probs = model.classify(feats_t, protos_t)
            probs = F.softmax(log_probs, dim=1)
            softmax_max = probs.max(dim=1, keepdim=True)[0]
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
            feat_norm = (feats_t ** 2).sum(dim=1, keepdim=True).sqrt()
            dist_to_center = torch.cdist(feats_t, fc, p=2)
            proto_score_var = probs.var(dim=1, keepdim=True)
            cluster_feats = torch.from_numpy(
                _cluster_feats(feats_t.cpu().numpy(), gmm_m)).to(feats_t.device)
            return torch.cat([
                min_d, min_d / (second_d + 1e-6), softmax_max, entropy, feat_norm,
                second_d, second_d - min_d, dist_to_center,
                proto_score_var, min_d / (dist_to_center + 1e-6),
                cluster_feats
            ], dim=1).detach().numpy()

    known_ood_np = _ood_feats_v2(known_feats, adapted_protos, gmm)
    unknown_ood_np = _ood_feats_v2(unknown_feats, adapted_protos, gmm)

    X = np.vstack([known_ood_np, unknown_ood_np])
    y = np.concatenate([np.ones(len(known_ood_np)), np.zeros(len(unknown_ood_np))])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', solver='lbfgs', l1_ratio=0)
    clf.fit(X_scaled, y)

    known_ood_scores = torch.from_numpy(clf.predict_proba(scaler.transform(known_ood_np))[:, 1]).float()
    unknown_ood_scores = torch.from_numpy(clf.predict_proba(scaler.transform(unknown_ood_np))[:, 1]).float()

    # Z-score 归一化
    anti_mu, anti_std = known_anti.mean().item(), known_anti.std().item() + 1e-8
    ood_mu, ood_std = known_ood_scores.mean().item(), known_ood_scores.std().item() + 1e-8

    norm_known_anti = (known_anti - anti_mu) / anti_std
    norm_known_ood = (known_ood_scores - ood_mu) / ood_std
    norm_unknown_anti = (unknown_anti - anti_mu) / anti_std
    norm_unknown_ood = (unknown_ood_scores - ood_mu) / ood_std

    # 融合权重搜索
    alphas = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    target_fpr = 0.05
    j_scores, tpr_scores, tnr_scores = [], [], []

    for alpha in alphas:
        combined_known = alpha * norm_known_anti + (1 - alpha) * norm_known_ood
        combined_unknown = alpha * norm_unknown_anti + (1 - alpha) * norm_unknown_ood

        sorted_k = combined_known.sort()[0]
        idx = min(int(len(sorted_k) * target_fpr), len(sorted_k) - 1)
        tau = sorted_k[idx].item()
        tnr = (combined_known >= tau).float().mean().item()
        tpr = (combined_unknown < tau).float().mean().item()
        j = tnr + tpr - 1.0

        j_scores.append(j)
        tpr_scores.append(tpr * 100)
        tnr_scores.append(tnr * 100)

    j_scores = np.array(j_scores)
    tpr_scores = np.array(tpr_scores)
    tnr_scores = np.array(tnr_scores)

    best_idx = np.argmax(j_scores)
    best_alpha = alphas[best_idx]
    print(f"  校准集最优 alpha={best_alpha:.2f}, Youden J={j_scores[best_idx]:.3f}")
    print(f"  (实际10轮测试集评估中, alpha*=0.60, TPR=24.94%)")

    # ---- 绘图 ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_j = '#1f77b4'
    color_tpr = '#2ca02c'
    color_tnr = '#d62728'

    ax1.set_xlabel(r'Fusion weight $\alpha$ (anti-prototype weight)', fontsize=12)
    ax1.set_ylabel('Youden J Statistic', color=color_j, fontsize=12)
    ax1.plot(alphas, j_scores, 'o-', color=color_j, linewidth=2, markersize=6,
             label='Youden J')
    ax1.tick_params(axis='y', labelcolor=color_j)

    # 标注实际实验最优 alpha=0.60
    actual_alpha = 0.60
    ax1.axvline(x=actual_alpha, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    j_at_06 = j_scores[list(alphas).index(0.6)]
    ax1.annotate(f'$\\alpha^*$=0.60\nJ={j_at_06:.3f}\n(Test: TPR=24.94%)',
                 xy=(actual_alpha, j_at_06),
                 xytext=(actual_alpha - 0.22, j_at_06 + 0.01),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='red'),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Rate (%)', fontsize=12)
    line_tpr, = ax2.plot(alphas, tpr_scores, 's--', color=color_tpr, linewidth=1.5,
                         markersize=5, label='TPR (%)')
    line_tnr, = ax2.plot(alphas, tnr_scores, '^--', color=color_tnr, linewidth=1.5,
                         markersize=5, label='TNR (%)')
    ax2.set_ylim([80, 100])

    # 手动创建 legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=color_j, marker='o', linewidth=2, label='Youden J'),
        Line2D([0], [0], color=color_tpr, marker='s', linewidth=1.5, linestyle='--', label='TPR (%)'),
        Line2D([0], [0], color=color_tnr, marker='^', linewidth=1.5, linestyle='--', label='TNR (%)'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=1.5, label=r'$\alpha^*=0.60$'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax1.set_title('Fusion Weight Search via Youden J Criterion (TNR=95%)',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'fig3_5_fusion_weight_search.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# ============ 图3.6: 两种信号的互补性可视化 ============

def plot_fig3_6(model, calib_features, calib_labels, train_features, train_labels):
    """【图3.6】两种信号的互补性可视化 — s_anti vs s_ood 散点图"""
    print("生成 【图3.6】 两种信号的互补性可视化 ...")

    prototypes = compute_prototypes(train_features, train_labels, K_shot=50)
    adapted_feats, adapted_protos = apply_model(model, calib_features, prototypes)

    known_mask = torch.tensor([c in BASE_CLASSES for c in calib_labels.numpy()])
    known_feats = adapted_feats[known_mask]
    unknown_feats = adapted_feats[~known_mask]

    # 反原型评分
    def score_anti(feats, protos):
        feat_center = feats.mean(dim=0, keepdim=True)
        anti_protos = 2 * feat_center - protos
        dists_proto = torch.cdist(feats, protos, p=2)
        dists_anti = torch.cdist(feats, anti_protos, p=2)
        return -dists_proto.min(dim=1).values + dists_anti.min(dim=1).values

    known_anti = score_anti(known_feats, adapted_protos)
    unknown_anti = score_anti(unknown_feats, adapted_protos)

    # OOD Head V2 评分（使用 GMM）
    gmm = GaussianMixture(n_components=12, covariance_type='full',
                          max_iter=200, random_state=SEED, reg_covar=1e-4)
    gmm.fit(known_feats.numpy())

    def extract_cluster_features_np(x_np, gmm_model):
        log_probs = gmm_model._estimate_weighted_log_prob(x_np)
        log_prob_max = log_probs.max(axis=1, keepdims=True)
        posteriors = np.exp(log_probs - log_prob_max)
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
        mahal_dists = []
        for k in range(gmm_model.n_components):
            diff = x_np - gmm_model.means_[k]
            prec = gmm_model.precisions_[k]
            mahal = np.sqrt(np.sum(diff @ prec * diff, axis=1))
            mahal_dists.append(mahal)
        mahal_dists = np.stack(mahal_dists, axis=1)
        cluster_mahal = mahal_dists.min(axis=1, keepdims=True)
        cluster_post_max = posteriors.max(axis=1, keepdims=True)
        posteriors_clipped = np.clip(posteriors, 1e-10, 1.0)
        cluster_entropy = -(posteriors_clipped * np.log(posteriors_clipped)).sum(axis=1, keepdims=True)
        return np.concatenate([cluster_mahal, cluster_post_max, cluster_entropy], axis=1)

    def extract_ood_features_v2(feats_t, protos_t, gmm_model):
        with torch.no_grad():
            feat_center = protos_t.mean(dim=0, keepdim=True)
            dists = torch.cdist(feats_t, protos_t, p=2)
            sorted_d, _ = dists.sort(dim=1)
            min_d = sorted_d[:, 0:1]
            second_d = sorted_d[:, 1:2]
            dist_ratio = min_d / (second_d + 1e-6)
            score_gap = second_d - min_d
            log_probs = model.classify(feats_t, protos_t)
            probs = F.softmax(log_probs, dim=1)
            softmax_max = probs.max(dim=1, keepdim=True)[0]
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
            feat_norm = (feats_t ** 2).sum(dim=1, keepdim=True).sqrt()
            dist_to_center = torch.cdist(feats_t, feat_center, p=2)
            proto_score_var = probs.var(dim=1, keepdim=True)
            norm_dist_ratio = min_d / (dist_to_center + 1e-6)
            cluster_feats = torch.from_numpy(
                extract_cluster_features_np(feats_t.cpu().numpy(), gmm_model)
            ).to(feats_t.device)
            return torch.cat([
                min_d, dist_ratio, softmax_max, entropy, feat_norm,
                second_d, score_gap, dist_to_center,
                proto_score_var, norm_dist_ratio, cluster_feats
            ], dim=1).detach()

    known_ood_feats = extract_ood_features_v2(known_feats, adapted_protos, gmm).numpy()
    unknown_ood_feats = extract_ood_features_v2(unknown_feats, adapted_protos, gmm).numpy()

    X = np.vstack([known_ood_feats, unknown_ood_feats])
    y = np.concatenate([np.ones(len(known_ood_feats)), np.zeros(len(unknown_ood_feats))])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', solver='lbfgs', l1_ratio=0)
    clf.fit(X_scaled, y)

    known_ood_scores = clf.predict_proba(scaler.transform(known_ood_feats))[:, 1]
    unknown_ood_scores = clf.predict_proba(scaler.transform(unknown_ood_feats))[:, 1]

    # Z-score 归一化
    anti_mu, anti_std = known_anti.mean().item(), known_anti.std().item() + 1e-8
    ood_mu, ood_std = known_ood_scores.mean(), known_ood_scores.std() + 1e-8

    norm_known_anti = ((known_anti - anti_mu) / anti_std).numpy()
    norm_unknown_anti = ((unknown_anti - anti_mu) / anti_std).numpy()
    norm_known_ood = (known_ood_scores - ood_mu) / ood_std
    norm_unknown_ood = (unknown_ood_scores - ood_mu) / ood_std

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(10, 8))

    rng = np.random.RandomState(SEED)
    # 随机采样以便可视化
    kn_idx = rng.choice(len(norm_known_anti), min(800, len(norm_known_anti)), replace=False)
    uk_idx = rng.choice(len(norm_unknown_anti), min(500, len(norm_unknown_anti)), replace=False)

    ax.scatter(norm_known_anti[kn_idx], norm_known_ood[kn_idx],
               c='steelblue', alpha=0.3, s=15, label='Known', zorder=2)
    ax.scatter(norm_unknown_anti[uk_idx], norm_unknown_ood[uk_idx],
               c='salmon', alpha=0.4, s=20, marker='^', label='Unknown', zorder=3)

    # 画融合决策边界（alpha=0.6）
    alpha = 0.6
    x_range = np.linspace(norm_known_anti.min() - 0.5, norm_unknown_anti.max() + 0.5, 100)

    # 计算阈值线: alpha * s_anti + (1-alpha) * s_ood = tau
    # 在校准集已知类上计算阈值
    combined_known = alpha * norm_known_anti + (1 - alpha) * norm_known_ood
    sorted_k = np.sort(combined_known)
    tau_idx = min(int(len(sorted_k) * 0.05), len(sorted_k) - 1)
    tau = sorted_k[tau_idx]

    y_range = (tau - alpha * x_range) / (1 - alpha)
    ax.plot(x_range, y_range, 'k--', linewidth=1.5, alpha=0.7,
            label=f'Decision boundary ($\\alpha$={alpha:.1f}, $\\tau$={tau:.2f})')

    ax.set_xlabel(r'$\hat{s}_{\mathrm{anti}}$ (Z-score normalized)', fontsize=12)
    ax.set_ylabel(r'$\hat{s}_{\mathrm{ood}}$ (Z-score normalized)', fontsize=12)
    ax.set_title('Complementarity of Geometric and Statistical Signals', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    # 添加分布直方图（边缘）
    # 上方：s_anti 的分布
    divider_x = ax.inset_axes([0, 1.0, 1, 0.15])
    divider_x.hist(norm_known_anti[kn_idx], bins=40, color='steelblue', alpha=0.5, density=True)
    divider_x.hist(norm_unknown_anti[uk_idx], bins=40, color='salmon', alpha=0.5, density=True)
    divider_x.set_xlim(ax.get_xlim())
    divider_x.axis('off')

    # 右侧：s_ood 的分布
    divider_y = ax.inset_axes([1.0, 0, 0.15, 1])
    divider_y.hist(norm_known_ood[kn_idx], bins=40, color='steelblue', alpha=0.5,
                   density=True, orientation='horizontal')
    divider_y.hist(norm_unknown_ood[uk_idx], bins=40, color='salmon', alpha=0.5,
                   density=True, orientation='horizontal')
    divider_y.set_ylim(ax.get_ylim())
    divider_y.axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'fig3_6_signal_complementarity.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {save_path}")


# ============ 主函数 ============

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("第三章图片生成脚本")
    print("=" * 60)

    # 加载数据
    print("\n加载特征数据...")
    calib_features, calib_labels = load_features('calib')
    train_features, train_labels = load_features('train')
    print(f"  训练集: {train_features.shape}, 校准集: {calib_features.shape}")

    # 加载模型
    print("\n加载最佳模型...")
    model = load_model()
    print("  模型加载完成")

    # 生成各图
    print("\n" + "=" * 60)
    plot_fig3_3(model, calib_features, calib_labels, train_features, train_labels)

    print("\n" + "=" * 60)
    plot_fig3_4(model, calib_features, calib_labels, train_features, train_labels)

    print("\n" + "=" * 60)
    plot_fig3_5(model, calib_features, calib_labels, train_features, train_labels)

    print("\n" + "=" * 60)
    plot_fig3_6(model, calib_features, calib_labels, train_features, train_labels)

    print("\n" + "=" * 60)
    print("所有图片生成完毕!")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
