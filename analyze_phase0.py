#!/usr/bin/env python3
"""
Phase 0 特征质量分析脚本

诊断 YAMNet → FC 特征提取器的分类能力
分析维度：
  1. 原始 YAMNet 特征 (521-dim) 的类间区分度
  2. FC 变换后特征 (64-dim) 的类间区分度
  3. kNN / 线性探测 基线准确率
  4. 各类别特征分布分析
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import os

from utils.TAU22 import TAUDataset
import yamnet_PT_inference as yamnet_infer
from torch_audioset.yamnet.model import yamnet as torch_yamnet
from episodic_trainer import FeatureExtractor

torch.backends.cudnn.benchmark = True

def extract_features_at_levels(dataset, feature_extractor, device, max_samples=6000):
    """提取不同层级的特征"""
    feature_extractor.eval()

    raw_yamnet_feats = []  # 521-dim
    fc1_feats = []         # 256-dim
    fc2_feats = []         # 128-dim
    final_feats = []       # 64-dim
    labels = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

    count = 0
    with torch.no_grad():
        for item in loader:
            if count >= max_samples:
                break
            audio = item['source_audio'].to(device)
            target = item['target'].squeeze(1).argmax(dim=1)

            # 原始 YAMNet 特征
            x = yamnet_infer.waveform_to_log_mel_patches(audio, sample_rate=16000)
            raw = feature_extractor.feature_model(x, to_prob=False)  # (B, 521)

            # 各层 FC 特征
            h1 = feature_extractor.drop1(feature_extractor.se1(feature_extractor.fc1(raw)))
            h2 = feature_extractor.drop2(feature_extractor.se2(feature_extractor.fc2(h1)))
            h3 = feature_extractor.fc3(h2)

            s1 = feature_extractor.proj_s1(h1)
            s2 = feature_extractor.proj_s2(h2)
            final = feature_extractor.fusion(torch.cat([s1, s2, h3], dim=1))

            raw_yamnet_feats.append(raw.cpu())
            fc1_feats.append(h1.cpu())
            fc2_feats.append(h2.cpu())
            final_feats.append(final.cpu())
            labels.append(target)
            count += len(audio)

    return {
        'raw_yamnet': torch.cat(raw_yamnet_feats),
        'fc1': torch.cat(fc1_feats),
        'fc2': torch.cat(fc2_feats),
        'final_64d': torch.cat(final_feats),
        'labels': torch.cat(labels),
    }


def compute_class_metrics(features, labels, name):
    """计算类间/类内距离比和 silhouette score"""
    feats_np = features.numpy()
    labels_np = labels.numpy()

    # Subsample for speed
    if len(feats_np) > 3000:
        idx = np.random.choice(len(feats_np), 3000, replace=False)
        feats_np = feats_np[idx]
        labels_np = labels_np[idx]

    # Silhouette score (越大越好, 范围 [-1, 1])
    sil = silhouette_score(feats_np, labels_np, sample_size=min(2000, len(feats_np)))

    # 类内/类间距离
    unique_labels = np.unique(labels_np)
    intra_dists = []
    inter_dists = []

    for c in unique_labels:
        mask = labels_np == c
        class_feats = feats_np[mask]
        if len(class_feats) < 2:
            continue
        # 类内: 随机采样 pairwise distance
        sample_idx = np.random.choice(len(class_feats), min(100, len(class_feats)), replace=False)
        sampled = class_feats[sample_idx]
        dists = cdist(sampled, sampled)
        np.fill_diagonal(dists, np.nan)
        intra_dists.append(np.nanmean(dists))

    # 类间: 每个类的 centroid 到其他类 centroid 的距离
    centroids = []
    for c in unique_labels:
        mask = labels_np == c
        centroids.append(feats_np[mask].mean(axis=0))
    centroids = np.array(centroids)
    inter_dists_mat = cdist(centroids, centroids)
    np.fill_diagonal(inter_dists_mat, np.nan)
    mean_inter = np.nanmean(inter_dists_mat)
    mean_intra = np.mean(intra_dists)

    ratio = mean_inter / (mean_intra + 1e-8)

    print(f"  {name:20s} | Silhouette: {sil:+.4f} | Intra: {mean_intra:.2f} | Inter: {mean_inter:.2f} | Ratio: {ratio:.2f}")
    return sil, ratio


def baseline_classify(features, labels, name):
    """kNN 和线性探测基线"""
    feats_np = features.numpy()
    labels_np = labels.numpy()

    # 80/20 split
    n = len(feats_np)
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = feats_np[train_idx], feats_np[test_idx]
    y_train, y_test = labels_np[train_idx], labels_np[test_idx]

    # 标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # kNN (k=5)
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_s, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test_s))

    # 线性探测 (Logistic Regression)
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X_train_s, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test_s))

    print(f"  {name:20s} | kNN-5: {knn_acc:.2%} | Linear Probe: {lr_acc:.2%}")
    return knn_acc, lr_acc


def per_class_analysis(features, labels, class_names=None):
    """各类别准确率分析 (使用最近质心分类器)"""
    feats_np = features.numpy()
    labels_np = labels.numpy()

    unique = np.unique(labels_np)

    # 计算每个类的 centroid
    centroids = {}
    for c in unique:
        mask = labels_np == c
        centroids[c] = feats_np[mask].mean(axis=0)

    # 最近质心分类
    centroid_arr = np.array([centroids[c] for c in unique])
    dists = cdist(feats_np, centroid_arr)
    preds = unique[dists.argmin(axis=1)]

    print("\n  Per-class accuracy (Nearest Centroid):")
    total_correct = 0
    for c in unique:
        mask = labels_np == c
        correct = (preds[mask] == c).sum()
        total = mask.sum()
        acc = correct / total
        total_correct += correct
        name = class_names[c] if class_names else f"Class {c}"
        print(f"    {name:20s}: {acc:.2%} ({correct}/{total})")

    print(f"    {'Overall':20s}: {total_correct/len(labels_np):.2%}")


def analyze_feature_stats(features, labels, name):
    """分析特征统计特性"""
    feats_np = features.numpy()
    labels_np = labels.numpy()

    print(f"\n  {name} statistics:")
    print(f"    Mean norm:  {np.linalg.norm(feats_np, axis=1).mean():.4f}")
    print(f"    Std of norms: {np.linalg.norm(feats_np, axis=1).std():.4f}")
    print(f"    Feature std (per dim): {feats_np.std(axis=0).mean():.4f}")

    # 检查特征是否坍缩
    cos_sims = []
    for i in range(min(1000, len(feats_np)-1)):
        a, b = feats_np[i], feats_np[i+1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        cos_sims.append(cos_sim)
    mean_cos = np.mean(cos_sims)
    print(f"    Mean cosine sim (random pairs): {mean_cos:.4f}",
          " ← HIGH = feature collapse!" if mean_cos > 0.9 else "")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_names = {
        0: "Airport", 1: "Tram", 2: "Bus", 3: "Public Square",
        4: "Shopping Mall", 5: "Street Pedestrian"
    }

    # 加载数据和模型
    print("\nLoading dataset...")
    dataset = TAUDataset(split='train')
    base_classes = [0, 1, 2, 3, 4, 5]

    # 过滤 base 类
    indices = []
    for i in range(len(dataset)):
        item = dataset[i]
        label_idx = item['target'].argmax().item()
        if label_idx in base_classes:
            indices.append(i)
    dataset = torch.utils.data.Subset(dataset, indices)
    print(f"Base class samples: {len(dataset)}")

    # 加载训练好的特征提取器
    print("\nLoading trained feature extractor...")
    fe = FeatureExtractor(pretrained_path='experiment/fewshot/base_feature_extractor.pth')
    fe = fe.to(device).eval()

    # 提取各层特征
    print("\nExtracting features at all levels...")
    data = extract_features_at_levels(dataset, fe, device, max_samples=6000)
    print(f"Extracted {len(data['labels'])} samples")

    # ===== 分析 =====
    print("\n" + "="*70)
    print("ANALYSIS 1: Feature Space Quality (Silhouette & Distance Ratio)")
    print("="*70)
    print("  (Silhouette: higher=better, Range [-1,1])")
    print("  (Ratio = inter/intra: higher=better)")
    print()

    for key in ['raw_yamnet', 'fc1', 'fc2', 'final_64d']:
        compute_class_metrics(data[key], data['labels'], key)

    print("\n" + "="*70)
    print("ANALYSIS 2: Baseline Classification Accuracy")
    print("="*70)
    print("  (Upper bound for downstream few-shot classifiers)")
    print()

    for key in ['raw_yamnet', 'fc1', 'fc2', 'final_64d']:
        baseline_classify(data[key], data['labels'], key)

    print("\n" + "="*70)
    print("ANALYSIS 3: Per-Class Accuracy (Final 64-dim features)")
    print("="*70)
    per_class_analysis(data['final_64d'], data['labels'], class_names)

    print("\n" + "="*70)
    print("ANALYSIS 4: Feature Statistics & Collapse Detection")
    print("="*70)
    for key in ['raw_yamnet', 'final_64d']:
        analyze_feature_stats(data[key], data['labels'], key)

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    # 自动诊断
    raw_sil, _ = compute_class_metrics_silent(data['raw_yamnet'], data['labels'])
    final_sil, _ = compute_class_metrics_silent(data['final_64d'], data['labels'])

    if raw_sil < 0.05:
        print("\n  [CRITICAL] Raw YAMNet features have very low class separability!")
        print("  → YAMNet (AudioSet pretrained) features are NOT discriminative for acoustic scenes.")
        print("  → RECOMMENDATION: Fine-tune YAMNet layers or use a scene-specific pretrained model.")
    elif raw_sil > 0.1 and final_sil < raw_sil:
        print("\n  [ISSUE] FC layers are DEGRADING the features!")
        print(f"  → Raw YAMNet silhouette: {raw_sil:.4f}, After FC: {final_sil:.4f}")
        print("  → RECOMMENDATION: Simplify FC architecture or improve FC training.")
    else:
        print("\n  [INFO] Feature quality is acceptable but accuracy is limited.")
        print("  → RECOMMENDATION: Try unfreezing some YAMNet layers for fine-tuning.")

    # 检查特征坍缩
    final_np = data['final_64d'].numpy()
    cos_sims = []
    for i in range(min(1000, len(final_np)-1)):
        a, b = final_np[i], final_np[i+1]
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        cos_sims.append(cos_sim)
    if np.mean(cos_sims) > 0.9:
        print("\n  [CRITICAL] Feature collapse detected! All features are nearly identical.")
        print("  → RECOMMENDATION: Check FC initialization, reduce regularization, or use contrastive loss.")


def compute_class_metrics_silent(features, labels):
    """静默版本，返回指标不打印"""
    feats_np = features.numpy()
    labels_np = labels.numpy()
    if len(feats_np) > 3000:
        idx = np.random.choice(len(feats_np), 3000, replace=False)
        feats_np = feats_np[idx]
        labels_np = labels_np[idx]
    sil = silhouette_score(feats_np, labels_np, sample_size=min(2000, len(feats_np)))
    unique_labels = np.unique(labels_np)
    centroids = [feats_np[labels_np == c].mean(axis=0) for c in unique_labels]
    centroids = np.array(centroids)
    inter = np.nanmean(cdist(centroids, centroids) + np.eye(len(centroids)) * np.nan)
    intra_list = []
    for c in unique_labels:
        cf = feats_np[labels_np == c]
        if len(cf) >= 2:
            idx2 = np.random.choice(len(cf), min(100, len(cf)), replace=False)
            d = cdist(cf[idx2], cf[idx2])
            np.fill_diagonal(d, np.nan)
            intra_list.append(np.nanmean(d))
    ratio = inter / (np.mean(intra_list) + 1e-8)
    return sil, ratio


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
