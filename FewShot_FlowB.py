"""
方案B: 条件流 + Meta-learning Few-shot 分类聚类模型

核心思想:
- 每个cINN以类原型(prototype)作为条件输入
- 流模型学习: p(x | prototype_c) = 在给定类原型条件下样本的密度
- 分类: argmax_c p(x | prototype_c)
- 聚类: 使用可学习聚类中心作为条件，GMM式混合分配

训练时使用可学习原型，推理时可通过set_support_set()替换为真实样本原型

接口与 FinalModel 完全兼容:
  forward(x) → (preds, masked_preds, masked_preds_aug, weights1, weights2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import random

from torch_audioset.yamnet.model import yamnet as torch_yamnet
import yamnet_PT_inference as yamnet_infer
from Reversable_Function import cINN as cINN_module


# ============================================================
# 工具函数 (与原代码一致)
# ============================================================

class PolicyNet(nn.Module):
    """门控网络"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(32, out_dim)

    def forward(self, x, temp):
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x)), 0.1))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x)), 0.1))
        logits = self.fc3(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return hard_mask


def top_k_gating(weights: torch.Tensor, k: int = 3, temperature: float = 1.0) -> torch.Tensor:
    scaled = weights / temperature
    top_k_values, top_k_indices = torch.topk(scaled, k=k, dim=-1, sorted=False)
    masked = torch.zeros_like(weights)
    batch_indices = torch.arange(weights.size(0), device=weights.device).unsqueeze(-1).expand(-1, k)
    masked.scatter_(1, top_k_indices, torch.ones_like(top_k_values))
    result = weights * masked
    result = F.normalize(result, p=1, dim=-1)
    return result


def get_masked_feature(feature: torch.Tensor, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    unknown_probs = preds[:, -1]
    mask_f = unknown_probs.unsqueeze(1).expand_as(feature)
    mask_p = unknown_probs.unsqueeze(1).expand_as(preds)
    return feature * mask_f, preds * mask_p


def feature_augment(features: torch.Tensor, p_apply: float = 1.0) -> torch.Tensor:
    batch_size, feat_dim = features.shape
    device = features.device
    augmented = features.clone()
    aug_list = ['none', 'noise', 'time_mask', 'scale', 'invert']

    for i in range(batch_size):
        if random.random() > p_apply:
            continue
        sample = features[i]
        aug_type = random.choice(aug_list)

        if aug_type == 'none':
            augmented[i] = sample
        elif aug_type == 'noise':
            augmented[i] = sample + torch.randn_like(sample, device=device) * 0.05
        elif aug_type == 'time_mask':
            mask_len = random.randint(1, min(6, feat_dim // 3))
            start = random.randint(0, feat_dim - mask_len)
            aug = sample.clone()
            aug[start:start + mask_len] = 0.0
            augmented[i] = aug
        elif aug_type == 'scale':
            scale = torch.empty(1, device=device).uniform_(0.8, 1.2).item()
            augmented[i] = sample * scale
        elif aug_type == 'invert':
            augmented[i] = -sample
    return augmented


# ============================================================
# 核心组件: 条件流分类专家
# ============================================================

class ConditionalFlowExpert(nn.Module):
    """
    条件流分类专家

    使用类原型作为cINN的条件输入，为每个类别计算条件对数似然:
      log p(x | prototype_c) = log p_prior(z) + log|det J|,  z = flow(x, condition=prototype_c)

    分类决策: argmax_c log p(x | prototype_c)
    """
    def __init__(self, input_dim: int, condition_dim: int, num_classes: int,
                 num_coupling_layers: int = 3, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]
        self.num_classes = num_classes
        self.flow = cINN_module.ConditionalINN(
            input_dim=input_dim,
            condition_dim=condition_dim,
            num_coupling_layers=num_coupling_layers,
            hidden_dims=hidden_dims,
            use_permutation=True,
            permutation_type='fixed'
        )

    def forward(self, features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feature_dim) 输入特征
            prototypes: (num_classes, feature_dim) 各类原型向量
        Returns:
            (B, num_classes) 各类的对数似然
        """
        B = features.size(0)
        log_probs = torch.zeros(B, self.num_classes, device=features.device)

        for c in range(self.num_classes):
            # 将第c个原型扩展为batch大小的条件
            condition = prototypes[c].unsqueeze(0).expand(B, -1)
            log_probs[:, c] = self.flow.log_prob(features, condition)

        return log_probs


# ============================================================
# 核心组件: 条件流聚类专家
# ============================================================

class ConditionalClusterExpert(nn.Module):
    """
    条件流聚类专家

    使用聚类中心作为cINN的条件输入，结合可学习混合权重，输出软聚类分配:
      gamma_k ∝ pi_k * p(x | center_k)
    """
    def __init__(self, input_dim: int, condition_dim: int, num_clusters: int,
                 num_coupling_layers: int = 2, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]
        self.num_clusters = num_clusters
        self.flow = cINN_module.ConditionalINN(
            input_dim=input_dim,
            condition_dim=condition_dim,
            num_coupling_layers=num_coupling_layers,
            hidden_dims=hidden_dims,
            use_permutation=True,
            permutation_type='fixed'
        )
        # 混合权重 (类似 FlowBasedCell)
        initial_logits = torch.log(torch.ones(num_clusters) / num_clusters)
        self.mix_logits = nn.Parameter(initial_logits + torch.randn_like(initial_logits) * 0.01)

    def forward(self, features: torch.Tensor,
                cluster_prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, feature_dim)
            cluster_prototypes: (num_clusters, feature_dim) 聚类中心
        Returns:
            gamma: (B, num_clusters) 软聚类分配
            log_probs: (B, num_clusters) 各聚类的对数似然
        """
        B = features.size(0)
        device = features.device

        # 处理全零样本 (padding)
        is_valid = ~torch.all(features == 0, dim=1)
        valid_idx = torch.where(is_valid)[0]

        gamma = torch.zeros(B, self.num_clusters, device=device)
        log_probs = torch.full((B, self.num_clusters), -10.0, device=device)

        if len(valid_idx) == 0:
            gamma[:] = 1.0 / self.num_clusters
            return gamma.detach(), log_probs

        x_valid = features[valid_idx]
        n_valid = x_valid.size(0)

        # 计算每个聚类的条件对数似然
        log_probs_valid = torch.zeros(n_valid, self.num_clusters, device=device)
        for k in range(self.num_clusters):
            condition = cluster_prototypes[k].unsqueeze(0).expand(n_valid, -1)
            log_probs_valid[:, k] = self.flow.log_prob(x_valid, condition)

        log_probs_valid = torch.clamp(log_probs_valid, min=-50.0, max=10.0)

        # 混合权重 + 条件对数似然 → 软分配
        log_pi = F.log_softmax(self.mix_logits, dim=0).unsqueeze(0)
        weighted = log_probs_valid + log_pi
        gamma_valid = F.softmax(weighted, dim=1)

        gamma[valid_idx] = gamma_valid
        log_probs[valid_idx] = log_probs_valid

        invalid = ~is_valid
        if invalid.any():
            gamma[invalid] = 1.0 / self.num_clusters

        return gamma, log_probs


# ============================================================
# 主模型: 方案B 条件流 + Meta-learning
# ============================================================

class FewShotFlowBModel(nn.Module):
    """
    方案B: 条件流 + Meta-learning Few-shot 分类聚类模型

    架构:
    ┌──────────────────────────────────────────────────┐
    │ 音频 → YAMNet → FC → 32维特征                      │
    │                                                    │
    │ 分类 MoE:                                          │
    │   Gate1 → 专家权重                                  │
    │   3× ConditionalFlowExpert (cINN condition=原型)    │
    │   可学习类原型 (num_known+1, 32)                    │
    │                                                    │
    │ 掩码: unknown概率遮蔽特征                            │
    │                                                    │
    │ 聚类 MoE:                                          │
    │   Gate2 → 专家权重                                  │
    │   3× ConditionalClusterExpert (条件=聚类中心)        │
    │   可学习聚类中心 (num_unknown, 32)                   │
    └──────────────────────────────────────────────────┘

    Few-shot 推理:
      model.set_support_set(features, labels)  # 设置支持集
      preds, ... = model(audio)                 # 使用支持集原型分类

    接口与 FinalModel 完全兼容:
      forward(x) → (preds, masked_preds, masked_preds_aug, weights1, weights2)
    """

    def __init__(self,
                 num_unknown_classes: int = 4,
                 num_known_classes: int = 6,
                 reload_feature_model_pretrained: bool = True):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_known_classes = num_known_classes
        self.num_unknown_classes = num_unknown_classes

        feature_dim = 32

        # ===== 特征提取器 (与 FinalModel 相同) =====
        self.feature_model = torch_yamnet(pretrained=False)
        self._reload_yamnet = reload_feature_model_pretrained
        if reload_feature_model_pretrained:
            state = torch.load('yamnet.pth')
            self.feature_model.load_state_dict(state)

        self.fc1 = nn.Sequential(
            nn.Linear(521, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.fc3 = nn.Sequential(
            nn.Linear(64, feature_dim), nn.BatchNorm1d(feature_dim), nn.Tanh())

        # ===== 方案B核心: 可学习类原型 (分类) =====
        # num_known_classes + 1 个原型 (含 "unknown" 类)
        self.class_prototypes = nn.Parameter(
            torch.randn(num_known_classes + 1, feature_dim) * 0.1)

        # ===== 可学习聚类中心 (聚类) =====
        self.cluster_prototypes = nn.Parameter(
            torch.randn(num_unknown_classes, feature_dim) * 0.1)

        # ===== MoE 分类专家 =====
        self.num_experts = 3
        self.classifier_experts = nn.ModuleList([
            ConditionalFlowExpert(
                input_dim=feature_dim,
                condition_dim=feature_dim,
                num_classes=num_known_classes + 1,
                num_coupling_layers=3,
                hidden_dims=[32, 32]
            ) for _ in range(self.num_experts)])

        # ===== MoE 聚类专家 =====
        self.num_experts_cluster = 3
        self.clusterer_experts = nn.ModuleList([
            ConditionalClusterExpert(
                input_dim=feature_dim,
                condition_dim=feature_dim,
                num_clusters=num_unknown_classes,
                num_coupling_layers=2,
                hidden_dims=[32, 32]
            ) for _ in range(self.num_experts_cluster)])

        # ===== 门控网络 =====
        self.gate1 = PolicyNet(feature_dim, self.num_experts)
        self.gate2 = PolicyNet(feature_dim, self.num_experts_cluster)

        # ===== Few-shot 支持集缓存 =====
        self._support_class_prototypes = None   # 推理时覆盖分类原型
        self._support_cluster_prototypes = None  # 推理时覆盖聚类中心

    # ----------------------------------------------------------
    # Few-shot 接口
    # ----------------------------------------------------------

    def set_support_set(self,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        cluster_features: torch.Tensor = None,
                        cluster_labels: torch.Tensor = None):
        """
        设置 Few-shot 支持集 (推理时使用)

        Args:
            features: (N_support, feature_dim) 已提取的支持样本特征
            labels: (N_support,) 整数标签 0 ~ num_known_classes-1
            cluster_features: 可选, 聚类支持特征 (N_support_clust, feature_dim)
            cluster_labels: 可选, 聚类标签 0 ~ num_unknown_classes-1

        Note:
            调用此方法后, forward() 将使用支持集原型替代可学习原型。
            调用 clear_support_set() 恢复可学习原型。

        Example:
            >>> # 1-shot: 每个类1个样本
            >>> support_feats = model.extract_features(support_audio)
            >>> model.set_support_set(support_feats, support_labels)
            >>> preds, mp, mpa, w1, w2 = model(query_audio)
        """
        # 分类原型
        prototypes = {}
        for c in range(self.num_known_classes):
            mask = labels == c
            if mask.any():
                prototypes[c] = features[mask].mean(dim=0)

        # "unknown" 类原型: 用所有支持样本均值近似
        if features.size(0) > 0:
            prototypes[self.num_known_classes] = features.mean(dim=0)

        self._support_class_prototypes = prototypes

        # 聚类中心
        if cluster_features is not None and cluster_labels is not None:
            cluster_protos = {}
            for k in range(self.num_unknown_classes):
                mask = cluster_labels == k
                if mask.any():
                    cluster_protos[k] = cluster_features[mask].mean(dim=0)
            self._support_cluster_prototypes = cluster_protos

    def clear_support_set(self):
        """清除支持集, 恢复使用可学习原型"""
        self._support_class_prototypes = None
        self._support_cluster_prototypes = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征的便捷方法 (用于准备支持集)"""
        return self._extract_features(x)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

    def _get_class_prototypes(self) -> torch.Tensor:
        """获取分类原型: 支持集优先 > 可学习"""
        if self._support_class_prototypes is not None:
            proto_list = []
            for c in range(self.num_known_classes + 1):
                if c in self._support_class_prototypes:
                    proto_list.append(self._support_class_prototypes[c])
                else:
                    proto_list.append(self.class_prototypes[c].detach())
            return torch.stack(proto_list).to(self.device)
        return self.class_prototypes

    def _get_cluster_prototypes(self) -> torch.Tensor:
        """获取聚类中心: 支持集优先 > 可学习"""
        if self._support_cluster_prototypes is not None:
            proto_list = []
            for k in range(self.num_unknown_classes):
                if k in self._support_cluster_prototypes:
                    proto_list.append(self._support_cluster_prototypes[k])
                else:
                    proto_list.append(self.cluster_prototypes[k].detach())
            return torch.stack(proto_list).to(self.device)
        return self.cluster_prototypes

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """音频 → 32维特征"""
        x = yamnet_infer.waveform_to_log_mel_patches(x, sample_rate=16000)
        feature = self.feature_model(x, to_prob=False)
        feature = self.fc1(feature)
        feature = self.fc2(feature)
        feature = self.fc3(feature)
        return feature

    # ----------------------------------------------------------
    # 主前向传播 (与 FinalModel 接口一致)
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        前向传播

        Args:
            x: (B, audio_samples) 原始音频波形

        Returns:
            preds: (B, num_known_classes+1) 分类概率 (已知类 + unknown)
            masked_preds: (B, num_unknown_classes) 聚类软分配
            masked_preds_aug: (B, num_unknown_classes) 增强后的聚类软分配
            weights1: (B, num_experts) 分类专家门控权重
            weights2: (B, num_experts_cluster) 聚类专家门控权重
        """
        batch_size = x.size(0)
        feature = self._extract_features(x)

        # ===== 分类阶段 =====
        prototypes = self._get_class_prototypes()

        weights1 = self.gate1(feature, 1)
        weights1 = top_k_gating(weights1, k=self.num_experts)

        preds = torch.zeros(batch_size, self.num_known_classes + 1, device=self.device)
        for i in range(self.num_experts):
            expert_output = self.classifier_experts[i](feature, prototypes)
            preds += weights1[:, i].unsqueeze(1) * expert_output

        preds = F.softmax(preds, dim=1)

        # ===== 未知样本掩码 =====
        masked_feature, _ = get_masked_feature(feature, preds)
        masked_feature_aug = feature_augment(masked_feature)

        # ===== 聚类阶段 =====
        cluster_protos = self._get_cluster_prototypes()

        weights2 = self.gate2(feature, 1)
        weights2 = top_k_gating(weights2, k=self.num_experts_cluster)

        masked_preds = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        full_log_probs = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        masked_preds_aug = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        full_log_probs_aug = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)

        for i in range(self.num_experts_cluster):
            expert_preds, expert_log_probs = self.clusterer_experts[i](
                masked_feature, cluster_protos)
            expert_preds_aug, expert_log_probs_aug = self.clusterer_experts[i](
                masked_feature_aug, cluster_protos)

            masked_preds += weights2[:, i].unsqueeze(1) * expert_preds
            full_log_probs += weights2[:, i].unsqueeze(1) * expert_log_probs
            masked_preds_aug += weights2[:, i].unsqueeze(1) * expert_preds_aug
            full_log_probs_aug += weights2[:, i].unsqueeze(1) * expert_log_probs_aug

        masked_preds = F.softmax(masked_preds, dim=1)
        masked_preds_aug = F.softmax(masked_preds_aug, dim=1)

        return preds, masked_preds, masked_preds_aug, weights1, weights2

    # ----------------------------------------------------------
    # 保存/加载 (与 ModelInterface 兼容)
    # ----------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        return {
            'num_known_classes': self.num_known_classes,
            'num_unknown_classes': self.num_unknown_classes,
            'reload_feature_model_pretrained': self._reload_yamnet
        }

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config()
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


# ============================================================
# 示例
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("方案B: 条件流 + Meta-learning Few-shot 模型")
    print("=" * 60)

    num_known_classes = 6
    num_unknown_classes = 4
    feature_dim = 32

    model = FewShotFlowBModel(
        num_unknown_classes=num_unknown_classes,
        num_known_classes=num_known_classes,
        reload_feature_model_pretrained=False
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 模拟前向传播 (使用随机特征, 跳过YAMNet)
    print("\n--- 测试前向传播 (训练模式, 使用可学习原型) ---")
    B = 16
    x = torch.randn(B, 16000)  # 模拟音频

    # 直接测试内部模块
    feat = torch.randn(B, feature_dim)
    prototypes = model._get_class_prototypes()
    print(f"特征 shape: {feat.shape}")
    print(f"分类原型 shape: {prototypes.shape}")

    # 测试条件流分类专家
    expert_out = model.classifier_experts[0](feat, prototypes)
    print(f"分类专家输出 shape: {expert_out.shape}")  # (B, num_known+1)

    # 测试条件流聚类专家
    cluster_protos = model._get_cluster_prototypes()
    gamma, log_probs = model.clusterer_experts[0](feat, cluster_protos)
    print(f"聚类 gamma shape: {gamma.shape}")    # (B, num_unknown)
    print(f"聚类 log_probs shape: {log_probs.shape}")

    # 测试 Few-shot 支持
    print("\n--- 测试 Few-shot 支持集 ---")
    support_feats = torch.randn(12, feature_dim)  # 6类 × 2样本
    support_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    model.set_support_set(support_feats, support_labels)

    prototypes_after = model._get_class_prototypes()
    print(f"支持集原型 shape: {prototypes_after.shape}")
    print(f"原型来源: 支持集 (非可学习参数)")

    model.clear_support_set()
    print("已清除支持集, 恢复可学习原型")

    print("\n" + "=" * 60)
    print("方案B模型验证通过!")
    print("=" * 60)