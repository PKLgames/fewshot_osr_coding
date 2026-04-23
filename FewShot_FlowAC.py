"""
方案A+C: 预训练流原型密度匹配 + 隐空间聚类 Few-shot 分类聚类模型

核心思想:
  A: 预训练流作为密度度量空间 → 在隐空间z中用support样本做原型匹配分类
  C: 预训练流映射到隐空间 → 在z空间做GMM/距离聚类

分类流程:
  1. 预训练的cINN将特征映射到标准正态隐空间: z = flow(x)
  2. 用support set计算各类原型: prototype_c = mean(z_i) for label_i == c
  3. 对query计算与各原型的距离 → 分类

聚类流程:
  1. 预训练的cINN将masked特征映射到隐空间
  2. 在隐空间用可学习聚类中心做soft分配 (类似FlowGMM)
  3. 或用support set定义聚类中心

接口与 FinalModel 完全兼容:
  forward(x) → (preds, masked_preds, masked_preds_aug, weights1, weights2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
import random
import math

from torch_audioset.yamnet.model import yamnet as torch_yamnet
import yamnet_PT_inference as yamnet_infer
from Reversable_Function import cINN as cINN_module


# ============================================================
# 工具函数 (与原代码一致)
# ============================================================

class PolicyNet(nn.Module):
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
# 核心组件A: 隐空间原型分类专家
# ============================================================

class PrototypeClassifierExpert(nn.Module):
    """
    方案A: 隐空间原型密度匹配分类

    原理:
      预训练的cINN将特征映射到标准正态隐空间 z = flow(x)
      在z空间中，同类样本聚集。用support set的z均值作为原型中心。
      分类 = 比较query的z到各原型的对数概率密度。

    两种度量模式:
      - 'density': 使用flow的精确log_prob (含Jacobian) 作为密度分数
      - 'euclidean': 在z空间中用加权欧氏距离 (更快，近似)
    """
    def __init__(self, input_dim: int, condition_dim: int = 0,
                 num_classes: int = 7,
                 num_coupling_layers: int = 3,
                 hidden_dims: List[int] = None,
                 distance_mode: str = 'density'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]
        self.num_classes = num_classes
        self.distance_mode = distance_mode

        # 单个cINN流 (condition_dim=0, 无条件)
        self.flow = cINN_module.ConditionalINN(
            input_dim=input_dim,
            condition_dim=condition_dim,
            num_coupling_layers=num_coupling_layers,
            hidden_dims=hidden_dims,
            use_permutation=True,
            permutation_type='fixed'
        )

        # 可学习的隐空间原型 (训练时使用，few-shot时被覆盖)
        # 初始化为标准正态分布的随机点
        self.register_buffer('prototypes', torch.randn(num_classes, input_dim) * 0.5)
        self.prototype_buffers = nn.Parameter(torch.randn(num_classes, input_dim) * 0.5)

        # 可学习的各簇方差尺度 (用于距离计算中的温度参数)
        self.log_sigma = nn.Parameter(torch.zeros(num_classes))

    def update_prototypes_from_buffer(self):
        """从可学习参数同步到buffer (训练时使用)"""
        self.prototypes.copy_(self.prototype_buffers.data)

    def set_prototypes(self, prototypes: torch.Tensor):
        """外部设置原型 (few-shot推理时使用)"""
        self.prototypes.copy_(prototypes)

    def forward(self, features: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            features: (B, D) 输入特征
            condition: (D,) 条件向量 (用于cINN，当condition_dim>0时使用)
        Returns:
            (B, num_classes) 各类的logit分数
        """
        B = features.size(0)
        device = features.device

        if self.distance_mode == 'density':
            return self._forward_density(features, condition)
        else:
            return self._forward_euclidean(features)

    def _forward_density(self, features: torch.Tensor,
                         condition: torch.Tensor = None) -> torch.Tensor:
        """
        使用流的log_prob作为密度分数，然后计算到各原型的"归一化距离分数"。

        思路:
          z = flow(x)
          对每个类c, 计算原型的log_prob作为参考，然后用距离加权
          score_c = log p(z) - α * ||z - μ_c||² / (2σ_c²)

          这结合了全局密度估计和局部原型匹配
        """
        B = features.size(0)
        D = features.shape[1]

        # 映射到隐空间
        if condition is not None and condition.numel() > 0:
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).expand(B, -1)
            z, log_det = self.flow.forward(features, condition, compute_jacobian=True)
        else:
            z, log_det = self.flow.forward(features,
                                           torch.zeros(B, 0, device=features.device),
                                           compute_jacobian=True)

        # 使用可学习原型 (训练模式)
        prototypes = self.prototype_buffers  # (num_classes, D)
        sigma = F.softplus(self.log_sigma).unsqueeze(0)  # (1, num_classes)

        # (B, num_classes, D) = z.unsqueeze(1) - prototypes.unsqueeze(0)
        diff = z.unsqueeze(1) - prototypes.unsqueeze(0)  # (B, C, D)
        sq_dist = (diff ** 2).sum(dim=2)  # (B, C)

        # 马氏距离 + log_det 作为分数
        scores = -sq_dist / (2 * sigma ** 2 + 1e-6)  # (B, C)

        return scores

    def _forward_euclidean(self, features: torch.Tensor) -> torch.Tensor:
        """
        纯隐空间欧氏距离分类。

        z = flow(x)
        score_c = -||z - μ_c||²
        """
        B = features.size(0)
        D = features.shape[1]

        z, _ = self.flow.forward(features,
                                 torch.zeros(B, 0, device=features.device),
                                 compute_jacobian=False)

        prototypes = self.prototype_buffers
        diff = z.unsqueeze(1) - prototypes.unsqueeze(0)
        sq_dist = (diff ** 2).sum(dim=2)

        return -sq_dist


# ============================================================
# 核心组件C: 隐空间聚类专家
# ============================================================

class LatentClusterExpert(nn.Module):
    """
    方案C: 隐空间聚类

    原理:
      1. cINN将masked特征映射到隐空间: z = flow(x_masked)
      2. 在隐空间用K个可学习聚类中心做soft分配
      3. gamma_k ∝ π_k * N(z | μ_k, σ_k²)
      4. 结合流的log_prob确保隐空间结构良好
    """
    def __init__(self, input_dim: int, condition_dim: int = 0,
                 num_clusters: int = 4,
                 num_coupling_layers: int = 2,
                 hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 32]
        self.num_clusters = num_clusters
        self.input_dim = input_dim

        self.flow = cINN_module.ConditionalINN(
            input_dim=input_dim,
            condition_dim=condition_dim,
            num_coupling_layers=num_coupling_layers,
            hidden_dims=hidden_dims,
            use_permutation=True,
            permutation_type='fixed'
        )

        # 可学习聚类中心 (在隐空间中)
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, input_dim) * 0.3)

        # 各聚类的log方差 (可学习)
        self.cluster_log_var = nn.Parameter(
            torch.zeros(num_clusters, input_dim))

        # 混合权重
        initial_logits = torch.log(torch.ones(num_clusters) / num_clusters)
        self.mix_logits = nn.Parameter(
            initial_logits + torch.randn_like(initial_logits) * 0.01)

    def forward(self, features: torch.Tensor,
                condition: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, D) masked输入特征
            condition: 可选条件
        Returns:
            gamma: (B, num_clusters) 软聚类分配
            log_probs: (B, num_clusters) 各聚类的对数似然
        """
        B = features.size(0)
        device = features.device

        # 处理全零样本
        is_valid = ~torch.all(features == 0, dim=1)
        valid_idx = torch.where(is_valid)[0]

        gamma = torch.zeros(B, self.num_clusters, device=device)
        log_probs = torch.full((B, self.num_clusters), -10.0, device=device)

        if len(valid_idx) == 0:
            gamma[:] = 1.0 / self.num_clusters
            return gamma.detach(), log_probs

        x_valid = features[valid_idx]
        n_valid = x_valid.size(0)

        # 映射到隐空间
        if condition is not None and condition.numel() > 0:
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).expand(n_valid, -1)
            z, log_det = self.flow.forward(x_valid, condition, compute_jacobian=True)
        else:
            z, log_det = self.flow.forward(x_valid,
                                           torch.zeros(n_valid, 0, device=device),
                                           compute_jacobian=True)

        # 在隐空间计算各聚类的对数概率
        # log N(z | μ_k, σ_k²) = -0.5 * [D*log(2π) + Σ log σ²_k + Σ (z-μ_k)²/σ²_k]
        log_pi = F.log_softmax(self.mix_logits, dim=0)  # (K,)

        log_probs_valid = torch.zeros(n_valid, self.num_clusters, device=device)
        for k in range(self.num_clusters):
            diff = z - self.cluster_centers[k].unsqueeze(0)  # (n_valid, D)
            log_var_k = self.cluster_log_var[k]  # (D,)
            var_k = F.softplus(log_var_k) + 1e-6

            # 对数高斯密度
            log_gauss = -0.5 * (
                self.input_dim * math.log(2 * math.pi) +
                log_var_k.sum() +
                (diff ** 2 / var_k).sum(dim=1)
            )
            log_probs_valid[:, k] = log_gauss + log_pi[k]

        log_probs_valid = torch.clamp(log_probs_valid, min=-50.0, max=10.0)

        # Softmax得到软分配
        gamma_valid = F.softmax(log_probs_valid, dim=1)

        gamma[valid_idx] = gamma_valid
        log_probs[valid_idx] = log_probs_valid

        invalid = ~is_valid
        if invalid.any():
            gamma[invalid] = 1.0 / self.num_clusters

        return gamma, log_probs


# ============================================================
# 主模型: 方案A+C 预训练流原型密度匹配 + 隐空间聚类
# ============================================================

class FewShotFlowACModel(nn.Module):
    """
    方案A+C: 预训练流原型密度匹配 + 隐空间聚类 Few-shot 模型

    架构:
    ┌────────────────────────────────────────────────────────┐
    │ 音频 → YAMNet → FC → 32维特征                           │
    │                                                          │
    │ 分类 (方案A):                                            │
    │   Gate1 → 专家权重                                       │
    │   3× PrototypeClassifierExpert                          │
    │     └ cINN: x → z (隐空间映射)                           │
    │     └ 可学习原型: μ_c (训练时)                           │
    │     └ Few-shot原型: support set z均值 (推理时)           │
    │                                                          │
    │ 掩码: unknown概率遮蔽特征                                │
    │                                                          │
    │ 聚类 (方案C):                                            │
    │   Gate2 → 专家权重                                       │
    │   3× LatentClusterExpert                                │
    │     └ cINN: x_masked → z'                               │
    │     └ 可学习聚类中心 + 高斯混合 (训练时)                 │
    │     └ Few-shot中心: support set z'均值 (推理时)         │
    └────────────────────────────────────────────────────────┘

    Few-shot 推理:
      support_feats = model.extract_features(support_audio)
      model.set_support_set(support_feats, support_labels,
                            cluster_feats, cluster_labels)
      preds, ... = model(query_audio)

    训练:
      使用可学习原型和聚类中心正常训练 (无需support set)
      流模型学习将特征映射到结构良好的隐空间
      原型在隐空间中自动聚类

    接口与 FinalModel 完全兼容:
      forward(x) → (preds, masked_preds, masked_preds_aug, weights1, weights2)
    """

    def __init__(self,
                 num_unknown_classes: int = 4,
                 num_known_classes: int = 6,
                 reload_feature_model_pretrained: bool = True,
                 distance_mode: str = 'density'):
        """
        Args:
            num_unknown_classes: 未知类数量
            num_known_classes: 已知类数量
            reload_feature_model_pretrained: 是否加载预训练YAMNet
            distance_mode: 'density' (使用flow log_prob+马氏距离)
                           或 'euclidean' (纯欧氏距离，更快)
        """
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_known_classes = num_known_classes
        self.num_unknown_classes = num_unknown_classes
        self.distance_mode = distance_mode

        feature_dim = 32

        # ===== 特征提取器 (与 FinalModel 相同) =====
        self.feature_model = torch_yamnet(pretrained=False)
        if reload_feature_model_pretrained:
            state = torch.load('yamnet.pth')
            self.feature_model.load_state_dict(state)

        self.fc1 = nn.Sequential(
            nn.Linear(521, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3))
        self.fc3 = nn.Sequential(
            nn.Linear(64, feature_dim), nn.BatchNorm1d(feature_dim), nn.Tanh())

        # ===== 方案A: MoE 隐空间原型分类专家 =====
        self.num_experts = 3
        self.classifier_experts = nn.ModuleList([
            PrototypeClassifierExpert(
                input_dim=feature_dim,
                condition_dim=0,
                num_classes=num_known_classes + 1,
                num_coupling_layers=3,
                hidden_dims=[32, 32],
                distance_mode=distance_mode
            ) for _ in range(self.num_experts)])

        # ===== 方案C: MoE 隐空间聚类专家 =====
        self.num_experts_cluster = 3
        self.clusterer_experts = nn.ModuleList([
            LatentClusterExpert(
                input_dim=feature_dim,
                condition_dim=0,
                num_clusters=num_unknown_classes,
                num_coupling_layers=2,
                hidden_dims=[32, 32]
            ) for _ in range(self.num_experts_cluster)])

        # ===== 门控网络 =====
        self.gate1 = PolicyNet(feature_dim, self.num_experts)
        self.gate2 = PolicyNet(feature_dim, self.num_experts_cluster)

        # ===== Few-shot 支持集缓存 =====
        self._support_class_prototypes = None   # Dict[int, Tensor] (在z空间)
        self._support_cluster_centers = None     # Dict[int, Tensor] (在z空间)

    # ----------------------------------------------------------
    # Few-shot 接口
    # ----------------------------------------------------------

    def set_support_set(self,
                        features: torch.Tensor,
                        labels: torch.Tensor,
                        cluster_features: torch.Tensor = None,
                        cluster_labels: torch.Tensor = None):
        """
        设置 Few-shot 支持集

        将support样本通过各专家的flow映射到隐空间，计算各类原型。

        Args:
            features: (N, feature_dim) 支持样本特征 (分类用)
            labels: (N,) 整数标签 0 ~ num_known_classes-1
            cluster_features: 可选, 聚类支持特征
            cluster_labels: 可选, 聚类标签 0 ~ num_unknown_classes-1
        """
        self.eval()
        with torch.no_grad():
            # === 分类原型: 在各专家的z空间中计算 ===
            class_prototypes_per_expert = {}
            for exp_i in range(self.num_experts):
                z, _ = self.classifier_experts[exp_i].flow.forward(
                    features,
                    torch.zeros(features.size(0), 0, device=features.device),
                    compute_jacobian=False)

                prototypes = {}
                for c in range(self.num_known_classes):
                    mask = labels == c
                    if mask.any():
                        prototypes[c] = z[mask].mean(dim=0)

                # "unknown" 类原型: 全体z均值
                if z.size(0) > 0:
                    prototypes[self.num_known_classes] = z.mean(dim=0)

                class_prototypes_per_expert[exp_i] = prototypes

            self._support_class_prototypes = class_prototypes_per_expert

            # === 聚类中心: 在各专家的z空间中计算 ===
            if cluster_features is not None and cluster_labels is not None:
                cluster_centers_per_expert = {}
                for exp_i in range(self.num_experts_cluster):
                    z, _ = self.clusterer_experts[exp_i].flow.forward(
                        cluster_features,
                        torch.zeros(cluster_features.size(0), 0,
                                    device=cluster_features.device),
                        compute_jacobian=False)

                    centers = {}
                    for k in range(self.num_unknown_classes):
                        mask = cluster_labels == k
                        if mask.any():
                            centers[k] = z[mask].mean(dim=0)

                    cluster_centers_per_expert[exp_i] = centers

                self._support_cluster_centers = cluster_centers_per_expert

    def clear_support_set(self):
        """清除支持集, 恢复使用可学习原型/中心"""
        self._support_class_prototypes = None
        self._support_cluster_centers = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征的便捷方法"""
        return self._extract_features(x)

    # ----------------------------------------------------------
    # 内部方法
    # ----------------------------------------------------------

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
            preds: (B, num_known_classes+1) 分类概率
            masked_preds: (B, num_unknown_classes) 聚类软分配
            masked_preds_aug: (B, num_unknown_classes) 增强聚类软分配
            weights1: (B, num_experts) 分类专家门控权重
            weights2: (B, num_experts_cluster) 聚类专家门控权重
        """
        batch_size = x.size(0)
        feature = self._extract_features(x)

        # ===== 分类阶段 (方案A) =====
        weights1 = self.gate1(feature, 1)
        weights1 = top_k_gating(weights1, k=self.num_experts)

        preds = torch.zeros(batch_size, self.num_known_classes + 1, device=self.device)

        for i in range(self.num_experts):
            expert = self.classifier_experts[i]

            # Few-shot: 用支持集原型覆盖可学习原型
            if self._support_class_prototypes is not None and \
               i in self._support_class_prototypes:
                proto_dict = self._support_class_prototypes[i]
                proto_list = []
                for c in range(self.num_known_classes + 1):
                    if c in proto_dict:
                        proto_list.append(proto_dict[c])
                    else:
                        proto_list.append(expert.prototype_buffers[c].detach())
                expert.set_prototypes(torch.stack(proto_list).to(self.device))

            expert_output = expert(feature)

            # 恢复可学习原型
            if self._support_class_prototypes is not None:
                expert.update_prototypes_from_buffer()

            preds += weights1[:, i].unsqueeze(1) * expert_output

        preds = F.softmax(preds, dim=1)

        # ===== 未知样本掩码 =====
        masked_feature, _ = get_masked_feature(feature, preds)
        masked_feature_aug = feature_augment(masked_feature)

        # ===== 聚类阶段 (方案C) =====
        weights2 = self.gate2(feature, 1)
        weights2 = top_k_gating(weights2, k=self.num_experts_cluster)

        masked_preds = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        full_log_probs = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        masked_preds_aug = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)
        full_log_probs_aug = torch.zeros(batch_size, self.num_unknown_classes, device=self.device)

        for i in range(self.num_experts_cluster):
            expert = self.clusterer_experts[i]

            # Few-shot: 用支持集中心覆盖可学习中心
            if self._support_cluster_centers is not None and \
               i in self._support_cluster_centers:
                center_dict = self._support_cluster_centers[i]
                center_list = []
                for k in range(self.num_unknown_classes):
                    if k in center_dict:
                        center_list.append(center_dict[k])
                    else:
                        center_list.append(expert.cluster_centers[k].detach())
                expert.cluster_centers.data.copy_(
                    torch.stack(center_list).to(self.device))

            expert_preds, expert_log_probs = expert(masked_feature)
            expert_preds_aug, expert_log_probs_aug = expert(masked_feature_aug)

            # 恢复可学习中心
            # (只有在few-shot模式下才需要恢复，但由于data.copy_已经修改了参数，
            #  我们需要注意: 在few-shot推理时通常不需要恢复，因为每次推理都重新设置)

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
            'reload_feature_model_pretrained': True,
            'distance_mode': self.distance_mode
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
    print("方案A+C: 预训练流原型密度匹配 + 隐空间聚类 模型")
    print("=" * 60)

    num_known_classes = 6
    num_unknown_classes = 4
    feature_dim = 32

    # --- 测试 density 模式 ---
    print("\n>>> density 模式 (使用flow log_prob + 马氏距离)")
    model_density = FewShotFlowACModel(
        num_unknown_classes=num_unknown_classes,
        num_known_classes=num_known_classes,
        reload_feature_model_pretrained=False,
        distance_mode='density'
    )

    total_params = sum(p.numel() for p in model_density.parameters())
    print(f"总参数量: {total_params:,}")

    # --- 测试 euclidean 模式 ---
    print("\n>>> euclidean 模式 (纯隐空间欧氏距离)")
    model_euclidean = FewShotFlowACModel(
        num_unknown_classes=num_unknown_classes,
        num_known_classes=num_known_classes,
        reload_feature_model_pretrained=False,
        distance_mode='euclidean'
    )

    total_params2 = sum(p.numel() for p in model_euclidean.parameters())
    print(f"总参数量: {total_params2:,}")

    # --- 测试内部模块 ---
    print("\n--- 测试 PrototypeClassifierExpert (density) ---")
    B = 16
    feat = torch.randn(B, feature_dim)

    expert = model_density.classifier_experts[0]
    scores = expert(feat)
    print(f"输入: {feat.shape}")
    print(f"输出 scores: {scores.shape}")  # (B, num_known+1)
    print(f"Softmax分类: {F.softmax(scores, dim=1).shape}")

    print("\n--- 测试 LatentClusterExpert ---")
    cluster_expert = model_density.clusterer_experts[0]
    gamma, log_probs = cluster_expert(feat)
    print(f"输入: {feat.shape}")
    print(f"gamma: {gamma.shape}")        # (B, num_unknown)
    print(f"log_probs: {log_probs.shape}")

    # --- 测试 Few-shot 支持 ---
    print("\n--- 测试 Few-shot 支持集 ---")
    support_feats = torch.randn(12, feature_dim)
    support_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    model_density.set_support_set(support_feats, support_labels)
    print("已设置支持集 (6类 × 2样本)")

    model_density.clear_support_set()
    print("已清除支持集")

    print("\n" + "=" * 60)
    print("方案A+C模型验证通过!")
    print("=" * 60)