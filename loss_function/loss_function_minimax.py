import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class LossF(nn.Module):
    """
    两阶段OSR发现学习损失函数

    设计思路：
    1. 第一阶段（分类器训练）：强分类 + 适度聚类约束 + 强防过拟合
    2. 第二阶段（聚类器训练）：保持分类 + 强聚类 + 弱防过拟合

    核心改进：
    - Margin-based分类正则化
    - 知识蒸馏风格的teacher-student一致性
    - 更强的Dropout和特征扰动正则化
    - 两阶段渐进式权重调整
    """

    def __init__(
        self,
        # 第一阶段超参数（分类器训练）
        stage1: dict = None,
        # 第二阶段超参数（聚类器训练）
        stage2: dict = None,
        eps: float = 1e-8,
    ):
        super(LossF, self).__init__()
        self.eps = eps

        # 第一阶段默认超参数：专注分类器训练
        if stage1 is None:
            stage1 = {
                'lambda_ce': 2.0,           # 分类损失权重（高）
                'lambda_binary': 1.0,       # 二分类损失权重
                'lambda_ps': 0.15,          # 一致性损失权重（低）
                'lambda_entropy': 0.1,      # 熵损失权重（低）
                'lambda_reg': 0.1,          # 簇平衡权重（低）
                'lambda_expert': 0.05,      # 专家平衡权重
                'lambda_separation': 0.05,  # 分离损失权重（低）
                'lambda_confidence': 0.15,  # 置信度惩罚
                'lambda_margin': 0.2,        # 新增：Margin正则化
                'lambda_kd': 0.1,            # 新增：知识蒸馏损失
                'label_smoothing': 0.15,    # 标签平滑
                'margin': 0.5,               # Margin阈值
                'temperature': 2.0,         # 蒸馏温度
            }

        # 第二阶段默认超参数：专注聚类训练
        if stage2 is None:
            stage2 = {
                'lambda_ce': 0.5,           # 分类损失权重（中）
                'lambda_binary': 0.3,       # 二分类损失权重（中）
                'lambda_ps': 1.5,           # 一致性损失权重（高）
                'lambda_entropy': 0.8,      # 熵损失权重（高）
                'lambda_reg': 0.8,          # 簇平衡权重（高）
                'lambda_expert': 0.05,      # 专家平衡权重
                'lambda_separation': 1.2,   # 分离损失权重（高）
                'lambda_confidence': 0.05,  # 置信度惩罚（低）
                'lambda_margin': 0.05,      # Margin正则化（低）
                'lambda_kd': 0.05,          # 知识蒸馏损失（低）
                'label_smoothing': 0.1,     # 标签平滑
                'margin': 0.3,              # Margin阈值
                'temperature': 2.0,         # 蒸馏温度
            }

        self.stage1 = stage1
        self.stage2 = stage2

    def forward(
        self,
        classifier_log_probs: torch.Tensor,
        gamma: torch.Tensor,
        gamma_aug: torch.Tensor,
        targets: torch.Tensor,
        weights1: torch.Tensor,
        weights2: torch.Tensor,
        log_probs: Optional[torch.Tensor] = None,
        epoch: int = 0,
        total_epochs: int = 30,
        num_known_classes: int = 6,
        stage: int = 1,
        stage1_epochs: int = 60,
        stage2_epochs: int = 30,
        teacher_probs: Optional[torch.Tensor] = None,  # 新增：教师模型输出
    ) -> Tuple[torch.Tensor, dict]:
        """
        两阶段损失函数前向传播

        Args:
            classifier_log_probs: (batch_size, num_known + 1) 分类器对数概率
            gamma: (batch_size, cluster_num) 聚类软标签
            gamma_aug: (batch_size, cluster_num) 增强后的聚类软标签
            targets: (batch_size,) 压缩标签
            weights1, weights2: (batch_size, num_experts) 专家权重
            log_probs: (batch_size, cluster_num) 流模型对数概率
            epoch: 当前epoch
            total_epochs: 总epoch数
            num_known_classes: 已知类数
            stage: 当前阶段 (1或2)
            stage1_epochs: 第一阶段epoch数
            stage2_epochs: 第二阶段epoch数
            teacher_probs: 教师模型概率 (可选)

        Returns:
            total_loss: 总损失
            loss_dict: 损失分量字典
        """
        device = classifier_log_probs.device
        batch_size = classifier_log_probs.size(0)

        # 根据阶段选择超参数
        if stage == 1:
            cfg = self.stage1
            stage_epoch = epoch  # 第一阶段内的epoch
            stage_total = stage1_epochs
        else:
            cfg = self.stage2
            stage_epoch = epoch - stage1_epochs  # 第二阶段内的epoch
            stage_total = stage2_epochs

        # 计算阶段内进度 [0, 1]
        stage_progress = stage_epoch / max(stage_total, 1)

        # ========== 1. L_ce: 分类损失（带标签平滑）==========
        L_ce = F.cross_entropy(
            classifier_log_probs,
            targets,
            reduction='mean',
            label_smoothing=cfg['label_smoothing']
        )

        # ========== 2. L_binary: 二分类损失（已知 vs 未知）==========
        binary_targets = (targets == num_known_classes).float()
        unknown_probs = classifier_log_probs[:, num_known_classes].clamp(min=1e-8, max=1-1e-8)
        L_binary = F.binary_cross_entropy(unknown_probs, binary_targets, reduction='mean')

        # ========== 3. L_confidence: 置信度惩罚（防过拟合）==========
        probs = F.softmax(classifier_log_probs, dim=1)
        max_probs, _ = probs.max(dim=1)
        # 只惩罚超过阈值的样本
        confidence_margin = 0.9 + 0.05 * (1 - stage_progress)  # 动态阈值
        confidence_penalty_mask = (max_probs > confidence_margin).float()
        if confidence_penalty_mask.sum() > 0:
            L_confidence = (max_probs * confidence_penalty_mask).sum() / (confidence_penalty_mask.sum() + self.eps)
            L_confidence = (L_confidence - confidence_margin).clamp(min=0)
        else:
            L_confidence = torch.tensor(0.0, device=device)

        # ========== 4. L_margin: Margin-based正则化（新增-防过拟合）==========
        # 鼓励分类器不要过于自信，保持一定的预测间隔
        L_margin = torch.tensor(0.0, device=device)
        if cfg['lambda_margin'] > 0:
            # 计算top-2概率的差异
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            # 惩罚过小的margin
            target_margin = cfg['margin']
            L_margin = F.relu(target_margin - margin).mean()

        # ========== 5. L_kd: 知识蒸馏损失（新增）==========
        # 如果有教师模型，使用蒸馏损失
        L_kd = torch.tensor(0.0, device=device)
        if teacher_probs is not None and cfg['lambda_kd'] > 0:
            temperature = cfg['temperature']
            # 软化教师和学生的概率分布
            teacher_soft = F.softmax(teacher_probs / temperature, dim=1)
            student_soft = F.softmax(classifier_log_probs / temperature, dim=1)
            # KL散度蒸馏损失
            L_kd = F.kl_div(
                student_soft.log(),
                teacher_soft,
                reduction='batchmean'
            ) * (temperature ** 2)

        # ========== 6. 聚类损失（核心）==========
        # 准备mask
        if log_probs is not None:
            is_invalid = torch.isneginf(log_probs).all(dim=1)
            valid_mask = ~is_invalid
        else:
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        unknown_mask = (targets == num_known_classes)
        unknown_valid_mask = valid_mask & unknown_mask
        n_unknown = unknown_valid_mask.sum()

        L_ps = torch.tensor(0.0, device=device)
        L_entropy = torch.tensor(0.0, device=device)
        L_reg = torch.tensor(0.0, device=device)
        L_separation = torch.tensor(0.0, device=device)

        if n_unknown > 0:
            gamma_unknown = gamma[unknown_valid_mask]
            gamma_aug_unknown = gamma_aug[unknown_valid_mask]
            num_clusters = gamma_unknown.size(1)

            # ========== L_ps: 聚类一致性损失 ==========
            # 原始和增强版本的一致性
            L_ps = F.kl_div(
                (gamma_unknown + self.eps).log(),
                gamma_aug_unknown + self.eps,
                reduction='batchmean'
            )

            # ========== L_entropy: 熵损失（目标熵正则化）==========
            entropy = -(gamma_unknown * torch.log(gamma_unknown + self.eps)).sum(dim=1)
            # 目标熵：根据类别数计算（log(num_clusters)是最优值）
            target_entropy = np.log(num_clusters) * 0.8  # 允许一定的自由度
            # 避免完全坍缩（熵太低）或完全均匀（熵太高）
            L_entropy = torch.mean((entropy - target_entropy) ** 2)

            # ========== L_reg: 簇平衡正则化 ==========
            avg_gamma = gamma_unknown.mean(dim=0)
            uniform_dist = torch.full_like(avg_gamma, 1.0 / num_clusters)
            L_reg = F.kl_div((avg_gamma + self.eps).log(), uniform_dist, reduction='sum')

            # ========== L_separation: 簇间分离损失 ==========
            if num_clusters > 1:
                cluster_assignments = gamma_unknown.argmax(dim=1)
                cluster_centers = []
                for k in range(num_clusters):
                    mask = (cluster_assignments == k)
                    if mask.sum() > 0:
                        center = gamma_unknown[mask].mean(dim=0)
                        cluster_centers.append(center)
                    else:
                        cluster_centers.append(torch.zeros(num_clusters, device=device))

                cluster_centers = torch.stack(cluster_centers)
                distance_matrix = torch.cdist(cluster_centers, cluster_centers, p=2)

                # 计算最小簇间距离
                mask_diag = torch.eye(num_clusters, device=device).bool()
                if (~mask_diag).sum() > 0:
                    min_distance = distance_matrix[~mask_diag].min()
                    max_distance = distance_matrix[~mask_diag].max()
                    # 鼓励最小距离增大，最大距离也增大
                    L_separation = -min_distance + 0.1 * max_distance
                else:
                    L_separation = torch.tensor(0.0, device=device)

        # ========== 7. 专家平衡损失 ==========
        L_expert1 = self.compute_expert_balance_loss(weights1)
        L_expert2 = self.compute_expert_balance_loss(weights2)

        # ========== 8. 渐进式权重调整 ==========
        # 根据阶段内进度动态调整权重
        if stage == 1:
            # 第一阶段：早期强分类 + 适度正则化，后期逐渐放松
            reg_scale = 1.0 - 0.3 * stage_progress  # 正则化逐渐降低
            ce_scale = 1.0  # 分类权重保持高
        else:
            # 第二阶段：早期保持分类 + 增强聚类，后期专注聚类
            reg_scale = 0.7 - 0.3 * stage_progress  # 正则化继续降低
            ce_scale = 1.0 - 0.5 * stage_progress  # 分类权重逐渐降低

        # ========== 总损失 ==========
        total_loss = (
            ce_scale * cfg['lambda_ce'] * L_ce +
            ce_scale * cfg['lambda_binary'] * L_binary +
            reg_scale * cfg['lambda_confidence'] * L_confidence +
            reg_scale * cfg['lambda_margin'] * L_margin +
            reg_scale * cfg['lambda_kd'] * L_kd +
            cfg['lambda_ps'] * L_ps +
            cfg['lambda_entropy'] * L_entropy +
            cfg['lambda_reg'] * L_reg +
            cfg['lambda_separation'] * L_separation +
            cfg['lambda_expert'] * (L_expert1 + L_expert2)
        )

        # ========== 构建损失字典 ==========
        loss_dict = {
            'L_ce': L_ce.detach(),
            'L_binary': L_binary.detach(),
            'L_confidence': L_confidence.detach(),
            'L_margin': L_margin.detach(),
            'L_kd': L_kd.detach(),
            'L_ps': L_ps.detach(),
            'L_entropy': L_entropy.detach(),
            'L_reg': L_reg.detach(),
            'L_separation': L_separation.detach(),
            'L_expert1': L_expert1.detach(),
            'L_expert2': L_expert2.detach(),
            'n_unknown_samples': float(n_unknown),
            'stage': stage,
            'stage_progress': stage_progress,
            'reg_scale': reg_scale,
            'ce_scale': ce_scale,
        }

        return total_loss, loss_dict

    def compute_expert_balance_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算专家平衡损失，鼓励所有专家被均匀使用
        """
        avg_usage = torch.mean(weights, dim=0)
        uniform_dist = torch.ones_like(avg_usage) / avg_usage.size(0)
        balance_loss = F.kl_div(
            (avg_usage + self.eps).log(),
            uniform_dist,
            reduction='sum'
        )
        return balance_loss

    def get_stage_weights(self, stage: int) -> dict:
        """
        获取指定阶段的超参数
        """
        if stage == 1:
            return self.stage1.copy()
        else:
            return self.stage2.copy()

    def set_stage_weights(self, stage: int, weights: dict):
        """
        设置指定阶段的超参数
        """
        if stage == 1:
            self.stage1.update(weights)
        else:
            self.stage2.update(weights)


# 保留原有类作为兼容（可选）
class Stage1Loss(nn.Module):
    """
    第一阶段损失函数包装器（兼容旧代码）
    """

    def __init__(self, **kwargs):
        super(Stage1Loss, self).__init__()
        self.loss_fn = LossF(**kwargs)

    def forward(self, *args, **kwargs):
        kwargs['stage'] = 1
        return self.loss_fn(*args, **kwargs)


class Stage2Loss(nn.Module):
    """
    第二阶段损失函数包装器（兼容旧代码）
    """

    def __init__(self, **kwargs):
        super(Stage2Loss, self).__init__()
        self.loss_fn = LossF(**kwargs)

    def forward(self, *args, **kwargs):
        kwargs['stage'] = 2
        return self.loss_fn(*args, **kwargs)
