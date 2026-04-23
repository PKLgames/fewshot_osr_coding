import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class LossF(nn.Module):
    """开放集识别(Open Set Recognition)损失函数

    专为 Final_classify_Model 设计的简化损失函数，输入仅需:
    - classifier_log_probs: 模型输出 (softmax 概率), shape [B, num_known_classes+1]
    - targets: 类别索引, shape [B], 其中 targets==num_known_classes 表示未知类
    - weights1: 门控网络权重, shape [B, num_experts]

    基于 loss_function_v4.py 的 _s1 方法简化而来，移除了 gamma/gamma_aug/weights2 等不需要的参数。

    损失组件:
    1. L_ce   - 焦点交叉熵损失 (Focal CE): 对已知类分类 + 未知类识别的监督信号
    2. L_bin  - 二元未知检测损失: 平衡的BCE，增强对未知类的敏感度
    3. L_unk  - 未知类编码损失: 基于间隔的损失，推动未知类样本的高概率和高置信度
    4. L_conf - 置信度惩罚: 信息熵正则化，防止过度自信
    5. L_exp  - 专家平衡损失: KL散度约束门控权重分布均匀
    6. L_overconf - 过度置信惩罚: 惩罚预测概率过度集中，防止过拟合
    """

    def __init__(self,
                 lambda_ce: float = 1.5,
                 lambda_binary: float = 1.0,
                 lambda_unk_enc: float = 0.8,
                 lambda_conf_penalty: float = 0.0,
                 lambda_expert: float = 0.1,
                 lambda_max_conf: float = 0.0,
                 label_smoothing: float = 0.25,
                 focal_gamma: float = 2.0,
                 eps: float = 1e-8):
        """
        Args:
            lambda_ce: 交叉熵损失权重
            lambda_binary: 二元未知检测损失权重
            lambda_unk_enc: 未知类编码损失权重
            lambda_conf_penalty: 置信度惩罚权重 (0表示禁用)
            lambda_expert: 专家平衡损失权重
            lambda_max_conf: 过度置信惩罚权重，防止模型预测过度集中
            label_smoothing: 标签平滑系数
            focal_gamma: 焦点损失的gamma参数，越大越关注难样本
            eps: 数值稳定常数
        """
        super(LossF, self).__init__()
        self.lambda_ce = lambda_ce
        self.lambda_bin = lambda_binary
        self.lambda_unk = lambda_unk_enc
        self.lambda_conf = lambda_conf_penalty
        self.lambda_exp = lambda_expert
        self.lambda_max_conf = lambda_max_conf
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        self.eps = eps

    def forward(
        self,
        classifier_log_probs: torch.Tensor,
        targets: torch.Tensor,
        weights1: torch.Tensor,
        epoch: int = None,
        total_epochs: int = None,
        num_known_classes: int = 6
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算开放集识别总损失

        Args:
            classifier_log_probs: 模型输出概率, shape [B, num_known_classes+1]
                                  (虽然名字是log_probs，实际接收softmax概率)
            targets: 类别索引, shape [B], 值域 [0, num_known_classes]
                     其中 num_known_classes 表示未知类
            weights1: 门控网络权重, shape [B, num_experts]
            epoch: 当前训练轮次 (None=测试阶段，使用prog=1.0)
            total_epochs: 总训练轮次 (None=测试阶段，使用prog=1.0)
            num_known_classes: 已知类别数

        Returns:
            total_loss: 标量总损失
            info: 各子损失的字典，用于日志记录
        """
        dev = classifier_log_probs.device
        B = classifier_log_probs.size(0)
        K = num_known_classes
        # 训练阶段: prog随epoch线性增长; 测试阶段(epoch=None): prog=1.0使用最终训练参数
        if epoch is not None and total_epochs is not None:
            prog = min(epoch / max(total_epochs - 1, 1), 1.0)
        else:
            prog = 1.0

        # 将概率转换为log概率以适配cross_entropy
        # classifier_log_probs 实际上是 softmax 概率，取log后作为 log_softmax 使用
        log_probs = classifier_log_probs.clamp(min=self.eps).log()
        probs = classifier_log_probs.clamp(min=self.eps, max=1.0 - self.eps)

        # ============================================================
        # 1. 焦点交叉熵损失 (Focal Cross-Entropy Loss)
        # ============================================================
        # 使用NLL loss (输入已是log概率) + 标签平滑
        smooth = self.label_smoothing
        if smooth > 0:
            # 手动实现带标签平滑的NLL损失
            num_classes = log_probs.size(1)  # K+1 (已知类 + 未知类)
            one_hot = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_labels = one_hot * (1.0 - smooth) + smooth / num_classes
            ce_raw = -(smooth_labels * log_probs).sum(dim=1)  # [B]
        else:
            ce_raw = F.nll_loss(log_probs, targets, reduction='none')  # [B]

        # Focal weighting: 对难样本(低置信度)加大权重
        with torch.no_grad():
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # 目标类概率
            focal_weight = (1.0 - pt) ** self.focal_gamma  # 难样本权重高

        L_ce = (focal_weight * ce_raw).mean()

        # ============================================================
        # 2. 二元未知检测损失 (Binary Unknown Detection Loss)
        # ============================================================
        unk_mask = (targets == K)  # 未知类掩码
        kn_mask = ~unk_mask        # 已知类掩码
        unk_prob = probs[:, K]     # 未知类概率列
        n_unk = unk_mask.sum()
        n_kn = kn_mask.sum()

        if n_unk > 0 and n_kn > 0:
            # 平衡权重：根据类别数量自适应调整
            w_pos = (B / (2.0 * n_unk.float())).detach()
            w_neg = (B / (2.0 * n_kn.float())).detach()
            L_bin = -(
                w_pos * unk_mask.float() * torch.log(unk_prob) +
                w_neg * kn_mask.float() * torch.log(1.0 - unk_prob)
            ).mean()
        elif n_unk == 0:
            # 批次中没有未知样本
            L_bin = F.binary_cross_entropy(
                unk_prob, unk_mask.float(), reduction='mean'
            )
        else:
            # 批次中全是未知样本
            L_bin = -torch.log(unk_prob).mean()

        # ============================================================
        # 3. 未知类编码损失 (Unknown Encoding Loss)
        # ============================================================
        L_unk = torch.tensor(0.0, device=dev)

        if n_unk > 0:
            # 未知类样本：推动unk_prob高于动态间隔
            margin = 0.4 + 0.2 * prog  # 间隔随训练进度增加（降低强度防过拟合）
            L_unk = F.relu(margin - unk_prob[unk_mask]).mean()

            if n_kn > 0:
                # 已知类样本：惩罚过高的未知类概率
                L_unk = L_unk + 0.2 * F.relu(unk_prob[kn_mask] - 0.15).mean()
        elif n_kn > 0:
            # 没有未知样本时，仍约束已知样本的未知概率
            L_unk = 0.5 * F.relu(unk_prob[kn_mask] - 0.1).mean()

        # ============================================================
        # 4. 置信度惩罚 (Confidence Penalty)
        # ============================================================
        L_conf = torch.tensor(0.0, device=dev)
        if self.lambda_conf > 0:
            # 计算预测分布的信息熵，鼓励模型保持适度不确定性
            entropy = -(probs * log_probs).sum(dim=1).mean()
            L_conf = entropy

        # ============================================================
        # 5. 专家平衡损失 (Expert Balance Loss)
        # ============================================================
        L_exp = torch.tensor(0.0, device=dev)
        if weights1 is not None and weights1.numel() > 0:
            # 约束门控权重分布趋向均匀，防止专家崩塌
            avg_w = weights1.mean(dim=0)  # [num_experts]
            num_experts = avg_w.size(0)
            uniform = torch.ones_like(avg_w) / num_experts
            L_exp = F.kl_div(
                (avg_w + self.eps).log(), uniform, reduction='sum'
            )

        # ============================================================
        # 6. 过度置信惩罚 (Overconfidence Penalty)
        # ============================================================
        L_overconf = torch.tensor(0.0, device=dev)
        if self.lambda_max_conf > 0:
            max_prob = probs.max(dim=1)[0]  # [B]
            # 动态阈值：训练初期宽松(0.9)，后期收紧(0.7)
            threshold = 0.9 - 0.2 * prog
            L_overconf = F.relu(max_prob - threshold).mean()

        # ============================================================
        # 总损失计算
        # ============================================================
        # 自适应聚类权重：训练初期较弱，后期逐渐增强
        a_clu = 0.1 + 0.3 * prog

        total = (self.lambda_ce * L_ce +
                 self.lambda_bin * L_bin +
                 self.lambda_unk * L_unk +
                 self.lambda_conf * L_conf +
                 self.lambda_max_conf * L_overconf +
                 a_clu * self.lambda_exp * L_exp)

        info = {
            'L_ce': L_ce.detach(),
            'L_binary': L_bin.detach(),
            'L_unk_enc': L_unk.detach(),
            'L_conf': L_conf.detach(),
            'L_expert': L_exp.detach(),
            'L_overconf': L_overconf.detach(),
            'n_unknown': float(n_unk),
            'n_known': float(n_kn),
            'total_loss': total.detach(),
            'prog': prog,
        }

        return total, info