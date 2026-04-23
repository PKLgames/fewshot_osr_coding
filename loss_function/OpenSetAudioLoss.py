import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenSetAudioLoss(nn.Module):
    def __init__(self, lambda_gate=1e-3, threshold_for_unknown=-10.0, label_smoothing=0.0):
        super(OpenSetAudioLoss, self).__init__()
        self.lambda_gate = lambda_gate # 门控正则化权重
        self.threshold_for_unknown = threshold_for_unknown # 判定为未知的似然阈值（调试用）
        self.label_smoothing = label_smoothing  # 标签平滑系数，防止过拟合

    def forward(self, model_outputs, labels, gate_weights, mode='train'):
        """
        Args:
            model_outputs: 来自 FlowBasedTissue 的输出 (log_probs)。
                           Shape: [Batch, num_known_classes + 1] (最后一维通常是未知类)
            labels: 真实标签。对于未知类，标签设为 num_known_classes (即最后一类)。
            gate_weights: 门控网络输出的权重，用于稀疏正则化。
            mode: 'train_known' 或 'train_open' (包含未知类)
            
        Returns:
            total_loss
        """
        device = model_outputs.device
        
        # --- 1. 负对数似然损失 (NLL) ---
        # model_outputs 是 Flow 模型输出的原始 log_prob (对数概率密度)
        # 这些值量级很大（如 -20 ~ -30），因为是对 input_dim=32 维度的联合对数概率
        # 必须先通过 log_softmax 归一化为类别概率的对数，再使用 NLLLoss
        
        # 处理标签：假设 labels 中数值等于 num_known_classes 的样本为 "未知类"
        # 在你的 FinalModel 中，preds shape 是 [batch, num_known_classes+1]
        batch_size, num_classes = model_outputs.shape
        num_known_classes = num_classes - 1
        
        # 关键修复：将原始 log_prob 归一化为分类 log-probabilities
        # log_softmax 在类别维度上归一化，使值域变为 [-inf, 0]，量级合理
        normalized_log_probs = F.log_softmax(model_outputs, dim=1)
        
        # 检查是否有未知类样本 (label == num_known_classes)
        is_unknown_mask = (labels >= num_known_classes)
        
        # --- 情况 A: 仅计算已知类损失 (如果存在已知类样本) ---
        known_loss = 0.0
        if (~is_unknown_mask).sum() > 0:
            # 提取已知类样本的归一化预测和标签
            known_preds = normalized_log_probs[~is_unknown_mask]
            known_lbls = labels[~is_unknown_mask]
            
            # 计算NLL损失，支持标签平滑
            # 标签平滑：将硬标签转为软标签，防止模型过度自信
            if self.label_smoothing > 0:
                n_known = (~is_unknown_mask).sum()
                # 创建软标签 one-hot
                smooth_labels = torch.zeros_like(known_preds)
                smooth_labels.scatter_(1, known_lbls.unsqueeze(1), 1.0)
                # 应用标签平滑
                smooth_labels = smooth_labels * (1 - self.label_smoothing) + \
                                self.label_smoothing / num_classes
                # 使用 KL 散度等价于手动计算的平滑交叉熵
                known_loss = -(smooth_labels * known_preds).sum(dim=1).mean()
            else:
                criterion = nn.NLLLoss(reduction='mean')
                known_loss = criterion(known_preds, known_lbls)
        
        # --- 情况 B: 处理未知类损失 (Open-Set Loss) ---
        # 对于未知类样本，我们希望其最大类别概率尽量低（即推向均匀分布）
        unknown_loss = 0.0
        if is_unknown_mask.sum() > 0 and mode in ('train', 'train_open'):
            unknown_preds = normalized_log_probs[is_unknown_mask] # [N_unknown, num_classes]
            
            # 使用归一化后的 log_probs，量级合理（[-inf, 0]）
            # 最小化未知样本的最大 log_prob → 让模型对未知样本不自信
            # 由于 log_softmax 值在 [-inf, 0]，取 max 后得到最大的类别概率（对数）
            # 对未知样本，我们希望这个值尽可能小（接近 log(1/num_classes)）
            max_log_prob, _ = torch.max(unknown_preds, dim=1)
            unknown_loss = -torch.mean(max_log_prob)  # 负号使其为正，最小化 = 最大化不确定性
            
        # 组合损失
        # 权重分配：如果有未知类样本，同时计算 known 和 unknown 损失
        alpha = 1.0
        beta = 1.0 # 未知类损失的权重，可根据数据量调整
        
        nll_loss = alpha * known_loss + beta * unknown_loss

        # --- 2. 门控网络稀疏正则化 (Gating Regularization) ---
        # 防止门控网络输出过于平均，强制稀疏性 (Top-k)
        # gate_weights: [Batch, num_experts]
        if gate_weights is not None:
            # 计算门控权重的熵，越稀疏熵越小
            # 但我们要最小化 Loss，所以取负熵作为正则项的一部分
            # 或者直接惩罚 L1/L2 范数，或者惩罚非 Top-k 的权重
            # 这里使用：Top-k 稀疏性惩罚
            # top_k_values, _ = torch.topk(gate_weights, k=1, dim=1) # 取最大的1个
            # gate_penalty = -torch.mean(torch.sum(top_k_values, dim=1)) # 鼓励最大值变大
            
            # 更简单的做法：L2 正则化，鼓励权重集中在少数专家
            # 或者使用辅助损失，让门控权重接近 one-hot
            gate_penalty = torch.mean(torch.sum(gate_weights ** 2, dim=1))
        else:
            gate_penalty = 0.0

        # --- 3. 总损失 ---
        total_loss = nll_loss + self.lambda_gate * gate_penalty
        
        # 调试信息 (可选)
        # print(f"Known Loss: {known_loss.item():.4f}, Unknown Loss: {unknown_loss.item():.4f}, Gate Loss: {gate_penalty.item():.4f}")
        
        return total_loss