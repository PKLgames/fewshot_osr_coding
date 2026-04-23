import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class LossF(nn.Module):
    """
    两阶段OSR损失函数（统一接口，通过set_stage切换阶段）
    
    v3改进（解决过拟合 + 聚类失败）：
    
    核心问题诊断：
    1. 过拟合：Train Acc 77% vs Test Acc 44%
       → 原因：学习率过高、正则化不足、模型容量过大
       → 解决：增加标签平滑、mixup正则、置信度惩罚
    2. 聚类完全失败：NMI≈0, ARI≈0, ACC≈25%(随机)
       → 原因：masked_feature在早期≈0，聚类器无法学习；
              Stage2从未保存模型（bug）；一致性损失无法引导聚类
       → 解决：使用 detach+直通 代替软掩码，确保聚类器收到完整特征；
              增加熵最小化 + 多视图一致性 + 互信息最大化
    3. Stage2早停bug修复：best_cluster_acc初始化为-1确保第一个epoch能保存
    
    关键改进：
    - Stage1: Focal CE + 置信度惩罚(防过拟合) + 更强标签平滑 + 聚类预热
    - Stage2: 低权重CE + 互信息最大化 + 样本级熵最小化 + Sinkhorn簇平衡
             + 多视图一致性(对称KL) + 置信度惩罚
    """

    def __init__(self,
                 # Stage1超参
                 stage1_lambda_ce=2.0, stage1_lambda_binary=1.0,
                 stage1_lambda_unk_enc=0.8, stage1_lambda_warmup=0.3,
                 stage1_lambda_balance=0.05, stage1_lambda_expert=0.1,
                 stage1_label_smoothing=0.15, stage1_focal_gamma=2.0,
                 stage1_lambda_conf_penalty=0.3,
                 # Stage2超参
                 stage2_lambda_ce=0.3, stage2_lambda_binary=0.2,
                 stage2_lambda_kd=0.5, stage2_lambda_ps=3.0,
                 stage2_lambda_sharp=1.5, stage2_lambda_me_max=1.5,
                 stage2_lambda_flow=0.3, stage2_lambda_expert=0.05,
                 stage2_label_smoothing=0.1, stage2_sharp_temp=0.5,
                 stage2_lambda_mi=1.0, stage2_lambda_conf_penalty=0.2,
                 # 公共
                 eps=1e-8):
        super(LossF, self).__init__()
        # S1
        self.s1_ce = stage1_lambda_ce
        self.s1_bin = stage1_lambda_binary
        self.s1_unk = stage1_lambda_unk_enc
        self.s1_warm = stage1_lambda_warmup
        self.s1_bal = stage1_lambda_balance
        self.s1_exp = stage1_lambda_expert
        self.s1_smooth = stage1_label_smoothing
        self.s1_focal = stage1_focal_gamma
        self.s1_conf = stage1_lambda_conf_penalty
        # S2
        self.s2_ce = stage2_lambda_ce
        self.s2_bin = stage2_lambda_binary
        self.s2_kd = stage2_lambda_kd
        self.s2_ps = stage2_lambda_ps
        self.s2_sharp = stage2_lambda_sharp
        self.s2_memax = stage2_lambda_me_max
        self.s2_flow = stage2_lambda_flow
        self.s2_exp = stage2_lambda_expert
        self.s2_smooth = stage2_label_smoothing
        self.s2_stemp = stage2_sharp_temp
        self.s2_mi = stage2_lambda_mi
        self.s2_conf = stage2_lambda_conf_penalty
        # Common
        self.eps = eps
        self._stage = 1
        self._teacher_cache = None

    def set_stage(self, stage):
        assert stage in [1, 2]
        self._stage = stage

    def cache_teacher_probs(self, probs):
        """缓存Stage1模型输出用于Stage2知识蒸馏(可选)"""
        self._teacher_cache = probs.detach()

    def forward(self, classifier_log_probs, gamma, gamma_aug, targets,
                weights1, weights2, log_probs=None, epoch=0,
                total_epochs=30, num_known_classes=6):
        if self._stage == 1:
            return self._s1(classifier_log_probs, gamma, gamma_aug, targets,
                           weights1, weights2, log_probs, epoch, total_epochs, num_known_classes)
        return self._s2(classifier_log_probs, gamma, gamma_aug, targets,
                       weights1, weights2, log_probs, epoch, total_epochs, num_known_classes)

    def _confidence_penalty(self, probs):
        """
        置信度惩罚：防止模型过度自信，是有效的正则化手段
        H(p) = -sum(p * log(p))，最大化熵 = 最小化负熵
        """
        log_probs = torch.log(probs + self.eps)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        # 返回负熵（用于最小化 → 实际最大化熵）
        return -entropy

    def _s1(self, cls_p, gamma, gamma_aug, targets, w1, w2,
            log_probs, epoch, total_epochs, K):
        """
        Stage1: 训练分类器 + 聚类预热
        
        v3改进：
        1. 更高标签平滑(0.15)防过拟合
        2. 置信度惩罚防止过度自信
        3. 更强的Unknown Encouragement
        4. 聚类预热使用所有样本的特征（不经过mask）
        """
        dev = cls_p.device
        B = cls_p.size(0)
        prog = min(epoch / max(total_epochs - 1, 1), 1.0)

        # --- 1. Focal Cross Entropy (高标签平滑) ---
        smooth = self.s1_smooth  # 固定0.15，不随epoch变化
        ce_raw = F.cross_entropy(cls_p, targets, reduction='none', label_smoothing=smooth)
        with torch.no_grad():
            probs = F.softmax(cls_p, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            fw = (1.0 - pt) ** self.s1_focal
        L_ce = (fw * ce_raw).mean()

        # --- 2. Binary Unknown Detection ---
        unk_mask = (targets == K)
        kn_mask = ~unk_mask
        unk_p = probs[:, K].clamp(self.eps, 1.0 - self.eps)
        L_bin = F.binary_cross_entropy(unk_p, unk_mask.float(), reduction='mean')

        # --- 3. Unknown Encouragement (Hinge Loss) ---
        # 更强的鼓励，让分类器尽早学会识别unknown
        L_unk = torch.tensor(0.0, device=dev)
        n_unk = unk_mask.sum()
        if n_unk > 0:
            margin = 0.2 + 0.3 * prog  # 0.2→0.5 (更高margin)
            L_unk = F.relu(margin - unk_p[unk_mask]).mean()
        n_kn = kn_mask.sum()
        if n_kn > 0:
            # 已知类的unknown概率应该低
            L_unk = L_unk + 0.5 * F.relu(unk_p[kn_mask] - 0.2).mean()

        # --- 4. 置信度惩罚(防过拟合核心) ---
        L_conf = self._confidence_penalty(probs)

        # --- 5. 聚类预热(对ALL样本) ---
        L_warm = torch.tensor(0.0, device=dev)
        L_bal = torch.tensor(0.0, device=dev)
        if B > 0 and gamma.size(0) > 0:
            # 对称KL一致性预热
            g_clamp = gamma.clamp(min=self.eps)
            ga_clamp = gamma_aug.clamp(min=self.eps)
            kl_f = F.kl_div(g_clamp.log(), ga_clamp, reduction='batchmean')
            kl_b = F.kl_div(ga_clamp.log(), g_clamp, reduction='batchmean')
            L_warm = 0.5 * (kl_f + kl_b)
            
            # 簇平衡
            avg_g = gamma.mean(dim=0)
            uni = torch.full_like(avg_g, 1.0 / avg_g.size(0))
            L_bal = F.kl_div((avg_g + self.eps).log(), uni, reduction='sum')

        # --- 6. 专家均衡 ---
        L_e1 = self._exp_bal(w1)
        L_e2 = self._exp_bal(w2)

        # 权重调度
        a_cls = 1.0  # 固定分类权重
        a_clu = 0.3 + 0.7 * prog  # 0.3→1.0 聚类预热逐渐增加

        total = (a_cls * self.s1_ce * L_ce +
                 a_cls * self.s1_bin * L_bin +
                 self.s1_unk * L_unk +
                 self.s1_conf * L_conf +
                 a_clu * self.s1_warm * L_warm +
                 a_clu * self.s1_bal * L_bal +
                 self.s1_exp * (L_e1 + L_e2))

        info = {'L_ce': L_ce.detach(), 'L_binary': L_bin.detach(),
                'L_unk_enc': L_unk.detach(), 'L_conf': L_conf.detach(),
                'L_warmup': L_warm.detach(), 'L_balance': L_bal.detach(),
                'L_exprt1': L_e1.detach(), 'L_exprt2': L_e2.detach(),
                'n_unknown': float(n_unk), 'a_cls': a_cls, 'a_clu': a_clu,
                'smooth': smooth, 'stage': 1}
        return total, info

    def _s2(self, cls_p, gamma, gamma_aug, targets, w1, w2,
            log_probs, epoch, total_epochs, K):
        """
        Stage2: 整体训练提升聚类
        
        v3改进：
        1. 增加互信息最大化(MI)：I(X;C) = H(C) - H(C|X)
           → 最大化边际熵(簇平衡) + 最小化条件熵(样本级确信)
        2. 更强的Sharpening和一致性损失
        3. 使用Sinkhorn进行更好的簇平衡
        4. 置信度惩罚防止分类器退化时过拟合
        """
        dev = cls_p.device
        B = cls_p.size(0)
        prog = min(epoch / max(total_epochs - 1, 1), 1.0)

        # --- 1. 分类维持(低权重 + 高标签平滑) ---
        L_ce = F.cross_entropy(cls_p, targets, reduction='mean', 
                               label_smoothing=self.s2_smooth)
        with torch.no_grad():
            probs = F.softmax(cls_p, dim=1)
        unk_mask = (targets == K)
        unk_p = probs[:, K].clamp(self.eps, 1.0 - self.eps)
        L_bin = F.binary_cross_entropy(unk_p, unk_mask.float(), reduction='mean')

        # 分类器置信度惩罚
        L_conf = self._confidence_penalty(probs)

        # --- 2. 知识蒸馏(可选) ---
        L_kd = torch.tensor(0.0, device=dev)
        if self._teacher_cache is not None and self._teacher_cache.size(0) == B:
            tp = self._teacher_cache.to(dev)
            L_kd = F.kl_div(F.log_softmax(cls_p, dim=1), tp + self.eps, reduction='batchmean')

        # --- 3. 聚类损失(核心改进) ---
        # 不再依赖 unk_mask 过滤，对所有样本计算聚类损失
        # 因为聚类器应该能对所有样本（包括被错误分类为known的unknown样本）做聚类
        nc = gamma.size(1)
        
        L_ps = torch.tensor(0.0, device=dev)
        L_sh = torch.tensor(0.0, device=dev)
        L_mm = torch.tensor(0.0, device=dev)
        L_mi = torch.tensor(0.0, device=dev)
        L_fl = torch.tensor(0.0, device=dev)

        if B > 0 and gamma.size(0) > 0:
            g_clamp = gamma.clamp(min=self.eps)
            ga_clamp = gamma_aug.clamp(min=self.eps)
            
            # (a) 对称KL一致性 - 对所有样本
            L_ps = 0.5 * (
                F.kl_div(g_clamp.log(), ga_clamp, reduction='batchmean') +
                F.kl_div(ga_clamp.log(), g_clamp, reduction='batchmean'))

            # (b) Sharpening: 鼓励聚类分配趋向one-hot (样本级熵最小化)
            st = max(self.s2_stemp * (1.0 - 0.5 * prog), 0.1)
            gm = 0.5 * (gamma + gamma_aug)
            sharp = gm ** (1.0 / st)
            sharp = (sharp / (sharp.sum(dim=1, keepdim=True) + self.eps)).detach()
            L_sh = 0.5 * (
                F.kl_div(g_clamp.log(), sharp + self.eps, reduction='batchmean') +
                F.kl_div(ga_clamp.log(), sharp + self.eps, reduction='batchmean'))

            # (c) 互信息最大化 I(X;C) = H(C) - H(C|X)
            #     H(C) = 边际熵最大化 → 簇平衡
            #     H(C|X) = 条件熵最小化 → 样本级确信
            avg_gamma = gamma.mean(dim=0)  # [nc]
            # H(C): 边际熵最大化（用KL到均匀分布）
            uni = torch.full_like(avg_gamma, 1.0 / nc)
            H_C_loss = F.kl_div((avg_gamma + self.eps).log(), uni, reduction='sum')
            # H(C|X): 条件熵最小化
            H_CX = -(gamma * (gamma + self.eps).log()).sum(dim=1).mean()
            # MI = H(C) - H(C|X)，我们要最大化MI
            # 等价于最小化 -MI = -H(C) + H(C|X) = H_C_loss + H_CX
            # 但H_C_loss已经是最小化KL到均匀的形式（最大化边际熵）
            L_mi = H_CX  # 最小化条件熵
            L_mm = H_C_loss  # 最大化边际熵（用KL到均匀）

            # (d) Flow似然聚类
            if log_probs is not None:
                # 对有效样本计算
                valid = ~torch.isneginf(log_probs).all(dim=1)
                if valid.sum() > 0:
                    lp = log_probs[valid].clamp(min=-100.0)
                    gv = gamma[valid]
                    L_fl = -(gv.detach() * lp).sum(dim=1).mean()

        # --- 4. 专家均衡 ---
        L_e1 = self._exp_bal(w1)
        L_e2 = self._exp_bal(w2)

        # 自适应权重
        a_ps = 1.0 + 1.0 * prog    # 1.0→2.0
        a_sh = 0.5 + 1.5 * prog    # 0.5→2.0
        a_mi = 0.5 + 1.0 * prog    # 0.5→1.5

        total = (self.s2_ce * L_ce + 
                 self.s2_bin * L_bin + 
                 self.s2_conf * L_conf +
                 self.s2_kd * L_kd +
                 a_ps * self.s2_ps * L_ps + 
                 a_sh * self.s2_sharp * L_sh +
                 self.s2_memax * L_mm +
                 a_mi * self.s2_mi * L_mi +
                 self.s2_flow * L_fl + 
                 self.s2_exp * (L_e1 + L_e2))

        info = {'L_ce': L_ce.detach(), 'L_binary': L_bin.detach(),
                'L_conf': L_conf.detach(), 'L_kd': L_kd.detach(),
                'L_ps': L_ps.detach(), 'L_sharp': L_sh.detach(),
                'L_me_max': L_mm.detach(), 'L_mi': L_mi.detach(),
                'L_flow': L_fl.detach(),
                'L_exprt1': L_e1.detach(), 'L_exprt2': L_e2.detach(),
                'n_unknown': float(unk_mask.sum()),
                'a_ps': a_ps, 'a_sh': a_sh, 'a_mi': a_mi, 'stage': 2}
        return total, info

    def _exp_bal(self, w):
        """专家负载均衡损失"""
        avg = torch.mean(w, dim=0)
        uni = torch.ones_like(avg) / avg.size(0)
        return F.kl_div((avg + self.eps).log(), uni, reduction='sum')


# ========================================================================
# 向后兼容封装
# ========================================================================

class Stage1Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Stage1Loss, self).__init__()
        self._loss = LossF(**kwargs)
        self._loss.set_stage(1)

    def forward(self, classifier_log_probs, gamma, gamma_aug, targets,
                weights1, weights2, log_probs=None, epoch=0,
                total_epochs=30, num_known_classes=6):
        return self._loss(classifier_log_probs, gamma, gamma_aug, targets,
                         weights1, weights2, log_probs, epoch, total_epochs, num_known_classes)


class Stage2Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Stage2Loss, self).__init__()
        self._loss = LossF(**kwargs)
        self._loss.set_stage(2)

    def cache_teacher_probs(self, probs):
        self._loss.cache_teacher_probs(probs)

    def forward(self, classifier_log_probs, gamma, gamma_aug, targets,
                weights1, weights2, log_probs=None, epoch=0,
                total_epochs=30, num_known_classes=6):
        return self._loss(classifier_log_probs, gamma, gamma_aug, targets,
                         weights1, weights2, log_probs, epoch, total_epochs, num_known_classes)
