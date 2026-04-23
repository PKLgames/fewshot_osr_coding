import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class LossF(nn.Module):
    def __init__(self,
                 stage1_lambda_ce=1.0, stage1_lambda_binary=1.5,
                 stage1_lambda_unk_enc=1.0, stage1_lambda_warmup=0.3,
                 stage1_lambda_balance=0.1, stage1_lambda_expert=0.1,
                 stage1_label_smoothing=0.2, stage1_focal_gamma=1.5,
                 stage1_lambda_conf_penalty=0.1,
                 stage2_lambda_ce=0.3, stage2_lambda_binary=0.2,
                 stage2_lambda_kd=0.0, stage2_lambda_ps=2.0,
                 stage2_lambda_sharp=1.0, stage2_lambda_me_max=1.0,
                 stage2_lambda_flow=0.3, stage2_lambda_expert=0.05,
                 stage2_label_smoothing=0.1, stage2_sharp_temp=0.5,
                 stage2_lambda_mi=1.0, stage2_lambda_conf_penalty=0.5,
                 stage2_lambda_compactness=0.0, stage2_lambda_separation=0.0,
                 stage2_sinkhorn_iters=10, stage2_sinkhorn_eps=0.01,
                 eps=1e-8, grad_clip=1.0):
        super(LossF, self).__init__()
        self.s1_ce = stage1_lambda_ce
        self.s1_bin = stage1_lambda_binary
        self.s1_unk = stage1_lambda_unk_enc
        self.s1_warm = stage1_lambda_warmup
        self.s1_bal = stage1_lambda_balance
        self.s1_exp = stage1_lambda_expert
        self.s1_smooth = stage1_label_smoothing
        self.s1_focal = stage1_focal_gamma
        self.s1_conf = stage1_lambda_conf_penalty
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
        self.s2_compact = stage2_lambda_compactness
        self.s2_sep = stage2_lambda_separation
        self.s2_sinkhorn_iters = stage2_sinkhorn_iters
        self.s2_sinkhorn_eps = stage2_sinkhorn_eps
        self.eps = eps
        self.grad_clip = grad_clip
        self._stage = 1
        self._teacher_cache = None

    def set_stage(self, stage):
        assert stage in [1, 2]
        self._stage = stage

    def cache_teacher_probs(self, probs):
        self._teacher_cache = probs.detach()

    def forward(self, classifier_log_probs, gamma, gamma_aug, targets,
                weights1, weights2, log_probs=None, epoch=0,
                total_epochs=30, num_known_classes=6, embeddings=None):
        if self._stage == 1:
            return self._s1(classifier_log_probs, gamma, gamma_aug, targets,
                           weights1, weights2, log_probs, epoch, total_epochs, num_known_classes)
        return self._s2(classifier_log_probs, gamma, gamma_aug, targets,
                       weights1, weights2, log_probs, epoch, total_epochs, num_known_classes,
                       embeddings)

    def _confidence_penalty(self, probs, target_entropy=None):
        log_probs = torch.log(probs + self.eps)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        if target_entropy is not None:
            return torch.abs(entropy - target_entropy)
        return entropy

    def _sinkhorn_knopp(self, gamma, n_iters=10, eps=0.01):
        B, K = gamma.shape
        gamma = gamma.clamp(min=eps)
        for _ in range(n_iters):
            row_sum = gamma.sum(dim=1, keepdim=True)
            gamma = gamma / (row_sum + eps)
            col_sum = gamma.sum(dim=0, keepdim=True)
            target_col_sum = B / K
            gamma = gamma * (target_col_sum / (col_sum + eps))
        gamma = gamma / (gamma.sum(dim=1, keepdim=True) + eps)
        return gamma

    def _s1(self, cls_p, gamma, gamma_aug, targets, w1, w2,
            log_probs, epoch, total_epochs, K):
        dev = cls_p.device
        B = cls_p.size(0)
        prog = min(epoch / max(total_epochs - 1, 1), 1.0)

        smooth = self.s1_smooth
        ce_raw = F.cross_entropy(cls_p, targets, reduction='none', label_smoothing=smooth)
        with torch.no_grad():
            probs = F.softmax(cls_p, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            fw = (1.0 - pt) ** self.s1_focal
        L_ce = (fw * ce_raw).mean()

        unk_mask = (targets == K)
        kn_mask = ~unk_mask
        unk_p = probs[:, K].clamp(self.eps, 1.0 - self.eps)
        n_unk = unk_mask.sum()
        n_kn = kn_mask.sum()

        if n_unk > 0 and n_kn > 0:
            w_pos = (B / (2.0 * n_unk.float())).detach()
            w_neg = (B / (2.0 * n_kn.float())).detach()
            L_bin = -(
                w_pos * unk_mask.float() * torch.log(unk_p) +
                w_neg * kn_mask.float() * torch.log(1.0 - unk_p)
            ).mean()
        elif n_unk == 0 and self.training:
            L_bin = torch.tensor(0.0, device=dev)
        else:
            L_bin = F.binary_cross_entropy(unk_p, unk_mask.float(), reduction='mean')

        L_unk = torch.tensor(0.0, device=dev)
        if n_unk > 0:
            margin = 0.5 + 0.3 * prog
            L_unk = F.relu(margin - unk_p[unk_mask]).mean()
            if n_kn > 0:
                L_unk = L_unk + 0.3 * F.relu(unk_p[kn_mask] - 0.1).mean()
        elif n_kn > 0:
            L_unk = 0.5 * F.relu(unk_p[kn_mask] - 0.1).mean()

        L_conf = self._confidence_penalty(probs)

        L_warm = torch.tensor(0.0, device=dev)
        L_bal = torch.tensor(0.0, device=dev)
        L_cons = torch.tensor(0.0, device=dev)

        if B > 0 and gamma.size(0) > 0 and not torch.isnan(gamma).any():
            g_clamp = gamma.clamp(min=self.eps)
            ga_clamp = gamma_aug.clamp(min=self.eps)
            kl_f = F.kl_div(g_clamp.log(), ga_clamp, reduction='batchmean')
            kl_b = F.kl_div(ga_clamp.log(), g_clamp, reduction='batchmean')
            L_warm = 0.5 * (kl_f + kl_b)
            avg_g = gamma.mean(dim=0)
            uni = torch.full_like(avg_g, 1.0 / avg_g.size(0))
            L_bal = F.kl_div((avg_g + self.eps).log(), uni, reduction='batchmean')
            L_cons = F.mse_loss(gamma, ga_clamp)

        L_e1 = self._exp_bal(w1)
        L_e2 = self._exp_bal(w2)

        a_cls = 1.0
        a_clu = 0.1 + 0.3 * prog

        total = (self.s1_ce * L_ce +
                 self.s1_bin * L_bin +
                 self.s1_unk * L_unk +
                 self.s1_conf * L_conf +
                 a_clu * self.s1_warm * L_warm +
                 a_clu * self.s1_bal * L_bal +
                 0.5 * L_cons +
                 self.s1_exp * (L_e1 + L_e2))

        info = {'L_ce': L_ce.detach(), 'L_binary': L_bin.detach(),
                'L_unk_enc': L_unk.detach(), 'L_conf': L_conf.detach(),
                'L_warmup': L_warm.detach(), 'L_balance': L_bal.detach(),
                'L_exprt1': L_e1.detach(), 'L_exprt2': L_e2.detach(),
                'n_unknown': float(n_unk), 'a_cls': a_cls, 'a_clu': a_clu,
                'smooth': smooth, 'stage': 1, 'prog': prog}
        return total, info

    def _s2(self, cls_p, gamma, gamma_aug, targets, w1, w2,
            log_probs, epoch, total_epochs, K, embeddings=None):
        dev = cls_p.device
        B = cls_p.size(0)
        prog = min(epoch / max(total_epochs - 1, 1), 1.0)

        L_ce = F.cross_entropy(cls_p, targets, reduction='mean', 
                               label_smoothing=self.s2_smooth)
        with torch.no_grad():
            probs = F.softmax(cls_p, dim=1)
        unk_mask = (targets == K)
        unk_p = probs[:, K].clamp(self.eps, 1.0 - self.eps)
        L_bin = F.binary_cross_entropy(unk_p, unk_mask.float(), reduction='mean')

        L_conf = torch.tensor(0.0, device=dev)
        L_kd = torch.tensor(0.0, device=dev)

        nc = gamma.size(1) if gamma.size(0) > 0 else 1
        
        L_ps = torch.tensor(0.0, device=dev)
        L_sh = torch.tensor(0.0, device=dev)
        L_mm = torch.tensor(0.0, device=dev)
        L_mi = torch.tensor(0.0, device=dev)
        L_fl = torch.tensor(0.0, device=dev)

        if B > 0 and gamma.size(0) > 0:
            g_clamp = gamma.clamp(min=self.eps)
            ga_clamp = gamma_aug.clamp(min=self.eps)
            
            kl_forward = F.kl_div(g_clamp.log(), ga_clamp, reduction='batchmean')
            kl_backward = F.kl_div(ga_clamp.log(), g_clamp, reduction='batchmean')
            L_ps = 0.5 * (kl_forward + kl_backward)
            L_ps = L_ps.clamp(max=10.0)

            st = max(self.s2_stemp * (1.0 - 0.3 * prog), 0.25)
            gm = 0.5 * (gamma + gamma_aug)
            with torch.no_grad():
                sharp = gm ** (1.0 / st)
                sharp = sharp / (sharp.sum(dim=1, keepdim=True) + self.eps)
            
            L_sh = 0.5 * (
                F.kl_div(g_clamp.log(), sharp, reduction='batchmean') +
                F.kl_div(ga_clamp.log(), sharp, reduction='batchmean'))
            L_sh = L_sh.clamp(max=10.0)

            avg_gamma = gamma.mean(dim=0)
            uni = torch.ones(nc, device=dev) / nc
            L_mm = F.kl_div((avg_gamma + self.eps).log(), uni, reduction='sum')
            L_mm = L_mm / max(B, 1.0)
            L_mm = L_mm.clamp(max=10.0)

            H_CX = -(gamma * g_clamp.log()).sum(dim=1).mean()
            L_mi = H_CX
            L_mi = L_mi.clamp(min=0.0, max=10.0)

            if log_probs is not None:
                valid = ~torch.isneginf(log_probs).all(dim=1)
                if valid.sum() > 0:
                    lp = log_probs[valid].clamp(min=-50.0, max=5.0)
                    gv = gamma[valid].detach()
                    lp_mean = lp.mean()
                    lp_std = lp.std()
                    if lp_std > 0.01:
                        lp_norm = (lp - lp_mean) / (lp_std + self.eps)
                    else:
                        lp_norm = lp - lp_mean
                    L_fl = -(gv * lp_norm).sum(dim=1).mean() * 0.1
                    L_fl = L_fl.clamp(min=-5.0, max=5.0)

        L_e1 = self._exp_bal(w1)
        L_e2 = self._exp_bal(w2)

        a_ps = 1.0 + 0.5 * prog
        a_sh = 0.8 + 0.4 * prog
        a_mi = 1.0 + 0.5 * prog

        total = (self.s2_ce * L_ce +
                 self.s2_bin * L_bin +
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
                'a_ps': a_ps, 'a_sh': a_sh, 'a_mi': a_mi, 'prog': prog, 'stage': 2}
        return total, info

    def _exp_bal(self, w):
        avg = torch.mean(w, dim=0)
        uni = torch.ones_like(avg) / avg.size(0)
        return F.kl_div((avg + self.eps).log(), uni, reduction='sum')


class LossF_V4(nn.Module):
    def __init__(self, **kwargs):
        super(LossF_V4, self).__init__()
        self._loss = LossF(**kwargs)
        self._loss.set_stage(1)

    def set_stage(self, stage):
        self._loss.set_stage(stage)

    def cache_teacher_probs(self, probs):
        self._loss.cache_teacher_probs(probs)

    def forward(self, classifier_log_probs, gamma, gamma_aug, targets,
                weights1, weights2, log_probs=None, epoch=0,
                total_epochs=30, num_known_classes=6, embeddings=None):
        return self._loss(classifier_log_probs, gamma, gamma_aug, targets,
                         weights1, weights2, log_probs, epoch, total_epochs, 
                         num_known_classes, embeddings)


class Stage1Loss_V4(nn.Module):
    def __init__(self, **kwargs):
        super(Stage1Loss_V4, self).__init__()
        self._loss = LossF_V4(**kwargs)
        self._loss.set_stage(1)

    def forward(self, classifier_log_probs, gamma, gamma_aug, targets,
                weights1, weights2, log_probs=None, epoch=0,
                total_epochs=30, num_known_classes=6):
        return self._loss(classifier_log_probs, gamma, gamma_aug, targets,
                         weights1, weights2, log_probs, epoch, total_epochs, num_known_classes)


class Stage2Loss_V4(nn.Module):
    def __init__(self, **kwargs):
        super(Stage2Loss_V4, self).__init__()
        self._loss = LossF_V4(**kwargs)
        self._loss.set_stage(2)

    def cache_teacher_probs(self, probs):
        self._loss.cache_teacher_probs(probs)

    def forward(self, classifier_log_probs, gamma, gamma_aug, targets,
                weights1, weights2, log_probs=None, epoch=0,
                total_epochs=30, num_known_classes=6, embeddings=None):
        return self._loss(classifier_log_probs, gamma, gamma_aug, targets,
                         weights1, weights2, log_probs, epoch, total_epochs, 
                         num_known_classes, embeddings)
