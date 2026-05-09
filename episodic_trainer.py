#!/usr/local/miniconda3/envs/py312/bin/python
"""
Episodic Meta-Trainer for Few-Shot Open Set Recognition

Three-phase training pipeline:
  Phase 1: Pre-extract features using backbone + FC (frozen)
  Phase 2: Episodic meta-training of prototypical classifier
  Phase 3: Calibrate OSR threshold via feature-space methods

Usage:
  python episodic_trainer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import os
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import json
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
from transformers import ASTModel as HF_ASTModel
import transformers.utils.logging as hf_logging
hf_logging.set_verbosity_error()
import logging as _logging
_logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
import torchaudio
import torchaudio.transforms as T_audio

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from utils.TAU22 import TAUDataset
import yamnet_PT_inference as yamnet_infer
from torch_audioset.yamnet.model import yamnet as torch_yamnet
# osr21: Flow reintroduced as feature generator and transformer (NOT for density scoring)
from Reversable_Function.cINN import ConditionalINN


# ============================================================
# 0. Base Class Pretrainer (无数据泄露的特征提取器训练)
# ============================================================

class BaseClassPretrainer:
    """
    Phase 0: 只在 base 类上训练特征提取器 (backbone + FC)

    解决数据泄露问题:
      - osr_training_model.py 的 calib_epoch 使用了全部10类(含unknown)
      - FC层权重已编码unknown类信息 → 泄露到 few-shot 特征
      - 本模块只在 train split 的 base 类(0-5)上训练 → 纯净特征
    """

    def __init__(self, feature_extractor,
                 base_classes: List[int],
                 device: str = 'cuda',
                 lr: float = 1e-4,
                 epochs: int = 30,
                 backbone_unfreeze_layers: List[str] = None):
        """
        Args:
            backbone_unfreeze_layers: backbone layer prefixes to fine-tune.
                e.g. ['encoder.layer.11'] for AST, ['layer14'] for YAMNet.
                None or [] = freeze all backbone layers.
        """
        self.model = feature_extractor.to(device)
        self.base_classes = base_classes
        self.device = device
        self.num_classes = len(base_classes)
        self.backbone_unfreeze_layers = backbone_unfreeze_layers or []

        # 分类头: 64 → 32 → num_classes (两层MLP, 更好的训练信号)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_classes)
        ).to(device)

        # 冻结 backbone, 但解冻指定层 (用于 fine-tune 适配声景任务)
        for name, param in self.model.feature_model.named_parameters():
            should_unfreeze = any(
                name.startswith(prefix) for prefix in self.backbone_unfreeze_layers)
            param.requires_grad = should_unfreeze

        # 分组学习率: backbone fine-tune 用低 LR, FC 层和分类头用正常 LR
        backbone_params = []
        fc_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('feature_model.'):
                backbone_params.append(param)
            else:
                fc_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params, 'lr': lr / 10,
                'name': 'backbone_finetune'})
        param_groups.append({
            'params': fc_params, 'lr': lr, 'name': 'fc_layers'})
        param_groups.append({
            'params': list(self.classifier.parameters()), 'lr': lr,
            'name': 'classifier'})

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=1e-3, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-7)
        self.epochs = epochs

    def _make_base_loader(self, dataset, batch_size=128):
        """
        从 TAUDataset 中过滤出只有 base 类的子集.
        使用 DataLoader 的 collate 机制, 在线过滤.
        """
        base_set = set(self.base_classes)

        # 预先过滤索引
        indices = []
        for i in range(len(dataset)):
            item = dataset[i]
            label_idx = item['target'].argmax().item()
            if label_idx in base_set:
                indices.append(i)

        subset = torch.utils.data.Subset(dataset, indices)
        return DataLoader(subset, batch_size=batch_size, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=False)

    def _remap_label(self, one_hot: torch.Tensor) -> torch.Tensor:
        """将原始 one-hot 标签映射为 0..num_classes-1"""
        orig_labels = one_hot.argmax(dim=1)  # (B,)
        remapped = torch.zeros_like(orig_labels)
        for new_id, orig_id in enumerate(self.base_classes):
            remapped[orig_labels == orig_id] = new_id
        return remapped

    def train(self, dataset, save_path: str):
        """在只含 base 类的数据上训练特征提取器 + 分类头"""
        loader = self._make_base_loader(dataset)

        print(f"\n{'='*70}")
        print("Phase 0b: Base Class Pre-training (no data leakage)")
        print(f"{'='*70}")
        print(f"Base classes: {self.base_classes}")
        print(f"Training samples: {len(loader.dataset)}")
        if self.backbone_unfreeze_layers:
            backbone_type = getattr(self.model, 'BACKBONE_TYPE', 'unknown')
            print(f"{backbone_type.upper()} fine-tune layers: {self.backbone_unfreeze_layers} "
                  f"(LR={self.optimizer.param_groups[0]['lr']:.1e})")
        else:
            print("Backbone: fully frozen")
        print(f"FC + Classifier (LR={self.optimizer.param_groups[-1]['lr']:.1e})")
        print(f"Epochs: {self.epochs}")
        print(f"{'='*70}\n")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.classifier.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for item in loader:
                audio = item['source_audio'].to(self.device)
                targets = item['target'].squeeze(1)  # (B, 10)
                labels = self._remap_label(targets).to(self.device)

                # NOTE: No autocast! YAMNet BatchNorm1d produces NaN in fp16.
                # Use pure fp32 for base class supervised fine-tuning.
                features = self.model(audio)          # (B, 64)
                logits = self.classifier(features)    # (B, num_classes)

                loss = criterion(logits, labels)

                # Mixup augmentation (30% probability per batch) for smoother decision boundaries
                if random.random() < 0.3 and features.size(0) > 1:
                    lam = np.random.beta(0.4, 0.4)
                    perm = torch.randperm(features.size(0), device=self.device)
                    mixed_features = lam * features + (1 - lam) * features[perm]
                    mixed_logits = self.classifier(mixed_features.detach())
                    mixed_labels = lam * F.one_hot(labels, self.num_classes).float() + \
                                   (1 - lam) * F.one_hot(labels[perm], self.num_classes).float()
                    loss = loss + 0.3 * -(mixed_labels * F.log_softmax(mixed_logits, dim=1)).sum(dim=1).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad] +
                    list(self.classifier.parameters()),
                    max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            self.scheduler.step()
            acc = 100. * correct / total
            avg_loss = running_loss / len(loader)
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:>2d}/{self.epochs} | "
                  f"Loss: {avg_loss:.4f} Acc: {acc:.2f}% | LR: {lr:.2e}")

        # 保存 (只保存特征提取器, 不保存分类头)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        backbone = getattr(self.model, 'BACKBONE_TYPE', 'yamnet')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'backbone': backbone,
                'pretrained_on': 'base_classes_only',
                'base_classes': self.base_classes,
            }
        }, save_path)
        print(f"\nFeature extractor saved: {save_path}")

        # 从保存的文件加载 (保持接口一致)
        self.model.load_state_dict(
            torch.load(save_path, map_location=self.device,
                       weights_only=False)['model_state_dict'])

        return self.model


# ============================================================
# 1. Audio Augmentation (for Contrastive Learning)
# ============================================================

class AudioAugmentation:
    """Vectorized waveform-level augmentations for SimCLR contrastive learning.

    Creates two different 'views' of the same audio clip so the model
    learns invariant, discriminative features without labels.
    All augmentations are applied in batch (no per-sample Python loop).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate

    def __call__(self, batch_audio: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to a batch of audio waveforms (vectorized).

        Args:
            batch_audio: (B, T) waveform tensor on any device
        Returns:
            augmented: (B, T) waveform tensor (same device)
        """
        B, T = batch_audio.shape
        device = batch_audio.device
        augmented = batch_audio.clone()

        # 1. Random volume scaling (0.5 ~ 1.5) — loudness invariance (p=0.8)
        gain_mask = torch.rand(B, device=device) < 0.8
        gains = 0.5 + torch.rand(B, device=device)  # [0.5, 1.5)
        augmented[gain_mask] = augmented[gain_mask] * gains[gain_mask].unsqueeze(1)

        # 2. Additive Gaussian noise — noise robustness (p=0.5)
        noise_mask = torch.rand(B, device=device) < 0.5
        noise_levels = 0.001 + torch.rand(B, device=device) * 0.009  # [0.001, 0.01)
        noise = torch.randn_like(augmented)
        augmented[noise_mask] = augmented[noise_mask] + noise[noise_mask] * noise_levels[noise_mask].unsqueeze(1)

        # 3. Random time shift (±10%) — temporal invariance (p=0.5)
        shift_mask = torch.rand(B, device=device) < 0.5
        shifts = torch.randint(-int(T * 0.1), int(T * 0.1) + 1, (B,), device=device)
        for i in shift_mask.nonzero(as_tuple=True)[0]:
            augmented[i] = torch.roll(augmented[i], shifts[i].item())

        # 4. Random time masking (2%~10%) — robustness to missing segments (p=0.3)
        mask4 = torch.rand(B, device=device) < 0.3
        mask_lens = torch.randint(int(T * 0.02), int(T * 0.1) + 1, (B,), device=device)
        mask_starts = torch.randint(0, T, (B,), device=device)
        for i in mask4.nonzero(as_tuple=True)[0]:
            ml = mask_lens[i].item()
            ms = min(mask_starts[i].item(), max(T - ml - 1, 0))
            augmented[i, ms:ms + ml] = 0

        # 5. Speed perturbation (0.85~1.15x) — tempo/rate invariance (p=0.4)
        speed_mask = torch.rand(B, device=device) < 0.4
        speeds = 0.85 + torch.rand(B, device=device) * 0.30  # [0.85, 1.15)
        for i in speed_mask.nonzero(as_tuple=True)[0]:
            speed = speeds[i].item()
            new_len = max(int(T / speed), 1)
            resampled = F.interpolate(
                augmented[i].unsqueeze(0).unsqueeze(0),
                size=new_len, mode='linear', align_corners=False
            ).squeeze(0).squeeze(0)
            if new_len >= T:
                augmented[i] = resampled[:T]
            else:
                augmented[i] = torch.cat([
                    resampled, torch.zeros(T - new_len, device=device)])

        # 6. Random segment fade-out — simulates varying signal quality (p=0.3)
        fade_mask = torch.rand(B, device=device) < 0.3
        fade_lens = torch.randint(int(T * 0.05), int(T * 0.2) + 1, (B,), device=device)
        fade_starts = torch.randint(0, T, (B,), device=device)
        for i in fade_mask.nonzero(as_tuple=True)[0]:
            fl = fade_lens[i].item()
            fs = min(fade_starts[i].item(), max(T - fl - 1, 0))
            fade_curve = torch.linspace(1.0, 0.0, fl, device=device)
            augmented[i, fs:fs + fl] *= fade_curve

        return augmented


# ============================================================
# 1a. Contrastive Pretrainer (SimCLR)
# ============================================================

class ContrastivePretrainer:
    """
    Phase 0a: SimCLR-style contrastive pretraining of the feature extractor.

    Trains on ALL training data (including unknown classes) since it is
    unsupervised — no label leakage.  Two random augmentations of the same
    audio clip form a positive pair; NT-Xent loss pulls their representations
    together while pushing apart representations from different clips.

    This produces features with stronger inter-class discriminability before
    the supervised base-class fine-tuning (Phase 0b).
    """

    def __init__(self, feature_extractor,
                 device: str = 'cuda',
                 lr: float = 3e-4,
                 epochs: int = 20,
                 temperature: float = 0.07,
                 proj_dim: int = 128,
                 batch_size: int = 2048,
                 num_workers: int = 16,
                 prefetch_factor: int = 4,
                 backbone_unfreeze_layers: List[str] = None,
                 supervised_contrastive: bool = True,
                 base_classes: List[int] = None):
        self.model = feature_extractor.to(device)
        self.device = device
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.backbone_unfreeze_layers = backbone_unfreeze_layers or []
        self.supervised_contrastive = supervised_contrastive
        self.base_classes = base_classes or []  # class indices for SupCon

        feat_dim = 64  # output dim of feature extractor

        # Projection head: feat_dim → proj_dim (discarded after pretraining)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        ).to(device)

        # Freeze backbone, except specified layers
        for name, param in self.model.feature_model.named_parameters():
            should_unfreeze = any(
                name.startswith(pfx) for pfx in self.backbone_unfreeze_layers)
            param.requires_grad = should_unfreeze

        # Separate param groups: backbone (low LR) vs FC + projection (normal LR)
        backbone_params = []
        fc_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('feature_model.'):
                backbone_params.append(param)
            else:
                fc_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({
                'params': backbone_params, 'lr': lr / 10,
                'name': 'backbone_finetune'})
        param_groups.append({
            'params': fc_params, 'lr': lr, 'name': 'fc_layers'})
        param_groups.append({
            'params': list(self.projection.parameters()), 'lr': lr,
            'name': 'projection'})

        self.optimizer = torch.optim.AdamW(
            param_groups, weight_decay=1e-3, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-7)

    @staticmethod
    def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        For each sample i in the batch, the positive pair is (z1[i], z2[i]).
        All other 2*(B-1) samples are treated as negatives.
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)           # (2B, D)
        # eps prevents NaN from zero-norm vectors (silent audio clips)
        z = F.normalize(z, dim=1, eps=1e-8)

        # Force float32 for the large similarity matrix to prevent fp16 overflow
        z = z.float()
        sim = torch.mm(z, z.T) / temperature     # (2B, 2B)
        # Mask out self-similarity on the diagonal (use fp16-safe value)
        sim.masked_fill_(
            torch.eye(2 * B, device=z.device).bool(), -1e4)

        # Label: sample i's positive is i+B; sample i+B's positive is i
        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B)
        ]).to(z.device)

        return F.cross_entropy(sim, labels)

    @staticmethod
    def supcon_loss(features: torch.Tensor, labels: torch.Tensor,
                    temperature: float = 0.1) -> torch.Tensor:
        """
        Supervised Contrastive (SupCon) loss.

        Args:
            features: (2B, D) two augmented views concatenated
            labels: (B,) class labels for first view; -1 = exclude from loss
            temperature: scaling temperature
        """
        device = features.device
        B = labels.shape[0]

        # Force float32 for numerical stability
        features = features.float()

        # Repeat labels for both views
        labels_2b = labels.repeat(2)  # (2B,)

        # Valid mask: exclude samples with label -1 (unknown classes)
        valid = labels_2b >= 0

        # Positive pair mask: same label, both valid, not self
        label_eq = labels_2b.unsqueeze(0) == labels_2b.unsqueeze(1)  # (2B, 2B)
        diag_mask = ~torch.eye(2 * B, dtype=torch.bool, device=device)
        pos_mask = label_eq & valid.unsqueeze(1) & valid.unsqueeze(0) & diag_mask

        # At least one positive pair?
        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.sum() == 0:
            # Fallback: treat as NT-Xent (pair i ↔ i+B)
            return F.cross_entropy(
                F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / temperature,
                torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(device))

        # Cosine similarity matrix
        feats_norm = F.normalize(features, dim=1, eps=1e-8)
        sim = torch.mm(feats_norm, feats_norm.T) / temperature  # (2B, 2B)
        sim.masked_fill_(~diag_mask, -1e4)  # mask diagonal

        # log-softmax over negatives (all j ≠ i)
        log_sum_exp = torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob = sim - log_sum_exp

        # Mean log-prob over positives, for anchors that have positives
        mean_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / (pos_mask.float().sum(dim=1) + 1e-8)
        loss = -mean_log_prob[has_pos].mean()
        return loss

    def train(self, dataset, save_path: str = None):
        """Run contrastive pretraining: SimCLR first half → SupCon second half (osr17)."""
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            prefetch_factor=self.prefetch_factor)

        supcon_start = self.epochs // 2  # switch at halfway point
        use_supcon = self.supervised_contrastive and len(self.base_classes) > 0

        print(f"\n{'='*70}")
        print("Phase 0a: Contrastive Pre-training")
        print(f"{'='*70}")
        print(f"Samples:         {len(loader.dataset)}")
        print(f"Epochs:          {self.epochs}")
        print(f"Batch size:      {self.batch_size}")
        print(f"Temperature:     {self.temperature}")
        print(f"Backbone unfreeze: {self.backbone_unfreeze_layers}")
        if use_supcon:
            print(f"Mode:            SimCLR (ep 1-{supcon_start}) → "
                  f"SupCon (ep {supcon_start+1}-{self.epochs})")
            print(f"Base classes:    {self.base_classes}")
        else:
            print(f"Mode:            SimCLR (all epochs)")
        print(f"{'='*70}\n")

        augment = AudioAugmentation(sample_rate=16000)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.projection.train()

            running_loss = 0.0
            num_batches = 0
            use_supcon_epoch = use_supcon and epoch > supcon_start

            for item in loader:
                audio = item['source_audio'].to(self.device)

                # Fix zero-norm samples (silent clips) with tiny noise
                # instead of skipping — keeps all data in training
                audio_norms = audio.abs().amax(dim=1)  # (B,)
                silent_mask = audio_norms < 1e-8
                if silent_mask.any():
                    audio[silent_mask] = torch.randn_like(audio[silent_mask]) * 1e-6

                # Two independently augmented views
                view1 = augment(audio)
                view2 = augment(audio)

                # NOTE: No autocast here! YAMNet backbone has BatchNorm1d layers
                # that produce NaN in fp16 (output range [-45,45] overflows BN stats).
                # Use pure fp32 for contrastive pretraining — YAMNet is small enough
                # that mixed precision provides negligible speedup.
                h1 = self.model(view1)          # (B, 64)
                h2 = self.model(view2)          # (B, 64)

                # Project
                z1 = self.projection(h1)        # (B, proj_dim)
                z2 = self.projection(h2)        # (B, proj_dim)

                if use_supcon_epoch:
                    # Extract integer class labels from one-hot target
                    # target shape: (B, 1, num_classes) — squeeze middle dim
                    targets = item['target'].squeeze(1)  # (B, num_classes)
                    labels = targets.argmax(dim=1).to(self.device)  # (B,)
                    # Mark non-base classes as -1 (excluded from SupCon)
                    base_set = set(self.base_classes)
                    label_mask = torch.tensor(
                        [l.item() in base_set for l in labels],
                        dtype=torch.bool, device=self.device)
                    labels_for_supcon = labels.clone()
                    labels_for_supcon[~label_mask] = -1

                    features = torch.cat([z1, z2], dim=0)  # (2B, D)
                    loss = self.supcon_loss(features, labels_for_supcon, self.temperature)
                else:
                    loss = self.nt_xent_loss(z1, z2, self.temperature)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    list(self.projection.parameters()),
                    max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()
                num_batches += 1

            self.scheduler.step()
            avg_loss = running_loss / max(num_batches, 1)
            lr = self.optimizer.param_groups[-1]['lr']
            mode_str = "SupCon" if use_supcon_epoch else "SimCLR"
            print(f"Epoch {epoch:>2d}/{self.epochs} [{mode_str}] | "
                  f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            backbone = getattr(self.model, 'BACKBONE_TYPE', 'yamnet')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {
                    'backbone': backbone,
                    'pretrained_on': 'contrastive_simclr_supcon' if use_supcon else 'contrastive_simclr',
                    'backbone_unfreeze_layers': self.backbone_unfreeze_layers,
                }
            }, save_path)
            print(f"\nContrastive pretrained model saved: {save_path}")

        return self.model


# ============================================================
# 2. Feature Extraction Pipeline
# ============================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise feature recalibration"""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class FeatureExtractor(nn.Module):
    """YAMNet + FC layers, frozen during episodic training

    防止数据泄露: 总是从 yamnet.pth 加载预训练 YAMNet 权重 + 随机 FC 层,
    不从 OSR 模型加载 FC 权重 (OSR 模型的 FC 层已在全类别上训练过).
    """

    BACKBONE_TYPE = 'yamnet'

    def __init__(self, pretrained_path: str = None,
                 load_yamnet_pretrained: bool = True):
        super().__init__()
        self.feature_model = torch_yamnet(pretrained=False)
        # YAMNet 521-dim → multi-scale feature extraction with SE attention
        self.fc1 = nn.Sequential(
            nn.Linear(521, 256), nn.BatchNorm1d(256), nn.GELU())
        self.se1 = SEBlock(256, reduction=4)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU())
        self.se2 = SEBlock(128, reduction=4)
        self.drop2 = nn.Dropout(0.15)

        self.fc3 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64))

        # 始终从 yamnet.pth 加载预训练权重, FC 层保持随机初始化
        if load_yamnet_pretrained and os.path.exists('yamnet.pth'):
            state = torch.load('yamnet.pth', map_location='cpu')
            self.feature_model.load_state_dict(state)
            print("FeatureExtractor: loaded YAMNet pretrained weights from yamnet.pth "
                  "(FC layers random)")

        # pretrained_path 用于加载 Phase 0 训练好的完整特征提取器
        # (含 YAMNet + 已在 base 类上训练的 FC 层)
        if pretrained_path and os.path.exists(pretrained_path):
            self.load_from_base_checkpoint(pretrained_path)

    def load_from_base_checkpoint(self, path: str):
        """Load full feature extractor from a Phase 0 base-class checkpoint"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)

        own_state = self.state_dict()
        loaded = 0
        for name, param in state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        print(f"FeatureExtractor: loaded {loaded}/{len(own_state)} parameters from {path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Audio waveform -> 64-dim features (SE-attended, no fusion)"""
        x = yamnet_infer.waveform_to_log_mel_patches(x, sample_rate=16000)
        feature = self.feature_model(x, to_prob=False)

        h1 = self.drop1(self.se1(self.fc1(feature)))   # (B, 256)
        h2 = self.drop2(self.se2(self.fc2(h1)))          # (B, 128)
        h3 = self.fc3(h2)                                 # (B, 64)
        return h3

    def freeze(self):
        """Freeze all parameters for episodic training"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self


class DistilASTFeatureExtractor(nn.Module):
    """Distil-AST (6-layer distilled) + GPU mel extraction + FC layers

    Replaces full AST (12 layers, 86M params, 3min/epoch) with distilled
    version (6 layers, 44M params) + torchaudio GPU-based mel spectrogram.
    ~1.8x faster than full AST due to:
      - GPU mel extraction (~3x faster than CPU numpy)
      - 2x fewer transformer layers
    Output: same 64-dim features, downstream pipeline unchanged.
    """

    BACKBONE_TYPE = 'distil_ast'

    def __init__(self, pretrained_path: str = None,
                 load_pretrained: bool = True):
        super().__init__()

        # Load Distil-AST backbone (no classification head)
        self.feature_model = HF_ASTModel.from_pretrained(
            'bookbot/distil-ast-audioset',
            attn_implementation='sdpa')

        # Freeze by default (unfreezing done in Phase 0 trainers)
        for param in self.feature_model.parameters():
            param.requires_grad = False

        # GPU mel extraction (replaces CPU numpy HF processor)
        # AST params: n_fft=400, hop=160, n_mels=128, 16kHz
        # Use htk mel scale to avoid filterbank warning
        self.mel_transform = T_audio.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160,
            n_mels=128, f_min=0, f_max=8000, power=2.0,
            mel_scale='htk', norm=None)
        # Normalization constants from AST training on AudioSet
        self._mel_mean = -4.2677393
        self._mel_std = 4.5689974

        # Projection: 768 → 256 → 128 → 64 (same as original AST)
        self.fc1 = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.GELU())
        self.se1 = SEBlock(256, reduction=4)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU())
        self.se2 = SEBlock(128, reduction=4)
        self.drop2 = nn.Dropout(0.15)

        self.fc3 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64))

        if pretrained_path and os.path.exists(pretrained_path):
            self.load_from_base_checkpoint(pretrained_path)

    def load_from_base_checkpoint(self, path: str):
        """Load full feature extractor from a Phase 0 checkpoint"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)

        config = checkpoint.get('config', {})
        if config.get('backbone', 'yamnet') != self.BACKBONE_TYPE:
            print(f"WARNING: checkpoint backbone is '{config.get('backbone')}', "
                  f"expected '{self.BACKBONE_TYPE}'. Skipping load.")
            return

        own_state = self.state_dict()
        loaded = 0
        for name, param in state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        print(f"DistilASTFeatureExtractor: loaded {loaded}/{len(own_state)} "
              f"parameters from {path}")

    def _waveform_to_mel_gpu(self, waveform: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated mel extraction matching AST's expected input format.
        Args:
            waveform: (B, samples) at 16kHz, on GPU
        Returns:
            (B, 1024, 128) normalized mel spectrograms (padded to AST's max_length)
        """
        with torch.amp.autocast('cuda', enabled=False):
            mel = self.mel_transform(waveform.float())            # (B, 128, T)
            log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))  # power_to_db
            log_mel = (log_mel - self._mel_mean) / self._mel_std  # normalize
            # Pad time axis to 1024 frames (AST's max_length)
            if log_mel.size(2) < 1024:
                pad = 1024 - log_mel.size(2)
                log_mel = F.pad(log_mel, (0, pad))  # pad right (time axis)
            return log_mel[:, :, :1024].transpose(1, 2)  # (B, 1024, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Audio waveform -> 64-dim features (SE-attended)
        Args:
            x: (B, samples) raw waveform at 16kHz
        Returns:
            (B, 64) feature vectors
        """
        input_values = self._waveform_to_mel_gpu(x)  # (B, 1024, 128)

        ast_needs_grad = any(
            p.requires_grad for p in self.feature_model.parameters())
        if ast_needs_grad:
            outputs = self.feature_model(input_values)
        else:
            with torch.no_grad():
                outputs = self.feature_model(input_values)

        cls_embedding = outputs.pooler_output  # (B, 768)

        h1 = self.drop1(self.se1(self.fc1(cls_embedding)))   # (B, 256)
        h2 = self.drop2(self.se2(self.fc2(h1)))               # (B, 128)
        h3 = self.fc3(h2)                                      # (B, 64)
        return h3

    def freeze(self):
        """Freeze all parameters for episodic training"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self


# ---- PANNs Cnn14_16k Feature Extractor ----

class PANNsConvBlock(nn.Module):
    """Conv block from PANNs (Cnn14 architecture): 2x Conv2d + BN + ReLU + AvgPool"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=2)
        return x


class PANNsCnn14_16k(nn.Module):
    """PANNs Cnn14 backbone for 16kHz audio (no spectrogram extractor).

    Architecture: bn0 + 6 ConvBlocks (1→64→128→256→512→1024→2048) + global pool.
    Output: (B, 2048) embedding vector.
    Pretrained on AudioSet, mAP=0.438.
    """
    def __init__(self):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = PANNsConvBlock(1, 64)
        self.conv_block2 = PANNsConvBlock(64, 128)
        self.conv_block3 = PANNsConvBlock(128, 256)
        self.conv_block4 = PANNsConvBlock(256, 512)
        self.conv_block5 = PANNsConvBlock(512, 1024)
        self.conv_block6 = PANNsConvBlock(1024, 2048)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, 1, time, 64) log-mel spectrogram"""
        # BatchNorm on frequency axis
        x = x.transpose(1, 3)   # (B, 64, time, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)   # (B, 1, time, 64)

        x = self.conv_block1(x)  # (B, 64, T/2, 32)
        x = self.conv_block2(x)  # (B, 128, T/4, 16)
        x = self.conv_block3(x)  # (B, 256, T/8, 8)
        x = self.conv_block4(x)  # (B, 512, T/16, 4)
        x = self.conv_block5(x)  # (B, 1024, T/32, 2)
        x = self.conv_block6(x)  # (B, 2048, T/64, 1)

        # Global average + max pooling over frequency (avg+max strategy from PANNs)
        x = torch.mean(x, dim=3)  # (B, 2048, T') — avg over freq
        x1, _ = torch.max(x, dim=2)  # max over remaining time
        x2 = torch.mean(x, dim=2)    # avg over remaining time
        x = x1 + x2                   # (B, 2048)
        return x


class PANNsFeatureExtractor(nn.Module):
    """PANNs Cnn14_16k + GPU mel + FC projection → 64-dim features.

    CNN-based backbone (~5M trainable params) pretrained on AudioSet.
    Much faster than transformer-based AST/Distil-AST (~50x faster per batch).
    Expected: ~45s/epoch for Phase 0b pretraining.

    Mel params match PANNs Cnn14_16k training config:
      n_fft=512, hop=160, n_mels=64, f_min=50, f_max=8000, 16kHz
    """
    BACKBONE_TYPE = 'panns_cnn14'
    WEIGHTS_FILE = 'panns_cnn14_16k.pth'
    WEIGHTS_URL = ('https://zenodo.org/api/records/3987831/files/'
                   'Cnn14_16k_mAP=0.438.pth/content')

    def __init__(self, pretrained_path: str = None,
                 load_pretrained: bool = True):
        super().__init__()

        self.feature_model = PANNsCnn14_16k()

        # GPU mel extraction (PANNs Cnn14_16k params)
        self.mel_transform = T_audio.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=160,
            n_mels=64, f_min=50, f_max=8000, power=2.0,
            mel_scale='slaney', norm='slaney')

        # Load PANNs pretrained weights
        if load_pretrained:
            self._load_panns_weights()

        # Freeze backbone by default
        for param in self.feature_model.parameters():
            param.requires_grad = False

        # Projection: 2048 → 256 → 128 → 64
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 256), nn.BatchNorm1d(256), nn.GELU())
        self.se1 = SEBlock(256, reduction=4)
        self.drop1 = nn.Dropout(0.2)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU())
        self.se2 = SEBlock(128, reduction=4)
        self.drop2 = nn.Dropout(0.15)

        self.fc3 = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64))

        if pretrained_path and os.path.exists(pretrained_path):
            self.load_from_base_checkpoint(pretrained_path)

    def _load_panns_weights(self):
        """Load Cnn14_16k pretrained weights, downloading if needed."""
        if not os.path.exists(self.WEIGHTS_FILE):
            print(f"PANNs weights not found at {self.WEIGHTS_FILE}. "
                  f"Downloading from Zenodo (~359MB)...")
            import urllib.request
            urllib.request.urlretrieve(self.WEIGHTS_URL, self.WEIGHTS_FILE)
            print(f"Downloaded: {self.WEIGHTS_FILE}")

        state = torch.load(self.WEIGHTS_FILE, map_location='cpu',
                           weights_only=False)
        # PANNs checkpoint may wrap state in 'model' or be a plain state_dict
        if isinstance(state, dict) and 'model' in state:
            state = state['model']

        own_state = self.feature_model.state_dict()
        loaded = 0
        for name, param in state.items():
            # Only load backbone weights (bn0 + conv_blocks)
            if any(name.startswith(p) for p in ['bn0', 'conv_block']):
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    loaded += 1
        print(f"PANNsFeatureExtractor: loaded {loaded}/{len(own_state)} "
              f"backbone parameters from {self.WEIGHTS_FILE}")

    def load_from_base_checkpoint(self, path: str):
        """Load full feature extractor from a Phase 0 checkpoint"""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)

        config = checkpoint.get('config', {})
        if config.get('backbone', 'yamnet') != self.BACKBONE_TYPE:
            print(f"WARNING: checkpoint backbone is '{config.get('backbone')}', "
                  f"expected '{self.BACKBONE_TYPE}'. Skipping load.")
            return

        own_state = self.state_dict()
        loaded = 0
        for name, param in state.items():
            if name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
        print(f"PANNsFeatureExtractor: loaded {loaded}/{len(own_state)} "
              f"parameters from {path}")

    def _waveform_to_mel_gpu(self, waveform: torch.Tensor) -> torch.Tensor:
        """GPU mel extraction matching PANNs Cnn14_16k input format.
        Args:
            waveform: (B, samples) at 16kHz
        Returns:
            (B, 1, time, 64) log-mel spectrogram
        """
        with torch.amp.autocast('cuda', enabled=False):
            mel = self.mel_transform(waveform.float())  # (B, 64, T)
            log_mel = torch.log10(torch.clamp(mel, min=1e-10))  # PANNs uses log10
            # Reshape to (B, 1, T, 64) for PANNs CNN
            return log_mel.transpose(1, 2).unsqueeze(1)  # (B, 1, T, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Audio waveform -> 64-dim features
        Args:
            x: (B, samples) raw waveform at 16kHz
        Returns:
            (B, 64) feature vectors
        """
        mel_input = self._waveform_to_mel_gpu(x)  # (B, 1, T, 64)

        backbone_needs_grad = any(
            p.requires_grad for p in self.feature_model.parameters())
        if backbone_needs_grad:
            features = self.feature_model(mel_input)
        else:
            with torch.no_grad():
                features = self.feature_model(mel_input)  # (B, 2048)

        h1 = self.drop1(self.se1(self.fc1(features)))   # (B, 256)
        h2 = self.drop2(self.se2(self.fc2(h1)))          # (B, 128)
        h3 = self.fc3(h2)                                 # (B, 64)
        return h3

    def freeze(self):
        """Freeze all parameters for episodic training"""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self


# ============================================================
# 2. Feature Cache
# ============================================================

class FeatureCache:
    """Pre-extracted features organized by class for fast episode sampling

    缓存版本控制: 当特征提取器改变时 (如切换backbone),
    旧缓存自动失效, 重新提取特征.
    """

    CACHE_VERSION = 'v20_tau22relabel_yamnet_normalized'  # Same features as osr20 (same backbone)

    def __init__(self, cache_dir: str = 'experiment/fewshot_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.features_by_class: Dict[int, torch.Tensor] = {}
        self.labels: Optional[torch.Tensor] = None
        self.features: Optional[torch.Tensor] = None
        self.class_list: List[int] = []

    def extract_and_cache(self, feature_extractor: nn.Module,
                          dataset, split: str, device: str,
                          batch_size: int = 512,
                          norm_mean: torch.Tensor = None,
                          norm_std: torch.Tensor = None):
        """Pre-extract features from dataset and save to disk.

        Args:
            norm_mean/norm_std: Per-dim statistics from training set.
                When provided, features are standardized: (f - mean) / (std + eps).
                For training split, pass None to auto-compute.
        """
        cache_path = os.path.join(self.cache_dir, f'{split}_features.pt')

        if os.path.exists(cache_path):
            data = torch.load(cache_path, weights_only=True)
            cached_version = data.get('version', 'v1_osr_leaked')
            if cached_version == self.CACHE_VERSION:
                print(f"Loading cached features from {cache_path} (version: {cached_version})")
                self.features = data['features']
                self.labels = data['labels']

        if self.features is None or self.labels is None:
            print(f"Extracting features for '{split}' split...")
            feature_extractor = feature_extractor.to(device)
            feature_extractor.eval()

            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=False,
                num_workers=8, pin_memory=True, drop_last=False)

            all_features = []
            all_labels = []

            with torch.no_grad():
                for item in tqdm(loader, desc=f"Extracting {split}"):
                    audio = item['source_audio'].to(device)
                    targets = item['target'].squeeze(1)       # (B, 10)
                    labels = targets.argmax(dim=1)            # (B,)

                    feats = feature_extractor(audio)          # (B, 64)
                    all_features.append(feats.cpu())
                    all_labels.append(labels.cpu())

            self.features = torch.cat(all_features, dim=0)
            self.labels = torch.cat(all_labels, dim=0)

            # osr17d: Per-dim standardization using training set statistics
            if split == 'train':
                norm_mean = self.features.mean(dim=0)
                norm_std = self.features.std(dim=0)
                print(f"  Feature normalization: mean_range=[{norm_mean.min():.4f}, {norm_mean.max():.4f}] "
                      f"std_range=[{norm_std.min():.4f}, {norm_std.max():.4f}]")

            if norm_mean is not None and norm_std is not None:
                self.features = (self.features - norm_mean) / (norm_std + 1e-6)

            torch.save(
                {'features': self.features, 'labels': self.labels,
                 'version': self.CACHE_VERSION,
                 'norm_mean': norm_mean, 'norm_std': norm_std},
                cache_path)
            print(f"Cached {len(self.features)} features -> {cache_path}")

        # Organize by class
        self.features_by_class = {}
        for i in range(len(self.labels)):
            c = self.labels[i].item()
            if c not in self.features_by_class:
                self.features_by_class[c] = []
            self.features_by_class[c].append(self.features[i])

        self.features_by_class = {
            k: torch.stack(v) for k, v in self.features_by_class.items()}
        self.class_list = sorted(self.features_by_class.keys())

        print(f"  Classes: {self.class_list}")
        for c in self.class_list:
            print(f"    Class {c}: {len(self.features_by_class[c])} samples")

    def get_class_features(self, class_id: int) -> torch.Tensor:
        return self.features_by_class.get(class_id, torch.zeros(0))

    def load(self, cache_path: str):
        """Load pre-cached features from disk (for eval-only mode)."""
        data = torch.load(cache_path, weights_only=True)
        self.features = data['features']
        self.labels = data['labels']

        # Organize by class
        self.features_by_class = {}
        for i in range(len(self.labels)):
            c = self.labels[i].item()
            if c not in self.features_by_class:
                self.features_by_class[c] = []
            self.features_by_class[c].append(self.features[i])

        self.features_by_class = {
            k: torch.stack(v) for k, v in self.features_by_class.items()}
        self.class_list = sorted(self.features_by_class.keys())

        print(f"  Loaded {cache_path}: {len(self.features)} samples, "
              f"classes={self.class_list}")
        for c in self.class_list:
            print(f"    Class {c}: {len(self.features_by_class[c])} samples")


# ============================================================
# 3. Episode Sampler
# ============================================================

class EpisodeSampler:
    """N-way K-shot Q-query episode sampler"""

    def __init__(self, feature_cache: FeatureCache,
                 available_classes: List[int],
                 N_way: int = 5, K_shot: int = 5, Q_query: int = 15,
                 support_drop_rate: float = 0.0):
        self.cache = feature_cache
        self.N = N_way
        self.K = K_shot
        self.Q = Q_query
        self.support_drop_rate = support_drop_rate

        # Only keep classes with enough samples
        min_samples = K_shot + Q_query
        self.valid_classes = [
            c for c in available_classes
            if len(feature_cache.get_class_features(c)) >= min_samples]

        if len(self.valid_classes) < N_way:
            raise ValueError(
                f"Need {N_way} classes with >= {min_samples} samples, "
                f"only {len(self.valid_classes)} available: {self.valid_classes}")

    def sample_episode(self, device: str = 'cpu') -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
            support_feats:   (N*K, D)
            support_labels:  (N*K,)  remapped to 0..N-1
            query_feats:     (N*Q, D)
            query_labels:    (N*Q,)  remapped to 0..N-1
            prototypes:      (N, D)  computed from support set
            episode_classes: list of original class ids
        """
        episode_classes = random.sample(self.valid_classes, self.N)

        support_feats, support_labels = [], []
        query_feats, query_labels = [], []

        for new_label, orig_class in enumerate(episode_classes):
            class_feats = self.cache.get_class_features(orig_class)
            indices = random.sample(range(len(class_feats)), self.K + self.Q)

            # Task augmentation: randomly drop support samples
            support_indices = list(indices[:self.K])
            if self.support_drop_rate > 0 and self.K > 1:
                support_indices = [idx for idx in support_indices
                                   if random.random() > self.support_drop_rate]
                if not support_indices:
                    support_indices = [indices[0]]

            actual_k = len(support_indices)
            support_feats.append(class_feats[support_indices])
            support_labels.extend([new_label] * actual_k)
            query_feats.append(class_feats[indices[self.K:]])
            query_labels.extend([new_label] * self.Q)

        support_feats = torch.cat(support_feats, dim=0).to(device)
        support_labels = torch.tensor(support_labels, dtype=torch.long, device=device)
        query_feats = torch.cat(query_feats, dim=0).to(device)
        query_labels = torch.tensor(query_labels, dtype=torch.long, device=device)

        # Compute prototypes
        prototypes = torch.zeros(self.N, support_feats.size(1), device=device)
        for c in range(self.N):
            mask = support_labels == c
            if mask.any():
                prototypes[c] = support_feats[mask].mean(dim=0)

        return {
            'support_feats': support_feats,
            'support_labels': support_labels,
            'query_feats': query_feats,
            'query_labels': query_labels,
            'prototypes': prototypes,
            'episode_classes': episode_classes,
        }

    @staticmethod
    def generate_pseudo_novel(features_by_class: Dict[int, torch.Tensor],
                              class_list: List[int],
                              num_samples: int,
                              device: str = 'cpu') -> torch.Tensor:
        """
        Generate pseudo-novel (simulated unknown) features.

        Methods:
          1. Inter-class mixup: convex combinations of features from different classes
          2. Strong perturbation: large Gaussian noise on random base features

        These teach the flow to assign LOW likelihood to out-of-distribution inputs,
        which is critical for OSR (unknown detection).
        """
        pseudo = []

        # Method 1: Inter-class mixup (60% of samples)
        n_mixup = int(num_samples * 0.6)
        for _ in range(n_mixup):
            c1, c2 = random.sample(class_list, 2)
            f1 = features_by_class[c1][random.randint(0, len(features_by_class[c1]) - 1)]
            f2 = features_by_class[c2][random.randint(0, len(features_by_class[c2]) - 1)]
            alpha = random.uniform(0.2, 0.8)
            pseudo.append(alpha * f1 + (1 - alpha) * f2)

        # Method 2: Strong perturbation (40% of samples)
        n_perturb = num_samples - n_mixup
        for _ in range(n_perturb):
            c = random.choice(class_list)
            f = features_by_class[c][random.randint(0, len(features_by_class[c]) - 1)]
            noise = torch.randn_like(f) * random.uniform(0.3, 0.8)
            pseudo.append(f + noise)

        return torch.stack(pseudo).to(device)


# ============================================================
# 3b. Flow OOD Generator (osr21a) — 条件Flow生成伪OOD样本
# ============================================================

class FlowOODGenerator:
    """
    osr21a: 用条件Flow生成高质量伪OOD特征样本, 替代GMM低密度采样.

    核心思路: 训练条件Flow学习已知类特征分布 p(x|class),
    然后通过在z空间操作(插值/外推/条件不匹配)生成边界样本.

    与osr18/19的区别: Flow不参与OSR评分, 只用于数据生成.
    """

    def __init__(self, base_features: torch.Tensor, class_labels: torch.Tensor,
                 device: str = 'cuda', n_coupling_layers: int = 3,
                 hidden_dim: int = 64, epochs: int = 200, lr: float = 1e-3):
        self.device = device
        self.feature_dim = base_features.shape[1]
        self.unique_classes = sorted(class_labels.unique().tolist())
        self.n_classes = len(self.unique_classes)

        # Per-class Flow models (轻量化: 每个类一个条件Flow)
        self.flows = {}
        self.class_means = {}
        self.class_stds = {}

        for c in self.unique_classes:
            mask = class_labels == c
            feats_c = base_features[mask]
            self.class_means[c] = feats_c.mean(dim=0)
            self.class_stds[c] = feats_c.std(dim=0).clamp(min=0.1)

            # 轻量ConditionalINN: input=feature_dim, condition=0 (无条件, 每类独立)
            flow = ConditionalINN(
                input_dim=self.feature_dim,
                condition_dim=0,  # 无条件: 每类独立建模
                num_coupling_layers=n_coupling_layers,
                hidden_dims=[hidden_dim, hidden_dim],
                use_permutation=True,
                permutation_type='fixed',
                dropout=0.1,
                s_clamp_max=2.0
            ).to(device)

            # 训练: MLE on known features
            feats_dev = feats_c.to(device)
            optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

            flow.train()
            prior = torch.distributions.Normal(
                torch.zeros(self.feature_dim, device=device),
                torch.ones(self.feature_dim, device=device))

            best_loss = float('inf')
            best_state = None
            for epoch in range(epochs):
                # Mini-batch training
                perm = torch.randperm(len(feats_dev))
                batch_size = min(64, len(feats_dev))
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, len(feats_dev), batch_size):
                    idx = perm[start:start + batch_size]
                    batch = feats_dev[idx]
                    c_empty = torch.zeros(batch.size(0), 0, device=device)

                    z, log_det = flow(batch, c_empty, compute_jacobian=True)
                    log_prob = prior.log_prob(z).sum(dim=1) + log_det
                    loss = -log_prob.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                scheduler.step()
                avg_loss = epoch_loss / max(n_batches, 1)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = {k: v.clone() for k, v in flow.state_dict().items()}

            if best_state is not None:
                flow.load_state_dict(best_state)
            flow.eval()
            self.flows[c] = flow

            n_params = sum(p.numel() for p in flow.parameters())
            print(f"  Flow class {c}: {len(feats_c)} samples, "
                  f"best NLL={best_loss:.2f}, params={n_params:,}")

    def _inverse_batch(self, flow: ConditionalINN, z: torch.Tensor) -> torch.Tensor:
        """Inverse in batches to avoid OOM."""
        results = []
        batch_size = 1024
        c_empty = torch.zeros(z.size(0), 0, device=z.device)
        for start in range(0, len(z), batch_size):
            z_batch = z[start:start + batch_size]
            c_batch = torch.zeros(z_batch.size(0), 0, device=z.device)
            x_batch = flow.inverse(z_batch, c_batch)
            results.append(x_batch)
        return torch.cat(results, dim=0)

    def generate_interpolation(self, n_samples: int,
                                alpha: float = 0.5) -> torch.Tensor:
        """策略A: z空间插值 → 类间边界样本.
        从两个不同类的Flow生成, 在z空间插值后逆变换."""
        samples = []
        for _ in range(n_samples):
            c1, c2 = random.sample(self.unique_classes, 2)
            z1 = torch.randn(1, self.feature_dim, device=self.device)
            z2 = torch.randn(1, self.feature_dim, device=self.device)
            z_interp = alpha * z1 + (1 - alpha) * z2
            # 逆变换到c1的特征空间
            c_empty = torch.zeros(1, 0, device=self.device)
            x = self.flows[c1].inverse(z_interp, c_empty)
            samples.append(x.squeeze(0))
        return torch.stack(samples)

    def generate_low_density(self, n_samples: int,
                              z_scale: float = 2.0) -> torch.Tensor:
        """策略B: 远离原点的z → 特征空间低密度样本.
        z ~ N(0, z_scale²·I), z_scale > 1 意味着采样点远离训练分布."""
        samples = []
        for _ in range(n_samples):
            c = random.choice(self.unique_classes)
            z = torch.randn(1, self.feature_dim, device=self.device) * z_scale
            c_empty = torch.zeros(1, 0, device=self.device)
            x = self.flows[c].inverse(z, c_empty)
            samples.append(x.squeeze(0))
        return torch.stack(samples)

    def generate_cross_class(self, n_samples: int) -> torch.Tensor:
        """策略C: 条件不匹配 → 精确边界样本.
        从c1的z采样, 但用c2的Flow逆变换 → 不属于任何类的边界样本."""
        samples = []
        for _ in range(n_samples):
            c1, c2 = random.sample(self.unique_classes, 2)
            z = torch.randn(1, self.feature_dim, device=self.device)
            # 用c2的Flow逆变换c1的z → 生成不匹配样本
            c_empty = torch.zeros(1, 0, device=self.device)
            x = self.flows[c2].inverse(z, c_empty)
            samples.append(x.squeeze(0))
        return torch.stack(samples)

    def sample(self, n: int, strategy: str = 'mix') -> torch.Tensor:
        """混合采样: 默认三种策略各1/3."""
        if strategy == 'mix':
            n1 = n // 3
            n2 = n // 3
            n3 = n - n1 - n2
            parts = []
            if n1 > 0:
                parts.append(self.generate_interpolation(n1, alpha=random.uniform(0.3, 0.7)))
            if n2 > 0:
                parts.append(self.generate_low_density(n2, z_scale=random.uniform(1.5, 3.0)))
            if n3 > 0:
                parts.append(self.generate_cross_class(n3))
            return torch.cat(parts, dim=0)
        elif strategy == 'interpolation':
            return self.generate_interpolation(n)
        elif strategy == 'low_density':
            return self.generate_low_density(n)
        elif strategy == 'cross_class':
            return self.generate_cross_class(n)
        else:
            return self.generate_low_density(n)


# ============================================================
# 3c. Lightweight Flow Transform (osr21b) — 可学习特征变换
# ============================================================

class LightweightFlowTransform(nn.Module):
    """
    osr21b: 轻量可学习Flow特征变换.

    与osr18 ConditionalINN的区别:
      - 无condition (不依赖class_id)
      - 使用固定条件c=0 (无条件Flow)
      - 2层coupling, 更轻量
      - 训练目标: 分类loss驱动, 非MLE

    用途: 将特征变换到线性可分性更好的空间,
    使anti_prototype的反射操作更准确.
    """
    def __init__(self, input_dim: int, n_coupling_layers: int = 2,
                 hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.flow = ConditionalINN(
            input_dim=input_dim,
            condition_dim=0,  # 无条件
            num_coupling_layers=n_coupling_layers,
            hidden_dims=[hidden_dim, hidden_dim],
            use_permutation=True,
            permutation_type='fixed',
            dropout=0.1,
            s_clamp_max=2.0
        )
        self._c_empty = None  # cached empty condition

    def _get_c_empty(self, batch_size: int, device: torch.device):
        if self._c_empty is None or self._c_empty.size(0) != batch_size:
            self._c_empty = torch.zeros(batch_size, 0, device=device)
        return self._c_empty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transform: features → transformed features."""
        c = self._get_c_empty(x.size(0), x.device)
        z, _ = self.flow(x, c, compute_jacobian=False)
        return z

    def forward_with_logdet(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with log determinant (for osr21c Jacobian scoring)."""
        c = self._get_c_empty(x.size(0), x.device)
        return self.flow(x, c, compute_jacobian=True)

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transform."""
        c = self._get_c_empty(z.size(0), z.device)
        return self.flow.inverse(z, c)


# ============================================================
# 4. Episodic Flow Classifier
# ============================================================

class EpisodicFlowClassifier(nn.Module):
    """
    Prototypical distance classifier for few-shot learning.
    OOD detection via feature-space methods (Mahalanobis, anti-prototype, etc.)
    """

    def __init__(self, input_dim: int = 64, condition_dim: int = 64,
                 use_flow_transform: bool = True,
                 use_ood_head: bool = True,
                 use_reciprocal: bool = True,
                 use_threshold: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.use_flow_transform = use_flow_transform
        # Ablation flags
        self.use_ood_head = use_ood_head
        self.use_reciprocal = use_reciprocal
        self.use_threshold = use_threshold

        # Feature adapter: trainable transform to refine frozen cached features
        # Learns a class-agnostic rotation/scaling of the feature space
        # that improves few-shot discrimination for both base and novel classes
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        # osr21b: Lightweight Flow transform for non-linear feature space mapping
        if use_flow_transform:
            self.feature_transform = LightweightFlowTransform(
                input_dim=input_dim,
                n_coupling_layers=2,
                hidden_dim=input_dim // 2
            )

        # Prototypical distance head (strong few-shot classification signal)
        # Operates in full 64-dim space for classification quality
        self.distance_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(input_dim, input_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(10.0))

        # Binary OOD detection head: operates on feature-space statistics (5-dim)
        # [min_dist, dist_ratio, softmax_max, entropy, feat_norm]
        if use_ood_head:
            self.ood_head = nn.Sequential(
                nn.Linear(5, 32), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, 1),  # logit for known(1) / unknown(0)
            )

        # osr17a: Learnable OSR threshold (trained during episodic training)
        if use_threshold:
            self.osr_threshold = LearnableOSRThreshold(num_scores=1, init_value=0.0)

        # osr17b: Reciprocal points for OSR scoring (one per max possible class)
        if use_reciprocal:
            self.num_max_classes = 20
            self.reciprocal_points = nn.Parameter(
                torch.randn(self.num_max_classes, input_dim) * 0.1
            )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature adapter."""
        return self.feature_adapter(x)

    def adapt_features(self, features: torch.Tensor,
                       prototypes: torch.Tensor):
        """Apply feature adapter to both features and prototypes."""
        adapted_feats = self.feature_adapter(features)
        adapted_protos = self.feature_adapter(prototypes)
        return adapted_feats, adapted_protos

    def classify(self, features: torch.Tensor,
                 prototypes: torch.Tensor) -> torch.Tensor:
        """
        Classification scores via prototypical distance.
        Returns: scores (B, N)  higher = more likely that class
        """
        # --- Feature adaptation (refine cached features) ---
        features, prototypes = self.adapt_features(features, prototypes)

        # osr21b: Optional Flow transform for non-linear boundary
        if self.use_flow_transform:
            features = self.feature_transform(features)
            prototypes = self.feature_transform(prototypes)

        # --- Prototypical distance path (64-dim) ---
        dist_feats = self.distance_head(features)     # (B, D)
        dist_protos = self.distance_head(prototypes)   # (N, D)
        dists = torch.cdist(dist_feats, dist_protos, p=2) ** 2  # (B, N)
        proto_scores = -dists * self.temperature.abs()           # (B, N)

        # Smooth clamp: prevents extreme scores → CE explosions
        combined = 10.0 * torch.tanh(proto_scores / 10.0)
        return combined

    def refine_prototypes_transductive(self, support_feats: torch.Tensor,
                                        support_labels: torch.Tensor,
                                        query_feats: torch.Tensor,
                                        prototypes: torch.Tensor,
                                        num_iters: int = 3,
                                        confidence_threshold: float = 0.9) -> torch.Tensor:
        """EM-style transductive prototype refinement.

        Uses high-confidence query predictions to expand the support set,
        then recomputes prototypes. Improves few-shot accuracy especially
        for novel classes where initial K-shot prototypes are noisy.

        Args:
            support_feats: (S, D) support set features
            support_labels: (S,) support set labels (0..N-1)
            query_feats: (Q, D) query features
            prototypes: (N, D) initial prototypes from support set
            num_iters: number of EM iterations
            confidence_threshold: min softmax prob to include a query
        Returns:
            refined_prototypes: (N, D)
        """
        current_protos = prototypes.clone()
        N = prototypes.size(0)

        for _ in range(num_iters):
            log_probs = self.classify(query_feats, current_protos)
            probs = F.softmax(log_probs, dim=1)
            max_probs, pred_labels = probs.max(dim=1)

            confident_mask = max_probs >= confidence_threshold
            if not confident_mask.any():
                break

            confident_feats = query_feats[confident_mask]
            confident_labels = pred_labels[confident_mask]

            for c in range(N):
                support_mask = support_labels == c
                support_c = support_feats[support_mask]
                query_c = confident_feats[confident_labels == c]
                if len(query_c) > 0:
                    current_protos[c] = torch.cat(
                        [support_c, query_c], dim=0).mean(dim=0)

        return current_protos

    @staticmethod
    def l2_normalize(features: torch.Tensor,
                     prototypes: torch.Tensor):
        """L2 normalize features and prototypes (inference only).
        Converts Euclidean distance to cosine distance for better transfer."""
        return (F.normalize(features, p=2, dim=1),
                F.normalize(prototypes, p=2, dim=1))

    def compute_reciprocal_loss(self, features, prototypes, labels):
        """osr17e: Reciprocal point loss (per-dim normalized).
        Move reciprocal points TOWARDS prototypes (indirectly pushes away from features).
        osr17d: raw L2² in 64-dim ≈ 50, dominated Total loss.
        osr17e: divide by input_dim → per-dim average ≈ 0.78, balanced scale."""
        N = prototypes.shape[0]
        R = self.reciprocal_points[:N]

        dist_R_to_proto = ((R - prototypes) ** 2).sum(dim=1)
        L = dist_R_to_proto.mean() / self.input_dim  # per-dim normalization

        return L

    def score_reciprocal(self, features, prototypes):
        """osr17b: Reciprocal point OSR score.
        High score = known (close to prototype, far from reciprocal).
        Low score = unknown."""
        N = prototypes.shape[0]
        R = self.reciprocal_points[:N]
        dist_proto = torch.cdist(features, prototypes, p=2).min(dim=1).values
        dist_R = torch.cdist(features, R, p=2).min(dim=1).values
        return -dist_proto + dist_R  # known high, unknown low

    def score_anti_prototype(self, features: torch.Tensor,
                              prototypes: torch.Tensor) -> torch.Tensor:
        """osr17h: Anti-prototype scoring with query-dependent center (reverted from osr17g).
        osr17g used prototypes.mean() as fixed center → test TPR dropped 17.45%→8.32%.
        Query-dependent center creates per-class reference frames that per-round
        recalibration compensates for, yielding better test generalization.
        This is empirically validated, not a bug."""
        # Query-dependent center: each class batch uses its own reference
        feat_center = features.mean(dim=0, keepdim=True)  # (1, D)
        # Anti-prototypes: reflection of proto through center
        anti_protos = 2 * feat_center - prototypes  # (N, D)
        dist_proto = torch.cdist(features, prototypes, p=2).min(dim=1).values
        dist_anti = torch.cdist(features, anti_protos, p=2).min(dim=1).values
        return -dist_proto + dist_anti  # known high, unknown low

    def score_anti_prototype_enhanced(self, features: torch.Tensor,
                                       prototypes: torch.Tensor) -> torch.Tensor:
        """Anti-prototype scoring with confidence-weighted center."""
        # Confidence-weighted center: use feature norms as proxy for confidence
        feat_norms = (features ** 2).sum(dim=1, keepdim=True).sqrt()  # (B, 1)
        total_norm = feat_norms.sum() + 1e-8
        weights = feat_norms / total_norm  # (B, 1) normalized confidence
        feat_center = (features * weights).sum(dim=0, keepdim=True)  # (1, D)

        anti_protos = 2 * feat_center - prototypes
        dist_proto = torch.cdist(features, prototypes, p=2).min(dim=1).values
        dist_anti = torch.cdist(features, anti_protos, p=2).min(dim=1).values
        return -dist_proto + dist_anti


class LearnableOSRThreshold(nn.Module):
    """Learnable OSR threshold: replaces fixed percentile calibration.
    Trained end-to-end during episodic training to produce known/unknown
    probabilities from a continuous OSR score (e.g. Mahalanobis distance)."""
    def __init__(self, num_scores: int = 1, init_value: float = 0.0):
        super().__init__()
        self.thresholds = nn.Parameter(torch.full((num_scores,), init_value))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """Convert continuous score → known probability via sigmoid."""
        return torch.sigmoid((scores - self.thresholds) * self.temperature.abs())

    def get_threshold(self) -> float:
        return self.thresholds.item()


# ============================================================
# 5. Episodic Loss
# ============================================================

class EpisodicLoss(nn.Module):
    """
    Meta-training loss for episodic few-shot learning.

    Components:
      1. L_ce             - classification NLL on query predictions
      2. L_density        - maximize log p(query | correct prototype)
      3. L_entropy        - encourage confident predictions (minimize entropy)
      4. L_separation     - maximize inter-prototype distance in feature space
    """

    def __init__(self, N_way: int = 5,
                 lambda_density: float = 0.1,
                 lambda_entropy: float = 0.05,
                 lambda_separation: float = 0.02,
                 ce_cap: float = 10.0):
        super().__init__()
        self.N_way = N_way
        self.ce_cap = ce_cap
        self.lambda_density = lambda_density
        self.lambda_entropy = lambda_entropy
        self.lambda_separation = lambda_separation

    def forward(self, log_probs: torch.Tensor,
                query_labels: torch.Tensor,
                prototypes: torch.Tensor = None,
                scale_cls: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        B, N = log_probs.shape

        # 1. Classification loss with smooth tanh cap to prevent gradient explosions
        log_preds = F.log_softmax(log_probs, dim=1)
        L_ce_raw = F.nll_loss(log_preds, query_labels)
        L_ce = self.ce_cap * torch.tanh(L_ce_raw / self.ce_cap)

        # 2. Density: push target class score up
        target_lp = log_probs.gather(1, query_labels.unsqueeze(1)).mean()
        L_density = torch.clamp(-target_lp, min=-5.0, max=5.0)

        # 3. Entropy: encourage confident predictions
        probs = F.softmax(log_probs, dim=1)
        entropy = -(probs * log_preds).sum(dim=1).mean()
        L_entropy = entropy

        # 4. Prototype separation: maximize pairwise distance between prototypes
        L_separation = torch.tensor(0.0, device=log_probs.device)
        if prototypes is not None and prototypes.size(0) > 1:
            pdist = torch.pdist(prototypes)
            L_separation = torch.clamp(-pdist.mean(), min=-2.0, max=0.0)

        total = (L_ce
                 + scale_cls * self.lambda_entropy * L_entropy
                 + scale_cls * self.lambda_separation * L_separation
                 + scale_cls * self.lambda_density * L_density)

        info = {
            'L_ce': L_ce.detach(),
            'L_density': L_density.detach(),
            'L_entropy': L_entropy.detach(),
            'L_separation': L_separation.detach(),
            'total': total.detach(),
        }
        return total, info


# ============================================================
# 5b. GMM Boundary Sampler — 生成更真实的伪OOD样本
# ============================================================

class GMMBoundarySampler:
    """
    从GMM低密度区域采样伪OOD样本，替代纯随机噪声。

    核心思路：真实unknown类不是随机噪声，而是"接近base类但不同"的分布。
    用GMM拟合base类特征后，从低密度区域采样得到的样本更接近真实unknown：
    - 在特征流形边界附近（不是完全脱离流形的随机噪声）
    - 不属于任何已知类的核心区域（GMM低密度 = 远离所有类中心）

    流程：
    1. 在CPU上用base类特征拟合GMM（一次性）
    2. 生成大量候选样本，按GMM对数似然排序
    3. 保留低密度样本（低于训练数据10th percentile），缓存到GPU
    4. 训练时从缓存中随机抽取
    """

    def __init__(self, base_features: torch.Tensor, device: str = 'cuda',
                 n_components: int = 6, pool_size: int = 5000,
                 density_percentile: float = 10.0):
        """
        Args:
            base_features: (N, D) base类特征（已在GPU上）
            device: 'cuda' or 'cpu'
            n_components: GMM高斯分量数（= base类数）
            pool_size: 预采样的低密度样本池大小
            density_percentile: 低密度阈值（percentile），越低越接近边界
        """
        self.device = device
        self.pool_size = pool_size

        # 在CPU上拟合GMM（sklearn不支持GPU）
        feats_cpu = base_features.cpu().numpy()
        print(f"  GMM Boundary Sampler: fitting GMM({n_components}) on "
              f"{feats_cpu.shape[0]} samples, dim={feats_cpu.shape[1]}")

        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            max_iter=200,
            random_state=42,
            reg_covar=1e-6  # Add regularization to fix covariance warning
        )
        self.gmm.fit(feats_cpu)

        # 训练数据的对数似然分布
        train_ll = self.gmm.score_samples(feats_cpu)
        self.threshold = np.percentile(train_ll, density_percentile)
        print(f"  GMM train log-likelihood: "
              f"mean={train_ll.mean():.2f}, "
              f"10th percentile={self.threshold:.2f}")

        # 生成候选样本并筛选低密度样本
        self._build_pool(feats_cpu.shape[1])

    def _build_pool(self, feature_dim: int):
        """预采样低密度样本池，缓存到GPU"""
        # 多生成一些候选，再筛选
        n_candidates = self.pool_size * 10
        candidates, _ = self.gmm.sample(n_candidates)
        candidates = candidates.astype(np.float32)

        # 按GMM对数似然排序，取低密度样本
        ll = self.gmm.score_samples(candidates)
        low_density_mask = ll < self.threshold
        low_density_samples = candidates[low_density_mask]

        if len(low_density_samples) < self.pool_size:
            # 低密度样本不够，放宽阈值
            sorted_idx = np.argsort(ll)
            low_density_samples = candidates[sorted_idx[:self.pool_size]]
        else:
            # 随机取pool_size个
            rng = np.random.RandomState(42)
            idx = rng.choice(len(low_density_samples), self.pool_size, replace=False)
            low_density_samples = low_density_samples[idx]

        self._pool = torch.from_numpy(low_density_samples).to(self.device)
        print(f"  GMM boundary pool: {self._pool.shape[0]} samples, "
              f"dim={self._pool.shape[1]}, device={self.device}")

    def sample(self, n: int) -> torch.Tensor:
        """从低密度池中随机抽取n个样本"""
        idx = torch.randint(0, len(self._pool), (n,), device=self.device)
        return self._pool[idx]


class CurriculumOODSampler:
    """
    osr17b: Curriculum OOD采样器 - 3级难度递进

    Level 1 (ep 1-1000): 简单扰动 - mixup + 高斯噪声
        - 伪未知样本 = 随机混合 base类特征 + 添加噪声
        - 目的：让 OOD head 学习基本边界

    Level 2 (ep 1001-2000): 中等难度 - 特征插值 + 低密度样本
        - 伪未知样本 = GMM边界样本（类间低密度区域）
        - 目的：让 OOD head 学习更精确的边界

    Level 3 (ep 2001+): 困难模式 - 对抗样本
        - 伪未知样本 = 靠近决策边界的混淆样本
        - 目的：让 OOD head 学习鲁棒性

    自动切换：根据当前 episode 选择合适的难度
    """

    def __init__(self, base_features: torch.Tensor, base_classes: List[int],
                 device: str = 'cuda', n_components: int = 6,
                 use_flow_generator: bool = True):
        """
        Args:
            base_features: (N, D) base类特征
            base_classes: base类ID列表
            device: 'cuda' or 'cpu'
            n_components: GMM分量数
            use_flow_generator: osr21a — 使用Flow生成器替代GMM
        """
        self.device = device
        self.base_classes = base_classes
        self.feature_dim = base_features.shape[1]

        # Level 1: 简单扰动不需要预计算
        print("  CurriculumOODSampler Level 1: Simple perturbations (mixup + noise)")

        # Level 2: osr21a Flow生成器 or GMM边界采样器
        self._flow_generator = None
        self._gmm_sampler = None
        self.use_flow_generator = use_flow_generator

        if use_flow_generator:
            print("  CurriculumOODSampler Level 2: osr21a Flow OOD Generator...")
            # 构建类标签
            samples_per_class = len(base_features) // len(base_classes)
            class_labels = torch.cat([
                torch.full((samples_per_class,), c) for c in base_classes
            ])
            self._flow_generator = FlowOODGenerator(
                base_features, class_labels, device,
                n_coupling_layers=3, hidden_dim=64, epochs=200, lr=1e-3)
        else:
            print("  CurriculumOODSampler Level 2: Initializing GMM boundary sampler...")
            self._gmm_sampler = GMMBoundarySampler(
                base_features, device, n_components=n_components, pool_size=5000)

        # Level 3: 对抗采样器（延迟初始化，需要分类器）
        self._adversarial_sampler = None
        self._flow_classifier = None

        # 缓存 base_features 用于 Level 1 和 Level 3
        self._base_features = base_features
        self._base_by_class = {
            c: base_features[i * (len(base_features) // len(base_classes)):
                      (i + 1) * (len(base_features) // len(base_classes))]
            for i, c in enumerate(base_classes)
        }

    def set_flow_classifier(self, classifier):
        """设置分类器用于 Level 3 对抗采样"""
        self._flow_classifier = classifier

    def get_level(self, current_episode: int) -> int:
        """
        根据当前 episode 返回 curriculum 难度等级
        Level 1: ep 1-1000
        Level 2: ep 1001-2000
        Level 3: ep 2001+
        """
        if current_episode <= 1000:
            return 1
        elif current_episode <= 2000:
            return 2
        else:
            return 3

    def sample(self, n: int, level: int = None, current_episode: int = 0) -> torch.Tensor:
        """
        采样 n 个伪 OOD 样本

        Args:
            n: 样本数量
            level: 指定难度等级（None则自动根据episode计算）
            current_episode: 当前episode（用于自动计算level）

        Returns:
            (n, D) 伪 OOD 特征
        """
        if level is None:
            level = self.get_level(current_episode)

        if level == 1:
            return self._sample_level1(n)
        elif level == 2:
            return self._sample_level2(n)
        elif level == 3:
            return self._sample_level3(n)
        else:
            raise ValueError(f"Unknown level: {level}")

    def _sample_level1(self, n: int) -> torch.Tensor:
        """
        Level 1: 简单扰动 - mixup + 高斯噪声

        40% mixup: 随机混合两个不同类的特征
        60% perturbation: 单个特征加高斯噪声
        """
        pseudo = []

        # Mixup样本 (40%)
        n_mixup = int(0.4 * n)
        for _ in range(n_mixup):
            c1, c2 = random.sample(self.base_classes, 2)
            f1 = self._base_by_class[c1][
                random.randint(0, len(self._base_by_class[c1]) - 1)]
            f2 = self._base_by_class[c2][
                random.randint(0, len(self._base_by_class[c2]) - 1)]
            alpha = random.uniform(0.2, 0.8)
            pseudo.append(alpha * f1 + (1 - alpha) * f2)

        # 强扰动样本 (60%)
        n_perturb = n - n_mixup
        for _ in range(n_perturb):
            c = random.choice(self.base_classes)
            f = self._base_by_class[c][
                random.randint(0, len(self._base_by_class[c]) - 1)]
            noise = torch.randn_like(f) * random.uniform(0.3, 0.8)
            pseudo.append(f + noise)

        return torch.stack(pseudo).to(self.device)

    def _sample_level2(self, n: int) -> torch.Tensor:
        """Level 2: osr21a Flow生成 or GMM边界样本"""
        if self._flow_generator is not None:
            return self._flow_generator.sample(n, strategy='mix')
        return self._gmm_sampler.sample(n)

    def _sample_level3(self, n: int) -> torch.Tensor:
        """
        Level 3: 困难模式 - 靠近决策边界的对抗样本

        策略：在特征空间中找到"让分类器不确定"的点
        - 随机采样候选点 (优先使用Flow生成器)
        - 计算分类熵（熵越高 = 越不确定）
        - 选择熵最高的点作为伪OOD
        """
        if self._flow_classifier is None:
            # 如果分类器未设置，回退到 Level 2
            return self._sample_level2(n)

        # 生成候选样本 (Flow生成 or GMM边界附近)
        if self._flow_generator is not None:
            candidates = self._flow_generator.sample(n * 5, strategy='low_density')
        else:
            candidates = self._gmm_sampler.sample(n * 5)  # 多生成一些

        # 计算每个候选的分类熵
        with torch.no_grad():
            # 使用随机原型计算分数
            random_protos = torch.stack(
                [self._base_by_class[c][:5].mean(dim=0)
                 for c in self.base_classes[:5]]
            ).to(self.device)

            log_probs = self._flow_classifier.classify(candidates, random_protos)
            probs = F.softmax(log_probs, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

            # 选择熵最高的 n 个样本（最不确定的）
            _, indices = torch.topk(entropy, n)

        return candidates[indices]


# ============================================================
# 6. Episodic Trainer
# ============================================================

class EpisodicTrainer:
    """
    Episodic meta-trainer for few-shot open set recognition.

    Training loop:
      1. Sample N-way K-shot episode from base class features
      2. Compute prototypes from support set
      3. Add noise to prototypes (robustness)
      4. Forward: compute log p(query | proto_c) for each class c
      5. Loss = classification CE + density regularization
      6. Update flow parameters only (feature extractor frozen)
    """

    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 flow_classifier: EpisodicFlowClassifier,
                 train_cache: FeatureCache,
                 calib_cache: FeatureCache,
                 test_cache: FeatureCache,
                 base_classes: List[int],
                 unknown_classes: List[int],
                 ood_features_by_class: Dict[int, torch.Tensor] = None,
                 N_way: int = 5,
                 K_shot: int = 5,
                 Q_query: int = 15,
                 lr: float = 1e-4,
                 proto_noise_std: float = 0.15,
                 ood_ratio: float = 0.3,
                 warmup_episodes: int = 500,
                 task_aug_drop_rate: float = 0.0,
                 gradient_accum_steps: int = 4,
                 device: str = 'cuda'):

        self.device = device
        self.base_classes = base_classes
        self.unknown_classes = unknown_classes
        self.N_way = N_way
        self.K_shot = K_shot
        self.Q_query = Q_query
        self.proto_noise_std = proto_noise_std
        self.ood_ratio = ood_ratio
        self.warmup_episodes = warmup_episodes
        self.base_lr = lr
        self.gradient_accum_steps = gradient_accum_steps
        self.training_augment = True

        self.stage2_len = warmup_episodes * 2

        # Real OOD features for outlier exposure (e.g., calib unknown class features)
        self.ood_features_by_class = ood_features_by_class or {}
        if self.ood_features_by_class:
            total_ood = sum(len(v) for v in self.ood_features_by_class.values())
            print(f"Real OOD exposure: {total_ood} samples from "
                  f"classes {list(self.ood_features_by_class.keys())}")

        # Pre-cache concatenated tensors on GPU (avoids per-episode CPU rebuild + transfer)
        self._ood_all = (
            torch.cat(list(self.ood_features_by_class.values()), dim=0).to(device)
            if self.ood_features_by_class else None)
        self._base_all = torch.cat(
            [train_cache.get_class_features(c) for c in base_classes], dim=0
        ).to(device)

        # osr17b: Curriculum OOD sampler with 3 difficulty levels
        print("Initializing Curriculum OOD Sampler (3-level progressive training)...")
        self._ood_sampler = CurriculumOODSampler(
            self._base_all, base_classes, device, n_components=len(base_classes),
            use_flow_generator=False)  # osr22: 禁用Flow，使用GMM边界采样器

        # Freeze feature extractor
        self.feature_extractor = feature_extractor.freeze()

        # Flow classifier (trainable, includes projection head)
        self.flow_classifier = flow_classifier.to(device)
        print(f"Flow classifier params: "
              f"{sum(p.numel() for p in self.flow_classifier.parameters()):,}")

        # Episode samplers (task augmentation only on train)
        self.train_sampler = EpisodeSampler(
            train_cache, base_classes, N_way, K_shot, Q_query,
            support_drop_rate=task_aug_drop_rate)
        self.val_sampler = EpisodeSampler(
            calib_cache, base_classes, N_way, K_shot, Q_query)

        # Caches for OSR calibration later
        self.train_cache = train_cache
        self.calib_cache = calib_cache
        self.test_cache = test_cache

        # Loss: classification losses only (CE + density + entropy + separation)
        self.criterion = EpisodicLoss(
            N_way=self.N_way,
            lambda_density=0.0,
            lambda_entropy=0.0,
            lambda_separation=0.05,
        )

        # Optimizer & scheduler (T_max = episodes after warmup)
        self.optimizer = torch.optim.AdamW(
            self.flow_classifier.parameters(),
            lr=lr, weight_decay=1e-2, amsgrad=True)
        # Adjust T_max: cosine completes earlier to avoid sustained high LR mid-training
        total_training = 6000 // self.gradient_accum_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_training, eta_min=1e-7)

        # Tracking
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
        }
        self.best_val_acc = 0.0
        self.patience_episodes = 3000  # osr17b: increased from 2000 → allow Phase 3 full training
        self.early_stop_counter = 0

    # ----------------------------------------------------------
    # Single training episode
    # ----------------------------------------------------------
    def train_episode(self, current_episode: int = 0,
                      accum_scale: float = 1.0) -> Tuple[float, float, float, dict]:
        """Run one training episode, return (loss, accuracy, ce_loss, stats)."""
        self.flow_classifier.train()

        episode = self.train_sampler.sample_episode(self.device)
        prototypes = episode['prototypes']
        query_feats = episode['query_feats']
        query_labels = episode['query_labels']

        # Add Gaussian noise to prototypes for robustness
        if self.proto_noise_std > 0:
            prototypes = prototypes + torch.randn_like(prototypes) * self.proto_noise_std

        # Feature-level augmentation
        if self.training_augment:
            scale = 0.8 + 0.40 * torch.rand_like(query_feats)
            query_feats = query_feats * scale
            query_feats = query_feats + 0.02 * torch.randn_like(query_feats)

        # Classification via prototypical distance
        log_probs = self.flow_classifier.classify(query_feats, prototypes)

        # OOD exposure
        ood_feats = None
        if self._ood_all is not None and self.ood_ratio > 0:
            num_ood = max(1, int(query_feats.size(0) * self.ood_ratio))
            num_real = int(num_ood * 0.6)
            num_pseudo = num_ood - num_real

            real_idx = torch.randint(0, len(self._ood_all), (num_real,), device=self.device)
            ood_parts = [self._ood_all[real_idx]]

            if num_pseudo > 0:
                curriculum_ood = self._ood_sampler.sample(
                    num_pseudo, current_episode=current_episode)
                ood_parts.append(curriculum_ood)

            ood_feats = torch.cat(ood_parts, dim=0)

        # Project prototypes for separation loss
        projected_protos = self.flow_classifier.project(prototypes)

        # Staged loss ramping
        w = self.warmup_episodes
        s2 = self.stage2_len

        if current_episode <= w:
            scale_cls = 0.0
        elif current_episode <= w + s2:
            progress = (current_episode - w) / s2
            scale_cls = min(1.0, progress * 0.8)
        else:
            scale_cls = 1.0

        loss, info = self.criterion(
            log_probs, query_labels,
            prototypes=projected_protos,
            scale_cls=scale_cls)

        # OOD head loss: binary classifier on 5-dim feature-space statistics
        if scale_cls > 0 and self.flow_classifier.use_ood_head and hasattr(self.flow_classifier, 'ood_head'):
            def _compute_ood_features_feat(feat_batch, proto_batch, log_probs_batch):
                """Extract 5-dim OOD feature vector."""
                dists = torch.cdist(feat_batch, proto_batch, p=2)  # (B, N)
                sorted_d, _ = dists.sort(dim=1)
                min_d = sorted_d[:, 0:1]
                second_d = sorted_d[:, 1:2]
                dist_ratio = min_d / (second_d + 1e-6)
                probs = F.softmax(log_probs_batch, dim=1)
                softmax_max = probs.max(dim=1, keepdim=True)[0]
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
                feat_norm = (feat_batch ** 2).sum(dim=1, keepdim=True).sqrt()
                return torch.cat([min_d, dist_ratio, softmax_max, entropy, feat_norm], dim=1)

            # Known samples: target = 1
            known_feats = _compute_ood_features_feat(query_feats, prototypes, log_probs)
            known_logits = self.flow_classifier.ood_head(known_feats)
            known_targets = torch.ones(query_feats.size(0), 1, device=self.device)
            ood_loss = F.binary_cross_entropy_with_logits(known_logits, known_targets)

            # Unknown samples: target = 0
            if ood_feats is not None and ood_feats.size(0) > 0:
                with torch.no_grad():
                    ood_log_probs = self.flow_classifier.classify(ood_feats, prototypes)
                unk_feats = _compute_ood_features_feat(ood_feats, prototypes, ood_log_probs)
                unk_logits = self.flow_classifier.ood_head(unk_feats)
                unk_targets = torch.zeros(ood_feats.size(0), 1, device=self.device)
                ood_loss = ood_loss + F.binary_cross_entropy_with_logits(unk_logits, unk_targets)

            loss = loss + scale_cls * 0.1 * ood_loss

        # Learnable threshold loss
        if scale_cls > 0 and self.flow_classifier.use_threshold and hasattr(self.flow_classifier, 'osr_threshold'):
            with torch.no_grad():
                adapted_q, adapted_p = self.flow_classifier.adapt_features(query_feats, prototypes)
                dist_q = torch.cdist(adapted_q, adapted_p, p=2).min(dim=1).values
                known_scores = -dist_q
            known_probs = self.flow_classifier.osr_threshold(known_scores)
            L_threshold = -torch.log(known_probs + 1e-8).mean()

            if ood_feats is not None and ood_feats.size(0) > 0:
                with torch.no_grad():
                    adapted_o, _ = self.flow_classifier.adapt_features(ood_feats, adapted_p)
                    dist_o = torch.cdist(adapted_o, adapted_p, p=2).min(dim=1).values
                    unknown_scores = -dist_o
                unknown_probs = self.flow_classifier.osr_threshold(unknown_scores)
                L_threshold = L_threshold + -torch.log(1 - unknown_probs + 1e-8).mean()

            loss = loss + 0.1 * L_threshold

        # Reciprocal point loss
        if scale_cls > 0 and self.flow_classifier.use_reciprocal and hasattr(self.flow_classifier, 'reciprocal_points') and query_labels.max() < self.flow_classifier.num_max_classes:
            L_reciprocal = self.flow_classifier.compute_reciprocal_loss(
                query_feats, prototypes, query_labels)
            loss = loss + 3.0 * L_reciprocal

        # Skip episode if loss is NaN/Inf
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()
            return 0.0, 1.0 / self.N_way, 0.0, {}

        # Backward with accumulation scaling
        scaled_loss = loss * accum_scale
        scaled_loss.backward()

        # Accuracy
        preds = log_probs.argmax(dim=1)
        acc = (preds == query_labels).float().mean().item()

        return loss.item(), acc, info['L_ce'].item(), {}

    # ----------------------------------------------------------
    # Single validation episode
    # ----------------------------------------------------------
    def validate_episode(self) -> Tuple[float, float]:
        """Run one validation episode (no grad)"""
        self.flow_classifier.eval()

        with torch.no_grad():
            episode = self.val_sampler.sample_episode(self.device)
            log_probs = self.flow_classifier.classify(
                episode['query_feats'], episode['prototypes'])
            loss, _ = self.criterion(log_probs, episode['query_labels'])
            acc = (log_probs.argmax(1) == episode['query_labels']).float().mean().item()

        return loss.item(), acc

    # ----------------------------------------------------------
    # Main training loop
    # ----------------------------------------------------------
    def train(self, num_episodes: int = 10000,
              eval_every: int = 200,
              num_val_episodes: int = 30,
              save_dir: str = 'experiment/fewshot',
              resume: bool = False):

        os.makedirs(save_dir, exist_ok=True)

        # Resume from checkpoint if requested
        start_episode = 1
        best_val_acc = 0.0
        early_stop_counter = 0

        if resume:
            checkpoint_path = os.path.join(save_dir, 'fewshot_best.pth')
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location=self.device,
                                  weights_only=False)
                self.flow_classifier.load_state_dict(ckpt['flow_state_dict'])
                if 'optimizer' in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer'])
                    # FIX: Add any new parameters to optimizer that weren't in the checkpoint
                    # This handles cases where model architecture was updated after checkpoint was saved
                    current_param_ids = set(id(p) for p in self.flow_classifier.parameters())
                    optim_param_ids = set()
                    for group in self.optimizer.param_groups:
                        for p in group['params']:
                            optim_param_ids.add(id(p))
                    missing_params = [p for p in self.flow_classifier.parameters()
                                     if id(p) not in optim_param_ids]
                    if missing_params:
                        print(f"  [FIX] Adding {len(missing_params)} new parameters to optimizer")
                        self.optimizer.add_param_group({'params': missing_params})
                start_episode = ckpt.get('episode', 1) + 1
                best_val_acc = ckpt.get('val_acc', 0.0)
                self.best_val_acc = best_val_acc
                print(f"\nResumed from episode {ckpt.get('episode', 1)} "
                      f"(Val Acc: {best_val_acc:.2%})")
                print(f"Continuing from episode {start_episode} to {num_episodes}")
            else:
                print(f"\nCheckpoint not found at {checkpoint_path}, starting from scratch")
                resume = False

        # osr17b: Set classifier for CurriculumOODSampler Level 3 (adversarial)
        if hasattr(self, '_ood_sampler'):
            self._ood_sampler.set_flow_classifier(self.flow_classifier)
            print("Curriculum OOD Sampler configured with classifier")

        print(f"\n{'='*70}")
        print(f"Episodic Meta-Training: {self.N_way}-way {self.K_shot}-shot")
        print(f"Base classes:     {self.base_classes}")
        print(f"Unknown classes:  {self.unknown_classes}")
        print(f"Episodes:         {num_episodes}")
        print(f"Accum steps:      {self.gradient_accum_steps} (eff.batch={self.gradient_accum_steps}x)")
        print(f"Eval every:       {eval_every} episodes")
        print(f"Proto noise std:  {self.proto_noise_std}")
        print(f"OOD ratio:          {self.ood_ratio}")
        print(f"LR warmup:        {self.warmup_episodes} episodes")
        print(f"Stage2 (cls):     {self.stage2_len} episodes")
        print(f"Gaussian prior:   delayed to Phase 2 (ep {self.warmup_episodes+1})")
        print(f"Patience:         {self.patience_episodes} episodes ({self.patience_episodes // eval_every} evals)")
        print(f"{'='*70}\n")

        running_loss = 0.0
        running_acc = 0.0
        running_ce = 0.0

        accum_steps = self.gradient_accum_steps

        for ep in range(start_episode, num_episodes + 1):
            # Zero gradients at start of each accumulation cycle
            if (ep - 1) % accum_steps == 0:
                self.optimizer.zero_grad()

            loss, acc, ce, z_stats = self.train_episode(
                current_episode=ep, accum_scale=1.0 / accum_steps)
            running_loss += loss
            running_acc += acc
            running_ce += ce

            # Step optimizer at end of accumulation cycle
            if ep % accum_steps == 0 or ep == num_episodes:
                # Gradient clipping: exclude reciprocal_points to allow them to train
                reciprocal = getattr(self.flow_classifier, 'reciprocal_points', None)
                params_to_clip = [p for p in self.flow_classifier.parameters()
                                 if p is not reciprocal]
                if params_to_clip:
                    torch.nn.utils.clip_grad_norm_(params_to_clip, 0.5)

                self.optimizer.step()
                self.optimizer.step()
                if ep <= self.warmup_episodes:
                    warmup_factor = ep / self.warmup_episodes
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.base_lr * warmup_factor
                else:
                    self.scheduler.step()

            self.history['train_loss'].append(loss)
            self.history['train_acc'].append(acc)

            # Periodic evaluation
            if ep % eval_every == 0:
                val_losses, val_accs = [], []
                for _ in range(num_val_episodes):
                    vl, va = self.validate_episode()
                    val_losses.append(vl)
                    val_accs.append(va)

                val_loss = np.mean(val_losses)
                val_acc = np.mean(val_accs)

                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                avg_train_loss = running_loss / eval_every
                avg_train_acc = running_acc / eval_every
                avg_ce = running_ce / eval_every
                running_loss = 0.0
                running_acc = 0.0
                running_ce = 0.0

                lr = self.optimizer.param_groups[0]['lr']
                w, s2 = self.warmup_episodes, self.stage2_len
                if ep <= w:
                    cur_cls = 0.0
                elif ep <= w + s2:
                    cur_cls = min(1.0, (ep - w) / s2 * 0.8)
                else:
                    cur_cls = 1.0
                print(f"Ep {ep:>5d}/{num_episodes} | "
                      f"CE: {avg_ce:.4f} Total: {avg_train_loss:.4f} Acc: {avg_train_acc:.2%} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
                      f"LR: {lr:.2e} cls:{cur_cls:.1f}")

                # Save best
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.best_val_acc = val_acc
                    early_stop_counter = 0
                    save_path = os.path.join(save_dir, 'fewshot_best.pth')
                    torch.save({
                        'flow_state_dict': self.flow_classifier.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'val_acc': float(val_acc),
                        'episode': ep,
                        'config': {
                            'input_dim': self.flow_classifier.input_dim,
                            'condition_dim': self.flow_classifier.condition_dim,
                        }
                    }, save_path)
                    print(f"  -> Best model saved (Val Acc: {val_acc:.2%})")
                else:
                    self.early_stop_counter += eval_every
                    if self.early_stop_counter >= self.patience_episodes:
                        print(f"\n!!! Early stopping at episode {ep} !!!")
                        break

        # Reload best
        best_path = os.path.join(save_dir, 'fewshot_best.pth')
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=self.device,
                              weights_only=False)
            self.flow_classifier.load_state_dict(ckpt['flow_state_dict'])
            print(f"\nLoaded best model (Val Acc: {ckpt['val_acc']:.2%}, "
                  f"Ep {ckpt['episode']})")

        print(f"\n{'='*70}")
        print("Episodic training complete!")
        print(f"Best validation accuracy: {self.best_val_acc:.2%}")
        print(f"{'='*70}")

        # Save episode-level history for plotting
        history_path = os.path.join(save_dir, 'episodic_history.json')
        os.makedirs(save_dir, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Episode history saved: {history_path}")


# ============================================================
# 7. OSR Calibrator
# ============================================================

class OSRCalibrator:
    """
    Calibrate OOD threshold for open set recognition.
    Feature-space methods only (no flow).
    """

    def __init__(self, flow_classifier: EpisodicFlowClassifier,
                 feature_cache: FeatureCache,
                 base_classes: List[int],
                 unknown_classes: List[int],
                 device: str = 'cuda'):
        self.flow = flow_classifier
        self.cache = feature_cache
        self.base_classes = base_classes
        self.unknown_classes = unknown_classes
        self.device = device
        self.threshold = None
        self.per_class_thresholds = None  # osr20a: per-class adaptive thresholds
        self.class_variances = None       # osr20a: per-class variance for distance normalization

    def compute_prototypes(self, K_shot: int = 50,
                           seed: int = 42) -> torch.Tensor:
        """Compute base class prototypes using K samples per class"""
        rng = random.Random(seed)
        prototypes = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            n = min(K_shot, len(feats))
            indices = rng.sample(range(len(feats)), n)
            prototypes.append(feats[indices].mean(dim=0))
        return torch.stack(prototypes).to(self.device)

    def compute_prototypes_rectified(self, K_shot: int = 50,
                                      seed: int = 42,
                                      prior_strength: float = 1.0) -> torch.Tensor:
        """osr20a: Bayesian prototype rectification.
        proto_c = (n * mean + m * prior) / (n + m)
        prior = global mean of all prototypes (shrinkage toward center)
        Reduces noise in few-shot prototype estimation."""
        rng = random.Random(seed)
        raw_prototypes = []
        sample_counts = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            n = min(K_shot, len(feats))
            indices = rng.sample(range(len(feats)), n)
            raw_prototypes.append(feats[indices].mean(dim=0))
            sample_counts.append(n)
        raw_prototypes = torch.stack(raw_prototypes)
        # Global prior = mean of all raw prototypes
        prior = raw_prototypes.mean(dim=0, keepdim=True)  # (1, D)
        # Rectify each prototype
        rectified = []
        for i in range(len(raw_prototypes)):
            m = prior_strength
            n = sample_counts[i]
            rectified.append((n * raw_prototypes[i] + m * prior.squeeze(0)) / (n + m))
        return torch.stack(rectified).to(self.device)

    def compute_prototypes_weighted(self, K_shot: int = 50,
                                     seed: int = 42) -> torch.Tensor:
        """osr20a: Confidence-weighted prototype estimation.
        Uses distance to class center as confidence weight;
        samples closer to center contribute more."""
        rng = random.Random(seed)
        prototypes = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            n = min(K_shot, len(feats))
            indices = rng.sample(range(len(feats)), n)
            samples = feats[indices]  # (n, D)
            # Initial mean
            proto = samples.mean(dim=0)
            # Confidence = inverse distance to initial mean
            dists = ((samples - proto.unsqueeze(0)) ** 2).sum(dim=1).sqrt()  # (n,)
            weights = F.softmax(-dists * 5.0, dim=0)  # temperature-scaled
            proto = (samples * weights.unsqueeze(1)).sum(dim=0)
            prototypes.append(proto)
        return torch.stack(prototypes).to(self.device)

    # ---- osr22a: Multi-prototype clustering ----

    def compute_multi_prototypes(self, K_shot: int = 50,
                                 seed: int = 42,
                                 K_sub: int = 2,
                                 adaptive_K: bool = False) -> torch.Tensor:
        """osr22a: Multi-prototype computation via K-means sub-clustering.
        Each class is split into K_sub sub-clusters, yielding K_sub*C prototypes total.

        Args:
            K_shot: samples per class for sub-clustering (use all available if > available)
            seed: random seed
            K_sub: number of sub-clusters per class (2-4 recommended)
            adaptive_K: if True, use silhouette score to select K per class

        Returns:
            multi_prototypes: (C * K_sub, D) tensor of sub-prototype centers
            cluster_labels: dict mapping class_id -> list of cluster indices
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score

        rng = random.Random(seed)
        all_sub_protos = []
        cluster_labels = {}
        sub_proto_idx = 0

        for c_idx, c_id in enumerate(self.base_classes):
            feats = self.cache.get_class_features(c_id)
            n = min(K_shot, len(feats))
            indices = rng.sample(range(len(feats)), n)
            samples = feats[indices].numpy()  # (n, D)

            # Adaptive K selection using silhouette analysis
            if adaptive_K and n >= 20:
                best_k, best_sil = 1, -1.0
                for k_trial in range(1, min(5, n // 5)):
                    if n < k_trial * 2:
                        continue
                    kmeans = MiniBatchKMeans(n_clusters=k_trial, random_state=seed, batch_size=32)
                    labels = kmeans.fit_predict(samples)
                    if len(set(labels)) > 1:
                        sil = silhouette_score(samples, labels)
                        if sil > best_sil:
                            best_k, best_sil = k_trial, sil
                K_class = best_k
            else:
                K_class = K_sub

            # Fit K-means with determined K
            if K_class == 1 or n < K_class * 2:
                # Fallback to single prototype if not enough samples
                sub_proto = samples.mean(axis=0)
                all_sub_protos.append(sub_proto)
                cluster_labels[c_id] = [sub_proto_idx]
                sub_proto_idx += 1
            else:
                kmeans = MiniBatchKMeans(n_clusters=K_class, random_state=seed, batch_size=32)
                kmeans.fit(samples)
                sub_protos = kmeans.cluster_centers_  # (K_class, D)
                for sub_proto in sub_protos:
                    all_sub_protos.append(sub_proto)
                    cluster_labels[c_id] = list(range(sub_proto_idx, sub_proto_idx + K_class))
                sub_proto_idx += K_class

        multi_protos = torch.from_numpy(np.stack(all_sub_protos)).float().to(self.device)
        self._multi_proto_cluster_labels = cluster_labels
        self._multi_proto_K_sub = K_sub

        return multi_protos

    def score_anti_prototype_multi(self, features: torch.Tensor,
                                    multi_prototypes: torch.Tensor,
                                    batch_size: int = 4096) -> torch.Tensor:
        """osr22a: Multi-prototype anti-prototype OOD scoring.
        Uses multiple sub-prototypes per class for more precise boundary estimation.

        Scoring: score = -dist_to_nearest_subproto + dist_to_nearest_anti_subproto
        where anti_subprotos are reflections of subprototypes through batch center.
        """
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)

                # Query-dependent center
                feat_center = batch.mean(dim=0, keepdim=True)  # (1, D)

                # Anti-sub-prototypes: reflection through center
                anti_protos = 2 * feat_center - multi_prototypes  # (M, D)

                # Distances to nearest sub-prototype and anti-sub-prototype
                dist_proto = torch.cdist(batch, multi_prototypes, p=2).min(dim=1).values
                dist_anti = torch.cdist(batch, anti_protos, p=2).min(dim=1).values

                # Score: close to proto (good), far from anti (good)
                scores = -dist_proto + dist_anti
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    # ---- Feature-space Mahalanobis scoring ----

    def compute_feat_stats(self):
        """Compute per-class means and per-class covariance in 64-dim feature space."""
        self.feat_class_means: Dict[int, torch.Tensor] = {}
        self.feat_cov_inv: Dict[int, torch.Tensor] = {}
        for c_idx, c_id in enumerate(self.base_classes):
            feats = self.cache.get_class_features(c_id)
            self.feat_class_means[c_idx] = feats.mean(dim=0).to(self.device)
            D = feats.size(1)
            cov = torch.cov(feats.T).to(self.device)
            cov += 1e-3 * torch.eye(D, device=self.device)
            self.feat_cov_inv[c_idx] = torch.linalg.inv(cov)

    def compute_class_variances(self, prototypes: torch.Tensor):
        """osr20a: Compute per-class intra-sample variance for distance normalization."""
        self.class_variances = []
        for c_idx, c_id in enumerate(self.base_classes):
            feats = self.cache.get_class_features(c_id)
            proto = prototypes[c_idx].unsqueeze(0)
            dists_sq = ((feats.to(self.device) - proto) ** 2).sum(dim=1)
            self.class_variances.append(dists_sq.mean().item() + 1e-6)

    def score_samples_feat_mahalanobis(self, features: torch.Tensor,
                                        prototypes: torch.Tensor = None,
                                        batch_size: int = 4096) -> torch.Tensor:
        """Feature-space Mahalanobis OOD score with per-class covariance.
        Higher = more known."""
        N = len(self.feat_class_means)
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                min_dist = torch.full((B,), float('inf'), device=self.device)
                for c_idx in range(N):
                    diff = batch - self.feat_class_means[c_idx].unsqueeze(0)
                    mahal = torch.sum(
                        (diff @ self.feat_cov_inv[c_idx]) * diff, dim=1).sqrt()
                    min_dist = torch.minimum(min_dist, mahal)
                all_scores.append(-min_dist.cpu())
        return torch.cat(all_scores, dim=0)

    def score_samples_feat_mahalanobis_relative(self, features: torch.Tensor,
                                                  prototypes: torch.Tensor = None,
                                                  batch_size: int = 4096) -> torch.Tensor:
        """Feature-space Mahalanobis + relative distance hybrid OOD score."""
        N = len(self.feat_class_means)
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                dists = torch.zeros(B, N, device=self.device)
                for c_idx in range(N):
                    diff = batch - self.feat_class_means[c_idx].unsqueeze(0)
                    dists[:, c_idx] = torch.sum(
                        (diff @ self.feat_cov_inv[c_idx]) * diff, dim=1).sqrt()
                sorted_dists, _ = dists.sort(dim=1)
                min_d = sorted_dists[:, 0]
                second_d = sorted_dists[:, 1]
                dist_ratio = min_d / (second_d + 1e-6)
                alpha = getattr(self, '_feat_mahal_rel_alpha', 0.5)
                score = alpha * (-min_d) + (1 - alpha) * (-dist_ratio * 10.0)
                all_scores.append(score.cpu())
        return torch.cat(all_scores, dim=0)

    # ---- Anti-prototype scoring ----

    def score_anti_prototype_all(self, features: torch.Tensor,
                                  prototypes: torch.Tensor,
                                  batch_size: int = 4096) -> torch.Tensor:
        """Anti-prototype OOD score. Higher = more known."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                scores = self.flow.score_anti_prototype(batch, prototypes)
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    def score_anti_prototype_enhanced(self, features: torch.Tensor,
                                       prototypes: torch.Tensor,
                                       batch_size: int = 4096) -> torch.Tensor:
        """Enhanced anti-prototype OOD score. Higher = more known."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                scores = self.flow.score_anti_prototype_enhanced(batch, prototypes)
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    # ---- Reciprocal point scoring ----

    def score_samples_reciprocal(self, features: torch.Tensor,
                                  prototypes: torch.Tensor,
                                  batch_size: int = 4096) -> torch.Tensor:
        """Reciprocal point OOD score. Higher = more known."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                scores = self.flow.score_reciprocal(batch, prototypes)
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    # ---- osr20a: Variance-normalized anti-prototype ----

    def score_anti_proto_var_norm(self, features: torch.Tensor,
                                   prototypes: torch.Tensor,
                                   batch_size: int = 4096) -> torch.Tensor:
        """osr20a: Anti-prototype with variance-normalized distance.
        score = (-dist_proto + dist_anti) / var(nearest_class)
        Accounts for different class spreads in feature space."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                feat_center = batch.mean(dim=0, keepdim=True)
                anti_protos = 2 * feat_center - prototypes
                dists_proto = torch.cdist(batch, prototypes, p=2)  # (B, N)
                dists_anti = torch.cdist(batch, anti_protos, p=2)
                # Per-sample: use nearest class variance
                nearest_class = dists_proto.argmin(dim=1)  # (B,)
                min_dist_proto = dists_proto.min(dim=1).values
                min_dist_anti = dists_anti.min(dim=1).values
                if self.class_variances is not None:
                    var_factors = torch.tensor(
                        [self.class_variances[c.item()] for c in nearest_class],
                        device=self.device)
                    scores = (-min_dist_proto + min_dist_anti) / var_factors.sqrt()
                else:
                    scores = -min_dist_proto + min_dist_anti
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    # ---- osr20a: Cosine-distance anti-prototype ----

    def score_anti_proto_cosine(self, features: torch.Tensor,
                                 prototypes: torch.Tensor,
                                 batch_size: int = 4096) -> torch.Tensor:
        """osr20a: Anti-prototype with cosine distance.
        Captures directional relationships in feature space."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                feat_center = batch.mean(dim=0, keepdim=True)
                anti_protos = 2 * feat_center - prototypes
                # Cosine distance = 1 - cosine_similarity
                cos_proto = F.cosine_similarity(batch.unsqueeze(1), prototypes.unsqueeze(0), dim=2)
                cos_anti = F.cosine_similarity(batch.unsqueeze(1), anti_protos.unsqueeze(0), dim=2)
                # Higher = more known: close to proto (high cos_sim), far from anti (low cos_sim)
                scores = cos_proto.max(dim=1).values - cos_anti.min(dim=1).values
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    # ---- osr20a: Anti-prototype with rectified prototypes ----

    def score_anti_proto_rectified(self, features: torch.Tensor,
                                    prototypes: torch.Tensor,
                                    batch_size: int = 4096) -> torch.Tensor:
        """osr20a: Anti-prototype using Bayesian-rectified prototypes.
        Same scoring as anti_prototype but with noise-reduced prototypes."""
        # prototypes should already be rectified via compute_prototypes_rectified()
        return self.score_anti_prototype_all(features, prototypes, batch_size)

    # ---- Geometric fusion ----

    def score_samples_geo_fusion(self, features: torch.Tensor,
                                  prototypes: torch.Tensor,
                                  batch_size: int = 4096) -> torch.Tensor:
        """Geometric fusion of anti_prototype + feat_mahalanobis."""
        anti_scores = self.score_anti_prototype_all(features, prototypes, batch_size)
        mahal_scores = self.score_samples_feat_mahalanobis(features, prototypes, batch_size)

        anti_mu = getattr(self, '_geo_anti_mu', anti_scores.mean().item())
        anti_std = getattr(self, '_geo_anti_std', anti_scores.std().item())
        mahal_mu = getattr(self, '_geo_mahal_mu', mahal_scores.mean().item())
        mahal_std = getattr(self, '_geo_mahal_std', mahal_scores.std().item())

        anti_normed = (anti_scores - anti_mu) / (anti_std + 1e-8)
        mahal_normed = (mahal_scores - mahal_mu) / (mahal_std + 1e-8)

        alpha = getattr(self, '_geo_fusion_alpha', 0.5)
        return alpha * anti_normed + (1 - alpha) * mahal_normed

    # ---- OOD head scoring ----

    def _extract_ood_features(self, features: torch.Tensor,
                               prototypes: torch.Tensor,
                               batch_size: int = 4096) -> torch.Tensor:
        """Extract 5-dim OOD feature vector."""
        self.flow.eval()
        all_feats = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                dists = torch.cdist(batch, prototypes, p=2)
                sorted_d, _ = dists.sort(dim=1)
                min_d = sorted_d[:, 0:1]
                second_d = sorted_d[:, 1:2]
                dist_ratio = min_d / (second_d + 1e-6)
                log_probs = self.flow.classify(batch, prototypes)
                probs = F.softmax(log_probs, dim=1)
                softmax_max = probs.max(dim=1, keepdim=True)[0]
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
                feat_norm = (batch ** 2).sum(dim=1, keepdim=True).sqrt()
                feats = torch.cat([min_d, dist_ratio, softmax_max, entropy, feat_norm], dim=1)
                all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    def _extract_ood_features_extended(self, features: torch.Tensor,
                                        prototypes: torch.Tensor,
                                        batch_size: int = 4096) -> torch.Tensor:
        """osr20a: Extended OOD feature vector (11-dim).
        Adds: dist_to_2nd_nearest, proto_score_variance, dist_to_center,
              score_gap, neighbor_density, norm_dist_ratio."""
        self.flow.eval()
        all_feats = []
        feat_center = prototypes.mean(dim=0, keepdim=True)  # global prototype center
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                dists = torch.cdist(batch, prototypes, p=2)  # (B, N)
                sorted_d, _ = dists.sort(dim=1)
                min_d = sorted_d[:, 0:1]                # 1: nearest proto distance
                second_d = sorted_d[:, 1:2]              # 2: 2nd nearest distance
                dist_ratio = min_d / (second_d + 1e-6)   # 3: distance ratio
                score_gap = (second_d - min_d)            # 4: gap between 1st and 2nd
                # Classification scores
                log_probs = self.flow.classify(batch, prototypes)
                probs = F.softmax(log_probs, dim=1)
                softmax_max = probs.max(dim=1, keepdim=True)[0]  # 5: max softmax
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)  # 6: entropy
                feat_norm = (batch ** 2).sum(dim=1, keepdim=True).sqrt()  # 7: feature norm
                # Distance to global center
                dist_to_center = torch.cdist(batch, feat_center, p=2)  # 8: dist to center
                # Score variance across classes
                proto_score_var = probs.var(dim=1, keepdim=True)  # 9: prediction variance
                # Normalized distance ratio (min_d / dist_to_center)
                norm_dist_ratio = min_d / (dist_to_center + 1e-6)  # 10: norm'd ratio
                # Neighbor density: how many protos within 2x min distance
                neighbor_density = (dists < (min_d * 2 + 1e-6)).float().sum(dim=1, keepdim=True)  # 11
                feats = torch.cat([
                    min_d, dist_ratio, softmax_max, entropy, feat_norm,  # original 5
                    second_d, score_gap, dist_to_center,                # new distances
                    proto_score_var, norm_dist_ratio, neighbor_density   # new stats
                ], dim=1)
                all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    def score_samples_ood_head(self, features: torch.Tensor,
                                prototypes: torch.Tensor,
                                batch_size: int = 4096) -> torch.Tensor:
        """Score using trained neural OOD head on 5-dim feature-space statistics."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                ood_feats = self._extract_ood_features(batch, prototypes, batch_size)
                ood_feats = ood_feats.to(self.device)
                logits = self.flow.ood_head(ood_feats)
                scores = torch.sigmoid(logits).squeeze(1)
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    # ---- osr20a: Extended OOD head (logistic regression on 11-dim features) ----

    def train_ood_head_extended(self, prototypes: torch.Tensor,
                                 target_fpr: float = 0.05,
                                 batch_size: int = 4096,
                                 verbose: bool = True):
        """osr20a: Train logistic regression OOD classifier on extended 11-dim features."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if verbose:
            print("Training extended OOD head (11-dim features)...")
        known_feats_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_feats_list.append(self._extract_ood_features_extended(feats, prototypes, batch_size))
        known_feats = torch.cat(known_feats_list).numpy()

        unknown_feats_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_feats_list.append(self._extract_ood_features_extended(feats, prototypes, batch_size))
        if not unknown_feats_list:
            if verbose:
                print("  No unknown samples, skipping.")
            return
        unknown_feats = torch.cat(unknown_feats_list).numpy()

        X = np.concatenate([known_feats, unknown_feats], axis=0)
        y = np.concatenate([np.ones(len(known_feats)), np.zeros(len(unknown_feats))], axis=0)

        # Standardize features
        self._ood_ext_scaler = StandardScaler()
        X_scaled = self._ood_ext_scaler.fit_transform(X)

        self._ood_ext_clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs')
        self._ood_ext_clf.fit(X_scaled, y)

        if verbose:
            feat_names = ['min_dist', 'dist_ratio', 'softmax_max', 'entropy', 'feat_norm',
                          '2nd_dist', 'score_gap', 'dist_to_center', 'proto_score_var',
                          'norm_dist_ratio', 'neighbor_density']
            w = self._ood_ext_clf.coef_[0]
            print("  Extended OOD head weights:")
            for name, wi in zip(feat_names, w):
                print(f"    {name:>18s}: {wi:+.4f}")

        known_probs = self._ood_ext_clf.predict_proba(
            self._ood_ext_scaler.transform(known_feats))[:, 1]
        sorted_probs = np.sort(known_probs)
        idx = min(int(len(sorted_probs) * target_fpr), len(sorted_probs) - 1)
        self._ood_ext_threshold = sorted_probs[idx]

        if verbose:
            unknown_probs = self._ood_ext_clf.predict_proba(
                self._ood_ext_scaler.transform(unknown_feats))[:, 1]
            tnr = (known_probs >= self._ood_ext_threshold).mean()
            tpr = (unknown_probs < self._ood_ext_threshold).mean()
            print(f"  Calib TNR: {tnr:.2%}  TPR: {tpr:.2%}  "
                  f"(threshold={self._ood_ext_threshold:.4f})")

    def score_ood_head_extended(self, features: torch.Tensor,
                                 prototypes: torch.Tensor,
                                 batch_size: int = 4096) -> torch.Tensor:
        """osr20a: Score using extended OOD head."""
        if not hasattr(self, '_ood_ext_clf'):
            raise RuntimeError("Call train_ood_head_extended() first")
        feats = self._extract_ood_features_extended(features, prototypes, batch_size).numpy()
        feats_scaled = self._ood_ext_scaler.transform(feats)
        probs = self._ood_ext_clf.predict_proba(feats_scaled)[:, 1]
        return torch.from_numpy(probs).float()

    # ---- osr22b: GMM cluster-enhanced OOD features ----

    def _fit_global_gmm(self, n_components: int = None, random_state: int = 42):
        """osr22b: Fit global Gaussian Mixture Model on all base class features.
        Models the overall feature space structure as a mixture of Gaussians."""
        from sklearn.mixture import GaussianMixture

        # Collect all base class features
        all_feats_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            all_feats_list.append(feats.numpy())
        all_feats = np.concatenate(all_feats_list, axis=0)

        # Default: 2x number of classes (captures sub-structure)
        if n_components is None:
            n_components = len(self.base_classes) * 2

        print(f"  Fitting GMM with {n_components} components on {len(all_feats)} samples...")
        self._gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='full',
            max_iter=200,
            random_state=random_state,
            reg_covar=1e-6
        )
        self._gmm.fit(all_feats)
        print(f"  GMM converged: {self._gmm.converged_}")

    def _extract_cluster_features(self, features: torch.Tensor,
                                   batch_size: int = 4096) -> torch.Tensor:
        """osr22b: Extract GMM-based cluster features (3-dim).
        Returns: [cluster_mahal, cluster_posterior_max, cluster_entropy]"""
        if not hasattr(self, '_gmm'):
            raise RuntimeError("Call _fit_global_gmm() first")

        self.flow.eval()
        all_cluster_feats = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].numpy()

                # Get weighted log probabilities for all clusters at once (B, K)
                log_probs = self._gmm._estimate_weighted_log_prob(batch)  # (B, K)

                # Stable posterior computation using log-sum-exp trick
                log_prob_max = log_probs.max(axis=1, keepdims=True)  # (B, 1)
                log_prob_shifted = log_probs - log_prob_max  # shift to avoid overflow
                posteriors = np.exp(log_prob_shifted)
                posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)  # normalize

                # Mahalanobis distance to nearest cluster
                mahal_dists = []
                for k in range(self._gmm.n_components):
                    mean = self._gmm.means_[k]  # (D,)
                    cov = self._gmm.covariances_[k]  # (D, D)
                    prec = self._gmm.precisions_[k]  # (D, D)

                    diff = batch - mean  # (B, D)
                    # Mahalanobis: sqrt(diff^T @ prec @ diff)
                    mahal = np.sqrt(np.sum(diff @ prec * diff, axis=1))
                    mahal_dists.append(mahal)

                mahal_dists = np.stack(mahal_dists, axis=1)  # (B, K)

                # 1. cluster_mahal: min Mahalanobis distance to any cluster
                cluster_mahal = mahal_dists.min(axis=1, keepdims=True)  # (B, 1)

                # 2. cluster_posterior_max: maximum posterior probability
                cluster_posterior_max = posteriors.max(axis=1, keepdims=True)  # (B, 1)

                # 3. cluster_entropy: entropy of posterior distribution
                # Entropy = -sum(p * log(p))
                posteriors_clipped = np.clip(posteriors, 1e-10, 1.0)  # avoid log(0)
                cluster_entropy = -(posteriors_clipped * np.log(posteriors_clipped)).sum(
                    axis=1, keepdims=True)  # (B, 1)

                cluster_feats = np.concatenate([
                    cluster_mahal,
                    cluster_posterior_max,
                    cluster_entropy
                ], axis=1)  # (B, 3)

                all_cluster_feats.append(cluster_feats)

        return torch.from_numpy(np.concatenate(all_cluster_feats, axis=0))

    def _extract_ood_features_extended_v2(self, features: torch.Tensor,
                                           prototypes: torch.Tensor,
                                           batch_size: int = 4096) -> torch.Tensor:
        """osr22b: Extended OOD features v2 (14-dim) - replaces neighbor_density with GMM cluster features.
        Features: [min_d, dist_ratio, softmax_max, entropy, feat_norm,
                   second_d, score_gap, dist_to_center, proto_score_var,
                   norm_dist_ratio,
                   cluster_mahal, cluster_posterior_max, cluster_entropy]"""
        # Original 10 features (excluding neighbor_density)
        self.flow.eval()
        all_feats = []
        feat_center = prototypes.mean(dim=0, keepdim=True)
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                dists = torch.cdist(batch, prototypes, p=2)
                sorted_d, _ = dists.sort(dim=1)
                min_d = sorted_d[:, 0:1]
                second_d = sorted_d[:, 1:2]
                dist_ratio = min_d / (second_d + 1e-6)
                score_gap = (second_d - min_d)

                log_probs = self.flow.classify(batch, prototypes)
                probs = F.softmax(log_probs, dim=1)
                softmax_max = probs.max(dim=1, keepdim=True)[0]
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
                feat_norm = (batch ** 2).sum(dim=1, keepdim=True).sqrt()

                dist_to_center = torch.cdist(batch, feat_center, p=2)
                proto_score_var = probs.var(dim=1, keepdim=True)
                norm_dist_ratio = min_d / (dist_to_center + 1e-6)

                # GMM cluster features (3-dim)
                cluster_feats = self._extract_cluster_features(
                    features[i:i+batch_size], batch_size).to(self.device)

                # Combine: 10 original + 3 cluster = 13 features
                feats = torch.cat([
                    min_d, dist_ratio, softmax_max, entropy, feat_norm,
                    second_d, score_gap, dist_to_center,
                    proto_score_var, norm_dist_ratio,
                    cluster_feats
                ], dim=1)
                all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    def train_ood_head_extended_v2(self, prototypes: torch.Tensor,
                                    target_fpr: float = 0.05,
                                    batch_size: int = 4096,
                                    n_gmm_components: int = None,
                                    verbose: bool = True):
        """osr22b: Train extended OOD head v2 with GMM cluster features (13-dim)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if verbose:
            print("Training extended OOD head v2 (13-dim with GMM clusters)...")

        # Fit GMM on base class features
        self._fit_global_gmm(n_components=n_gmm_components)

        known_feats_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_feats_list.append(self._extract_ood_features_extended_v2(
                feats, prototypes, batch_size))
        known_feats = torch.cat(known_feats_list).numpy()

        unknown_feats_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_feats_list.append(self._extract_ood_features_extended_v2(
                    feats, prototypes, batch_size))
        if not unknown_feats_list:
            if verbose:
                print("  No unknown samples, skipping.")
            return
        unknown_feats = torch.cat(unknown_feats_list).numpy()

        X = np.concatenate([known_feats, unknown_feats], axis=0)
        y = np.concatenate([np.ones(len(known_feats)), np.zeros(len(unknown_feats))], axis=0)

        self._ood_ext_v2_scaler = StandardScaler()
        X_scaled = self._ood_ext_v2_scaler.fit_transform(X)

        self._ood_ext_v2_clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs')
        self._ood_ext_v2_clf.fit(X_scaled, y)

        if verbose:
            feat_names = ['min_dist', 'dist_ratio', 'softmax_max', 'entropy', 'feat_norm',
                          '2nd_dist', 'score_gap', 'dist_to_center', 'proto_score_var',
                          'norm_dist_ratio', 'cluster_mahal', 'cluster_post_max', 'cluster_ent']
            w = self._ood_ext_v2_clf.coef_[0]
            print("  Extended OOD head v2 weights:")
            for name, wi in zip(feat_names, w):
                print(f"    {name:>20s}: {wi:+.4f}")

        known_probs = self._ood_ext_v2_clf.predict_proba(
            self._ood_ext_v2_scaler.transform(known_feats))[:, 1]
        sorted_probs = np.sort(known_probs)
        idx = min(int(len(sorted_probs) * target_fpr), len(sorted_probs) - 1)
        self._ood_ext_v2_threshold = sorted_probs[idx]

        if verbose:
            unknown_probs = self._ood_ext_v2_clf.predict_proba(
                self._ood_ext_v2_scaler.transform(unknown_feats))[:, 1]
            tnr = (known_probs >= self._ood_ext_v2_threshold).mean()
            tpr = (unknown_probs < self._ood_ext_v2_threshold).mean()
            print(f"  Calib TNR: {tnr:.2%}  TPR: {tpr:.2%}  "
                  f"(threshold={self._ood_ext_v2_threshold:.4f})")

    def score_ood_head_extended_v2(self, features: torch.Tensor,
                                    prototypes: torch.Tensor,
                                    batch_size: int = 4096) -> torch.Tensor:
        """osr22b: Score using extended OOD head v2 with GMM cluster features."""
        if not hasattr(self, '_ood_ext_v2_clf'):
            raise RuntimeError("Call train_ood_head_extended_v2() first")
        feats = self._extract_ood_features_extended_v2(features, prototypes, batch_size).numpy()
        feats_scaled = self._ood_ext_v2_scaler.transform(feats)
        probs = self._ood_ext_v2_clf.predict_proba(feats_scaled)[:, 1]
        return torch.from_numpy(probs).float()

    # ---- Unified scoring interface ----

    def score_samples(self, features: torch.Tensor,
                      prototypes: torch.Tensor,
                      method: str = 'feature_mahalanobis',
                      batch_size: int = 4096) -> torch.Tensor:
        """Score samples using specified method."""
        if method == 'feature_mahalanobis':
            return self.score_samples_feat_mahalanobis(features, prototypes, batch_size)
        if method == 'feat_mahalanobis_relative':
            return self.score_samples_feat_mahalanobis_relative(features, prototypes, batch_size)
        if method == 'anti_prototype':
            return self.score_anti_prototype_all(features, prototypes, batch_size)
        if method == 'anti_prototype_enhanced':
            return self.score_anti_prototype_enhanced(features, prototypes, batch_size)
        if method == 'reciprocal':
            return self.score_samples_reciprocal(features, prototypes, batch_size)
        if method == 'geo_fusion':
            return self.score_samples_geo_fusion(features, prototypes, batch_size)
        if method == 'ood_head':
            return self.score_samples_ood_head(features, prototypes, batch_size)
        if method == 'learned_ensemble':
            return self.score_samples_learned(features, prototypes, batch_size)
        # osr20a new methods
        if method == 'anti_proto_var_norm':
            return self.score_anti_proto_var_norm(features, prototypes, batch_size)
        if method == 'anti_proto_cosine':
            return self.score_anti_proto_cosine(features, prototypes, batch_size)
        if method == 'anti_proto_rectified':
            return self.score_anti_proto_rectified(features, prototypes, batch_size)
        if method == 'ood_head_extended':
            return self.score_ood_head_extended(features, prototypes, batch_size)
        # osr22b: extended OOD head v2 with GMM clusters
        if method == 'ood_head_extended_v2':
            return self.score_ood_head_extended_v2(features, prototypes, batch_size)
        # osr22a: multi-prototype anti_prototype
        if method == 'anti_prototype_multi':
            # Use cached multi_prototypes if available, else compute
            if not hasattr(self, '_cached_multi_protos'):
                K_sub = getattr(self, '_multi_proto_K_sub', 2)
                multi_protos = self.compute_multi_prototypes(K_shot=50, K_sub=K_sub)
                self._cached_multi_protos = multi_protos
            return self.score_anti_prototype_multi(features, self._cached_multi_protos, batch_size)
        # osr20b: ensemble
        if method == 'ensemble_anti_oodext':
            return self.score_ensemble_anti_oodext(features, prototypes, batch_size)
        # osr21b: anti_prototype in Flow-transformed space
        if method == 'anti_prototype_flow':
            return self.score_anti_prototype_flow(features, prototypes, batch_size)
        # osr21c: Flow Jacobian scoring
        if method == 'flow_jacobian':
            return self.score_flow_jacobian(features, prototypes, batch_size)
        # osr21c: triple ensemble
        if method == 'ensemble_triple':
            return self.score_ensemble_triple(features, prototypes, batch_size)
        # osr22c: adaptive ensemble fusion
        if method == 'ensemble_adaptive':
            return self.score_ensemble_adaptive(features, prototypes, batch_size)
        # osr22d: cluster boundary detection
        if method == 'cluster_boundary':
            return self.score_cluster_boundary(features, prototypes, batch_size)
        # osr22d: three-way ensemble with cluster boundary
        if method == 'ensemble_cluster':
            return self.score_ensemble_cluster(features, prototypes, batch_size)
        return self.score_samples_feat_mahalanobis(features, prototypes, batch_size)

    # ---- osr20b: Ensemble anti_prototype + ood_head_extended ----

    def train_ensemble_anti_oodext(self, prototypes: torch.Tensor,
                                    target_fpr: float = 0.05,
                                    batch_size: int = 4096):
        """osr20b: Train ensemble fusion of anti_prototype + ood_head_extended.
        Normalizes both scores and searches optimal fusion weight on calibration data."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        print("Training ensemble (anti_proto + ood_head_extended) fusion...")

        # Get anti_prototype scores
        known_anti_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_anti_list.append(self.score_anti_prototype_all(feats, prototypes, batch_size))
        known_anti = torch.cat(known_anti_list)

        unknown_anti_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_anti_list.append(self.score_anti_prototype_all(feats, prototypes, batch_size))
        unknown_anti = torch.cat(unknown_anti_list) if unknown_anti_list else torch.tensor([])

        # Get ood_head_extended scores (need to train first)
        self.train_ood_head_extended(prototypes, target_fpr, batch_size, verbose=False)
        known_ood_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_ood_list.append(self.score_ood_head_extended(feats, prototypes, batch_size))
        known_ood = torch.cat(known_ood_list)

        unknown_ood_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_ood_list.append(self.score_ood_head_extended(feats, prototypes, batch_size))
        unknown_ood = torch.cat(unknown_ood_list) if unknown_ood_list else torch.tensor([])

        # Store normalization stats
        self._ens_anti_mu = known_anti.mean().item()
        self._ens_anti_std = known_anti.std().item() + 1e-8
        self._ens_ood_mu = known_ood.mean().item()
        self._ens_ood_std = known_ood.std().item() + 1e-8

        # Search optimal alpha via Youden J
        norm_known_anti = (known_anti - self._ens_anti_mu) / self._ens_anti_std
        norm_known_ood = (known_ood - self._ens_ood_mu) / self._ens_ood_std

        has_unknown = len(unknown_anti) > 0 and len(unknown_ood) > 0
        if has_unknown:
            norm_unknown_anti = (unknown_anti - self._ens_anti_mu) / self._ens_anti_std
            norm_unknown_ood = (unknown_ood - self._ens_ood_mu) / self._ens_ood_std

        best_alpha, best_j = 0.5, -1.0
        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            combined_known = alpha * norm_known_anti + (1 - alpha) * norm_known_ood
            sorted_k = combined_known.sort()[0]
            idx = min(int(len(sorted_k) * target_fpr), len(sorted_k) - 1)
            tau = sorted_k[idx].item()
            tnr = (combined_known >= tau).float().mean().item()
            if has_unknown:
                combined_unknown = alpha * norm_unknown_anti + (1 - alpha) * norm_unknown_ood
                tpr = (combined_unknown < tau).float().mean().item()
            else:
                tpr = 0.0
            j = tnr + tpr - 1.0
            if j > best_j:
                best_j, best_alpha = j, alpha

        self._ens_fusion_alpha = best_alpha
        print(f"  Ensemble fusion alpha: {best_alpha:.2f} (Youden J={best_j:.3f})")

    def score_ensemble_anti_oodext(self, features: torch.Tensor,
                                    prototypes: torch.Tensor,
                                    batch_size: int = 4096) -> torch.Tensor:
        """osr20b: Ensemble score = alpha * norm(anti_proto) + (1-alpha) * norm(ood_ext)."""
        anti_scores = self.score_anti_prototype_all(features, prototypes, batch_size)
        ood_scores = self.score_ood_head_extended(features, prototypes, batch_size)

        anti_mu = getattr(self, '_ens_anti_mu', anti_scores.mean().item())
        anti_std = getattr(self, '_ens_anti_std', anti_scores.std().item() + 1e-8)
        ood_mu = getattr(self, '_ens_ood_mu', ood_scores.mean().item())
        ood_std = getattr(self, '_ens_ood_std', ood_scores.std().item() + 1e-8)

        norm_anti = (anti_scores - anti_mu) / anti_std
        norm_ood = (ood_scores - ood_mu) / ood_std

        alpha = getattr(self, '_ens_fusion_alpha', 0.7)
        return alpha * norm_anti + (1 - alpha) * norm_ood

    # ---- osr22c: Adaptive cluster-weighted ensemble fusion ----

    def train_ensemble_adaptive(self, prototypes: torch.Tensor,
                                 target_fpr: float = 0.05,
                                 batch_size: int = 4096,
                                 alpha_min: float = 0.3,
                                 alpha_max: float = 0.9):
        """osr22c: Train adaptive ensemble fusion with density-based alpha adjustment.
        Uses GMM cluster posterior to dynamically adjust fusion weight:
        - High density (high posterior) → trust anti_prototype → alpha close to alpha_max
        - Low density (low posterior) → trust ood_head → alpha close to alpha_min

        Args:
            prototypes: class prototypes
            target_fpr: target false positive rate for threshold
            batch_size: batch size for feature extraction
            alpha_min: minimum alpha (for sparse regions)
            alpha_max: maximum alpha (for dense regions)
        """
        from sklearn.preprocessing import StandardScaler

        print("Training adaptive ensemble fusion (GMM density-based alpha)...")

        # Fit GMM for density estimation
        self._fit_global_gmm()

        # Get anti_prototype scores
        known_anti_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_anti_list.append(self.score_anti_prototype_all(feats, prototypes, batch_size))
        known_anti = torch.cat(known_anti_list)

        # Get ood_head_extended_v2 scores (with GMM features)
        self.train_ood_head_extended_v2(prototypes, target_fpr, batch_size, verbose=False)
        known_ood_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_ood_list.append(self.score_ood_head_extended_v2(feats, prototypes, batch_size))
        known_ood = torch.cat(known_ood_list)

        # Get cluster posterior (density) for each sample
        known_density = self._extract_cluster_features(
            torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
            batch_size)[:, 1]  # cluster_posterior_max

        # Store normalization stats
        self._ens_anti_mu = known_anti.mean().item()
        self._ens_anti_std = known_anti.std().item() + 1e-8
        self._ens_ood_mu = known_ood.mean().item()
        self._ens_ood_std = known_ood.std().item() + 1e-8

        # Normalize density to [0, 1] for alpha interpolation
        density_min = known_density.min().item()
        density_max = known_density.max().item()
        self._ens_density_min = density_min
        self._ens_density_max = density_max - density_min + 1e-8

        # Normalize scores
        norm_known_anti = (known_anti - self._ens_anti_mu) / self._ens_anti_std
        norm_known_ood = (known_ood - self._ens_ood_mu) / self._ens_ood_std

        # Compute adaptive alpha for each sample
        norm_density = (known_density - density_min) / self._ens_density_max
        adaptive_alpha = alpha_min + (alpha_max - alpha_min) * norm_density

        # Adaptive fusion
        combined_known = adaptive_alpha * norm_known_anti + (1 - adaptive_alpha) * norm_known_ood

        # Set threshold at target FPR
        sorted_k, _ = combined_known.sort()
        idx = min(int(len(sorted_k) * target_fpr), len(sorted_k) - 1)
        self._ens_adaptive_threshold = sorted_k[idx].item()

        # Compute TNR
        tnr = (combined_known >= self._ens_adaptive_threshold).float().mean().item()

        # Compute TPR on unknown samples
        unknown_anti_list = []
        unknown_ood_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_anti_list.append(self.score_anti_prototype_all(feats, prototypes, batch_size))
                unknown_ood_list.append(self.score_ood_head_extended_v2(feats, prototypes, batch_size))

        if unknown_anti_list and unknown_ood_list:
            unknown_anti = torch.cat(unknown_anti_list)
            unknown_ood = torch.cat(unknown_ood_list)
            unknown_density = self._extract_cluster_features(
                torch.cat([self.cache.get_class_features(c) for c in self.unknown_classes
                           if len(self.cache.get_class_features(c)) > 0]),
                batch_size)[:, 1]

            norm_unknown_anti = (unknown_anti - self._ens_anti_mu) / self._ens_anti_std
            norm_unknown_ood = (unknown_ood - self._ens_ood_mu) / self._ens_ood_std
            norm_unknown_density = (unknown_density - density_min) / self._ens_density_max
            adaptive_alpha_unknown = alpha_min + (alpha_max - alpha_min) * norm_unknown_density

            combined_unknown = adaptive_alpha_unknown * norm_unknown_anti + (1 - adaptive_alpha_unknown) * norm_unknown_ood
            tpr = (combined_unknown < self._ens_adaptive_threshold).float().mean().item()
        else:
            tpr = 0.0

        j = tnr + tpr - 1.0
        self._ens_adaptive_alpha_min = alpha_min
        self._ens_adaptive_alpha_max = alpha_max
        print(f"  Adaptive ensemble: alpha in [{alpha_min:.2f}, {alpha_max:.2f}], "
              f"Youden J={j:.3f} (TNR={tnr:.2%}, TPR={tpr:.2%})")

    def score_ensemble_adaptive(self, features: torch.Tensor,
                                 prototypes: torch.Tensor,
                                 batch_size: int = 4096) -> torch.Tensor:
        """osr22c: Score using adaptive ensemble fusion with density-based alpha."""
        # Get scores
        anti_scores = self.score_anti_prototype_all(features, prototypes, batch_size)
        ood_scores = self.score_ood_head_extended_v2(features, prototypes, batch_size)

        # Get density (cluster posterior)
        density = self._extract_cluster_features(features, batch_size)[:, 1]  # cluster_posterior_max

        # Normalize
        anti_mu = getattr(self, '_ens_anti_mu', anti_scores.mean().item())
        anti_std = getattr(self, '_ens_anti_std', anti_scores.std().item() + 1e-8)
        ood_mu = getattr(self, '_ens_ood_mu', ood_scores.mean().item())
        ood_std = getattr(self, '_ens_ood_std', ood_scores.std().item() + 1e-8)

        norm_anti = (anti_scores - anti_mu) / anti_std
        norm_ood = (ood_scores - ood_mu) / ood_std

        # Compute adaptive alpha based on density
        density_min = getattr(self, '_ens_density_min', 0.0)
        density_range = getattr(self, '_ens_density_max', 1.0)
        alpha_min = getattr(self, '_ens_adaptive_alpha_min', 0.3)
        alpha_max = getattr(self, '_ens_adaptive_alpha_max', 0.9)

        norm_density = (density - density_min) / density_range
        norm_density = norm_density.clamp(0.0, 1.0)  # ensure in [0, 1]
        adaptive_alpha = alpha_min + (alpha_max - alpha_min) * norm_density

        return adaptive_alpha * norm_anti + (1 - adaptive_alpha) * norm_ood

    # ---- osr22d: Cluster boundary detection + three-way ensemble ----

    def train_cluster_boundary(self, prototypes: torch.Tensor,
                                K_shot: int = 50,
                                K_sub: int = 2,
                                beta: float = 1.5,
                                seed: int = 42):
        """osr22d: Train cluster boundary detector using per-class K-means sub-clustering.
        For each class, fits K_sub sub-clusters and defines a boundary radius.
        Samples within the boundary (radius) are considered "known".

        Args:
            prototypes: base class prototypes (not used for sub-clustering)
            K_shot: samples per class for sub-clustering
            K_sub: number of sub-clusters per class
            beta: boundary radius multiplier (radius = mean_dist + beta * std_dist)
            seed: random seed
        """
        from sklearn.cluster import MiniBatchKMeans

        print(f"Training cluster boundary detector (K_sub={K_sub}, beta={beta})...")

        self._cluster_boundaries = {}  # {class_id: [(center, radius, weight), ...]}
        rng = random.Random(seed)

        for c_id in self.base_classes:
            feats = self.cache.get_class_features(c_id)
            n = min(K_shot, len(feats))
            indices = rng.sample(range(len(feats)), n)
            samples = feats[indices].numpy()

            # Fit K-means sub-clusters
            if K_sub == 1 or n < K_sub * 2:
                # Single cluster fallback
                center = samples.mean(axis=0)
                dists = np.linalg.norm(samples - center, axis=1)
                mean_dist = dists.mean()
                std_dist = dists.std()
                radius = mean_dist + beta * std_dist
                self._cluster_boundaries[c_id] = [(center, radius, n)]
            else:
                kmeans = MiniBatchKMeans(n_clusters=K_sub, random_state=seed, batch_size=32)
                labels = kmeans.fit_predict(samples)

                boundaries = []
                for k in range(K_sub):
                    mask = labels == k
                    if mask.sum() == 0:
                        continue
                    cluster_samples = samples[mask]
                    center = cluster_samples.mean(axis=0)
                    dists = np.linalg.norm(cluster_samples - center, axis=1)
                    mean_dist = dists.mean()
                    std_dist = dists.std()
                    radius = mean_dist + beta * std_dist
                    weight = mask.sum()
                    boundaries.append((center, radius, weight))
                self._cluster_boundaries[c_id] = boundaries

        total_clusters = sum(len(v) for v in self._cluster_boundaries.values())
        print(f"  Cluster boundary detector: {total_clusters} sub-clusters across {len(self.base_classes)} classes")

    def score_cluster_boundary(self, features: torch.Tensor,
                                prototypes: torch.Tensor,
                                batch_size: int = 4096) -> torch.Tensor:
        """osr22d: Score using cluster boundary detection.
        Higher score = inside cluster boundary = known.
        Lower score = outside all cluster boundaries = unknown.

        Score = max(0, max_boundary_radius - min_distance_to_any_boundary_center)
        """
        if not hasattr(self, '_cluster_boundaries'):
            raise RuntimeError("Call train_cluster_boundary() first")

        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].numpy()
                B = batch.shape[0]  # batch is numpy array, use .shape

                batch_scores = np.full(B, -np.inf)

                # For each sample, find the nearest cluster and compute boundary score
                for c_id, boundaries in self._cluster_boundaries.items():
                    for center, radius, weight in boundaries:
                        # Distance from sample to this cluster center
                        dists = np.linalg.norm(batch - center, axis=1)  # (B,)
                        # Score = radius - distance (positive = inside boundary)
                        cluster_score = radius - dists
                        batch_scores = np.maximum(batch_scores, cluster_score)

                all_scores.append(torch.from_numpy(batch_scores).float())

        return torch.cat(all_scores)

    def train_ensemble_cluster(self, prototypes: torch.Tensor,
                                target_fpr: float = 0.05,
                                batch_size: int = 4096,
                                K_sub: int = 2,
                                beta: float = 1.5):
        """osr22d: Train three-way ensemble = anti_proto + ood_head_ext_v2 + cluster_boundary.
        Searches optimal fusion weights for combining the three signals."""
        print("Training three-way ensemble (anti_proto + ood_head_v2 + cluster_boundary)...")

        # Train base ensemble (anti + ood_v2)
        self.train_ensemble_adaptive(prototypes, target_fpr, batch_size)

        # Train cluster boundary
        self.train_cluster_boundary(prototypes, K_shot=50, K_sub=K_sub, beta=beta)

        # Get all three scores for known samples
        known_anti_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_anti_list.append(self.score_anti_prototype_all(feats, prototypes, batch_size))
        known_anti = torch.cat(known_anti_list)

        known_ood_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_ood_list.append(self.score_ood_head_extended_v2(feats, prototypes, batch_size))
        known_ood = torch.cat(known_ood_list)

        known_boundary_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_boundary_list.append(self.score_cluster_boundary(feats, prototypes, batch_size))
        known_boundary = torch.cat(known_boundary_list)

        # Get scores for unknown samples
        unknown_anti_list, unknown_ood_list, unknown_boundary_list = [], [], []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_anti_list.append(self.score_anti_prototype_all(feats, prototypes, batch_size))
                unknown_ood_list.append(self.score_ood_head_extended_v2(feats, prototypes, batch_size))
                unknown_boundary_list.append(self.score_cluster_boundary(feats, prototypes, batch_size))

        has_unknown = len(unknown_anti_list) > 0
        if has_unknown:
            unknown_anti = torch.cat(unknown_anti_list)
            unknown_ood = torch.cat(unknown_ood_list)
            unknown_boundary = torch.cat(unknown_boundary_list)

        # Normalize each component
        self._cluster_ens_anti_mu = known_anti.mean().item()
        self._cluster_ens_anti_std = known_anti.std().item() + 1e-8
        self._cluster_ens_ood_mu = known_ood.mean().item()
        self._cluster_ens_ood_std = known_ood.std().item() + 1e-8
        self._cluster_ens_boundary_mu = known_boundary.mean().item()
        self._cluster_ens_boundary_std = known_boundary.std().item() + 1e-8

        norm_known_anti = (known_anti - self._cluster_ens_anti_mu) / self._cluster_ens_anti_std
        norm_known_ood = (known_ood - self._cluster_ens_ood_mu) / self._cluster_ens_ood_std
        norm_known_boundary = (known_boundary - self._cluster_ens_boundary_mu) / self._cluster_ens_boundary_std

        if has_unknown:
            norm_unknown_anti = (unknown_anti - self._cluster_ens_anti_mu) / self._cluster_ens_anti_std
            norm_unknown_ood = (unknown_ood - self._cluster_ens_ood_mu) / self._cluster_ens_ood_std
            norm_unknown_boundary = (unknown_boundary - self._cluster_ens_boundary_mu) / self._cluster_ens_boundary_std

        # Search optimal three-way weights
        # Score = w1 * anti + w2 * ood + w3 * boundary, where w1 + w2 + w3 = 1
        best_weights, best_j = (0.5, 0.3, 0.2), -1.0

        for w1 in [0.4, 0.5, 0.6, 0.7]:  # anti_proto weight
            for w2 in [0.1, 0.2, 0.3, 0.4]:  # ood_head weight
                w3 = 1.0 - w1 - w2  # boundary weight
                if w3 < 0 or w3 > 0.5:
                    continue

                combined_known = w1 * norm_known_anti + w2 * norm_known_ood + w3 * norm_known_boundary
                sorted_k = combined_known.sort()[0]
                idx = min(int(len(sorted_k) * target_fpr), len(sorted_k) - 1)
                tau = sorted_k[idx].item()
                tnr = (combined_known >= tau).float().mean().item()

                if has_unknown:
                    combined_unknown = w1 * norm_unknown_anti + w2 * norm_unknown_ood + w3 * norm_unknown_boundary
                    tpr = (combined_unknown < tau).float().mean().item()
                else:
                    tpr = 0.0

                j = tnr + tpr - 1.0
                if j > best_j:
                    best_j, best_weights = j, (w1, w2, w3)

        self._cluster_ens_weights = best_weights
        print(f"  Three-way ensemble weights: anti={best_weights[0]:.2f}, "
              f"ood={best_weights[1]:.2f}, boundary={best_weights[2]:.2f}, "
              f"Youden J={best_j:.3f}")

    def score_ensemble_cluster(self, features: torch.Tensor,
                                 prototypes: torch.Tensor,
                                 batch_size: int = 4096) -> torch.Tensor:
        """osr22d: Score using three-way ensemble (anti_proto + ood_head_v2 + cluster_boundary)."""
        anti_scores = self.score_anti_prototype_all(features, prototypes, batch_size)
        ood_scores = self.score_ood_head_extended_v2(features, prototypes, batch_size)
        boundary_scores = self.score_cluster_boundary(features, prototypes, batch_size)

        # Normalize
        anti_mu = getattr(self, '_cluster_ens_anti_mu', anti_scores.mean().item())
        anti_std = getattr(self, '_cluster_ens_anti_std', anti_scores.std().item() + 1e-8)
        ood_mu = getattr(self, '_cluster_ens_ood_mu', ood_scores.mean().item())
        ood_std = getattr(self, '_cluster_ens_ood_std', ood_scores.std().item() + 1e-8)
        boundary_mu = getattr(self, '_cluster_ens_boundary_mu', boundary_scores.mean().item())
        boundary_std = getattr(self, '_cluster_ens_boundary_std', boundary_scores.std().item() + 1e-8)

        norm_anti = (anti_scores - anti_mu) / anti_std
        norm_ood = (ood_scores - ood_mu) / ood_std
        norm_boundary = (boundary_scores - boundary_mu) / boundary_std

        weights = getattr(self, '_cluster_ens_weights', (0.5, 0.3, 0.2))
        return weights[0] * norm_anti + weights[1] * norm_ood + weights[2] * norm_boundary

    # ---- osr21b: Anti-prototype in Flow-transformed space ----

    def score_anti_prototype_flow(self, features: torch.Tensor,
                                   prototypes: torch.Tensor,
                                   batch_size: int = 4096) -> torch.Tensor:
        """osr21b: Anti-prototype scoring in Flow-transformed feature space."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            # Transform prototypes once
            adapted_protos = self.flow.feature_adapter(prototypes)
            if self.flow.use_flow_transform:
                adapted_protos = self.flow.feature_transform(adapted_protos)

            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                # Same transform path as classify()
                adapted_batch = self.flow.feature_adapter(batch)
                if self.flow.use_flow_transform:
                    adapted_batch = self.flow.feature_transform(adapted_batch)

                feat_center = adapted_batch.mean(dim=0, keepdim=True)
                anti_protos = 2 * feat_center - adapted_protos
                dist_proto = torch.cdist(adapted_batch, adapted_protos, p=2).min(dim=1).values
                dist_anti = torch.cdist(adapted_batch, anti_protos, p=2).min(dim=1).values
                scores = -dist_proto + dist_anti
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    # ---- osr21c: Flow Jacobian (log_det) scoring ----

    def score_flow_jacobian(self, features: torch.Tensor,
                             prototypes: torch.Tensor,
                             batch_size: int = 4096) -> torch.Tensor:
        """osr21c: Use Flow Jacobian log_det as OSR signal.
        Known features → well-modeled → stable log_det.
        Unknown features → extrapolation → abnormal log_det."""
        self.flow.eval()
        if not self.flow.use_flow_transform:
            # No Flow transform available, return zeros
            return torch.zeros(len(features))

        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                adapted = self.flow.feature_adapter(batch)
                _, log_det = self.flow.feature_transform.forward_with_logdet(adapted)
                # Use absolute log_det as OOD signal
                # Higher |log_det| → more "stretched" → potentially OOD
                all_scores.append(-log_det.abs().cpu())
        return torch.cat(all_scores)

    # ---- osr21c: Triple ensemble (anti_proto + ood_ext + flow_jacobian) ----

    def train_ensemble_triple(self, prototypes: torch.Tensor,
                               target_fpr: float = 0.05,
                               batch_size: int = 4096):
        """osr21c: Three-way ensemble = anti_proto + ood_head_ext + flow_jacobian."""
        print("Training triple ensemble (anti_proto + ood_head_extended + flow_jacobian)...")

        # Ensure base ensemble is trained
        self.train_ensemble_anti_oodext(prototypes, target_fpr)

        # Get flow jacobian scores
        known_jac_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_jac_list.append(self.score_flow_jacobian(feats, prototypes, batch_size))
        known_jac = torch.cat(known_jac_list)

        unknown_jac_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_jac_list.append(self.score_flow_jacobian(feats, prototypes, batch_size))
        unknown_jac = torch.cat(unknown_jac_list) if unknown_jac_list else torch.tensor([])

        # Store Jacobian normalization stats
        self._jac_mu = known_jac.mean().item()
        self._jac_std = known_jac.std().item() + 1e-8

        # Search optimal three-way fusion weights
        # Fix the anti+oodext alpha, search for jacobian weight gamma
        # Score = (1-gamma) * [alpha*anti + (1-alpha)*ood] + gamma * jac
        alpha = getattr(self, '_ens_fusion_alpha', 0.7)
        has_unknown = len(unknown_jac) > 0

        # Pre-compute anti+oodext combined
        known_anti_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_anti_list.append(self.score_anti_prototype_all(feats, prototypes, batch_size))
        known_anti = torch.cat(known_anti_list)
        known_ood = torch.cat([
            self.score_ood_head_extended(self.cache.get_class_features(c), prototypes, batch_size)
            for c in self.base_classes
        ])

        ens_anti_mu = getattr(self, '_ens_anti_mu', known_anti.mean().item())
        ens_anti_std = getattr(self, '_ens_anti_std', known_anti.std().item() + 1e-8)
        ens_ood_mu = getattr(self, '_ens_ood_mu', known_ood.mean().item())
        ens_ood_std = getattr(self, '_ens_ood_std', known_ood.std().item() + 1e-8)

        known_base = alpha * (known_anti - ens_anti_mu) / ens_anti_std + \
                     (1 - alpha) * (known_ood - ens_ood_mu) / ens_ood_std
        norm_known_jac = (known_jac - self._jac_mu) / self._jac_std

        if has_unknown:
            unknown_anti = torch.cat([
                self.score_anti_prototype_all(self.cache.get_class_features(c), prototypes, batch_size)
                for c in self.unknown_classes if len(self.cache.get_class_features(c)) > 0
            ])
            unknown_ood = torch.cat([
                self.score_ood_head_extended(self.cache.get_class_features(c), prototypes, batch_size)
                for c in self.unknown_classes if len(self.cache.get_class_features(c)) > 0
            ])
            unknown_base = alpha * (unknown_anti - ens_anti_mu) / ens_anti_std + \
                          (1 - alpha) * (unknown_ood - ens_ood_mu) / ens_ood_std
            norm_unknown_jac = (unknown_jac - self._jac_mu) / self._jac_std

        best_gamma, best_j = 0.0, -1.0
        for gamma in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]:
            combined_known = (1 - gamma) * known_base + gamma * norm_known_jac
            sorted_k = combined_known.sort()[0]
            idx = min(int(len(sorted_k) * target_fpr), len(sorted_k) - 1)
            tau = sorted_k[idx].item()
            tnr = (combined_known >= tau).float().mean().item()

            if has_unknown:
                combined_unknown = (1 - gamma) * unknown_base + gamma * norm_unknown_jac
                tpr = (combined_unknown < tau).float().mean().item()
            else:
                tpr = 0.0

            j = tnr + tpr - 1.0
            if j > best_j:
                best_j, best_gamma = j, gamma

        self._triple_gamma = best_gamma
        self._triple_alpha = alpha
        print(f"  Triple ensemble: alpha(anti)={alpha:.2f}, "
              f"beta(ood)={1-alpha:.2f}, gamma(jac)={best_gamma:.2f}, "
              f"Youden J={best_j:.3f}")

    def score_ensemble_triple(self, features: torch.Tensor,
                               prototypes: torch.Tensor,
                               batch_size: int = 4096) -> torch.Tensor:
        """osr21c: Three-way ensemble scoring."""
        anti_scores = self.score_anti_prototype_all(features, prototypes, batch_size)
        ood_scores = self.score_ood_head_extended(features, prototypes, batch_size)
        jac_scores = self.score_flow_jacobian(features, prototypes, batch_size)

        alpha = getattr(self, '_triple_alpha', 0.7)
        gamma = getattr(self, '_triple_gamma', 0.0)

        ens_anti_mu = getattr(self, '_ens_anti_mu', 0.0)
        ens_anti_std = getattr(self, '_ens_anti_std', 1.0)
        ens_ood_mu = getattr(self, '_ens_ood_mu', 0.0)
        ens_ood_std = getattr(self, '_ens_ood_std', 1.0)
        jac_mu = getattr(self, '_jac_mu', 0.0)
        jac_std = getattr(self, '_jac_std', 1.0)

        norm_anti = (anti_scores - ens_anti_mu) / ens_anti_std
        norm_ood = (ood_scores - ens_ood_mu) / ens_ood_std
        norm_jac = (jac_scores - jac_mu) / jac_std

        base = alpha * norm_anti + (1 - alpha) * norm_ood
        return (1 - gamma) * base + gamma * norm_jac

    # ---- Learned OOD head (logistic regression) ----

    def train_ood_head(self, prototypes: torch.Tensor,
                       target_fpr: float = 0.05,
                       batch_size: int = 4096,
                       verbose: bool = True):
        """Train a logistic regression OOD classifier on calibration data."""
        from sklearn.linear_model import LogisticRegression

        if verbose:
            print("Training learned OOD head...")
        known_feats_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_feats_list.append(self._extract_ood_features(feats, prototypes, batch_size))
        known_feats = torch.cat(known_feats_list).numpy()

        unknown_feats_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_feats_list.append(self._extract_ood_features(feats, prototypes, batch_size))
        if not unknown_feats_list:
            print("  No unknown samples for training OOD head, skipping.")
            return
        unknown_feats = torch.cat(unknown_feats_list).numpy()

        X = np.concatenate([known_feats, unknown_feats], axis=0)
        y = np.concatenate([np.ones(len(known_feats)), np.zeros(len(unknown_feats))], axis=0)

        self._ood_clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs')
        self._ood_clf.fit(X, y)

        if verbose:
            w = self._ood_clf.coef_[0]
            feat_names = ['min_dist', 'dist_ratio', 'softmax_max', 'entropy', 'feat_norm']
            print("  Learned weights:")
            for name, wi in zip(feat_names, w):
                print(f"    {name:>12s}: {wi:+.4f}")

        known_probs = self._ood_clf.predict_proba(known_feats)[:, 1]
        sorted_probs = np.sort(known_probs)
        idx = min(int(len(sorted_probs) * target_fpr), len(sorted_probs) - 1)
        self._ood_threshold = sorted_probs[idx]

        if verbose:
            unknown_probs = self._ood_clf.predict_proba(unknown_feats)[:, 1]
            tnr = (known_probs >= self._ood_threshold).mean()
            tpr = (unknown_probs < self._ood_threshold).mean()
            print(f"  Calib TNR: {tnr:.2%}  TPR: {tpr:.2%}  "
                  f"(threshold={self._ood_threshold:.4f})")

    def score_samples_learned(self, features: torch.Tensor,
                               prototypes: torch.Tensor,
                               batch_size: int = 4096) -> torch.Tensor:
        """Score using learned OOD head."""
        if not hasattr(self, '_ood_clf'):
            raise RuntimeError("Call train_ood_head() first")
        feats = self._extract_ood_features(features, prototypes, batch_size).numpy()
        probs = self._ood_clf.predict_proba(feats)[:, 1]
        return torch.from_numpy(probs).float()

    # ---- Calibration ----

    def calibrate(self, target_fpr: float = 0.05,
                  K_shot: int = 50,
                  method: str = 'feature_mahalanobis') -> float:
        """Find OSR threshold on calibration data."""
        print(f"\n{'='*70}")
        print(f"OSR Calibration ({method.upper()})")
        print(f"{'='*70}")

        prototypes = self.compute_prototypes(K_shot)
        print(f"Prototypes: {prototypes.shape} from {len(self.base_classes)} base classes")

        # Compute stats needed for each method
        if method in ('feature_mahalanobis', 'feat_mahalanobis_relative', 'geo_fusion',
                       'anti_prototype', 'anti_prototype_enhanced', 'anti_prototype_multi',
                       'ood_head', 'learned_ensemble', 'anti_proto_var_norm', 'anti_proto_cosine',
                       'ood_head_extended', 'ood_head_extended_v2', 'cluster_boundary'):
            print("Computing feature-space statistics...")
            self.compute_feat_stats()
            print("Computing feature-space statistics...")
            self.compute_feat_stats()

        # osr20a: class variances for var-norm method
        if method == 'anti_proto_var_norm':
            self.compute_class_variances(prototypes)

        # Transductive refinement
        if method not in ('learned_ensemble', 'ood_head_extended', 'ood_head_extended_v2',
                          'anti_prototype_multi', 'cluster_boundary',
                          'ensemble_adaptive', 'ensemble_cluster'):
            support_feats_list = []
            support_labels_list = []
            query_feats_list = []
            for new_id, c_id in enumerate(self.base_classes):
                feats = self.cache.get_class_features(c_id)
                n = min(K_shot, len(feats))
                rng = random.Random(42)
                idx = rng.sample(range(len(feats)), min(len(feats), K_shot + 30))
                support_feats_list.append(feats[idx[:n]])
                support_labels_list.extend([new_id] * n)
                if len(idx) > n:
                    query_feats_list.append(feats[idx[n:]])
            if query_feats_list:
                support_feats_t = torch.cat(support_feats_list).to(self.device)
                support_labels_t = torch.tensor(support_labels_list, device=self.device)
                query_feats_t = torch.cat(query_feats_list).to(self.device)
                prototypes = self.flow.refine_prototypes_transductive(
                    support_feats_t, support_labels_t,
                    query_feats_t, prototypes,
                    num_iters=2, confidence_threshold=0.7)

        # Method-specific setup
        if method == 'feat_mahalanobis_relative':
            best_alpha, best_j = 0.5, 0.0
            for trial_alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
                self._feat_mahal_rel_alpha = trial_alpha
                trial_known = []
                trial_unknown = []
                for c in self.base_classes:
                    trial_known.append(self.score_samples_feat_mahalanobis_relative(
                        self.cache.get_class_features(c), prototypes))
                for c in self.unknown_classes:
                    trial_unknown.append(self.score_samples_feat_mahalanobis_relative(
                        self.cache.get_class_features(c), prototypes))
                trial_known = torch.cat(trial_known)
                trial_unknown = torch.cat(trial_unknown)
                sorted_k, _ = trial_known.sort()
                tnr_idx = min(int(len(sorted_k) * 0.05), len(sorted_k) - 1)
                tau = sorted_k[tnr_idx].item()
                tpr = (trial_unknown > tau).float().mean().item()
                j = 0.95 + tpr - 1.0
                if j > best_j:
                    best_j, best_alpha = j, trial_alpha
            self._feat_mahal_rel_alpha = best_alpha
            print(f"  Mahalanobis relative alpha: {best_alpha:.2f} (Youden J={best_j:.3f})")

        if method == 'geo_fusion':
            print("Computing geometric fusion normalization stats...")
            known_anti_all, known_mahal_all = [], []
            for c in self.base_classes:
                feats = self.cache.get_class_features(c)
                known_anti_all.append(self.score_anti_prototype_all(feats, prototypes))
                known_mahal_all.append(self.score_samples_feat_mahalanobis(feats, prototypes))
            all_anti = torch.cat(known_anti_all)
            all_mahal = torch.cat(known_mahal_all)
            self._geo_anti_mu = all_anti.mean().item()
            self._geo_anti_std = all_anti.std().item()
            self._geo_mahal_mu = all_mahal.mean().item()
            self._geo_mahal_std = all_mahal.std().item()

            norm_known_anti = (all_anti - self._geo_anti_mu) / (self._geo_anti_std + 1e-8)
            norm_known_mahal = (all_mahal - self._geo_mahal_mu) / (self._geo_mahal_std + 1e-8)

            unknown_anti_all, unknown_mahal_all = [], []
            for c in self.unknown_classes:
                feats = self.cache.get_class_features(c)
                if len(feats) > 0:
                    unknown_anti_all.append(self.score_anti_prototype_all(feats, prototypes))
                    unknown_mahal_all.append(self.score_samples_feat_mahalanobis(feats, prototypes))
            has_unknown = len(unknown_anti_all) > 0
            if has_unknown:
                norm_unknown_anti = (torch.cat(unknown_anti_all) - self._geo_anti_mu) / (self._geo_anti_std + 1e-8)
                norm_unknown_mahal = (torch.cat(unknown_mahal_all) - self._geo_mahal_mu) / (self._geo_mahal_std + 1e-8)

            best_alpha, best_j = 0.5, -1.0
            for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                combined_known = alpha * norm_known_anti + (1 - alpha) * norm_known_mahal
                sorted_k = combined_known.sort()[0]
                idx = min(int(len(sorted_k) * target_fpr), len(sorted_k) - 1)
                thresh = sorted_k[idx]
                if has_unknown:
                    combined_unknown = alpha * norm_unknown_anti + (1 - alpha) * norm_unknown_mahal
                    tpr = (combined_unknown > thresh).float().mean().item()
                else:
                    tpr = 0.0
                tnr = (combined_known >= thresh).float().mean().item()
                j = tnr + tpr - 1.0
                if j > best_j:
                    best_j, best_alpha = j, alpha
            self._geo_fusion_alpha = best_alpha
            print(f"  Fusion weight: alpha={best_alpha:.2f} (Youden J={best_j:.3f})")

        if method == 'learned_ensemble':
            self.train_ood_head(prototypes, target_fpr)
            self.threshold = self._ood_threshold
            known_scores = self.score_samples_learned(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes)
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_samples_learned(
                    torch.cat(unknown_feats), prototypes)
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # osr20a: extended OOD head calibration
        if method == 'ood_head_extended':
            self.train_ood_head_extended(prototypes, target_fpr)
            self.threshold = self._ood_ext_threshold
            known_scores = self.score_ood_head_extended(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes)
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_ood_head_extended(
                    torch.cat(unknown_feats), prototypes)
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # osr20b: ensemble calibration
        if method == 'ensemble_anti_oodext':
            self.train_ensemble_anti_oodext(prototypes, target_fpr)
            known_scores = self.score_ensemble_anti_oodext(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes)
            sorted_known, _ = known_scores.sort()
            idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
            self.threshold = sorted_known[idx].item()
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_ensemble_anti_oodext(
                    torch.cat(unknown_feats), prototypes)
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # ============ osr22 new methods calibration ============

        # osr22a: anti_prototype_multi calibration
        if method == 'anti_prototype_multi':
            known_scores = self.score_samples(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes, method='anti_prototype_multi')
            sorted_known, _ = known_scores.sort()
            idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
            self.threshold = sorted_known[idx].item()
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_samples(
                    torch.cat(unknown_feats), prototypes, method='anti_prototype_multi')
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # osr22b: ood_head_extended_v2 calibration
        if method == 'ood_head_extended_v2':
            self.train_ood_head_extended_v2(prototypes, target_fpr)
            known_scores = self.score_ood_head_extended_v2(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes)
            sorted_known, _ = known_scores.sort()
            idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
            self.threshold = sorted_known[idx].item()
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_ood_head_extended_v2(
                    torch.cat(unknown_feats), prototypes)
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # osr22c: ensemble_adaptive calibration
        if method == 'ensemble_adaptive':
            self.train_ensemble_adaptive(prototypes, target_fpr)
            known_scores = self.score_ensemble_adaptive(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes)
            sorted_known, _ = known_scores.sort()
            idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
            self.threshold = self._ens_adaptive_threshold
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_ensemble_adaptive(
                    torch.cat(unknown_feats), prototypes)
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # osr22d: cluster_boundary calibration
        if method == 'cluster_boundary':
            self.train_cluster_boundary(prototypes)
            known_scores = self.score_cluster_boundary(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes)
            sorted_known, _ = known_scores.sort()
            idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
            self.threshold = sorted_known[idx].item()
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_cluster_boundary(
                    torch.cat(unknown_feats), prototypes)
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # osr22d: ensemble_cluster calibration
        if method == 'ensemble_cluster':
            self.train_ensemble_cluster(prototypes, target_fpr)
            known_scores = self.score_ensemble_cluster(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes)
            sorted_known, _ = known_scores.sort()
            idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
            self.threshold = sorted_known[idx].item()
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_ensemble_cluster(
                    torch.cat(unknown_feats), prototypes)
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # Score known and unknown samples
        known_scores_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_scores_list.append(
                self.score_samples(feats, prototypes, method))
        known_scores = torch.cat(known_scores_list)

        unknown_scores_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_scores_list.append(
                    self.score_samples(feats, prototypes, method))
        unknown_scores = torch.cat(unknown_scores_list) if unknown_scores_list else torch.tensor([])

        print(f"Known samples:   {len(known_scores)}")
        print(f"Unknown samples: {len(unknown_scores)}")
        print(f"Known scores:   [{known_scores.min():.2f}, {known_scores.max():.2f}] "
              f"mean={known_scores.mean():.2f}")
        if len(unknown_scores) > 0:
            print(f"Unknown scores: [{unknown_scores.min():.2f}, {unknown_scores.max():.2f}] "
                  f"mean={unknown_scores.mean():.2f}")
            sep = known_scores.mean() - unknown_scores.mean()
            print(f"Mean separation: {sep:.2f} (positive = known higher = good)")

        sorted_known, _ = known_scores.sort()
        idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
        threshold = sorted_known[idx].item()

        known_correct = (known_scores >= threshold).float().mean()
        print(f"\nThreshold (tau): {threshold:.4f}")
        print(f"Known accepted (TNR): {known_correct:.2%}")
        if len(unknown_scores) > 0:
            unknown_detected = (unknown_scores < threshold).float().mean()
            print(f"Unknown detected (TPR): {unknown_detected:.2%}")

        self.threshold = threshold
        return threshold

    def evaluate_osr(self, threshold: float = None,
                     K_shot: int = 50,
                     num_rounds: int = 10,
                     method: str = 'feature_mahalanobis',
                     recalibrate_per_round: bool = True,
                     use_per_class_threshold: bool = False,
                     use_rectified_prototypes: bool = False,
                     prior_strength: float = 1.0) -> Dict:
        """Full OSR evaluation with few-shot prototype computation.

        osr20a additions:
          use_per_class_threshold: per-class adaptive thresholds instead of global
          use_rectified_prototypes: Bayesian prototype rectification
          prior_strength: regularization strength for rectification (m)
        """
        if not recalibrate_per_round:
            if threshold is None:
                threshold = self.threshold
            if threshold is None:
                raise ValueError("Run calibrate() first or provide threshold")

        print(f"\n{'='*70}")
        print(f"OSR Evaluation ({method.upper()})")
        if recalibrate_per_round:
            print(f"Mode: per-round recalibration (TNR-fixed at 95%)")
        if use_per_class_threshold:
            print(f"Threshold: per-class adaptive")
        if use_rectified_prototypes:
            print(f"Prototypes: Bayesian rectified (m={prior_strength})")
        print(f"{'='*70}")

        all_known_rates = []
        all_unknown_rates = []
        all_known_scores = []    # AUROC: collect raw known scores
        all_unknown_scores = []  # AUROC: collect raw unknown scores
        target_fpr = 0.05

        for round_i in range(num_rounds):
            # osr20a: optionally use rectified prototypes
            if use_rectified_prototypes:
                prototypes = self.compute_prototypes_rectified(
                    K_shot, seed=round_i, prior_strength=prior_strength)
            else:
                prototypes = self.compute_prototypes(K_shot, seed=round_i)

            # Transductive refinement
            if method not in ('feature_mahalanobis', 'feat_mahalanobis_relative',
                              'learned_ensemble', 'ood_head_extended',
                              'anti_prototype_multi', 'ood_head_extended_v2',
                              'cluster_boundary', 'ensemble_adaptive', 'ensemble_cluster'):
                support_feats_list = []
                support_labels_list = []
                query_feats_list = []
                for new_id, c_id in enumerate(self.base_classes):
                    feats = self.cache.get_class_features(c_id)
                    n = min(K_shot, len(feats))
                    rng = random.Random(round_i)
                    idx = rng.sample(range(len(feats)), min(len(feats), K_shot + 30))
                    support_feats_list.append(feats[idx[:n]])
                    support_labels_list.extend([new_id] * n)
                    if len(idx) > n:
                        query_feats_list.append(feats[idx[n:]])
                if query_feats_list:
                    support_feats_t = torch.cat(support_feats_list).to(self.device)
                    support_labels_t = torch.tensor(support_labels_list, device=self.device)
                    query_feats_t = torch.cat(query_feats_list).to(self.device)
                    prototypes = self.flow.refine_prototypes_transductive(
                        support_feats_t, support_labels_t,
                        query_feats_t, prototypes,
                        num_iters=2, confidence_threshold=0.7)

            self.compute_feat_stats()

            # osr20a: compute class variances for var-norm method
            if method == 'anti_proto_var_norm':
                self.compute_class_variances(prototypes)

            # Method-specific per-round setup
            if method == 'feat_mahalanobis_relative':
                best_alpha, best_j = 0.5, 0.0
                for trial_alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
                    self._feat_mahal_rel_alpha = trial_alpha
                    trial_known = []
                    trial_unknown = []
                    for c in self.base_classes:
                        trial_known.append(self.score_samples_feat_mahalanobis_relative(
                            self.cache.get_class_features(c), prototypes))
                    for c in self.unknown_classes:
                        trial_unknown.append(self.score_samples_feat_mahalanobis_relative(
                            self.cache.get_class_features(c), prototypes))
                    trial_known = torch.cat(trial_known)
                    trial_unknown = torch.cat(trial_unknown)
                    sorted_k, _ = trial_known.sort()
                    tnr_idx = min(int(len(sorted_k) * 0.05), len(sorted_k) - 1)
                    tau = sorted_k[tnr_idx].item()
                    tpr = (trial_unknown > tau).float().mean().item()
                    j = 0.95 + tpr - 1.0
                    if j > best_j:
                        best_j, best_alpha = j, trial_alpha
                self._feat_mahal_rel_alpha = best_alpha

            if method == 'geo_fusion':
                known_anti_all, known_mahal_all = [], []
                for c in self.base_classes:
                    feats = self.cache.get_class_features(c)
                    known_anti_all.append(self.score_anti_prototype_all(feats, prototypes))
                    known_mahal_all.append(self.score_samples_feat_mahalanobis(feats, prototypes))
                all_anti = torch.cat(known_anti_all)
                all_mahal = torch.cat(known_mahal_all)
                self._geo_anti_mu = all_anti.mean().item()
                self._geo_anti_std = all_anti.std().item()
                self._geo_mahal_mu = all_mahal.mean().item()
                self._geo_mahal_std = all_mahal.std().item()
                best_alpha, best_j = 0.5, 0.0
                norm_known_anti = (all_anti - self._geo_anti_mu) / (self._geo_anti_std + 1e-8)
                norm_known_mahal = (all_mahal - self._geo_mahal_mu) / (self._geo_mahal_std + 1e-8)
                unknown_anti_all, unknown_mahal_all = [], []
                for c in self.unknown_classes:
                    feats = self.cache.get_class_features(c)
                    if len(feats) > 0:
                        unknown_anti_all.append(self.score_anti_prototype_all(feats, prototypes))
                        unknown_mahal_all.append(self.score_samples_feat_mahalanobis(feats, prototypes))
                has_unknown = len(unknown_anti_all) > 0
                if has_unknown:
                    norm_unknown_anti = (torch.cat(unknown_anti_all) - self._geo_anti_mu) / (self._geo_anti_std + 1e-8)
                    norm_unknown_mahal = (torch.cat(unknown_mahal_all) - self._geo_mahal_mu) / (self._geo_mahal_std + 1e-8)
                for alpha in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                    combined_known = alpha * norm_known_anti + (1 - alpha) * norm_known_mahal
                    sorted_k = combined_known.sort()[0]
                    idx = min(int(len(sorted_k) * 0.05), len(sorted_k) - 1)
                    tau = sorted_k[idx].item()
                    if has_unknown:
                        combined_unknown = alpha * norm_unknown_anti + (1 - alpha) * norm_unknown_mahal
                        tpr = (combined_unknown > tau).float().mean().item()
                    else:
                        tpr = 0.0
                    j = 0.95 + tpr - 1.0
                    if j > best_j:
                        best_j, best_alpha = j, alpha
                self._geo_fusion_alpha = best_alpha

            if method == 'learned_ensemble':
                self.train_ood_head(prototypes, target_fpr, verbose=False)

            # osr20a: extended OOD head setup
            if method == 'ood_head_extended':
                self.train_ood_head_extended(prototypes, target_fpr, verbose=(round_i == 0))

            # osr20b: ensemble anti_proto + ood_head_extended setup
            if method == 'ensemble_anti_oodext':
                if round_i == 0:
                    self.train_ensemble_anti_oodext(prototypes, target_fpr)
                else:
                    self.train_ood_head_extended(prototypes, target_fpr, verbose=False)

            # ============ osr22 new methods per-round setup ============

            # osr22a: anti_prototype_multi (no per-round training needed)

            # osr22b: ood_head_extended_v2 setup
            if method == 'ood_head_extended_v2':
                if round_i == 0:
                    self.train_ood_head_extended_v2(prototypes, target_fpr, verbose=True)
                else:
                    # Still need to fit GMM each round for cluster features
                    self._fit_global_gmm()

            # osr22c: ensemble_adaptive setup
            if method == 'ensemble_adaptive':
                if round_i == 0:
                    self.train_ensemble_adaptive(prototypes, target_fpr)
                else:
                    # Still need to fit GMM each round
                    self._fit_global_gmm()
                    self.train_ood_head_extended_v2(prototypes, target_fpr, verbose=False)

            # osr22d: cluster_boundary setup
            if method == 'cluster_boundary':
                if round_i == 0:
                    self.train_cluster_boundary(prototypes)

            # osr22d: ensemble_cluster setup
            if method == 'ensemble_cluster':
                if round_i == 0:
                    self.train_ensemble_cluster(prototypes, target_fpr)
                else:
                    self._fit_global_gmm()
                    self.train_ood_head_extended_v2(prototypes, target_fpr, verbose=False)

            # Score known samples
            known_scores_list = []
            for c in self.base_classes:
                feats = self.cache.get_class_features(c)
                known_scores_list.append(self.score_samples(feats, prototypes, method))
            known_scores = torch.cat(known_scores_list)

            # osr20a: per-class adaptive thresholds
            if use_per_class_threshold and recalibrate_per_round:
                per_class_thresholds = {}
                for c_idx, c_id in enumerate(self.base_classes):
                    c_scores = self.score_samples(
                        self.cache.get_class_features(c_id), prototypes, method)
                    sorted_c, _ = c_scores.sort()
                    idx = min(int(len(sorted_c) * target_fpr), len(sorted_c) - 1)
                    per_class_thresholds[c_idx] = sorted_c[idx].item()

                # Known rate: per-sample uses its nearest class threshold
                with torch.no_grad():
                    all_known_feats = torch.cat([
                        self.cache.get_class_features(c) for c in self.base_classes
                    ]).to(self.device)
                    dists_to_protos = torch.cdist(all_known_feats, prototypes, p=2)
                    nearest_class = dists_to_protos.argmin(dim=1)  # (B,)
                    sample_thresholds = torch.tensor(
                        [per_class_thresholds[nc.item()] for nc in nearest_class])
                    known_rate = (known_scores >= sample_thresholds).float().mean().item()

                # Unknown detection with per-class thresholds
                unknown_detect_list = []
                for c in self.unknown_classes:
                    feats = self.cache.get_class_features(c)
                    if len(feats) == 0:
                        continue
                    scores = self.score_samples(feats, prototypes, method)
                    feats_dev = feats.to(self.device)
                    dists = torch.cdist(feats_dev, prototypes, p=2)
                    nearest_c = dists.argmin(dim=1)
                    unk_thresholds = torch.tensor(
                        [per_class_thresholds[nc.item()] for nc in nearest_c])
                    unknown_detect_list.append(
                        (scores < unk_thresholds).float().mean().item())
            else:
                # Original global threshold logic
                if recalibrate_per_round:
                    sorted_known, _ = known_scores.sort()
                    idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
                    round_threshold = sorted_known[idx].item()
                else:
                    round_threshold = threshold

                known_rate = (known_scores >= round_threshold).float().mean().item()

                unknown_detect_list = []
                for c in self.unknown_classes:
                    feats = self.cache.get_class_features(c)
                    if len(feats) == 0:
                        continue
                    scores = self.score_samples(feats, prototypes, method)
                    unknown_detect_list.append((scores < round_threshold).float().mean().item())

            all_known_rates.append(known_rate)
            if unknown_detect_list:
                unknown_rate = np.mean(unknown_detect_list)
                all_unknown_rates.append(unknown_rate)

            # AUROC: collect raw scores (known=0, unknown=1)
            all_known_scores.append(known_scores.cpu().numpy())
            unknown_raw_scores = []
            for c in self.unknown_classes:
                feats = self.cache.get_class_features(c)
                if len(feats) == 0:
                    continue
                scores = self.score_samples(feats, prototypes, method)
                unknown_raw_scores.append(scores.cpu().numpy())
            if unknown_raw_scores:
                all_unknown_scores.append(np.concatenate(unknown_raw_scores))

        mean_known = np.mean(all_known_rates)
        mean_unknown = np.mean(all_unknown_rates) if all_unknown_rates else 0.0
        osr_score = (mean_known + mean_unknown) / 2

        # AUROC: aggregate scores across all rounds
        auroc_score = 0.0
        if all_known_scores and all_unknown_scores:
            all_k = np.concatenate(all_known_scores)
            all_u = np.concatenate(all_unknown_scores)
            labels = np.concatenate([np.zeros(len(all_k)), np.ones(len(all_u))])
            scores = np.concatenate([all_k, all_u])
            # Higher score → more likely known → label=0, so negate for AUROC
            auroc_score = roc_auc_score(labels, -scores)

        print(f"\nResults over {num_rounds} rounds:")
        print(f"  Known accepted (TNR):     {mean_known:.2%}")
        print(f"  Unknown detected (TPR):   {mean_unknown:.2%}")
        print(f"  OSR Score (mean):         {osr_score:.2%}")
        print(f"  AUROC:                    {auroc_score:.4f}")

        results = {
            'known_tnr': mean_known,
            'unknown_tpr': mean_unknown,
            'osr_score': osr_score,
            'auroc': auroc_score,
            'threshold': threshold if not recalibrate_per_round else 'per-round',
            'num_rounds': num_rounds,
            'method': method,
            'recalibrate': recalibrate_per_round,
            'per_class_threshold': use_per_class_threshold,
            'rectified_prototypes': use_rectified_prototypes,
        }
        return results


# ============================================================
# 8. Few-Shot Evaluator
# ============================================================

class FewShotEvaluator:
    """Evaluate few-shot classification accuracy with confidence intervals"""

    def __init__(self, flow_classifier: EpisodicFlowClassifier,
                 device: str = 'cuda'):
        self.flow = flow_classifier
        self.device = device

    def evaluate(self, feature_cache: FeatureCache,
                 eval_classes: List[int],
                 N_way: int = 5, K_shot: int = 5,
                 num_episodes: int = 100,
                 transductive: bool = True) -> Dict:
        """
        Run few-shot evaluation over many episodes.

        Returns mean accuracy with 95% confidence interval.
        """
        self.flow.eval()

        sampler = EpisodeSampler(
            feature_cache, eval_classes,
            N_way=N_way, K_shot=K_shot, Q_query=15)

        all_accs = []
        all_confs = []

        with torch.no_grad():
            for _ in tqdm(range(num_episodes), desc=f"{N_way}w{K_shot}s eval"):
                episode = sampler.sample_episode(self.device)

                query_feats = episode['query_feats']
                prototypes = episode['prototypes']

                # L2 normalize at inference for cosine distance
                query_feats, prototypes = EpisodicFlowClassifier.l2_normalize(
                    query_feats, prototypes)

                # Transductive refinement: expand prototypes with confident queries
                if transductive:
                    # L2 normalize support feats to match normalized prototypes
                    support_feats_norm, _ = EpisodicFlowClassifier.l2_normalize(
                        episode['support_feats'],
                        episode['prototypes'])  # prototypes already normalized
                    prototypes = self.flow.refine_prototypes_transductive(
                        support_feats_norm, episode['support_labels'],
                        query_feats, prototypes,
                        num_iters=3, confidence_threshold=0.7)

                log_probs = self.flow.classify(query_feats, prototypes)

                preds = log_probs.argmax(dim=1)
                acc = (preds == episode['query_labels']).float().mean().item()
                all_accs.append(acc)

                probs = F.softmax(log_probs, dim=1)
                conf = probs.max(dim=1)[0].mean().item()
                all_confs.append(conf)

        results = {
            'mean_acc': np.mean(all_accs),
            'std_acc': np.std(all_accs),
            'ci95': 1.96 * np.std(all_accs) / np.sqrt(len(all_accs)),
            'mean_confidence': np.mean(all_confs),
            'N_way': N_way,
            'K_shot': K_shot,
            'num_episodes': num_episodes,
        }

        print(f"\n  {N_way}-way {K_shot}-shot: "
              f"Acc = {results['mean_acc']:.2%} +/- {results['ci95']:.2%} "
              f"(conf: {results['mean_confidence']:.2%})")

        return results


# ============================================================
# 9. Feature Visualization (t-SNE)
# ============================================================

def visualize_features_tsne(train_cache: FeatureCache,
                            test_cache: FeatureCache,
                            base_classes: List[int],
                            unknown_classes: List[int],
                            save_dir: str):
    """Generate t-SNE plot of base vs novel class features for domain shift analysis"""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn/matplotlib not available, skipping t-SNE visualization")
        return

    features, labels, splits = [], [], []
    for c in base_classes:
        feats = train_cache.get_class_features(c)
        if len(feats) > 0:
            features.append(feats)
            labels.extend([c] * len(feats))
            splits.extend(['base'] * len(feats))
    for c in unknown_classes:
        feats = test_cache.get_class_features(c)
        if len(feats) > 0:
            features.append(feats)
            labels.extend([c] * len(feats))
            splits.extend(['novel'] * len(feats))

    features = torch.cat(features, dim=0).numpy()
    labels = np.array(labels)
    splits = np.array(splits)

    # Subsample for speed
    max_samples = 5000
    if len(features) > max_samples:
        idx = np.random.choice(len(features), max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]
        splits = splits[idx]

    print(f"\nComputing t-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    class_names = {
        0: 'Airport', 1: 'Tram', 2: 'Bus', 3: 'Public Square',
        4: 'Shopping Mall', 5: 'Street Pedestrian',
        6: 'Metro Station', 7: 'Street Traffic', 8: 'Metro', 9: 'Park'
    }
    for c in sorted(set(labels)):
        mask = labels == c
        marker = 'o' if c in base_classes else '^'
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   label=f'{class_names.get(c, f"Class {c}")} '
                         f'({"base" if c in base_classes else "novel"})',
                   marker=marker, alpha=0.5, s=12)

    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.set_title('t-SNE: Base vs Novel Class Features')

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, 'tsne_features.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved: {path}")


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Episodic Meta-Trainer for Few-Shot OSR')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training, load checkpoint and run evaluation only')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to fewshot_final.pth (default: auto-detect from experiment_dir)')
    args = parser.parse_args()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # ============ Configuration ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 10 acoustic scene classes -> 6 base (known), 4 novel (unknown)
    # New mapping: 0=airport, 1=shopping_mall, 2=metro_station, 3=street_pedestrian,
    #              4=public_square, 5=street_traffic, 6=tram, 7=bus, 8=metro, 9=park
    # Base (known): airport(0), tram(6), bus(7), public_square(4), shopping_mall(1), street_pedestrian(3)
    # Unknown (novel): metro_station(2), street_traffic(5), metro(8), park(9)
    base_classes = [0, 1, 2, 3, 4, 5]
    unknown_classes = [6, 7, 8, 9]

    N_way = 6  # osr22: 使用全部6个base classes (原来是5-way)
    K_shot = 5
    Q_query = 100  # osr22: 大幅提升GPU利用率 (6-way * 100 = 600 queries/episode)
    num_episodes = 6000
    gradient_accum_steps = 1
    feature_dim = 64

    # Change this one variable to redirect all model output paths
    experiment_dir = 'experiment/yamnet_fewshot_osr22_relabel'  # osr22: resetlabel with new vocabulary mapping
    base_pretrained_path = os.path.join(experiment_dir, 'base_feature_extractor.pth')
    contrastive_pretrained_path = os.path.join(experiment_dir, 'contrastive_feature_extractor.pth')

    # ---- Backbone selection: 'yamnet', 'distil_ast' or 'panns_cnn14' ----
    backbone_choice = 'yamnet'

    if backbone_choice == 'yamnet':
        FeatureExtractorClass = FeatureExtractor
        backbone_unfreeze_layers = [
            'layer7', 'layer8', 'layer9', 'layer10',
            'layer11', 'layer12', 'layer13', 'layer14'
        ]
    elif backbone_choice == 'distil_ast':
        FeatureExtractorClass = DistilASTFeatureExtractor
        backbone_unfreeze_layers = [
            'encoder.layer.4', 'encoder.layer.5'  # last 2 of 6 Distil-AST layers
        ]
    elif backbone_choice == 'panns_cnn14':
        FeatureExtractorClass = PANNsFeatureExtractor
        backbone_unfreeze_layers = [
            'conv_block3', 'conv_block4', 'conv_block5', 'conv_block6'  # last 4 CNN blocks
        ]
    else:
        raise ValueError(f"Unknown backbone: {backbone_choice}")

    print(f"\n{'='*70}")
    print("Few-Shot Open Set Recognition - Episodic Training Pipeline")
    print(f"Backbone: {backbone_choice}")
    print(f"{'='*70}")

    # ============ Phase 0: Feature Extractor Pre-training ============
    print("\nLoading datasets...")
    train_dataset = TAUDataset(split='train')

    # Check if existing checkpoint matches selected backbone
    _need_retrain = False
    if os.path.exists(base_pretrained_path):
        ckpt = torch.load(base_pretrained_path, map_location='cpu', weights_only=False)
        ckpt_backbone = ckpt.get('config', {}).get('backbone', 'yamnet')
        if ckpt_backbone == FeatureExtractorClass.BACKBONE_TYPE:
            print(f"\nBase feature extractor already exists: {base_pretrained_path}")
            feature_extractor = FeatureExtractorClass(pretrained_path=base_pretrained_path)
        else:
            print(f"\nWARNING: Existing checkpoint has backbone='{ckpt_backbone}', "
                  f"but need '{FeatureExtractorClass.BACKBONE_TYPE}'. "
                  f"Will retrain Phase 0 without deleting existing files.")
            _need_retrain = True

    if not os.path.exists(base_pretrained_path) or _need_retrain:
        # Phase 0a: Contrastive pre-training (SimCLR → SupCon, osr17)
        print(f"\n{'='*70}")
        print(f"Phase 0a: Contrastive Pre-training (SimCLR → SupCon)")
        print("  - First half: SimCLR (all classes, no label leakage)")
        print("  - Second half: SupCon (base classes only, discriminative)")
        print(f"  - {backbone_choice} last layers unfrozen for acoustic scene adaptation")
        print(f"{'='*70}")
        feature_extractor = FeatureExtractorClass()
        contrastive_pretrainer = ContrastivePretrainer(
            feature_extractor, device,
            lr=3e-4, epochs=200,
            temperature=0.1,
            backbone_unfreeze_layers=backbone_unfreeze_layers,
            supervised_contrastive=True,
            base_classes=base_classes)
        feature_extractor = contrastive_pretrainer.train(train_dataset, contrastive_pretrained_path)

        # Phase 0b: Base class supervised fine-tuning
        print(f"\n{'='*70}")
        print(f"Phase 0b: Base Class Supervised Fine-tuning")
        print("  - Only base classes (no leakage)")
        print(f"  - Unfreeze: {backbone_unfreeze_layers}")
        print(f"{'='*70}")
        pretrainer = BaseClassPretrainer(
            feature_extractor, base_classes, device,
            lr=1e-5, epochs=20,
            backbone_unfreeze_layers=backbone_unfreeze_layers)
        feature_extractor = pretrainer.train(train_dataset, base_pretrained_path)

    feature_extractor = feature_extractor.to(device)

    if not args.eval_only:
        # ============ Phase 1: Feature Extraction ============
        print(f"\n{'='*70}")
        print("Phase 1: Feature Extraction")
        print(f"{'='*70}")

        calib_dataset = TAUDataset(split='calib')
        test_dataset = TAUDataset(split='test')

        train_cache = FeatureCache()
        train_cache.extract_and_cache(feature_extractor, train_dataset, 'train', device, batch_size=64)

        # osr17e: Load normalization stats from cache file (not from already-normalized features)
        # osr17d bug: recomputed mean/std from normalized features → ≈0/≈1, wrong for new extractions
        train_cache_data = torch.load(
            os.path.join('experiment/fewshot_cache', 'train_features.pt'), weights_only=True)
        train_norm_mean = train_cache_data.get('norm_mean', None)
        train_norm_std = train_cache_data.get('norm_std', None)
        if train_norm_mean is not None:
            print(f"  Loaded norm stats from cache: mean_range=[{train_norm_mean.min():.4f}, {train_norm_mean.max():.4f}] "
                  f"std_range=[{train_norm_std.min():.4f}, {train_norm_std.max():.4f}]")

        calib_cache = FeatureCache()
        calib_cache.extract_and_cache(feature_extractor, calib_dataset, 'calib', device, batch_size=64,
                                       norm_mean=train_norm_mean, norm_std=train_norm_std)

        test_cache = FeatureCache()
        test_cache.extract_and_cache(feature_extractor, test_dataset, 'test', device, batch_size=64,
                                      norm_mean=train_norm_mean, norm_std=train_norm_std)

        # --- t-SNE visualization: base vs novel feature distribution ---
        visualize_features_tsne(train_cache, test_cache,
                                base_classes, unknown_classes, experiment_dir)

        # ============ Phase 2: Episodic Meta-Training ============
        print(f"\n{'='*70}")
        print("Phase 2: Episodic Meta-Training")
        print(f"{'='*70}")

        flow_classifier = EpisodicFlowClassifier(
            input_dim=feature_dim,
            condition_dim=feature_dim,
            use_flow_transform=False,  # osr22: 纯聚类OSR，无Flow变换
        )

        # OOD exposure: provide unknown class features for open-set training
        ood_features = {}
        for c in unknown_classes:
            feats = test_cache.get_class_features(c)
            if len(feats) > 0:
                ood_features[c] = feats
        print(f"OOD exposure: {len(ood_features)} unknown classes, "
              f"{sum(len(v) for v in ood_features.values())} total samples")

        trainer = EpisodicTrainer(
            feature_extractor=feature_extractor,
            flow_classifier=flow_classifier,
            train_cache=train_cache,
            calib_cache=calib_cache,
            test_cache=test_cache,
            base_classes=base_classes,
            unknown_classes=unknown_classes,
            ood_features_by_class=ood_features,
            N_way=N_way,
            K_shot=K_shot,
            Q_query=Q_query,
            lr=5e-5,
            proto_noise_std=0.15,
            ood_ratio=0.3,
            warmup_episodes=200,
            task_aug_drop_rate=0.2,
            gradient_accum_steps=gradient_accum_steps,
            device=device
        )

        trainer.train(
            num_episodes=num_episodes,
            eval_every=500,  # 减少验证频率 (150→500)
            num_val_episodes=10,  # 减少验证episode数 (30→10)
            save_dir=experiment_dir,
            resume=True  # Resume from checkpoint if exists
        )

    else:
        # ============ Eval-only: load from checkpoint & cache ============
        print(f"\n{'='*70}")
        print("EVAL-ONLY MODE: Loading checkpoint & cached features")
        print(f"{'='*70}")

        # Determine checkpoint path
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            ckpt_path = os.path.join(experiment_dir, 'fewshot_final.pth')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Use --checkpoint <path> to specify, or run training first.")
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Reconstruct flow classifier from checkpoint
        flow_classifier = EpisodicFlowClassifier(
            input_dim=feature_dim,
            condition_dim=feature_dim,
            use_flow_transform=False,
        )
        flow_classifier.load_state_dict(ckpt['flow_state_dict'])
        flow_classifier = flow_classifier.to(device)
        print(f"Flow classifier loaded (epoch from checkpoint config)")

        # Load cached features
        cache_dir = 'experiment/fewshot_cache'
        for split_name in ['train', 'calib', 'test']:
            fpath = os.path.join(cache_dir, f'{split_name}_features.pt')
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Feature cache not found: {fpath}\n"
                    f"Run full training first to generate caches.")

        train_cache = FeatureCache()
        train_cache.load(os.path.join(cache_dir, 'train_features.pt'))

        calib_cache = FeatureCache()
        calib_cache.load(os.path.join(cache_dir, 'calib_features.pt'))

        test_cache = FeatureCache()
        test_cache.load(os.path.join(cache_dir, 'test_features.pt'))

        print(f"Caches loaded: train={len(train_cache.features)}, "
              f"calib={len(calib_cache.features)}, test={len(test_cache.features)}")

        # Reconstruct a minimal trainer (needed by FewShotEvaluator only)
        trainer = EpisodicTrainer(
            feature_extractor=feature_extractor,
            flow_classifier=flow_classifier,
            train_cache=train_cache,
            calib_cache=calib_cache,
            test_cache=test_cache,
            base_classes=base_classes,
            unknown_classes=unknown_classes,
            ood_features_by_class={},
            N_way=N_way,
            K_shot=K_shot,
            Q_query=Q_query,
            lr=0,  # not used in eval
            device=device
        )

    # ============ Phase 3: Evaluation & OSR Calibration ============
    print(f"\n{'='*70}")
    print("Phase 3: Evaluation & OSR Calibration")
    print(f"{'='*70}")

    # --- Few-shot classification accuracy ---
    evaluator = FewShotEvaluator(trainer.flow_classifier, device)

    print("\n--- Base class few-shot evaluation (train cache) ---")
    base_results = {}
    for k in [1, 5, 10]:
        base_results[k] = evaluator.evaluate(
            train_cache, base_classes, N_way=N_way, K_shot=k)

    print("\n--- Novel class few-shot evaluation (test cache) ---")
    novel_results = {}
    novel_N = min(len(unknown_classes), N_way)  # unknown类不够时自动降N
    for k in [1, 5, 10]:
        novel_results[k] = evaluator.evaluate(
            test_cache, unknown_classes, N_way=novel_N, K_shot=k)

    # --- OSR calibration & evaluation: feature-space methods only ---
    osr_results = {}
    # Core baselines (kept from osr20a)
    osr_methods = ['feature_mahalanobis', 'anti_prototype', 'geo_fusion',
                   'ood_head', 'ood_head_extended']

    for method in osr_methods:
        print(f"\n--- OSR {method.upper()} calibration (calib data) ---")
        calibrator = OSRCalibrator(
            trainer.flow_classifier, calib_cache,
            base_classes, unknown_classes, device)
        threshold = calibrator.calibrate(target_fpr=0.05, K_shot=50, method=method)

        print(f"\n--- OSR {method.upper()} evaluation (test data, per-round recalib) ---")
        test_calibrator = OSRCalibrator(
            trainer.flow_classifier, test_cache,
            base_classes, unknown_classes, device)
        osr_results[method] = test_calibrator.evaluate_osr(
            threshold=threshold, K_shot=50, num_rounds=10,
            method=method, recalibrate_per_round=True)

    # osr20a winner: anti_prototype with per-class threshold
    per_class_methods = ['anti_prototype', 'ood_head_extended']
    for method in per_class_methods:
        tag = f"{method}_per_class"
        print(f"\n--- OSR {tag.upper()} calibration (calib data) ---")
        calibrator = OSRCalibrator(
            trainer.flow_classifier, calib_cache,
            base_classes, unknown_classes, device)
        calibrator.calibrate(target_fpr=0.05, K_shot=50, method=method)

        print(f"\n--- OSR {tag.upper()} evaluation (test data, per-class threshold) ---")
        test_calibrator = OSRCalibrator(
            trainer.flow_classifier, test_cache,
            base_classes, unknown_classes, device)
        osr_results[tag] = test_calibrator.evaluate_osr(
            K_shot=50, num_rounds=10,
            method=method, recalibrate_per_round=True,
            use_per_class_threshold=True)

    # osr20b direction 1: Ensemble anti_prototype + ood_head_extended
    print(f"\n--- OSR ENSEMBLE_ANTI_OODEXT calibration (calib data) ---")
    calibrator = OSRCalibrator(
        trainer.flow_classifier, calib_cache,
        base_classes, unknown_classes, device)
    calibrator.calibrate(target_fpr=0.05, K_shot=50, method='ensemble_anti_oodext')

    print(f"\n--- OSR ENSEMBLE_ANTI_OODEXT evaluation (test data) ---")
    test_calibrator = OSRCalibrator(
        trainer.flow_classifier, test_cache,
        base_classes, unknown_classes, device)
    osr_results['ensemble_anti_oodext'] = test_calibrator.evaluate_osr(
        K_shot=50, num_rounds=10,
        method='ensemble_anti_oodext', recalibrate_per_round=True)

    # osr20b: ensemble + per-class threshold
    print(f"\n--- OSR ensemble_anti_oodext_per_class evaluation ---")
    test_calibrator = OSRCalibrator(
        trainer.flow_classifier, test_cache,
        base_classes, unknown_classes, device)
    osr_results['ensemble_anti_oodext_per_class'] = test_calibrator.evaluate_osr(
        K_shot=50, num_rounds=10,
        method='ensemble_anti_oodext', recalibrate_per_round=True,
        use_per_class_threshold=True)

    # ============ osr22 new methods: clustering-augmented OSR ============

    # osr22a: Multi-prototype anti_prototype (K-means sub-clusters)
    print(f"\n--- OSR ANTI_PROTOTYPE_MULTI calibration (calib data) ---")
    calibrator = OSRCalibrator(
        trainer.flow_classifier, calib_cache,
        base_classes, unknown_classes, device)
    calibrator._multi_proto_K_sub = 2
    calibrator.calibrate(target_fpr=0.05, K_shot=50, method='anti_prototype_multi')

    print(f"\n--- OSR ANTI_PROTOTYPE_MULTI evaluation (test data) ---")
    test_calibrator = OSRCalibrator(
        trainer.flow_classifier, test_cache,
        base_classes, unknown_classes, device)
    test_calibrator._multi_proto_K_sub = 2
    osr_results['anti_prototype_multi'] = test_calibrator.evaluate_osr(
        K_shot=50, num_rounds=10,
        method='anti_prototype_multi', recalibrate_per_round=True)

    # osr22b: Extended OOD head v2 (with GMM cluster features)
    print(f"\n--- OSR OOD_HEAD_EXTENDED_V2 calibration (calib data) ---")
    calibrator = OSRCalibrator(
        trainer.flow_classifier, calib_cache,
        base_classes, unknown_classes, device)
    calibrator.calibrate(target_fpr=0.05, K_shot=50, method='ood_head_extended_v2')

    print(f"\n--- OSR OOD_HEAD_EXTENDED_V2 evaluation (test data) ---")
    test_calibrator = OSRCalibrator(
        trainer.flow_classifier, test_cache,
        base_classes, unknown_classes, device)
    osr_results['ood_head_extended_v2'] = test_calibrator.evaluate_osr(
        K_shot=50, num_rounds=10,
        method='ood_head_extended_v2', recalibrate_per_round=True)

    # osr22c: Adaptive ensemble fusion (density-based alpha)
    print(f"\n--- OSR ENSEMBLE_ADAPTIVE calibration (calib data) ---")
    calibrator = OSRCalibrator(
        trainer.flow_classifier, calib_cache,
        base_classes, unknown_classes, device)
    calibrator.calibrate(target_fpr=0.05, K_shot=50, method='ensemble_adaptive')

    print(f"\n--- OSR ENSEMBLE_ADAPTIVE evaluation (test data) ---")
    test_calibrator = OSRCalibrator(
        trainer.flow_classifier, test_cache,
        base_classes, unknown_classes, device)
    osr_results['ensemble_adaptive'] = test_calibrator.evaluate_osr(
        K_shot=50, num_rounds=10,
        method='ensemble_adaptive', recalibrate_per_round=True)

    # osr22d: Cluster boundary detection
    print(f"\n--- OSR CLUSTER_BOUNDARY calibration (calib data) ---")
    calibrator = OSRCalibrator(
        trainer.flow_classifier, calib_cache,
        base_classes, unknown_classes, device)
    calibrator.calibrate(target_fpr=0.05, K_shot=50, method='cluster_boundary')

    print(f"\n--- OSR CLUSTER_BOUNDARY evaluation (test data) ---")
    test_calibrator = OSRCalibrator(
        trainer.flow_classifier, test_cache,
        base_classes, unknown_classes, device)
    osr_results['cluster_boundary'] = test_calibrator.evaluate_osr(
        K_shot=50, num_rounds=10,
        method='cluster_boundary', recalibrate_per_round=True)

    # osr22d: Three-way ensemble (anti_proto + ood_head_v2 + cluster_boundary)
    print(f"\n--- OSR ENSEMBLE_CLUSTER calibration (calib data) ---")
    calibrator = OSRCalibrator(
        trainer.flow_classifier, calib_cache,
        base_classes, unknown_classes, device)
    calibrator.calibrate(target_fpr=0.05, K_shot=50, method='ensemble_cluster')

    print(f"\n--- OSR ENSEMBLE_CLUSTER evaluation (test data) ---")
    test_calibrator = OSRCalibrator(
        trainer.flow_classifier, test_cache,
        base_classes, unknown_classes, device)
    osr_results['ensemble_cluster'] = test_calibrator.evaluate_osr(
        K_shot=50, num_rounds=10,
        method='ensemble_cluster', recalibrate_per_round=True)

    # Summary comparison
    print(f"\n{'='*70}")
    print("OSR Method Comparison (test data, per-round recalibration)")
    print(f"{'='*70}")
    print(f"  {'Method':<35s} {'TNR':>8s} {'TPR':>8s} {'OSR':>8s} {'AUROC':>8s}")
    for m, r in osr_results.items():
        auroc_str = f"{r['auroc']:.4f}" if 'auroc' in r else '  N/A '
        print(f"  {m:<35s} {r['known_tnr']:>7.2%} {r['unknown_tpr']:>7.2%} {r['osr_score']:>7.2%} {auroc_str:>8s}")

    # ============ Save Final Model ============
    best_method = max(osr_results, key=lambda m: osr_results[m]['osr_score'])
    save_dir = experiment_dir
    final_path = os.path.join(save_dir, 'fewshot_final.pth')
    torch.save({
        'flow_state_dict': trainer.flow_classifier.state_dict(),
        'osr_threshold': osr_results[best_method].get('threshold', 0),
        'config': {
            'input_dim': feature_dim,
            'condition_dim': feature_dim,
            'base_classes': base_classes,
            'unknown_classes': unknown_classes,
            'scoring_method': best_method,
            'use_flow_transform': False,  # osr22: clustering-based OSR, no Flow transform
        },
        'base_results': base_results,
        'novel_results': novel_results,
        'osr_results': osr_results,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")

    print(f"\n{'='*70}")
    print("Pipeline complete!")
    print(f"{'='*70}")

    # ============ Generate Analysis Plots ============
    try:
        from plot_training import generate_all_plots
        import glob
        # Find the latest log (redirected output or default)
        log_candidates = glob.glob('TAU22_*_osr*.log')
        if log_candidates:
            latest_log = max(log_candidates, key=os.path.getmtime)
            generate_all_plots(latest_log, experiment_dir)
    except Exception as e:
        print(f"Plot generation skipped: {e}")

    return trainer, calibrator, evaluator


if __name__ == "__main__":
    trainer, calibrator, evaluator = main()
