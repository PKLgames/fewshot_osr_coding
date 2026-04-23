#!/usr/local/miniconda3/envs/py312/bin/python
"""
Episodic Meta-Trainer for Few-Shot Open Set Recognition

Three-phase training pipeline:
  Phase 1: Pre-extract features using backbone + FC (frozen)
  Phase 2: Episodic meta-training of conditional flow (cINN)
  Phase 3: Calibrate OSR likelihood threshold

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
from Reversable_Function import NSF as NSF_module


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
            scaler = torch.amp.GradScaler('cuda', enabled=False)

            running_loss = 0.0
            correct = 0
            total = 0

            for item in loader:
                audio = item['source_audio'].to(self.device)
                targets = item['target'].squeeze(1)  # (B, 10)
                labels = self._remap_label(targets).to(self.device)

                with torch.amp.autocast('cuda'):
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
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad] +
                    list(self.classifier.parameters()),
                    max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

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
                 backbone_unfreeze_layers: List[str] = None):
        self.model = feature_extractor.to(device)
        self.device = device
        self.temperature = temperature
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.backbone_unfreeze_layers = backbone_unfreeze_layers or []

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
        z = F.normalize(z, dim=1)

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

    def train(self, dataset, save_path: str = None):
        """Run contrastive pretraining on the full dataset (unsupervised)."""
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
            prefetch_factor=self.prefetch_factor)

        print(f"\n{'='*70}")
        print("Phase 0a: Contrastive Pre-training (SimCLR)")
        print(f"{'='*70}")
        print(f"Samples:         {len(loader.dataset)}")
        print(f"Epochs:          {self.epochs}")
        print(f"Batch size:      {self.batch_size}")
        print(f"Temperature:     {self.temperature}")
        print(f"Backbone unfreeze: {self.backbone_unfreeze_layers}")
        print(f"{'='*70}\n")

        augment = AudioAugmentation(sample_rate=16000)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.projection.train()
            scaler = torch.amp.GradScaler('cuda', enabled=False)

            running_loss = 0.0
            num_batches = 0

            for item in loader:
                audio = item['source_audio'].to(self.device)

                # Two independently augmented views
                view1 = augment(audio)
                view2 = augment(audio)

                with torch.amp.autocast('cuda'):
                    # Encode
                    h1 = self.model(view1)          # (B, 64)
                    h2 = self.model(view2)          # (B, 64)

                    # Project
                    z1 = self.projection(h1)        # (B, proj_dim)
                    z2 = self.projection(h2)        # (B, proj_dim)

                    loss = self.nt_xent_loss(z1, z2, self.temperature)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) +
                    list(self.projection.parameters()),
                    max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item()
                num_batches += 1

            self.scheduler.step()
            avg_loss = running_loss / num_batches
            lr = self.optimizer.param_groups[-1]['lr']
            print(f"Epoch {epoch:>2d}/{self.epochs} | "
                  f"Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            backbone = getattr(self.model, 'BACKBONE_TYPE', 'yamnet')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': {
                    'backbone': backbone,
                    'pretrained_on': 'contrastive_simclr',
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

    CACHE_VERSION = 'v19_distil_ast_panns'  # Distil-AST + PANNs Cnn14_16k

    def __init__(self, cache_dir: str = 'experiment/fewshot_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.features_by_class: Dict[int, torch.Tensor] = {}
        self.labels: Optional[torch.Tensor] = None
        self.features: Optional[torch.Tensor] = None
        self.class_list: List[int] = []

    def extract_and_cache(self, feature_extractor: nn.Module,
                          dataset, split: str, device: str,
                          batch_size: int = 512):
        """Pre-extract features from dataset and save to disk"""
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

                    feats = feature_extractor(audio)          # (B, 32)
                    all_features.append(feats.cpu())
                    all_labels.append(labels.cpu())

            self.features = torch.cat(all_features, dim=0)
            self.labels = torch.cat(all_labels, dim=0)

            torch.save(
                {'features': self.features, 'labels': self.labels,
                 'version': self.CACHE_VERSION},
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
# 4. Episodic Flow Classifier
# ============================================================

class EpisodicFlowClassifier(nn.Module):
    """
    Hybrid classifier: prototypical distance + optional conditional flow density.

    When use_flow=True (legacy):
      Two parallel scoring paths: distance head + flow (cINN)
    When use_flow=False (recommended):
      Distance head only, no flow. OOD detection via feature-space methods.

    classify() returns combined scores for training.
    forward() returns flow-only log-likelihoods for OOD scoring (if use_flow).
    """

    def __init__(self, input_dim: int = 64, condition_dim: int = 64,
                 num_coupling_layers: int = 8,
                 hidden_dims: List[int] = None,
                 use_projection: bool = True,
                 s_clamp_max: float = 3.0,
                 flow_dim: int = 32,
                 use_flow: bool = False):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.use_projection = use_projection
        self.flow_dim = flow_dim
        self.use_flow = use_flow

        # Feature adapter: trainable transform to refine frozen cached features
        # Learns a class-agnostic rotation/scaling of the feature space
        # that improves few-shot discrimination for both base and novel classes
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        if use_flow:
            # Flow dimension adapter: projects 64-dim features to lower-dim for flow
            self.flow_dim_adapter = nn.Sequential(
                nn.Linear(input_dim, flow_dim),
                nn.LayerNorm(flow_dim),
            )

            # Pre-normalization: stabilize flow input distribution before projection
            self.flow_pre_norm = nn.LayerNorm(flow_dim)

            # Learnable projection: adapts flow-dim features for density estimation
            if use_projection:
                self.projection = nn.Sequential(
                    nn.Linear(flow_dim, flow_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(flow_dim * 2, flow_dim),
                )

            # Condition network: transforms prototype conditions before entering flow
            # Gives flow richer condition info without modifying NSF internals
            self.condition_network = nn.Sequential(
                nn.Linear(flow_dim, flow_dim * 2),
                nn.ReLU(),
                nn.Linear(flow_dim * 2, flow_dim),
                nn.LayerNorm(flow_dim),
            )

            self.flow = NSF_module.ConditionalNSF(
                input_dim=flow_dim,
                condition_dim=flow_dim,
                num_coupling_layers=num_coupling_layers,
                hidden_dims=hidden_dims,
                num_bins=8,
                bound=5.0,
                use_permutation=True,
                permutation_type='fixed',
                dropout=0.15,
                s_clamp_max=s_clamp_max
            )

            # Background flow: unconditional density model for likelihood ratio OSR
            self.bg_flow = NSF_module.ConditionalNSF(
                input_dim=flow_dim,
                condition_dim=1,
                num_coupling_layers=4,
                hidden_dims=[128, 128],
                num_bins=8,
                bound=5.0,
                use_permutation=True,
                permutation_type='fixed',
                dropout=0.1,
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

        # Binary OOD detection head: operates on feature-space statistics
        # Input: [min_dist, dist_ratio, softmax_max, entropy, feat_norm] = 5 features
        self.ood_head = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),  # logit for known(1) / unknown(0)
        )

    def _prepare_flow_features(self, features: torch.Tensor,
                                prototypes: torch.Tensor):
        """Prepare features for flow: adapt → dim-reduce → pre-norm → project.
        Returns (proj_feats, proj_protos) both in flow_dim space."""
        if not self.use_flow:
            raise RuntimeError("_prepare_flow_features called with use_flow=False")
        adapted_feats, adapted_protos = self.adapt_features(features, prototypes)
        flow_feats = self.flow_dim_adapter(adapted_feats)
        flow_protos = self.flow_dim_adapter(adapted_protos)
        flow_feats = self.flow_pre_norm(flow_feats)
        flow_protos = self.flow_pre_norm(flow_protos)
        if self.use_projection:
            flow_feats = self.projection(flow_feats)
            flow_protos = self.projection(flow_protos)
        # Transform prototype conditions via condition network (gives flow richer conditioning)
        flow_protos = self.condition_network(flow_protos)
        return flow_feats, flow_protos

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Apply flow_dim_adapter + projection head.
        When use_flow=False, returns adapted features directly."""
        if not self.use_flow:
            return self.feature_adapter(x)
        x = self.flow_dim_adapter(self.feature_adapter(x))
        if self.use_projection:
            return self.projection(x)
        return x

    def adapt_features(self, features: torch.Tensor,
                       prototypes: torch.Tensor):
        """Apply feature adapter to both features and prototypes."""
        adapted_feats = self.feature_adapter(features)
        adapted_protos = self.feature_adapter(prototypes)
        return adapted_feats, adapted_protos

    def _compute_flow_log_probs(self, proj_feats: torch.Tensor,
                                proj_protos: torch.Tensor) -> torch.Tensor:
        """Compute flow log-likelihoods from already-projected (flow_dim) features."""
        if not self.use_flow:
            raise RuntimeError("_compute_flow_log_probs called with use_flow=False")
        B = proj_feats.size(0)
        N = proj_protos.size(0)
        feats_exp = proj_feats.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        conds_exp = proj_protos.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
        log_probs = self.flow.log_prob(feats_exp, conds_exp).reshape(B, N)
        return log_probs

    def _compute_flow_with_z(self, proj_feats: torch.Tensor,
                              proj_protos: torch.Tensor):
        """Compute flow log-likelihoods AND latent z values in one pass."""
        if not self.use_flow:
            raise RuntimeError("_compute_flow_with_z called with use_flow=False")
        B = proj_feats.size(0)
        N = proj_protos.size(0)
        D = proj_feats.size(1)
        feats_exp = proj_feats.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        conds_exp = proj_protos.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
        z_all, log_det = self.flow.forward(
            feats_exp, conds_exp, compute_jacobian=True)
        prior = torch.distributions.Normal(
            torch.zeros(D, device=proj_feats.device),
            torch.ones(D, device=proj_feats.device))
        log_probs = (prior.log_prob(z_all).sum(dim=1) + log_det).reshape(B, N)
        z_all = z_all.reshape(B, N, D)
        return log_probs, z_all

    def forward(self, features: torch.Tensor,
                prototypes: torch.Tensor) -> torch.Tensor:
        """
        Flow-only log-likelihoods (for OOD scoring in Phase 3).
        Returns: log_probs (B, N). Only available when use_flow=True.
        """
        if not self.use_flow:
            raise RuntimeError("forward() requires use_flow=True")
        proj_feats, proj_protos = self._prepare_flow_features(features, prototypes)
        return self._compute_flow_log_probs(proj_feats, proj_protos)

    def classify(self, features: torch.Tensor,
                 prototypes: torch.Tensor) -> torch.Tensor:
        """
        Classification scores (for training and inference).

        When use_flow=True: combined prototypical distance + flow density.
        When use_flow=False: prototypical distance only.

        Returns: scores (B, N)  higher = more likely that class
        """
        # --- Feature adaptation (refine cached features) ---
        features, prototypes = self.adapt_features(features, prototypes)

        # --- Prototypical distance path (64-dim) ---
        dist_feats = self.distance_head(features)     # (B, D)
        dist_protos = self.distance_head(prototypes)   # (N, D)
        dists = torch.cdist(dist_feats, dist_protos, p=2) ** 2  # (B, N)
        proto_scores = -dists * self.temperature.abs()           # (B, N)

        if self.use_flow:
            # --- Flow density path (flow_dim) ---
            proj_feats, proj_protos = self._prepare_flow_features(features, prototypes)
            flow_lp = self._compute_flow_log_probs(proj_feats, proj_protos)
            # Center both per-sample for comparable scales, then sum
            proto_centered = proto_scores - proto_scores.mean(dim=1, keepdim=True)
            flow_centered = flow_lp - flow_lp.mean(dim=1, keepdim=True)
            combined = proto_centered + flow_centered
        else:
            combined = proto_scores

        # Smooth clamp: prevents extreme scores → CE explosions
        combined = 10.0 * torch.tanh(combined / 10.0)
        return combined

    def classify_with_z(self, features: torch.Tensor,
                        prototypes: torch.Tensor):
        """
        Same as classify() but also returns latent z values (when use_flow).
        Returns: (combined_scores (B, N), z_all (B, N, D) or None)
        """
        # --- Feature adaptation (refine cached features) ---
        features, prototypes = self.adapt_features(features, prototypes)

        # --- Prototypical distance path (64-dim) ---
        dist_feats = self.distance_head(features)
        dist_protos = self.distance_head(prototypes)
        dists = torch.cdist(dist_feats, dist_protos, p=2) ** 2
        proto_scores = -dists * self.temperature.abs()

        if self.use_flow:
            proj_feats, proj_protos = self._prepare_flow_features(features, prototypes)
            flow_lp, z_all = self._compute_flow_with_z(proj_feats, proj_protos)
            proto_centered = proto_scores - proto_scores.mean(dim=1, keepdim=True)
            flow_centered = flow_lp - flow_lp.mean(dim=1, keepdim=True)
            combined = proto_centered + flow_centered
            combined = 10.0 * torch.tanh(combined / 10.0)
            return combined, z_all
        else:
            combined = 10.0 * torch.tanh(proto_scores / 10.0)
            return combined, None

    def forward_to_latent(self, features: torch.Tensor,
                          prototypes: torch.Tensor) -> torch.Tensor:
        """Transform features to latent z-space conditioned on each prototype.
        Only available when use_flow=True."""
        if not self.use_flow:
            raise RuntimeError("forward_to_latent() requires use_flow=True")
        proj_feats, proj_protos = self._prepare_flow_features(features, prototypes)
        B = proj_feats.size(0)
        N = proj_protos.size(0)
        D = proj_feats.size(1)
        feats_exp = proj_feats.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        conds_exp = proj_protos.unsqueeze(0).expand(B, -1, -1).reshape(B * N, -1)
        z_all, _ = self.flow.forward(feats_exp, conds_exp, compute_jacobian=False)
        return z_all.reshape(B, N, D)

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

    def train_background_flow(self, base_features: torch.Tensor,
                               device: str = 'cuda',
                               epochs: int = 50,
                               lr: float = 1e-3,
                               batch_size: int = 512,
                               verbose: bool = True):
        """Train unconditional background flow on base class features.
        Only available when use_flow=True.
        """
        self.eval()
        # Project base features through the same pipeline
        all_proj = []
        with torch.no_grad():
            for i in range(0, len(base_features), batch_size):
                batch = base_features[i:i+batch_size].to(device)
                adapted = self.feature_adapter(batch)
                flow_feats = self.flow_dim_adapter(adapted)
                flow_feats = self.flow_pre_norm(flow_feats)
                if self.use_projection:
                    flow_feats = self.projection(flow_feats)
                all_proj.append(flow_feats.cpu())
        proj_feats = torch.cat(all_proj, dim=0).to(device)

        if verbose:
            print(f"\n  Background flow training: {len(proj_feats)} samples, {epochs} epochs")

        optimizer = torch.optim.Adam(self.bg_flow.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)
        dummy_cond = torch.zeros(batch_size, 1, device=device)

        self.bg_flow.train()
        for epoch in range(1, epochs + 1):
            # Shuffle
            perm = torch.randperm(len(proj_feats), device=device)
            total_ll = 0.0
            n_batches = 0
            for i in range(0, len(proj_feats), batch_size):
                batch = proj_feats[perm[i:i+batch_size]]
                bs = batch.size(0)
                cond = torch.zeros(bs, 1, device=device)
                log_probs = self.bg_flow.log_prob(batch, cond)
                loss = -log_probs.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.bg_flow.parameters(), 1.0)
                optimizer.step()

                total_ll += log_probs.mean().item()
                n_batches += 1

            scheduler.step()
            if verbose and (epoch % 10 == 0 or epoch == 1):
                avg_ll = total_ll / n_batches
                print(f"    BG Epoch {epoch:>2d}/{epochs} | "
                      f"log p(x): {avg_ll:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        self.bg_flow.eval()
        if verbose:
            print("  Background flow trained.")

    def score_likelihood_ratio(self, features: torch.Tensor,
                                prototypes: torch.Tensor) -> torch.Tensor:
        """Likelihood ratio OOD score: log p(x|class) - log p(x|background).
        Higher = more likely known. Only available when use_flow=True."""
        if not self.use_flow:
            raise RuntimeError("score_likelihood_ratio() requires use_flow=True")
        # Class-conditional log probs (B, N)
        log_probs_class = self.forward(features, prototypes)

        # Background log probs (B,)
        proj_feats, proj_protos = self._prepare_flow_features(features, prototypes)
        B = proj_feats.size(0)
        dummy_cond = torch.zeros(B, 1, device=proj_feats.device)
        log_probs_bg = self.bg_flow.log_prob(proj_feats, dummy_cond)  # (B,)

        # Likelihood ratio: subtract background
        return log_probs_class - log_probs_bg.unsqueeze(1)

    @staticmethod
    def l2_normalize(features: torch.Tensor,
                     prototypes: torch.Tensor):
        """L2 normalize features and prototypes (inference only).
        Converts Euclidean distance to cosine distance for better transfer."""
        return (F.normalize(features, p=2, dim=1),
                F.normalize(prototypes, p=2, dim=1))

    def compute_energy(self, features: torch.Tensor,
                       prototypes: torch.Tensor,
                       temperature: float = 1.0) -> torch.Tensor:
        """
        Energy-based OOD score: E(x) = -T * log(sum_c exp(log p(x|c) / T))

        Lower energy → more likely known.  Higher energy → more likely unknown.
        Returns -Energy (higher = more known) for threshold consistency.

        Args:
            features:   (B, D)
            prototypes: (N, D)
            temperature: softmax temperature
        Returns:
            neg_energy: (B,) higher = more likely known
        """
        log_probs = self.forward(features, prototypes)  # (B, N)
        # neg_energy = T * logsumexp(log_probs / T)
        neg_energy = temperature * torch.logsumexp(log_probs / temperature, dim=1)
        return neg_energy


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
      5. L_pseudo_novel   - push log-likelihood DOWN for pseudo-novel samples
      6. L_latent_repel   - push z_wrong away from z_correct in flow latent space
      7. L_gaussian       - explicit: push z_correct toward N(0,I)
      8. L_non_gaussian   - explicit: push z_wrong away from N(0,I) norm
    """

    def __init__(self, N_way: int = 5,
                 lambda_density: float = 0.1,
                 lambda_entropy: float = 0.05,
                 lambda_separation: float = 0.02,
                 lambda_pseudo_novel: float = 0.05,
                 lambda_latent_repel: float = 0.1,
                 latent_repel_margin: float = 64.0,
                 lambda_gaussian: float = 0.1,
                 lambda_non_gaussian: float = 0.05,
                 lambda_z_contrastive: float = 0.05,
                 lambda_z_contrastive_known: float = 0.7,
                 lambda_z_contrastive_unknown: float = 0.3,
                 z_contrastive_temperature: float = 0.1,
                 z_contrastive_margin_range: Tuple[float, float] = (1.5, 3.0),
                 feature_dim: int = 64,
                 ce_cap: float = 10.0):
        super().__init__()
        self.N_way = N_way
        self.ce_cap = ce_cap
        self.lambda_density = lambda_density
        self.lambda_entropy = lambda_entropy
        self.lambda_separation = lambda_separation
        self.lambda_pseudo_novel = lambda_pseudo_novel
        self.lambda_latent_repel = lambda_latent_repel
        self.latent_repel_margin = latent_repel_margin
        self.lambda_gaussian = lambda_gaussian
        self.lambda_non_gaussian = lambda_non_gaussian
        self.lambda_z_contrastive = lambda_z_contrastive
        self.lambda_z_contrastive_known = lambda_z_contrastive_known
        self.lambda_z_contrastive_unknown = lambda_z_contrastive_unknown
        self.z_contrastive_temperature = z_contrastive_temperature
        self.z_contrastive_margin_range = z_contrastive_margin_range
        self.feature_dim = feature_dim

    def forward(self, log_probs: torch.Tensor,
                query_labels: torch.Tensor,
                prototypes: torch.Tensor = None,
                pseudo_novel_log_probs: torch.Tensor = None,
                z_all: torch.Tensor = None,
                aux_scale: float = 1.0,
                scale_cls: float = 1.0,
                scale_struct: float = 1.0,
                scale_gaussian: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            log_probs:    (B, N) conditional log-probabilities from flow
            query_labels: (B,) ground truth labels 0..N-1
            prototypes:   (N, D) class prototypes (for separation loss)
            pseudo_novel_log_probs: (B_pseudo, N) log-probs for pseudo-novel samples
            z_all:        (B, N, D) latent z values from flow (for LCR loss)
            aux_scale:    legacy single scale (used when scale_cls/sange_struct not set)
            scale_cls:    scale for classification helper losses (density, entropy, separation)
            scale_struct: scale for structure + OSR losses (pseudo_novel, latent_repel, non_gaussian)
            scale_gaussian: separate scale for gaussian prior (starts from ep 1, not delayed)
        Returns:
            total_loss, info dict
        """
        B, N = log_probs.shape

        # 1. Classification loss with smooth tanh cap to prevent gradient explosions
        log_preds = F.log_softmax(log_probs, dim=1)
        L_ce_raw = F.nll_loss(log_preds, query_labels)
        L_ce = self.ce_cap * torch.tanh(L_ce_raw / self.ce_cap)

        # 2. Density: 推动目标类密度上升，但 clamp 防止无界增长
        target_lp = log_probs.gather(1, query_labels.unsqueeze(1)).mean()
        L_density = torch.clamp(-target_lp, min=-5.0, max=5.0)

        # 3. Entropy: 鼓励自信的预测分布
        probs = F.softmax(log_probs, dim=1)
        entropy = -(probs * log_preds).sum(dim=1).mean()
        L_entropy = entropy

        # 4. Prototype separation: maximize pairwise distance between prototypes
        L_separation = torch.tensor(0.0, device=log_probs.device)
        if prototypes is not None and prototypes.size(0) > 1:
            pdist = torch.pdist(prototypes)  # pairwise distances
            L_separation = torch.clamp(-pdist.mean(), min=-2.0, max=0.0)

        # 5. Pseudo-novel repulsion: minimize logsumexp(log_probs) for pseudo-novels
        #    Clamped to prevent runaway gradients (flow log-likelihoods can be unbounded)
        L_pseudo = torch.tensor(0.0, device=log_probs.device)
        if pseudo_novel_log_probs is not None and pseudo_novel_log_probs.size(0) > 0:
            raw_energy = torch.logsumexp(
                pseudo_novel_log_probs.clamp(max=10.0), dim=1)
            L_pseudo = torch.clamp(raw_energy, min=-5.0, max=5.0).mean()

        # 6. Latent Contrastive Repulsion (LCR): push z_wrong away from z_correct
        #    In flow latent z-space, z_correct ~ N(0,I) but z_wrong is unconstrained.
        #    LCR explicitly forces z_wrong to be FAR from z_correct by margin.
        L_latent_repel = torch.tensor(0.0, device=log_probs.device)
        if z_all is not None and z_all.size(1) > 1:
            # z_all: (B, N, D), query_labels: (B,)
            z_correct = z_all[torch.arange(B, device=z_all.device), query_labels]  # (B, D)
            # Expand for pairwise comparison: (B, N, D)
            z_correct_exp = z_correct.unsqueeze(1).expand_as(z_all)
            # Squared distance between each z_c and z_correct
            dist_sq = ((z_all - z_correct_exp) ** 2).sum(dim=2)  # (B, N)
            # Mask: zero out the correct class (we only penalize wrong classes)
            mask = torch.ones(B, N, device=z_all.device)
            mask[torch.arange(B, device=z_all.device), query_labels] = 0.0
            # Hinge loss: penalize if dist < margin
            repel = torch.clamp(self.latent_repel_margin - dist_sq, min=0.0)  # (B, N)
            L_latent_repel = (repel * mask).sum() / mask.sum().clamp(min=1.0)

        # 7. L_gaussian: explicit prior matching — push z_correct → N(0, I)
        #    Two components:
        #    a) Norm matching: push mean ||z||² toward D (= feature_dim)
        #    b) Per-dimension std matching: push each dim's std toward 1.0
        L_gaussian = torch.tensor(0.0, device=log_probs.device)
        if z_all is not None:
            z_correct = z_all[torch.arange(B, device=z_all.device), query_labels]  # (B, D)
            D = self.feature_dim

            # 7a. Norm matching: minimize |mean(||z||²) - D| / D
            #     When z ~ N(0,I), E[||z||²] = D, so this term → 0
            z_norm_sq = (z_correct ** 2).sum(dim=1)  # (B,)
            L_norm = (z_norm_sq.mean() - D).abs() / D

            # 7b. Per-dimension std matching: push each dim std toward 1.0
            #     Under N(0,I), each dim has std=1.0. When all dims are std=1.0, total
            #     std is also correct. This prevents z from collapsing to few active dims.
            z_std_per_dim = z_correct.std(dim=0)  # (D,) std per dimension across batch
            L_std = ((z_std_per_dim - 1.0) ** 2).mean()

            L_gaussian = L_norm + L_std

        # 8. L_non_gaussian: push z_wrong AWAY from N(0,I) norm
        #    Under N(0,I), E[||z||^2] = D. We penalize z_wrong whose norm is
        #    close to D (i.e., looks Gaussian). Hinge: want ||z_wrong||^2 > 1.5*D
        #    IMPORTANT: only active when z_correct has converged (scale_struct > 0)
        #    to avoid conflicting with L_gaussian during early training
        L_non_gaussian = torch.tensor(0.0, device=log_probs.device)
        if z_all is not None and z_all.size(1) > 1 and scale_struct > 0:
            z_wrong_norm_sq = (z_all ** 2).sum(dim=2)  # (B, N)
            target = self.feature_dim * 1.5  # 50% above Gaussian mean
            penalty = torch.clamp(target - z_wrong_norm_sq, min=0.0)  # (B, N)
            mask = torch.ones(B, N, device=z_all.device)
            mask[torch.arange(B, device=z_all.device), query_labels] = 0.0
            L_non_gaussian = (penalty * mask).sum() / mask.sum().clamp(min=1.0) / self.feature_dim

        # 9. L_z_contrastive: Create tighter known clusters + more compact unknown regions
        #    For known samples: z_correct (from correct proto) pulled toward class centroid
        #    For unknown samples: all z_proto pushed apart from each other (sparse regions)
        L_z_contrastive = torch.tensor(0.0, device=log_probs.device)
        L_z_contrastive_known = torch.tensor(0.0, device=log_probs.device)
        L_z_contrastive_unknown = torch.tensor(0.0, device=log_probs.device)

        if z_all is not None and z_all.size(1) > 1:
            # z_all: (B, N, D) where z_all[:, c] = z conditioned on prototype c

            # For known samples: encourage z_correct to be close to its class prototype in z-space
            z_correct = z_all[torch.arange(B, device=z_all.device), query_labels]  # (B, D)

            # Compute class-specific prototype centroids
            class_centroids = []
            for c in range(self.N_way):
                mask = (query_labels == c)
                if mask.sum() > 0:
                    class_centroids.append(z_all[mask, c].mean(dim=0))
                else:
                    class_centroids.append(prototypes[c])
            class_centroids = torch.stack(class_centroids, dim=0)  # (N_way, D)

            # Pull z_correct toward its class centroid — makes known region compact
            target_centroids = class_centroids[query_labels]  # (B, D)
            pos_dist = ((z_correct - target_centroids) ** 2).sum(dim=1).sqrt()  # (B,)
            L_z_contrastive_known = pos_dist.mean()

            # For unknown: push all z_proto far apart (encourage sparse regions for OOD)
            # Use all negative prototypes for each sample for better coverage
            n_samples = B
            k_negatives = min(self.N_way - 1, 5)  # Use up to 5 negatives

            contrastive_loss = []
            for b in range(n_samples):
                # Get all negative class indices (excluding the correct class)
                correct_class = query_labels[b].item()
                neg_class_indices = [c for c in range(self.N_way) if c != correct_class]
                if not neg_class_indices:
                    continue
                neg_class_tensor = torch.tensor(neg_class_indices, device=z_all.device)
                if len(neg_class_tensor) > k_negatives:
                    perm = torch.randperm(len(neg_class_tensor), device=z_all.device)[:k_negatives]
                    neg_class_tensor = neg_class_tensor[perm]

                # Push z_correct away from negative prototype z values
                neg_z = z_all[b, neg_class_tensor, :]  # (k, D)
                neg_dist = torch.cdist(z_correct[b:b+1], neg_z, p=2)

                # Adaptive margin based on positive distance
                margin = pos_dist[b].item() * 2.0  # Scale margin with positive distance
                margin = max(self.z_contrastive_margin_range[0],
                             min(self.z_contrastive_margin_range[1], margin))

                # Triplet-style loss with temperature scaling
                loss = torch.clamp(margin - neg_dist.min(), min=0.0).mean()
                loss = loss / self.z_contrastive_temperature  # Scale for gradient stability
                contrastive_loss.append(loss)

            if contrastive_loss:
                L_z_contrastive_unknown = torch.stack(contrastive_loss).mean()
                # Apply weighting factors from configuration
                L_z_contrastive = (self.lambda_z_contrastive_known * L_z_contrastive_known +
                                 self.lambda_z_contrastive_unknown * L_z_contrastive_unknown)

        # Staged loss weighting:
        #   scale_cls:      L_density + L_entropy + L_separation (classification helpers)
        #   scale_gaussian: L_gaussian (prior matching — starts from ep 1, co-trained with CE)
        #   scale_struct:   L_pseudo_novel + L_latent_repel + L_non_gaussian (OSR losses)
        # Fallback: if only aux_scale provided (backward compat), apply uniformly
        if aux_scale != 1.0 and scale_cls == 1.0 and scale_struct == 1.0 and scale_gaussian == 1.0:
            scale_cls = aux_scale
            scale_struct = aux_scale
            scale_gaussian = aux_scale

        total = (L_ce
                 + scale_cls * self.lambda_entropy * L_entropy
                 + scale_cls * self.lambda_separation * L_separation
                 + scale_cls * self.lambda_density * L_density
                 + scale_gaussian * self.lambda_gaussian * L_gaussian
                 + scale_struct * self.lambda_pseudo_novel * L_pseudo
                 + scale_struct * self.lambda_latent_repel * L_latent_repel
                 + scale_struct * self.lambda_non_gaussian * L_non_gaussian
                 + scale_gaussian * self.lambda_z_contrastive * L_z_contrastive)

        info = {
            'L_ce': L_ce.detach(),
            'L_density': L_density.detach(),
            'L_entropy': L_entropy.detach(),
            'L_separation': L_separation.detach(),
            'L_pseudo': L_pseudo.detach(),
            'L_latent_repel': L_latent_repel.detach(),
            'L_gaussian': L_gaussian.detach(),
            'L_non_gaussian': L_non_gaussian.detach(),
            'L_z_contrastive': L_z_contrastive.detach(),
            'L_z_contrastive_known': L_z_contrastive_known.detach(),
            'L_z_contrastive_unknown': L_z_contrastive_unknown.detach(),
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
        self.training_augment = True  # feature-level augmentation for generalization

        # Single-stage joint training: classification + OOD trained together from start
        # OOD modules ramp via lambda warmup (no Stage 1/Stage 2 split)
        # Staged aux loss ramping (缩短warmup, 让loss尽早全部激活):
        #   Phase 1 (ep ≤ warmup): CE + gaussian prior (从第1个episode就启动归一化)
        #   Phase 2 (warmup < ep ≤ warmup+stage2_len): classification helpers ramp 0→1
        #   Phase 3 (warmup+stage2_len < ep): structure + OSR losses ramp 0→1
        self.stage2_len = warmup_episodes * 2   # ep for density/entropy/separation
        self.stage3_len = warmup_episodes * 4   # ep for pseudo_novel/repel/non_gaussian

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

        # GMM boundary sampler: pre-computed low-density OOD samples
        print("Initializing GMM Boundary Sampler...")
        self._gmm_sampler = GMMBoundarySampler(
            self._base_all, device, n_components=len(base_classes))

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

        # Loss: explicit Gaussian matching + inter-class repulsion
        # Auxiliary losses ramp up after warmup (controlled by aux_scale in forward)
        # Key change: much stronger gaussian prior (0.03→0.3) — the foundation of OSR
        # pann_osr2: Simplified loss - only CE + separation
        # All z/flow losses disabled (set to 0)
        self.criterion = EpisodicLoss(
            N_way=self.N_way,
            lambda_density=0.0,          # pann_osr2: disabled
            lambda_entropy=0.0,           # pann_osr2: disabled
            lambda_separation=0.05,       # pann_osr2: increased for better inter-class separation
            lambda_pseudo_novel=0.0,       # pann_osr2: disabled
            lambda_latent_repel=0.0,      # pann_osr2: disabled
            latent_repel_margin=32.0,
            lambda_gaussian=0.0,           # pann_osr2: disabled (no flow)
            lambda_non_gaussian=0.0,       # pann_osr2: disabled
            lambda_z_contrastive=0.0,      # pann_osr2: disabled
            lambda_z_contrastive_known=0.0,
            lambda_z_contrastive_unknown=0.0,
            z_contrastive_temperature=0.1,
            z_contrastive_margin_range=(1.5, 3.0),
            feature_dim=32,
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
        self.patience_episodes = 2000  # OOD z-space需要更长时间收敛
        self.early_stop_counter = 0

    # ----------------------------------------------------------
    # Single training episode
    # ----------------------------------------------------------
    def train_episode(self, current_episode: int = 0,
                      accum_scale: float = 1.0) -> Tuple[float, float, float, dict]:
        """Run one training episode, return (loss, accuracy, ce_loss, z_stats).
        accum_scale: loss scaling for gradient accumulation (1/accum_steps)."""
        self.flow_classifier.train()

        # Reset spline L2 accumulator before forward pass (only if flow enabled)
        if self.flow_classifier.use_flow:
            self.flow_classifier.flow.reset_spline_l2()

        episode = self.train_sampler.sample_episode(self.device)
        prototypes = episode['prototypes']
        query_feats = episode['query_feats']
        query_labels = episode['query_labels']

        # Add Gaussian noise to prototypes for robustness
        if self.proto_noise_std > 0:
            prototypes = prototypes + torch.randn_like(prototypes) * self.proto_noise_std

        # Feature-level augmentation: random scaling + random masking + Gaussian noise
        # Simulates domain shift for better generalization to novel classes
        if self.training_augment:
            # Random scaling: multiply each feature by a random factor in [0.8, 1.2]
            scale = 0.8 + 0.40 * torch.rand_like(query_feats)
            query_feats = query_feats * scale
            # Gaussian noise injection: small perturbation for robustness
            query_feats = query_feats + 0.02 * torch.randn_like(query_feats)

        # Classification: combined prototypical distance + flow density + z values for LCR
        log_probs, z_all = self.flow_classifier.classify_with_z(query_feats, prototypes)

        # Real OOD exposure: flow-only for density-based OOD repulsion
        # Fully vectorized — no per-sample Python loop, uses GPU-cached tensors
        ood_log_probs = None
        if self._ood_all is not None and self.ood_ratio > 0:
            num_ood = max(1, int(query_feats.size(0) * self.ood_ratio))
            num_real = int(num_ood * 0.6)
            num_pseudo = num_ood - num_real

            # Vectorized real OOD: random index into pre-cached GPU tensor
            real_idx = torch.randint(0, len(self._ood_all), (num_real,), device=self.device)
            ood_parts = [self._ood_all[real_idx]]

            # Pseudo-OOD: GMM boundary sampling (替代纯随机噪声)
            # GMM低密度区样本更接近真实unknown分布（在特征边界附近，而非脱离流形）
            if num_pseudo > 0:
                gmm_ood = self._gmm_sampler.sample(num_pseudo)
                ood_parts.append(gmm_ood)

            ood_feats = torch.cat(ood_parts, dim=0)
            # v5: OOD forward through flow with stop-gradient
            # OOD samples don't update flow parameters — flow only models known distributions
            with torch.no_grad():
                ood_log_probs, _ = self.flow_classifier.classify_with_z(ood_feats, prototypes)

        # Project prototypes for separation loss (in projected space)
        projected_protos = self.flow_classifier.project(prototypes)

        # Staged aux loss ramping (single-stage joint training):
        #   Phase 1 (ep ≤ warmup): CE + gaussian prior (从第1个episode启动归一化)
        #   Phase 2 (warmup < ep ≤ warmup+stage2_len): classification helpers ramp 0→1
        #   Phase 3 (after stage2): structure + OSR losses ramp, gaussian at full weight
        w = self.warmup_episodes
        s2 = self.stage2_len
        s3 = self.stage3_len

        if current_episode <= w:
            # Phase 1: CE + gaussian (z空间从第1个episode就逐步对齐N(0,I))
            scale_cls = 0.0
            scale_struct = 0.0
            scale_gaussian = 0.5
        elif current_episode <= w + s2:
            # Phase 2: add classification helpers + gaussian继续增强
            progress = (current_episode - w) / s2
            scale_cls = min(1.0, progress * 0.8)        # gentler ramp (80% at end)
            scale_struct = 0.0
            scale_gaussian = min(1.0, 0.5 + progress * 0.5)  # 50%→100%
        else:
            # Phase 3: full classification + structural + gaussian
            scale_cls = 1.0
            scale_struct = min(1.0, (current_episode - w - s2) / s3)
            scale_gaussian = 1.0

        loss, info = self.criterion(
            log_probs, query_labels,
            prototypes=projected_protos,
            pseudo_novel_log_probs=ood_log_probs,
            z_all=z_all,
            scale_cls=scale_cls,
            scale_struct=scale_struct,
            scale_gaussian=scale_gaussian)

        # Spline derivative L2 regularization: penalize extreme transformations (only if flow enabled)
        if self.flow_classifier.use_flow:
            spline_l2 = self.flow_classifier.flow.get_spline_l2_reg()
            loss = loss + 0.01 * spline_l2

        # OOD head loss: binary classifier on feature-space statistics (known=1, unknown=0)
        # Uses feature-space distances and classification confidence, NOT z-space
        if scale_cls > 0:
            def _compute_ood_features_feat(feat_batch, proto_batch, log_probs_batch):
                """Extract 5-dim OOD feature vector from feature-space stats.
                1. proto_dist_min: min L2 distance to any prototype
                2. proto_dist_ratio: min_dist / second_min_dist (relative clarity)
                3. softmax_max: max classification confidence
                4. logit_entropy: prediction entropy
                5. feature_norm: L2 norm of the feature vector
                """
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
            if ood_log_probs is not None and ood_log_probs.size(0) > 0:
                unk_feats = _compute_ood_features_feat(ood_feats, prototypes, ood_log_probs)
                unk_logits = self.flow_classifier.ood_head(unk_feats)
                unk_targets = torch.zeros(ood_feats.size(0), 1, device=self.device)
                ood_loss = ood_loss + F.binary_cross_entropy_with_logits(unk_logits, unk_targets)

            loss = loss + scale_cls * 0.1 * ood_loss

        # Adaptive z_norm² penalty: pull z back when it diverges too far from N(0,I)
        # Only active when structure losses are ramping (stage 3+)
        if z_all is not None and scale_struct > 0:
            target_norm = self.criterion.feature_dim  # 32 for flow_dim=32
            with torch.no_grad():
                z_correct = z_all[torch.arange(z_all.size(0), device=z_all.device), query_labels]
                current_z_norm_sq = (z_correct ** 2).sum(dim=1).mean().item()
            if current_z_norm_sq > 2.0 * target_norm:
                overshoot = (current_z_norm_sq / target_norm - 2.0)
                # Extra pull on L_gaussian: proportional to overshoot severity
                adaptive_w = min(0.2 * overshoot, 1.0)  # cap at 1.0
                loss = loss + adaptive_w * info['L_gaussian']

        # Skip episode if loss is NaN/Inf (prevents model corruption)
        if not torch.isfinite(loss):
            self.optimizer.zero_grad()  # Reset accumulated gradients on NaN
            return 0.0, 1.0 / self.N_way, 0.0, {}

        # Backward with accumulation scaling (caller handles zero_grad + step)
        scaled_loss = loss * accum_scale
        scaled_loss.backward()

        # Accuracy
        preds = log_probs.argmax(dim=1)
        acc = (preds == query_labels).float().mean().item()

        # Z-space diagnostics (detached, for logging only)
        z_stats = {}
        if z_all is not None:
            with torch.no_grad():
                z_correct = z_all[torch.arange(z_all.size(0), device=z_all.device), query_labels]
                z_stats['z_norm_sq'] = (z_correct ** 2).sum(dim=1).mean().item()
                z_stats['z_std'] = z_correct.std(dim=0).mean().item()
                z_stats['gaussian'] = info['L_gaussian'].item()
                z_stats['non_gaussian'] = info['L_non_gaussian'].item()
        else:
            z_stats['z_norm_sq'] = 0.0
            z_stats['z_std'] = 0.0
            z_stats['gaussian'] = 0.0
            z_stats['non_gaussian'] = 0.0

        return loss.item(), acc, info['L_ce'].item(), z_stats

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
            loss, _ = self.criterion(log_probs, episode['query_labels'], aux_scale=1.0)
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
                start_episode = ckpt.get('episode', 1) + 1
                best_val_acc = ckpt.get('val_acc', 0.0)
                self.best_val_acc = best_val_acc
                print(f"\nResumed from episode {ckpt.get('episode', 1)} "
                      f"(Val Acc: {best_val_acc:.2%})")
                print(f"Continuing from episode {start_episode} to {num_episodes}")
            else:
                print(f"\nCheckpoint not found at {checkpoint_path}, starting from scratch")
                resume = False

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
        print(f"Stage3 (struct):  {self.stage3_len} episodes")
        print(f"Gaussian prior:   starts from ep 1 (co-trained with CE)")
        print(f"Patience:         {self.patience_episodes} episodes ({self.patience_episodes // eval_every} evals)")
        print(f"Training mode:    Single-stage joint (classification + OOD together)")
        print(f"{'='*70}\n")

        running_loss = 0.0
        running_acc = 0.0
        running_ce = 0.0
        running_z_norm_sq = 0.0
        running_z_std = 0.0
        n_z_stats = 0

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

            # Accumulate z-space diagnostics
            if z_stats:
                running_z_norm_sq += z_stats.get('z_norm_sq', 0)
                running_z_std += z_stats.get('z_std', 0)
                n_z_stats += 1

            # Step optimizer at end of accumulation cycle
            if ep % accum_steps == 0 or ep == num_episodes:
                torch.nn.utils.clip_grad_norm_(
                    self.flow_classifier.parameters(), 0.5)
                if ep <= self.warmup_episodes:
                    warmup_factor = ep / self.warmup_episodes
                    for pg in self.optimizer.param_groups:
                        pg['lr'] = self.base_lr * warmup_factor
                else:
                    self.scheduler.step()
                self.optimizer.step()

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
                avg_z_norm = running_z_norm_sq / max(n_z_stats, 1)
                avg_z_std = running_z_std / max(n_z_stats, 1)
                running_loss = 0.0
                running_acc = 0.0
                running_ce = 0.0
                running_z_norm_sq = 0.0
                running_z_std = 0.0
                n_z_stats = 0

                lr = self.optimizer.param_groups[0]['lr']
                # Compute current staged scales for display
                w, s2, s3 = self.warmup_episodes, self.stage2_len, self.stage3_len
                if ep <= w:
                    cur_cls, cur_struct, cur_gauss = 0.0, 0.0, 0.5
                elif ep <= w + s2:
                    progress = (ep - w) / s2
                    cur_cls = min(1.0, progress * 0.8)
                    cur_struct = 0.0
                    cur_gauss = min(1.0, 0.5 + progress * 0.5)
                else:
                    progress3 = min(1.0, (ep - w - s2) / s3)
                    cur_cls = 1.0
                    cur_struct = progress3
                    cur_gauss = 1.0
                # z-space diagnostics: ||z||^2 should ≈ feature_dim, z_std should ≈ 1.0
                z_target = self.criterion.feature_dim
                print(f"Ep {ep:>5d}/{num_episodes} | "
                      f"CE: {avg_ce:.4f} Total: {avg_train_loss:.4f} Acc: {avg_train_acc:.2%} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%} | "
                      f"LR: {lr:.2e} cls:{cur_cls:.1f} gau:{cur_gauss:.1f} str:{cur_struct:.1f} | "
                      f"z_norm²: {avg_z_norm:.1f}(≈{z_target}) z_std: {avg_z_std:.3f}(≈1.0)")

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

    Primary method: Flow-Latent Mahalanobis
      Maps features to latent z-space via cINN, computes class-specific
      Mahalanobis distance. Known samples → low distance (high neg-distance).

    Alternative: Energy-based OOD detection (available via method='energy')
      Score(x) = -E(x) = logsumexp_c log p(x | proto_c)
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
        self.energy_temperature = 1.0

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

    # ---- Energy-based scoring ----

    def score_samples_energy(self, features: torch.Tensor,
                             prototypes: torch.Tensor,
                             batch_size: int = 4096) -> torch.Tensor:
        """
        Energy-based OOD score: neg_energy = logsumexp_c log p(x | proto_c)
        Projection is applied inside compute_energy → forward().
        """
        self.flow.eval()
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                neg_e = self.flow.compute_energy(
                    batch, prototypes, self.energy_temperature)
                all_scores.append(neg_e.cpu())

        return torch.cat(all_scores, dim=0)

    # ---- Confidence gap scoring ----

    def score_samples_confidence_gap(self, features: torch.Tensor,
                                      prototypes: torch.Tensor,
                                      batch_size: int = 4096) -> torch.Tensor:
        """
        Confidence gap: max_c log_p(x|proto_c) - logsumexp_c log_p(x|proto_c)
        Known samples should have a clear best class (high gap),
        unknown samples should be uncertain (low gap).
        Higher score = more likely known.
        """
        self.flow.eval()
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                log_probs = self.flow(batch, prototypes)  # (B, N)
                max_lp, _ = log_probs.max(dim=1)          # (B,)
                lse_lp = torch.logsumexp(log_probs, dim=1) # (B,)
                gap = max_lp - lse_lp                      # (B,) higher = more confident
                all_scores.append(gap.cpu())

        return torch.cat(all_scores, dim=0)

    # ---- Multi-feature extraction for learnable OOD head ----

    def extract_ood_features(self, features: torch.Tensor,
                              prototypes: torch.Tensor,
                              batch_size: int = 4096) -> torch.Tensor:
        """
        Extract multi-dimensional OOD feature vector for each sample.
        Returns: (N, 4) tensor with columns:
          [0] flow_energy  - negative energy (higher = known)
          [1] confidence_gap - max_logp - logsumexp_logp (higher = known)
          [2] max_logp     - max class log-likelihood (higher = known)
          [3] logp_std     - std of log-probs across classes (lower = known)
        """
        self.flow.eval()
        all_feats = []

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                log_probs = self.flow(batch, prototypes)  # (B, N)

                max_lp, _ = log_probs.max(dim=1)
                lse_lp = torch.logsumexp(log_probs, dim=1)
                neg_energy = lse_lp  # energy score (T=1)
                gap = max_lp - lse_lp
                std_lp = log_probs.std(dim=1)

                feats = torch.stack([neg_energy, gap, max_lp, -std_lp], dim=1)  # (B, 4)
                all_feats.append(feats.cpu())

        return torch.cat(all_feats, dim=0)

    # ---- Mahalanobis scoring (primary) ----

    def compute_latent_stats(self, prototypes: torch.Tensor,
                             batch_size: int = 4096):
        """Compute per-class mean and covariance in flow latent z-space.
        Uses forward_to_latent() which handles adapt + dim-reduce + project internally."""
        self.flow.eval()
        self.class_means: Dict[int, torch.Tensor] = {}
        self.class_cov_inv: Dict[int, torch.Tensor] = {}

        for c_idx, c_id in enumerate(self.base_classes):
            feats = self.cache.get_class_features(c_id)
            all_z = []

            with torch.no_grad():
                for i in range(0, len(feats), batch_size):
                    batch = feats[i:i+batch_size].to(self.device)
                    # forward_to_latent handles adapt + flow_dim_adapter + projection
                    z_all = self.flow.forward_to_latent(
                        batch, prototypes[c_idx].unsqueeze(0))  # (B, 1, D)
                    all_z.append(z_all.squeeze(1).cpu())

            z_all = torch.cat(all_z, dim=0)
            self.class_means[c_idx] = z_all.mean(dim=0).to(self.device)

            D = z_all.size(1)
            cov = torch.cov(z_all.T).to(self.device)
            cov += 1e-3 * torch.eye(D, device=self.device)
            self.class_cov_inv[c_idx] = torch.linalg.inv(cov)

    def score_samples_mahalanobis(self, features: torch.Tensor,
                                   prototypes: torch.Tensor,
                                   batch_size: int = 4096) -> torch.Tensor:
        """Flow-Latent Mahalanobis OOD score (negated: higher = more known).
        Uses forward_to_latent() which handles adapt + dim-reduce + project internally."""
        self.flow.eval()
        N = prototypes.size(0)
        all_scores = []

        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                min_dist = torch.full((B,), float('inf'), device=self.device)

                for c_idx in range(N):
                    # forward_to_latent handles adapt + flow_dim_adapter + projection
                    z_all = self.flow.forward_to_latent(
                        batch, prototypes[c_idx].unsqueeze(0))  # (B, 1, D)
                    z_c = z_all.squeeze(1)  # (B, D)
                    diff = z_c - self.class_means[c_idx].unsqueeze(0)
                    mahal = torch.sum(
                        (diff @ self.class_cov_inv[c_idx]) * diff, dim=1
                    ).sqrt()
                    min_dist = torch.minimum(min_dist, mahal)

                all_scores.append(-min_dist.cpu())

        return torch.cat(all_scores, dim=0)

    # ---- Feature-space Mahalanobis scoring (bypasses flow entirely) ----

    def compute_feat_stats(self):
        """Compute per-class means and per-class covariance in 64-dim feature space.
        Per-class covariance is more precise than shared — different acoustic scenes
        have different distribution shapes."""
        self.feat_class_means: Dict[int, torch.Tensor] = {}
        self.feat_cov_inv: Dict[int, torch.Tensor] = {}
        for c_idx, c_id in enumerate(self.base_classes):
            feats = self.cache.get_class_features(c_id)
            self.feat_class_means[c_idx] = feats.mean(dim=0).to(self.device)
            D = feats.size(1)
            cov = torch.cov(feats.T).to(self.device)
            cov += 1e-3 * torch.eye(D, device=self.device)
            self.feat_cov_inv[c_idx] = torch.linalg.inv(cov)

    def score_samples_feat_mahalanobis(self, features: torch.Tensor,
                                        prototypes: torch.Tensor = None,
                                        batch_size: int = 4096) -> torch.Tensor:
        """Feature-space Mahalanobis OOD score with per-class covariance.
        Higher = more known. Uses min_c Mahalanobis_distance(x, class_c)."""
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
        """Feature-space Mahalanobis + relative distance hybrid OOD score.

        Two signals:
          1. min_dist: absolute distance to nearest class (known should be close)
          2. dist_ratio: min_dist / second_min_dist (known should be ≪1, unknown ≈1)
        Fusion: alpha * (-min_dist) + (1-alpha) * (-dist_ratio)
        Higher = more known.
        """
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

                # Sort distances: smallest and second smallest
                sorted_dists, _ = dists.sort(dim=1)
                min_d = sorted_dists[:, 0]        # (B,)
                second_d = sorted_dists[:, 1]      # (B,)
                # Relative distance ratio: how much closer to best vs second-best
                # Known → small ratio (clearly belongs to one class)
                # Unknown → ratio ≈ 1 (equally far from all classes)
                dist_ratio = min_d / (second_d + 1e-6)

                # Fusion: negative distances → higher = more known
                # alpha tuned by Youden J during calibration, default 0.5
                alpha = getattr(self, '_feat_mahal_rel_alpha', 0.5)
                score = alpha * (-min_d) + (1 - alpha) * (-dist_ratio * 10.0)
                all_scores.append(score.cpu())
        return torch.cat(all_scores, dim=0)

    # ---- Likelihood Ratio OSR (flow-based) ----

    def score_samples_likelihood_ratio(self, features: torch.Tensor,
                                        prototypes: torch.Tensor,
                                        batch_size: int = 4096) -> torch.Tensor:
        """Likelihood ratio OOD score: log p(x|class) - log p(x|background).
        Flow-based method that cancels input complexity, isolating class signal.
        Higher = more likely known."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                lr_scores = self.flow.score_likelihood_ratio(batch, prototypes)
                # Max across classes → higher means at least one class explains well
                max_lr, _ = lr_scores.max(dim=1)
                all_scores.append(max_lr.cpu())
        return torch.cat(all_scores, dim=0)

    # ---- Combined scoring: Flow + Mahalanobis ----

    def score_samples_combined(self, features: torch.Tensor,
                                prototypes: torch.Tensor,
                                batch_size: int = 4096,
                                alpha: float = 0.5) -> torch.Tensor:
        """
        Combined Flow + Mahalanobis OOD score.
        Z-normalizes both scores using calibration stats, then weighted fusion.

        Args:
            features: (N, D) query features
            prototypes: (M, D) class prototypes
            batch_size: batch size for inference
            alpha: weight for flow score (1-alpha for mahalanobis)
        Returns:
            combined_scores: (N,) higher = more likely known
        """
        flow_scores = self.score_samples_energy(features, prototypes, batch_size)
        mahal_scores = self.score_samples_mahalanobis(features, prototypes, batch_size)

        # Z-normalize using calibration stats
        if hasattr(self, '_flow_mu') and hasattr(self, '_flow_std'):
            flow_scores = (flow_scores - self._flow_mu) / (self._flow_std + 1e-8)
        if hasattr(self, '_mahal_mu') and hasattr(self, '_mahal_std'):
            mahal_scores = (mahal_scores - self._mahal_mu) / (self._mahal_std + 1e-8)

        return alpha * flow_scores + (1 - alpha) * mahal_scores

    # ---- Unified scoring interface ----

    def score_samples(self, features: torch.Tensor,
                      prototypes: torch.Tensor,
                      method: str = 'flow_mahalanobis',
                      batch_size: int = 4096) -> torch.Tensor:
        """Score samples using specified method.
        Flow-based:    'flow_energy', 'flow_mahalanobis', 'flow_confidence_gap',
                       'flow_likelihood_ratio', 'flow_hybrid'
        Feature-based: 'feature_mahalanobis'
        Learned:       'learned_ensemble'
        """
        if method == 'flow_energy':
            return self.score_samples_energy(features, prototypes, batch_size)
        if method == 'flow_hybrid':
            return self.score_samples_combined(features, prototypes, batch_size)
        if method == 'flow_confidence_gap':
            return self.score_samples_confidence_gap(features, prototypes, batch_size)
        if method == 'learned_ensemble':
            return self.score_samples_learned(features, prototypes, batch_size)
        if method == 'feature_mahalanobis':
            return self.score_samples_feat_mahalanobis(features, prototypes, batch_size)
        if method == 'feat_mahalanobis_relative':
            return self.score_samples_feat_mahalanobis_relative(features, prototypes, batch_size)
        if method == 'flow_likelihood_ratio':
            return self.score_samples_likelihood_ratio(features, prototypes, batch_size)
        if method == 'ood_head':
            return self.score_samples_ood_head(features, prototypes, batch_size)
        if method == 'z_norm':
            return self.score_samples_z_norm(features, prototypes, batch_size)
        if method == 'z_consistency':
            return self.score_samples_z_consistency(features, prototypes, batch_size)
        if method == 'z_reconstruction':
            return self.score_samples_z_reconstruction(features, prototypes, batch_size)
        if method == 'z_gap':
            return self.score_samples_z_gap(features, prototypes, batch_size)
        if method == 'z_mahal_fusion':
            return self.score_samples_z_mahal_fusion(features, prototypes, batch_size)
        return self.score_samples_mahalanobis(features, prototypes, batch_size)

    # ---- Neural OOD head (trained during episodic meta-training) ----

    def score_samples_ood_head(self, features: torch.Tensor,
                                prototypes: torch.Tensor,
                                batch_size: int = 4096) -> torch.Tensor:
        """Score using trained neural OOD head on feature-space statistics."""
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                log_probs = self.flow.classify(batch, prototypes)
                # Feature-space OOD features (same as training)
                dists = torch.cdist(batch, prototypes, p=2)
                sorted_d, _ = dists.sort(dim=1)
                min_d = sorted_d[:, 0:1]
                second_d = sorted_d[:, 1:2]
                dist_ratio = min_d / (second_d + 1e-6)
                probs = F.softmax(log_probs, dim=1)
                softmax_max = probs.max(dim=1, keepdim=True)[0]
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1, keepdim=True)
                feat_norm = (batch ** 2).sum(dim=1, keepdim=True).sqrt()
                ood_feats = torch.cat([min_d, dist_ratio, softmax_max, entropy, feat_norm], dim=1)
                logits = self.flow.ood_head(ood_feats)
                scores = torch.sigmoid(logits).squeeze(1)  # P(known)
                all_scores.append(scores.cpu())
        return torch.cat(all_scores)

    # ---- Z-space OOD scoring methods ----

    def score_samples_z_norm(self, features: torch.Tensor,
                              prototypes: torch.Tensor,
                              batch_size: int = 4096) -> torch.Tensor:
        """Z-space per-dimension deviation OOD score.

        For known samples conditioned on the correct prototype, z ~ N(0,I),
        so per-dim mean≈0, std≈1. For OOD samples, these statistics deviate.

        Score combines three signals (higher = more likely known):
          1. Per-dim mean deviation from 0 (known≈0, OOD≠0)
          2. Per-dim std deviation from 1.0 (known≈1, OOD≠1)
          3. Per-sample std across dims (known≈1, OOD may differ)
        """
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                # Get z values for all prototype conditionings
                z_all = self.flow.forward_to_latent(batch, prototypes)  # (B, N, D)
                # Get log_probs to find best matching prototype
                log_probs = self.flow(batch, prototypes)  # (B, N)
                best_c = log_probs.argmax(dim=1)  # (B,)
                z_best = z_all[torch.arange(B, device=self.device), best_c]  # (B, D)

                # Signal 1: per-dim mean deviation from 0
                mean_dev = z_best.mean(dim=0).abs().mean()  # scalar, ≈0 for known

                # Signal 2: per-dim std deviation from 1.0
                std_dev = (z_best.std(dim=0) - 1.0).abs().mean()  # scalar, ≈0 for known

                # Per-sample scores
                per_sample_mean_dev = z_best.abs().mean(dim=1)  # (B,) ≈0.80 for N(0,I)
                per_sample_std_dev = (z_best.std(dim=1) - 1.0).abs()  # (B,)

                # Combined: lower deviation = more likely known = higher score
                score = -(per_sample_mean_dev + per_sample_std_dev)
                all_scores.append(score.cpu())
        return torch.cat(all_scores)

    def score_samples_z_consistency(self, features: torch.Tensor,
                                     prototypes: torch.Tensor,
                                     batch_size: int = 4096) -> torch.Tensor:
        """Z-space consistency OOD score across all prototype conditionings.

        For known samples, at least one prototype conditioning produces z ≈ N(0,I).
        Score uses per-dim statistics: find the prototype whose conditioning gives
        z with per-dim statistics closest to N(0,I) (mean≈0, std≈1).
        Higher score = more likely known.
        """
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                z_all = self.flow.forward_to_latent(batch, prototypes)  # (B, N, D)

                # For each prototype conditioning, compute deviation from N(0,I)
                # per-sample: mean(|z|) + |std(z) - 1|
                per_sample_mean = z_all.abs().mean(dim=2)  # (B, N)
                # std per sample across dims
                per_sample_std = z_all.std(dim=2)  # (B, N)
                per_sample_std_dev = (per_sample_std - 1.0).abs()  # (B, N)

                deviation = per_sample_mean + per_sample_std_dev  # (B, N)

                # Find prototype with minimum deviation (closest to N(0,I))
                min_deviation, _ = deviation.min(dim=1)  # (B,)
                all_scores.append(-min_deviation.cpu())
        return torch.cat(all_scores)

    def score_samples_z_reconstruction(self, features: torch.Tensor,
                                        prototypes: torch.Tensor,
                                        batch_size: int = 4096) -> torch.Tensor:
        """Z-space reconstruction error OOD score.

        Forward: x_proj → z via flow(x_proj, proto). Inverse: x_hat = flow⁻¹(z, proto).
        Known samples reconstruct well (x_hat ≈ x_proj) because the flow is calibrated
        for known-class features. OOD samples reconstruct poorly.
        Higher score = more likely known (= -||x_proj - x_hat||).
        """
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                N = prototypes.size(0)

                # Project features through flow pipeline
                proj_feats, proj_protos = self.flow._prepare_flow_features(
                    batch, prototypes)

                # Find best matching prototype
                log_probs = self.flow(batch, prototypes)  # (B, N)
                best_c = log_probs.argmax(dim=1)  # (B,)

                # Forward to z using best prototype's conditioning
                best_protos = proj_protos[best_c]  # (B, D)
                z, _ = self.flow.flow.forward(
                    proj_feats, best_protos, compute_jacobian=False)  # (B, D)

                # Inverse: reconstruct from z
                x_hat = self.flow.flow.inverse(z, best_protos)  # (B, D)

                # Reconstruction error
                recon_error = (proj_feats - x_hat).norm(dim=1)  # (B,)
                all_scores.append(-recon_error.cpu())
        return torch.cat(all_scores)

    def score_samples_z_gap(self, features: torch.Tensor,
                             prototypes: torch.Tensor,
                             batch_size: int = 4096) -> torch.Tensor:
        """Z-space prototype gap OOD score.

        Known samples have ONE correct prototype → large gap between best
        and second-best z-space Gaussian fit. OOD samples match no prototype
        well → small gap. Uses the ratio of min to second-min deviation.

        Higher score = more likely known = larger gap.
        """
        self.flow.eval()
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = features[i:i+batch_size].to(self.device)
                B = batch.size(0)
                z_all = self.flow.forward_to_latent(batch, prototypes)  # (B, N, D)

                # Per-prototype deviation from N(0,I)
                per_sample_mean = z_all.abs().mean(dim=2)  # (B, N)
                per_sample_std = z_all.std(dim=2)  # (B, N)
                per_sample_std_dev = (per_sample_std - 1.0).abs()  # (B, N)
                deviation = per_sample_mean + per_sample_std_dev  # (B, N)

                # Sort deviations: smallest (best fit) first
                sorted_dev, _ = deviation.sort(dim=1)  # (B, N)
                best = sorted_dev[:, 0]    # (B,)
                second = sorted_dev[:, 1]  # (B,)

                # Gap ratio: known → best << second → small ratio (closer to 0)
                #            OOD  → best ≈ second → ratio ≈ 1.0
                # Score = -ratio (known gets higher score)
                gap_ratio = best / (second + 1e-6)
                all_scores.append(-gap_ratio.cpu())
        return torch.cat(all_scores)

    def score_samples_z_mahal_fusion(self, features: torch.Tensor,
                                       prototypes: torch.Tensor,
                                       batch_size: int = 4096) -> torch.Tensor:
        """Fusion of z_consistency (flow-based) + feature_mahalanobis (non-flow).

        Combines the two best-performing OSR methods. Feature mahalanobis
        captures distance in the original 64-dim space; z_consistency captures
        flow z-space Gaussian fit quality. These are complementary signals.
        Z-normalizes both and fuses with optimized alpha.
        """
        z_scores = self.score_samples_z_consistency(features, prototypes, batch_size)
        mahal_scores = self.score_samples_feat_mahalanobis(features, prototypes, batch_size)

        # Z-normalize
        z_mu, z_std = z_scores.mean().item(), z_scores.std().item()
        m_mu, m_std = mahal_scores.mean().item(), mahal_scores.std().item()
        z_normed = (z_scores - z_mu) / (z_std + 1e-8)
        m_normed = (mahal_scores - m_mu) / (m_std + 1e-8)

        alpha = getattr(self, '_z_mahal_alpha', 0.5)
        return alpha * z_normed + (1 - alpha) * m_normed

    # ---- Learned OOD head (logistic regression on multi-feature) ----

    def train_ood_head(self, prototypes: torch.Tensor,
                       target_fpr: float = 0.05,
                       batch_size: int = 4096,
                       verbose: bool = True):
        """
        Train a logistic regression OOD classifier on calibration data.
        Uses 4 features: [neg_energy, confidence_gap, max_logp, neg_logp_std]
        Learns weights that optimally separate known from unknown.
        """
        from sklearn.linear_model import LogisticRegression

        if verbose:
            print("Training learned OOD head...")
        # Extract features for known samples
        known_feats_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_feats_list.append(self.extract_ood_features(feats, prototypes, batch_size))
        known_feats = torch.cat(known_feats_list).numpy()

        # Extract features for unknown samples
        unknown_feats_list = []
        for c in self.unknown_classes:
            feats = self.cache.get_class_features(c)
            if len(feats) > 0:
                unknown_feats_list.append(self.extract_ood_features(feats, prototypes, batch_size))
        if not unknown_feats_list:
            print("  No unknown samples for training OOD head, skipping.")
            return
        unknown_feats = torch.cat(unknown_feats_list).numpy()

        # Labels: 1 = known, 0 = unknown
        X = np.concatenate([known_feats, unknown_feats], axis=0)
        y = np.concatenate([np.ones(len(known_feats)), np.zeros(len(unknown_feats))], axis=0)

        # Train logistic regression with class weight balancing
        self._ood_clf = LogisticRegression(
            C=1.0, max_iter=1000, class_weight='balanced', solver='lbfgs')
        self._ood_clf.fit(X, y)

        # Print feature weights
        if verbose:
            w = self._ood_clf.coef_[0]
            feat_names = ['neg_energy', 'conf_gap', 'max_logp', 'neg_std']
            print("  Learned weights:")
            for name, wi in zip(feat_names, w):
                print(f"    {name:>12s}: {wi:+.4f}")

        # Compute threshold from known sample scores at target FPR
        known_probs = self._ood_clf.predict_proba(known_feats)[:, 1]
        sorted_probs = np.sort(known_probs)
        idx = min(int(len(sorted_probs) * target_fpr), len(sorted_probs) - 1)
        self._ood_threshold = sorted_probs[idx]

        # Report calibration performance
        if verbose:
            unknown_probs = self._ood_clf.predict_proba(unknown_feats)[:, 1]
            tnr = (known_probs >= self._ood_threshold).mean()
            tpr = (unknown_probs < self._ood_threshold).mean()
            print(f"  Calib TNR: {tnr:.2%}  TPR: {tpr:.2%}  "
                  f"(threshold={self._ood_threshold:.4f})")

    def score_samples_learned(self, features: torch.Tensor,
                               prototypes: torch.Tensor,
                               batch_size: int = 4096) -> torch.Tensor:
        """
        Score samples using the learned OOD head.
        Returns P(known) from logistic regression (higher = more likely known).
        """
        if not hasattr(self, '_ood_clf'):
            raise RuntimeError("Call train_ood_head() first")
        feats = self.extract_ood_features(features, prototypes, batch_size).numpy()
        probs = self._ood_clf.predict_proba(feats)[:, 1]
        return torch.from_numpy(probs).float()

    # ---- Calibration ----

    def calibrate(self, target_fpr: float = 0.05,
                  K_shot: int = 50,
                  method: str = 'flow_hybrid') -> float:
        """Find OSR threshold on calibration data.
        Methods: 'flow_hybrid', 'flow_confidence_gap', 'flow_likelihood_ratio',
                 'flow_energy', 'flow_mahalanobis', 'feature_mahalanobis', 'learned_ensemble'
        """
        print(f"\n{'='*70}")
        print(f"OSR Calibration ({method.upper()})")
        print(f"{'='*70}")

        prototypes = self.compute_prototypes(K_shot)
        print(f"Prototypes: {prototypes.shape} from {len(self.base_classes)} base classes")

        if method in ('flow_mahalanobis', 'flow_hybrid'):
            print("Computing latent space statistics...")
            self.compute_latent_stats(prototypes)

        if method == 'feature_mahalanobis':
            print("Computing feature-space statistics...")
            self.compute_feat_stats()

        if method == 'flow_likelihood_ratio':
            print("Training background flow for likelihood ratio...")
            base_feats = torch.cat([
                self.cache.get_class_features(c) for c in self.base_classes])
            self.flow.train_background_flow(base_feats, self.device)
        # Learned OOD head: train logistic regression
        if method == 'learned_ensemble':
            self.train_ood_head(prototypes, target_fpr)
            self.threshold = self._ood_threshold
            # Print stats
            known_scores = self.score_samples_learned(
                torch.cat([self.cache.get_class_features(c) for c in self.base_classes]),
                prototypes)
            unknown_feats = [self.cache.get_class_features(c) for c in self.unknown_classes]
            unknown_feats = [f for f in unknown_feats if len(f) > 0]
            if unknown_feats:
                unknown_scores = self.score_samples_learned(
                    torch.cat(unknown_feats), prototypes)
                sep = known_scores.mean() - unknown_scores.mean()
                print(f"Known scores:   [{known_scores.min():.2f}, {known_scores.max():.2f}] "
                      f"mean={known_scores.mean():.2f}")
                print(f"Unknown scores: [{unknown_scores.min():.2f}, {unknown_scores.max():.2f}] "
                      f"mean={unknown_scores.mean():.2f}")
                print(f"Mean separation: {sep:.2f} (positive = known higher = good)")
            print(f"\nThreshold (tau): {self.threshold:.4f}")
            return self.threshold

        # For 'feat_mahalanobis_relative': compute stats + optimize alpha
        if method == 'feat_mahalanobis_relative':
            self.compute_feat_stats()
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

        # For 'z_mahal_fusion': compute feat stats + optimize alpha
        if method == 'z_mahal_fusion':
            self.compute_feat_stats()
            best_alpha, best_j = 0.5, 0.0
            for trial_alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
                self._z_mahal_alpha = trial_alpha
                trial_known = []
                trial_unknown = []
                for c in self.base_classes:
                    trial_known.append(self.score_samples_z_mahal_fusion(
                        self.cache.get_class_features(c), prototypes))
                for c in self.unknown_classes:
                    feats = self.cache.get_class_features(c)
                    if len(feats) > 0:
                        trial_unknown.append(self.score_samples_z_mahal_fusion(
                            feats, prototypes))
                trial_known = torch.cat(trial_known)
                if not trial_unknown:
                    continue
                trial_unknown = torch.cat(trial_unknown)
                sorted_k, _ = trial_known.sort()
                tnr_idx = min(int(len(sorted_k) * 0.05), len(sorted_k) - 1)
                tau = sorted_k[tnr_idx].item()
                tpr = (trial_unknown > tau).float().mean().item()
                j = 0.95 + tpr - 1.0
                if j > best_j:
                    best_j, best_alpha = j, trial_alpha
            self._z_mahal_alpha = best_alpha
            print(f"  Z-Mahal fusion alpha: {best_alpha:.2f} (Youden J={best_j:.3f})")

        # For 'flow_hybrid': compute both scores on known data to get normalization stats
        if method == 'flow_hybrid':
            print("Computing flow + mahalanobis normalization stats...")
            known_flow_all, known_mahal_all = [], []
            for c in self.base_classes:
                feats = self.cache.get_class_features(c)
                known_flow_all.append(self.score_samples_energy(feats, prototypes))
                known_mahal_all.append(self.score_samples_mahalanobis(feats, prototypes))
            all_flow = torch.cat(known_flow_all)
            all_mahal = torch.cat(known_mahal_all)
            self._flow_mu, self._flow_std = all_flow.mean().item(), all_flow.std().item()
            self._mahal_mu, self._mahal_std = all_mahal.mean().item(), all_mahal.std().item()

            # Z-normalize known scores for alpha optimization
            norm_known_flow = (all_flow - self._flow_mu) / (self._flow_std + 1e-8)
            norm_known_mahal = (all_mahal - self._mahal_mu) / (self._mahal_std + 1e-8)

            # Also compute normalized unknown scores for alpha optimization
            unknown_flow_all, unknown_mahal_all = [], []
            for c in self.unknown_classes:
                feats = self.cache.get_class_features(c)
                if len(feats) > 0:
                    unknown_flow_all.append(self.score_samples_energy(feats, prototypes))
                    unknown_mahal_all.append(self.score_samples_mahalanobis(feats, prototypes))
            has_unknown = len(unknown_flow_all) > 0
            if has_unknown:
                norm_unknown_flow = (torch.cat(unknown_flow_all) - self._flow_mu) / (self._flow_std + 1e-8)
                norm_unknown_mahal = (torch.cat(unknown_mahal_all) - self._mahal_mu) / (self._mahal_std + 1e-8)

            # Optimize alpha via Youden's J = TPR + TNR - 1
            best_alpha, best_j = 0.5, -1.0
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                combined_known = alpha * norm_known_flow + (1 - alpha) * norm_known_mahal
                sorted_k = combined_known.sort()[0]
                idx = min(int(len(sorted_k) * target_fpr), len(sorted_k) - 1)
                thresh = sorted_k[idx]
                tnr = (combined_known >= thresh).float().mean().item()
                if has_unknown:
                    combined_unknown = alpha * norm_unknown_flow + (1 - alpha) * norm_unknown_mahal
                    tpr = (combined_unknown < thresh).float().mean().item()
                else:
                    tpr = 0.0
                j = tnr + tpr - 1.0
                if j > best_j:
                    best_j, best_alpha = j, alpha

            self._combined_alpha = best_alpha
            print(f"  Flow scores:   mu={self._flow_mu:.3f} std={self._flow_std:.3f}")
            print(f"  Mahal scores:  mu={self._mahal_mu:.3f} std={self._mahal_std:.3f}")
            print(f"  Fusion weight: alpha={self._combined_alpha:.2f} (Youden J={best_j:.3f})")

        # Score known samples
        known_scores_list = []
        for c in self.base_classes:
            feats = self.cache.get_class_features(c)
            known_scores_list.append(
                self.score_samples(feats, prototypes, method))
        known_scores = torch.cat(known_scores_list)

        # Score unknown samples
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

        # Threshold: set at (1-FPR) percentile so that ~(1-FPR) of known are accepted
        # Sort ascending → pick FPR quantile → 95% of known scores >= threshold
        sorted_known, _ = known_scores.sort()
        idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
        threshold = sorted_known[idx].item()

        # Evaluate
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
                     method: str = 'flow_hybrid',
                     recalibrate_per_round: bool = True) -> Dict:
        """
        Full OSR evaluation with few-shot prototype computation.
        Runs multiple rounds with different random support sets and averages.

        Args:
            recalibrate_per_round: If True, recompute threshold from current round's
                known scores instead of using fixed calib threshold. This fixes
                calib→test distribution shift that caused test TPR to drop.
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
        print(f"{'='*70}")

        all_known_rates = []
        all_unknown_rates = []
        target_fpr = 0.05

        for round_i in range(num_rounds):
            prototypes = self.compute_prototypes(K_shot, seed=round_i)

            # Transductive refinement: use subset of known samples as queries
            # to refine prototypes before scoring
            if method not in ('feature_mahalanobis', 'feat_mahalanobis_relative', 'learned_ensemble'):
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

            if method in ('flow_mahalanobis', 'flow_hybrid'):
                self.compute_latent_stats(prototypes)

            if method == 'feature_mahalanobis':
                self.compute_feat_stats()

            if method == 'feat_mahalanobis_relative':
                self.compute_feat_stats()
                # Optimize alpha for best Youden J on calibration data
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
                    # Find threshold at 95% TNR
                    sorted_k, _ = trial_known.sort()
                    tnr_idx = min(int(len(sorted_k) * 0.05), len(sorted_k) - 1)
                    tau = sorted_k[tnr_idx].item()
                    tpr = (trial_unknown > tau).float().mean().item()
                    j = 0.95 + tpr - 1.0  # Youden J = TNR + TPR - 1
                    if j > best_j:
                        best_j, best_alpha = j, trial_alpha
                self._feat_mahal_rel_alpha = best_alpha

            if method == 'flow_likelihood_ratio':
                base_feats = torch.cat([
                    self.cache.get_class_features(c) for c in self.base_classes])
                self.flow.train_background_flow(
                    base_feats, self.device, epochs=30, verbose=False)

            # For 'z_mahal_fusion': compute feat stats + optimize alpha per round
            if method == 'z_mahal_fusion':
                self.compute_feat_stats()
                best_alpha, best_j = 0.5, 0.0
                for trial_alpha in [0.2, 0.35, 0.5, 0.65, 0.8]:
                    self._z_mahal_alpha = trial_alpha
                    trial_known = []
                    trial_unknown = []
                    for c in self.base_classes:
                        trial_known.append(self.score_samples_z_mahal_fusion(
                            self.cache.get_class_features(c), prototypes))
                    for c in self.unknown_classes:
                        feats = self.cache.get_class_features(c)
                        if len(feats) > 0:
                            trial_unknown.append(self.score_samples_z_mahal_fusion(
                                feats, prototypes))
                    trial_known = torch.cat(trial_known)
                    if not trial_unknown:
                        continue
                    trial_unknown = torch.cat(trial_unknown)
                    sorted_k, _ = trial_known.sort()
                    tnr_idx = min(int(len(sorted_k) * 0.05), len(sorted_k) - 1)
                    tau = sorted_k[tnr_idx].item()
                    tpr = (trial_unknown > tau).float().mean().item()
                    j = 0.95 + tpr - 1.0
                    if j > best_j:
                        best_j, best_alpha = j, trial_alpha
                self._z_mahal_alpha = best_alpha

            # For 'flow_hybrid': update normalization stats per round
            if method == 'flow_hybrid':
                known_flow_all, known_mahal_all = [], []
                for c in self.base_classes:
                    feats = self.cache.get_class_features(c)
                    known_flow_all.append(self.score_samples_energy(feats, prototypes))
                    known_mahal_all.append(self.score_samples_mahalanobis(feats, prototypes))
                all_flow = torch.cat(known_flow_all)
                all_mahal = torch.cat(known_mahal_all)
                self._flow_mu, self._flow_std = all_flow.mean().item(), all_flow.std().item()
                self._mahal_mu, self._mahal_std = all_mahal.mean().item(), all_mahal.std().item()

            # For 'learned_ensemble': retrain OOD head per round
            if method == 'learned_ensemble':
                self.train_ood_head(prototypes, target_fpr, verbose=False)

            # Score known samples
            known_scores_list = []
            for c in self.base_classes:
                feats = self.cache.get_class_features(c)
                known_scores_list.append(self.score_samples(feats, prototypes, method))
            known_scores = torch.cat(known_scores_list)

            # Per-round recalibration: compute threshold from this round's known scores
            if recalibrate_per_round:
                sorted_known, _ = known_scores.sort()
                idx = min(int(len(sorted_known) * target_fpr), len(sorted_known) - 1)
                round_threshold = sorted_known[idx].item()
            else:
                round_threshold = threshold

            # Known classes: rate of being correctly accepted as known
            known_rate = (known_scores >= round_threshold).float().mean().item()
            all_known_rates.append(known_rate)

            # Unknown classes: rate of being correctly rejected as unknown
            unknown_detect_list = []
            for c in self.unknown_classes:
                feats = self.cache.get_class_features(c)
                if len(feats) == 0:
                    continue
                scores = self.score_samples(feats, prototypes, method)
                unknown_detect_list.append((scores < round_threshold).float().mean().item())

            if unknown_detect_list:
                unknown_rate = np.mean(unknown_detect_list)
                all_unknown_rates.append(unknown_rate)

        mean_known = np.mean(all_known_rates)
        mean_unknown = np.mean(all_unknown_rates) if all_unknown_rates else 0.0
        osr_score = (mean_known + mean_unknown) / 2

        print(f"\nResults over {num_rounds} rounds:")
        print(f"  Known accepted (TNR):     {mean_known:.2%}")
        print(f"  Unknown detected (TPR):   {mean_unknown:.2%}")
        print(f"  OSR Score (mean):         {osr_score:.2%}")

        results = {
            'known_tnr': mean_known,
            'unknown_tpr': mean_unknown,
            'osr_score': osr_score,
            'threshold': threshold if not recalibrate_per_round else 'per-round',
            'num_rounds': num_rounds,
            'method': method,
            'recalibrate': recalibrate_per_round,
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
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # ============ Configuration ============
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 10 acoustic scene classes -> 6 base (known), 4 novel (unknown)
    # 0=airport, 1=tram, 2=bus, 3=public_square,
    # 4=shopping_mall, 5=street_pedestrian
    # 6=metro_station, 7=street_traffic, 8=metro, 9=park
    base_classes = [0, 1, 2, 3, 4, 5]
    unknown_classes = [6, 7, 8, 9]

    N_way = 5
    K_shot = 5
    Q_query = 30  # 15→30: 2x query samples提升GPU利用率 (5-way * 30 = 150 queries/episode)
    num_episodes = 10000
    gradient_accum_steps = 1  # 2→1: Q_query增大后不需要累积
    feature_dim = 64

    # Change this one variable to redirect all model output paths
    experiment_dir = 'experiment/ast_fewshot_osr1'
    base_pretrained_path = os.path.join(experiment_dir, 'base_feature_extractor.pth')
    contrastive_pretrained_path = os.path.join(experiment_dir, 'contrastive_feature_extractor.pth')

    # ---- Backbone selection: 'distil_ast' or 'panns_cnn14' ----
    # 'distil_ast':  Distil-AST (6-layer, 44M, ~1.7min/ep) — better than full AST
    # 'panns_cnn14': PANNs Cnn14_16k (CNN, ~5M, ~45s/ep) — fastest, CNN-based
    backbone_choice = 'distil_ast'

    if backbone_choice == 'distil_ast':
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
            print(f"\nExisting checkpoint has backbone='{ckpt_backbone}', "
                  f"need {FeatureExtractorClass.BACKBONE_TYPE} retrain. "
                  f"Removing old checkpoints.")
            os.remove(base_pretrained_path)
            if os.path.exists(contrastive_pretrained_path):
                os.remove(contrastive_pretrained_path)
            _need_retrain = True

    if not os.path.exists(base_pretrained_path) or _need_retrain:
        # Both Distil-AST and PANNs are AudioSet-pretrained → skip SimCLR
        # Only fine-tune projection head + last 2 backbone layers on base classes
        print(f"\n{'='*70}")
        print(f"Phase 0b: Base Class Supervised Fine-tuning ({backbone_choice}, no SimCLR)")
        print("  - Only base classes (no leakage)")
        print(f"  - Unfreeze: {backbone_unfreeze_layers}")
        print(f"{'='*70}")
        feature_extractor = FeatureExtractorClass()
        pretrainer = BaseClassPretrainer(
            feature_extractor, base_classes, device,
            lr=5e-6, epochs=20,
            backbone_unfreeze_layers=backbone_unfreeze_layers)
        feature_extractor = pretrainer.train(train_dataset, base_pretrained_path)

    feature_extractor = feature_extractor.to(device)

    # ============ Phase 1: Feature Extraction ============
    print(f"\n{'='*70}")
    print("Phase 1: Feature Extraction")
    print(f"{'='*70}")

    calib_dataset = TAUDataset(split='calib')
    test_dataset = TAUDataset(split='test')

    train_cache = FeatureCache()
    train_cache.extract_and_cache(feature_extractor, train_dataset, 'train', device, batch_size=64)

    calib_cache = FeatureCache()
    calib_cache.extract_and_cache(feature_extractor, calib_dataset, 'calib', device, batch_size=64)

    test_cache = FeatureCache()
    test_cache.extract_and_cache(feature_extractor, test_dataset, 'test', device, batch_size=64)

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
        num_coupling_layers=0,          # pann_osr2: disable flow
        hidden_dims=[128],
        use_projection=True,
        s_clamp_max=3.0,
        flow_dim=32,
        use_flow=False                  # pann_osr2: simple prototype network only
    )

    # pann_osr2: Disable OOD exposure (simplified approach)
    ood_features = {}  # Empty - no OOD exposure
    print("OOD exposure: disabled (pann_osr2 simplified approach)")

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
        proto_noise_std=0.15,       # pann_osr2: reset to default
        ood_ratio=0.0,              # pann_osr2: disabled (no OOD exposure)
        warmup_episodes=100,        # pann_osr2: shorter warmup (no complex losses)
        task_aug_drop_rate=0.2,
        gradient_accum_steps=gradient_accum_steps,
        device=device
    )

    trainer.train(
        num_episodes=num_episodes,
        eval_every=150,
        num_val_episodes=30,
        save_dir=experiment_dir,
        resume=True  # Resume from checkpoint if exists
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

    # --- OSR calibration & evaluation: compare all methods ---
    # feature_mahalanobis为首要方法; 精简掉效果最差的方法以加速评估
    osr_results = {}

    # Select OSR methods based on flow availability
    if trainer.flow_classifier.use_flow:
        osr_methods = ['feature_mahalanobis', 'z_consistency', 'z_gap', 'z_mahal_fusion', 'flow_hybrid', 'flow_energy', 'ood_head']
    else:
        # When use_flow=False, only use feature-space methods (skip all z/flow methods)
        osr_methods = ['feature_mahalanobis', 'ood_head']
        print("NOTE: use_flow=False, skipping flow-dependent methods (z_*, flow_*)")

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

    # Summary comparison
    print(f"\n{'='*70}")
    print("OSR Method Comparison (test data, per-round recalibration)")
    print(f"{'='*70}")
    print(f"  {'Method':<20s} {'TNR':>8s} {'TPR':>8s} {'OSR':>8s}")
    for m, r in osr_results.items():
        print(f"  {m:<20s} {r['known_tnr']:>7.2%} {r['unknown_tpr']:>7.2%} {r['osr_score']:>7.2%}")

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
            'flow_dim': 32,
            'base_classes': base_classes,
            'unknown_classes': unknown_classes,
            'scoring_method': best_method,
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
