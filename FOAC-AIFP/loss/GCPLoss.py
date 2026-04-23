"""
Gaussian Center-based Prototype Loss (GCPL).

Learns a single center (prototype) per class and trains with cosine-similarity
based cross-entropy plus a center-alignment regularisation term.

Interface consumed by Pretrainer:
    criterion = GCPLoss(**vars(args))
    logits, loss = criterion(features, fc_logits, labels)
    criterion.Dist.state_dict()  →  {'centers': Tensor[C, D]}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterDist(nn.Module):
    """Learnable per-class center bank."""

    def __init__(self, num_classes: int, feat_dim: int = 512):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        return self.centers[labels]


class GCPLoss(nn.Module):
    def __init__(self, train_classes: int = 60, feat_dim: int = 512,
                 weight_pl: float = 0.1, temp: float = 1.0, **_kwargs):
        super().__init__()
        self.Dist = CenterDist(train_classes, feat_dim)
        self.weight_pl = weight_pl
        self.temp = temp

    def forward(self, features: torch.Tensor, _logits: torch.Tensor,
                labels: torch.Tensor):
        """
        Args:
            features: [B, D] pooled backbone features
            _logits:  [B, C] FC logits (unused – kept for interface compat)
            labels:   [B]    integer class labels
        Returns:
            logits: [B, C] cosine-similarity logits
            loss:   scalar
        """
        centers = self.Dist.centers                         # [C, D]
        feat_norm = F.normalize(features, dim=-1)
        cent_norm = F.normalize(centers, dim=-1)
        logits = feat_norm @ cent_norm.t()                  # [B, C]
        logits_scaled = logits / self.temp

        # Cross-entropy on cosine similarity
        loss_ce = F.cross_entropy(logits_scaled, labels)

        # Center alignment: pull features toward their class center
        with torch.no_grad():
            batch_centers = centers[labels]                  # [B, D]
        loss_center = F.mse_loss(features, batch_centers)

        loss = loss_ce + self.weight_pl * loss_center
        return logits_scaled, loss
