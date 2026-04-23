"""
Reciprocal Point Loss (RPL).

Learns a "reciprocal point" (anti-prototype) per class that represents the
extra-class space.  Features are pushed *away* from the reciprocal point of
their own class, while the distance to reciprocal points serves as an
open-set score.

Interface consumed by Pretrainer:
    criterion = RPLoss(**vars(args))
    logits, loss = criterion(features, fc_logits, labels, epoch=epoch)
    criterion.Dist.state_dict()  →  {'centers': Tensor[C, D]}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReciprocalDist(nn.Module):
    """Learnable per-class reciprocal-point bank."""

    def __init__(self, num_classes: int, feat_dim: int = 512):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        return self.centers[labels]


class RPLoss(nn.Module):
    def __init__(self, train_classes: int = 60, feat_dim: int = 512,
                 weight_pl: float = 0.1, beta: float = 0.1,
                 temp: float = 1.0, num_centers: int = 1, **_kwargs):
        super().__init__()
        self.Dist = ReciprocalDist(train_classes, feat_dim)
        self.weight_pl = weight_pl
        self.beta = beta
        self.temp = temp
        self.num_centers = num_centers

    def forward(self, features: torch.Tensor, _logits: torch.Tensor,
                labels: torch.Tensor, epoch: int = 0):
        """
        Args:
            features: [B, D] pooled backbone features
            _logits:  [B, C] FC logits (unused – kept for interface compat)
            labels:   [B]    integer class labels
            epoch:    current epoch (for margin annealing, optional)
        Returns:
            logits: [B, C]  (higher → more likely this class)
            loss:   scalar
        """
        rp = self.Dist.centers                                # [C, D]

        # --- Negative distance to reciprocal points as class logits ---
        # Being FAR from a class's reciprocal point → likely belongs to that class
        feat_norm = F.normalize(features, dim=-1)
        rp_norm = F.normalize(rp, dim=-1)
        cosine = feat_norm @ rp_norm.t()                      # [B, C]
        # Negate: low cosine (far from reciprocal point) → high class score
        logits = -cosine / self.temp

        # --- RPL margin loss: push features away from their own-class RP ---
        batch_rp = rp[labels]                                  # [B, D]
        dist_sq = ((features - batch_rp) ** 2).sum(dim=-1)     # [B]

        # Margin that increases over epochs (encourages growing separation)
        margin = min(10.0 + epoch * 0.5, 50.0)
        loss_margin = F.relu(margin - dist_sq).mean()

        # --- Entropy regularisation on cosine scores ---
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-7)).sum(dim=-1).mean()

        # --- Cross-entropy to keep logits class-discriminative ---
        loss_ce = F.cross_entropy(logits, labels)

        loss = loss_ce + self.weight_pl * loss_margin + self.beta * entropy
        return logits, loss
