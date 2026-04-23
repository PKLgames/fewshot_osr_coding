"""
Neural Spline Flow (NSF) with Rational-Quadratic Splines

Replaces affine coupling layers with monotonic rational-quadratic spline transforms,
providing significantly more expressiveness per coupling layer.

Reference:
  Durkan et al., "Neural Spline Flows", NeurIPS 2019

Interface matches ConditionalINN:
  forward(x, c, compute_jacobian) -> (z, log_det)
  inverse(z, c) -> x
  log_prob(x, c) -> log_likelihood
  sample(c, num_samples) -> samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


class SplineCouplingLayer(nn.Module):
    """
    Coupling layer using rational-quadratic spline transforms.

    Each half of the input is transformed via a monotonic spline whose
    parameters are predicted by a neural network conditioned on the other
    half and an external condition vector.

    Compared to affine coupling (exp(s)*x + t), splines can represent
    arbitrary monotonic functions → much more expressive per layer.
    """

    def __init__(self,
                 input_dim: int,
                 condition_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 num_bins: int = 8,
                 bound: float = 5.0,
                 split_idx: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()

        if split_idx is None:
            self.split_idx = input_dim // 2
        else:
            self.split_idx = split_idx

        self.d1 = self.split_idx
        self.d2 = input_dim - self.split_idx
        self.num_bins = num_bins
        self.bound = bound

        # Spline params per output dim: K widths + K heights + (K+1) derivatives = 3K+1
        self.num_spline_params = 3 * num_bins + 1

        # Network for first half: conditioned on x2 + c
        self.net1 = self._build_network(
            self.d2 + condition_dim, self.d1 * self.num_spline_params,
            hidden_dims, dropout)

        # Network for second half: conditioned on z1 + c
        self.net2 = self._build_network(
            self.d1 + condition_dim, self.d2 * self.num_spline_params,
            hidden_dims, dropout)

        # Spline derivative L2 regularization accumulator
        self._deriv_l2_sum = None
        self._deriv_l2_count = 0

    def _build_network(self, input_dim: int, output_dim: int,
                       hidden_dims: List[int], dropout: float = 0.0) -> nn.Sequential:
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def _parse_spline_params(self, raw: torch.Tensor, dim: int):
        """
        Parse raw network output into spline parameters.

        Returns:
            widths:     (B, dim, K) — bin widths, sum = 2*bound
            heights:    (B, dim, K) — bin heights, sum = 2*bound
            derivatives:(B, dim, K+1) — knot derivatives, all positive
        """
        B = raw.size(0)
        K = self.num_bins

        raw = raw.view(B, dim, self.num_spline_params)

        widths_raw = raw[:, :, :K]
        heights_raw = raw[:, :, K:2*K]
        derivatives_raw = raw[:, :, 2*K:]

        # Softmax ensures each dim's bins sum to 2*bound
        widths = F.softmax(widths_raw, dim=2) * (2 * self.bound)
        heights = F.softmax(heights_raw, dim=2) * (2 * self.bound)

        # Derivatives must be positive for monotonicity
        derivatives = F.softplus(derivatives_raw) + 1e-3

        return widths, heights, derivatives

    @staticmethod
    def _rqs_forward(x, widths, heights, derivatives, bound):
        """
        Rational-quadratic spline forward: x → y.

        Knot positions start at -bound.  knot_x[..., 0]=-bound, knot_x[..., K]=+bound.
        x and knots share the same coordinate system.
        """
        B, D = x.shape
        K = widths.size(2)

        # Build knot x-positions: [-bound, ..., +bound]
        neg_b = torch.full((B, D, 1), -bound, device=x.device, dtype=x.dtype)
        knot_x = torch.cat([neg_b, neg_b + widths.cumsum(dim=2)], dim=2)  # (B,D,K+1)

        # Build knot y-positions: [-bound, ..., +bound]
        knot_y = torch.cat([neg_b, neg_b + heights.cumsum(dim=2)], dim=2)  # (B,D,K+1)

        # Bin search
        x_exp = x.unsqueeze(2).contiguous()
        bin_idx = torch.searchsorted(knot_x, x_exp).squeeze(2).clamp(1, K) - 1

        idx = bin_idx.unsqueeze(2).expand(-1, -1, 1)

        left_x  = torch.gather(knot_x, 2, idx).squeeze(2)
        right_x = torch.gather(knot_x, 2, idx + 1).squeeze(2)
        left_y  = torch.gather(knot_y, 2, idx).squeeze(2)
        right_y = torch.gather(knot_y, 2, idx + 1).squeeze(2)
        d_left  = torch.gather(derivatives, 2, idx).squeeze(2)
        d_right = torch.gather(derivatives, 2, idx + 1).squeeze(2)

        w = right_x - left_x
        h = right_y - left_y

        # ξ ∈ [0, 1]
        xi = ((x - left_x) / (w + 1e-8)).clamp(0.0, 1.0)
        s = h / (w + 1e-8)

        xi2 = xi * xi
        xi1m = xi * (1.0 - xi)

        numer = s * xi2 + d_left * xi1m
        denom = s + (d_right + d_left - 2.0 * s) * xi1m + 1e-8

        y = left_y + h * numer / denom

        # Log determinant
        alpha = d_left * (1.0 - xi) ** 2 + 2.0 * s * xi1m + d_right * xi2
        log_det = (torch.log(h + 1e-8) + torch.log(alpha + 1e-8)
                   - 2.0 * torch.log(denom + 1e-8) - torch.log(w + 1e-8))

        # Identity outside [-bound, bound]
        oor = (x < -bound) | (x > bound)
        y = torch.where(oor, x, y)
        log_det = torch.where(oor, torch.zeros_like(log_det), log_det)

        return y, log_det

    @staticmethod
    def _rqs_inverse(y, widths, heights, derivatives, bound):
        """
        Rational-quadratic spline inverse: y → x.
        """
        B, D = y.shape
        K = widths.size(2)

        neg_b = torch.full((B, D, 1), -bound, device=y.device, dtype=y.dtype)
        knot_x = torch.cat([neg_b, neg_b + widths.cumsum(dim=2)], dim=2)
        knot_y = torch.cat([neg_b, neg_b + heights.cumsum(dim=2)], dim=2)

        y_exp = y.unsqueeze(2).contiguous()
        bin_idx = torch.searchsorted(knot_y, y_exp).squeeze(2).clamp(1, K) - 1

        idx = bin_idx.unsqueeze(2).expand(-1, -1, 1)

        left_x  = torch.gather(knot_x, 2, idx).squeeze(2)
        right_x = torch.gather(knot_x, 2, idx + 1).squeeze(2)
        left_y  = torch.gather(knot_y, 2, idx).squeeze(2)
        right_y = torch.gather(knot_y, 2, idx + 1).squeeze(2)
        d_left  = torch.gather(derivatives, 2, idx).squeeze(2)
        d_right = torch.gather(derivatives, 2, idx + 1).squeeze(2)

        w = right_x - left_x
        h = right_y - left_y
        s = h / (w + 1e-8)

        theta = ((y - left_y) / (h + 1e-8)).clamp(0.0, 1.0)

        # Quadratic solve (Durkan et al. 2019 Eq. 10-12)
        a = d_right + d_left - 2.0 * s
        b = s - d_left + theta * a
        c = -theta * s

        disc = (b * b - 4.0 * a * c).clamp(min=0.0)

        a_tiny = a.abs() < 1e-6
        b_tiny = b.abs() < 1e-6
        xi_q = (-b + torch.sqrt(disc)) / (2.0 * a + 1e-8)
        xi_l = -c / (b + 1e-8)
        xi = torch.where(a_tiny & b_tiny, theta,
                         torch.where(a_tiny, xi_l, xi_q)).clamp(0.0, 1.0)

        x = left_x + w * xi

        oor = (y < -bound) | (y > bound)
        x = torch.where(oor, y, x)

        return x

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                compute_jacobian: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: x → z with spline transforms."""
        x1 = x[:, :self.d1]
        x2 = x[:, self.d1:]

        raw1 = self.net1(torch.cat([x2, c], dim=1))
        w1, h1, d1 = self._parse_spline_params(raw1, self.d1)
        z1, ld1 = self._rqs_forward(x1, w1, h1, d1, self.bound)

        raw2 = self.net2(torch.cat([z1, c], dim=1))
        w2, h2, d2 = self._parse_spline_params(raw2, self.d2)
        z2, ld2 = self._rqs_forward(x2, w2, h2, d2, self.bound)

        z = torch.cat([z1, z2], dim=1)

        # Accumulate L2 of spline derivatives for regularization
        deriv_l2 = d1.pow(2).mean() + d2.pow(2).mean()
        if self._deriv_l2_sum is None:
            self._deriv_l2_sum = deriv_l2
        else:
            self._deriv_l2_sum = self._deriv_l2_sum + deriv_l2
        self._deriv_l2_count += 1

        if compute_jacobian:
            return z, ld1.sum(dim=1) + ld2.sum(dim=1)
        else:
            return z, None

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Inverse pass: z → x using spline inverses."""
        z1 = z[:, :self.d1]
        z2 = z[:, self.d1:]

        # Invert z2 → x2 conditioned on (z1, c)
        raw2 = self.net2(torch.cat([z1, c], dim=1))
        w2, h2, d2 = self._parse_spline_params(raw2, self.d2)
        x2 = self._rqs_inverse(z2, w2, h2, d2, self.bound)

        # Invert z1 → x1 conditioned on (x2, c)
        raw1 = self.net1(torch.cat([x2, c], dim=1))
        w1, h1, d1 = self._parse_spline_params(raw1, self.d1)
        x1 = self._rqs_inverse(z1, w1, h1, d1, self.bound)

        return torch.cat([x1, x2], dim=1)

    def get_and_reset_deriv_l2(self) -> torch.Tensor:
        """Return mean spline derivative L2 and reset accumulator."""
        if self._deriv_l2_count == 0 or self._deriv_l2_sum is None:
            return torch.tensor(0.0)
        avg = self._deriv_l2_sum / self._deriv_l2_count
        self._deriv_l2_sum = None
        self._deriv_l2_count = 0
        return avg


class ConditionalNSF(nn.Module):
    """
    Neural Spline Flow: stack of spline coupling layers with permutations.

    Same interface as ConditionalINN from cINN.py.
    """

    def __init__(self,
                 input_dim: int,
                 condition_dim: int,
                 num_coupling_layers: int = 6,
                 hidden_dims: List[int] = None,
                 num_bins: int = 8,
                 bound: float = 5.0,
                 use_permutation: bool = True,
                 permutation_type: str = 'fixed',
                 dropout: float = 0.15,
                 s_clamp_max: float = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_coupling_layers = num_coupling_layers
        self.use_permutation = use_permutation
        self.s_clamp_max = s_clamp_max or 0.0

        self.coupling_layers = nn.ModuleList()
        if use_permutation and permutation_type == 'learnable':
            self.permutations = nn.ModuleList()
        self.permutation_type = permutation_type

        for i in range(num_coupling_layers):
            split_idx = input_dim // 2 if i % 2 == 0 else input_dim // 3
            self.coupling_layers.append(SplineCouplingLayer(
                input_dim=input_dim,
                condition_dim=condition_dim,
                hidden_dims=hidden_dims,
                num_bins=num_bins,
                bound=bound,
                split_idx=split_idx,
                dropout=dropout,
            ))

            if use_permutation and i < num_coupling_layers - 1:
                if permutation_type == 'learnable':
                    perm = nn.Conv1d(1, 1, 1, bias=False)
                    nn.init.eye_(perm.weight)
                    self.permutations.append(perm)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                compute_jacobian: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z = x
        total_log_det = torch.zeros(x.size(0), device=x.device) if compute_jacobian else None

        for i, layer in enumerate(self.coupling_layers):
            if compute_jacobian:
                z, ld = layer(z, c, compute_jacobian=True)
                total_log_det = total_log_det + ld
            else:
                z, _ = layer(z, c, compute_jacobian=False)

            if self.use_permutation and i < len(self.coupling_layers) - 1:
                if self.permutation_type == 'learnable':
                    z = self.permutations[i](z.unsqueeze(1)).squeeze(1)
                else:
                    z = torch.flip(z, dims=[1])

        return z, total_log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z
        for i in reversed(range(len(self.coupling_layers))):
            if self.use_permutation and i > 0:
                if self.permutation_type == 'learnable':
                    x = self.permutations[i - 1](x.unsqueeze(1)).squeeze(1)
                else:
                    x = torch.flip(x, dims=[1])
            x = self.coupling_layers[i].inverse(x, c)
        return x

    def reset_spline_l2(self):
        """Reset all coupling layer accumulators."""
        for layer in self.coupling_layers:
            layer._deriv_l2_sum = None
            layer._deriv_l2_count = 0

    def get_spline_l2_reg(self) -> torch.Tensor:
        """Get mean spline derivative L2 across all layers and reset."""
        total = torch.tensor(0.0,
                             device=next(self.parameters()).device)
        for layer in self.coupling_layers:
            total = total + layer.get_and_reset_deriv_l2()
        return total / max(len(self.coupling_layers), 1)

    def log_prob(self, x: torch.Tensor, c: torch.Tensor,
                 prior_dist=None) -> torch.Tensor:
        if prior_dist is None:
            prior_dist = torch.distributions.Normal(
                torch.zeros(self.input_dim, device=x.device),
                torch.ones(self.input_dim, device=x.device))

        z, log_det = self.forward(x, c, compute_jacobian=True)
        return prior_dist.log_prob(z).sum(dim=1) + log_det

    def sample(self, c: torch.Tensor, num_samples: int = 1,
               prior_dist=None) -> torch.Tensor:
        if c.dim() == 1:
            c = c.unsqueeze(0)
        B = c.size(0)

        if prior_dist is None:
            prior_dist = torch.distributions.Normal(
                torch.zeros(self.input_dim, device=c.device),
                torch.ones(self.input_dim, device=c.device))

        if num_samples > 1:
            c_exp = c.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.condition_dim)
            z = prior_dist.sample((B * num_samples,)).to(c.device)
            return self.inverse(z, c_exp).reshape(B, num_samples, -1)
        else:
            z = prior_dist.sample((B,)).to(c.device)
            return self.inverse(z, c)


if __name__ == "__main__":
    B, D = 32, 64

    model = ConditionalNSF(
        input_dim=D, condition_dim=D,
        num_coupling_layers=6, hidden_dims=[256, 256],
        num_bins=8, bound=5.0, dropout=0.15)

    print(f"NSF params: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(B, D)
    c = torch.randn(B, D)

    z, ld = model.forward(x, c, compute_jacobian=True)
    x_rec = model.inverse(z, c)
    mse = (x - x_rec).pow(2).mean().item()
    print(f"Reconstruction MSE: {mse:.2e}")
    print(f"z mean={z.mean():.3f} std={z.std():.3f}")
    print(f"log_det mean={ld.mean():.3f}")

    lp = model.log_prob(x, c)
    print(f"log_prob mean={lp.mean():.3f}")

    samples = model.sample(c, num_samples=3)
    print(f"Sample shape: {samples.shape}")
