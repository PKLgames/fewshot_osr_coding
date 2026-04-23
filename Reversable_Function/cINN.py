import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List


class ConditionalCouplingLayer(nn.Module):
    """
    条件可逆耦合层 (Affine coupling with condition)

    s_clamp_max controls the expressiveness:
      s_clamped = s_clamp_max * tanh(s / s_clamp_max)
      range: [-s_clamp_max, s_clamp_max]
      exp(s_clamped) range: [exp(-s_clamp_max), exp(s_clamp_max)]

      Default 2.0: per-layer scale [0.14, 7.39]
      Recommended 3.0: per-layer scale [0.05, 20.09] — better for complex distributions
    """
    def __init__(self,
                 input_dim: int,
                 condition_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 split_idx: Optional[int] = None,
                 dropout: float = 0.0,
                 s_clamp_max: float = 2.0):
        super(ConditionalCouplingLayer, self).__init__()

        if split_idx is None:
            self.split_idx = input_dim // 2
        else:
            self.split_idx = split_idx

        self.d1 = self.split_idx
        self.d2 = input_dim - self.split_idx
        self.s_clamp_max = s_clamp_max

        in_dim_1 = self.d2 + condition_dim
        in_dim_2 = self.d1 + condition_dim

        self.s1_net = self._build_network(in_dim_1, self.d1, hidden_dims, dropout)
        self.t1_net = self._build_network(in_dim_1, self.d1, hidden_dims, dropout)
        self.s2_net = self._build_network(in_dim_2, self.d2, hidden_dims, dropout)
        self.t2_net = self._build_network(in_dim_2, self.d2, hidden_dims, dropout)

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

    def _clamp_s(self, s: torch.Tensor) -> torch.Tensor:
        """Soft clamp: s_clamp_max * tanh(s / s_clamp_max)"""
        return self.s_clamp_max * torch.tanh(s / self.s_clamp_max)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                compute_jacobian: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]

        x1 = x[:, :self.d1]
        x2 = x[:, self.d1:]

        # First half transform: x1 → z1, conditioned on x2 and c
        s1 = self._clamp_s(self.s1_net(torch.cat([x2, c], dim=1)))
        t1 = self.t1_net(torch.cat([x2, c], dim=1))
        z1 = x1 * torch.exp(s1) + t1

        # Second half transform: x2 → z2, conditioned on z1 and c
        s2 = self._clamp_s(self.s2_net(torch.cat([z1, c], dim=1)))
        t2 = self.t2_net(torch.cat([z1, c], dim=1))
        z2 = x2 * torch.exp(s2) + t2

        z = torch.cat([z1, z2], dim=1)

        if compute_jacobian:
            log_det_jacobian = s1.sum(dim=1) + s2.sum(dim=1)
            return z, log_det_jacobian
        else:
            return z, None

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        z1 = z[:, :self.d1]
        z2 = z[:, self.d1:]

        # Inverse of second half
        s2 = self._clamp_s(self.s2_net(torch.cat([z1, c], dim=1)))
        t2 = self.t2_net(torch.cat([z1, c], dim=1))
        x2 = (z2 - t2) * torch.exp(-s2)

        # Inverse of first half
        s1 = self._clamp_s(self.s1_net(torch.cat([x2, c], dim=1)))
        t1 = self.t1_net(torch.cat([x2, c], dim=1))
        x1 = (z1 - t1) * torch.exp(-s1)

        return torch.cat([x1, x2], dim=1)


class ConditionalINN(nn.Module):
    """
    Conditional Invertible Neural Network (stack of coupling layers + permutations)

    Maps x → z conditioned on c, with z ~ N(0, I) when well-trained.
    Inter-layer permutations ensure all dimensions get transformed.

    Args:
        s_clamp_max: controls affine coupling scale range.
            2.0 = conservative (original), 3.0 = recommended for 64-dim features
    """
    def __init__(self,
                 input_dim: int,
                 condition_dim: int,
                 num_coupling_layers: int = 8,
                 hidden_dims: List[int] = [256, 256],
                 use_permutation: bool = True,
                 permutation_type: str = 'fixed',
                 dropout: float = 0.0,
                 s_clamp_max: float = 2.0):
        super(ConditionalINN, self).__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.num_coupling_layers = num_coupling_layers
        self.use_permutation = use_permutation
        self.s_clamp_max = s_clamp_max

        self.coupling_layers = nn.ModuleList()

        if use_permutation:
            self.permutations = nn.ModuleList() if permutation_type == 'learnable' else []
            self.permutation_type = permutation_type

        for i in range(num_coupling_layers):
            # Alternate split ratio for better mixing
            split_idx = input_dim // 2 if i % 2 == 0 else input_dim // 3

            coupling_layer = ConditionalCouplingLayer(
                input_dim=input_dim,
                condition_dim=condition_dim,
                hidden_dims=hidden_dims,
                split_idx=split_idx,
                dropout=dropout,
                s_clamp_max=s_clamp_max
            )
            self.coupling_layers.append(coupling_layer)

            if use_permutation and i < num_coupling_layers - 1:
                if permutation_type == 'learnable':
                    perm_layer = nn.Conv1d(1, 1, 1, bias=False)
                    nn.init.eye_(perm_layer.weight)
                    self.permutations.append(perm_layer)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                compute_jacobian: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size = x.shape[0]
        z = x
        total_log_det = torch.zeros(batch_size, device=x.device) if compute_jacobian else None

        for i, coupling_layer in enumerate(self.coupling_layers):
            if compute_jacobian:
                z, log_det = coupling_layer(z, c, compute_jacobian=True)
                total_log_det = total_log_det + log_det
            else:
                z, _ = coupling_layer(z, c, compute_jacobian=False)

            if self.use_permutation and i < len(self.coupling_layers) - 1:
                if self.permutation_type == 'learnable':
                    z_reshaped = z.unsqueeze(1)
                    z_permuted = self.permutations[i](z_reshaped)
                    z = z_permuted.squeeze(1)
                else:
                    z = torch.flip(z, dims=[1])

        return z, total_log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = z

        for i in reversed(range(len(self.coupling_layers))):
            if self.use_permutation and i > 0:
                if self.permutation_type == 'learnable':
                    x_reshaped = x.unsqueeze(1)
                    x_permuted = self.permutations[i-1](x_reshaped)
                    x = x_permuted.squeeze(1)
                else:
                    x = torch.flip(x, dims=[1])

            coupling_layer = self.coupling_layers[i]
            x = coupling_layer.inverse(x, c)

        return x

    def sample(self, c: torch.Tensor, num_samples: int = 1,
               prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        if c.dim() == 1:
            c = c.unsqueeze(0)

        batch_size = c.shape[0]

        if prior_dist is None:
            prior_dist = torch.distributions.Normal(
                torch.zeros(self.input_dim, device=c.device),
                torch.ones(self.input_dim, device=c.device)
            )

        if num_samples > 1:
            c_expanded = c.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, self.condition_dim)
            z = prior_dist.sample((batch_size * num_samples,)).to(c.device)
            samples = self.inverse(z, c_expanded)
            samples = samples.reshape(batch_size, num_samples, -1)
        else:
            z = prior_dist.sample((batch_size,)).to(c.device)
            samples = self.inverse(z, c)

        return samples

    def log_prob(self, x: torch.Tensor, c: torch.Tensor,
                 prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        if prior_dist is None:
            prior_dist = torch.distributions.Normal(
                torch.zeros(self.input_dim, device=x.device),
                torch.ones(self.input_dim, device=x.device)
            )

        z, log_det_jacobian = self.forward(x, c, compute_jacobian=True)

        log_prob_prior = prior_dist.log_prob(z).sum(dim=1)

        log_prob = log_prob_prior + log_det_jacobian

        return log_prob


if __name__ == "__main__":
    input_dim = 64
    condition_dim = 64
    batch_size = 32

    # Test with increased capacity
    model = ConditionalINN(
        input_dim=input_dim,
        condition_dim=condition_dim,
        num_coupling_layers=8,
        hidden_dims=[256, 256],
        use_permutation=True,
        permutation_type='fixed',
        s_clamp_max=3.0
    )

    print(f"Model structure:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    x = torch.randn(batch_size, input_dim)
    c = torch.randn(batch_size, condition_dim)

    z, log_det = model.forward(x, c, compute_jacobian=True)
    print(f"\nForward pass:")
    print(f"Input x shape: {x.shape}")
    print(f"Latent z shape: {z.shape}")
    print(f"Log det shape: {log_det.shape}")

    x_reconstructed = model.inverse(z, c)
    reconstruction_error = torch.mean((x - x_reconstructed) ** 2)
    print(f"\nReconstruction error (MSE): {reconstruction_error.item():.6f}")

    log_prob = model.log_prob(x, c)
    print(f"\nLog prob shape: {log_prob.shape}")
    print(f"Average log prob: {log_prob.mean().item():.4f}")

    samples = model.sample(c, num_samples=3)
    print(f"\nSampled data shape: {samples.shape}")
