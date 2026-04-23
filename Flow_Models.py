import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import Reversable_Function.cINN as cINN

class PrototypeFlowModel(nn.Module):
    """
    条件可逆神经网络（cINN）的组合模型，使用cluster_num个cINN模型按权重相加
    """
    def __init__(self, 
                 input_dim: int, 
                 condition_dim: int=0, 
                 num_coupling_layers: int = 12,
                 hidden_dims: List[int] = [256, 256],
                 use_permutation: bool = True,
                 permutation_type: str = 'fixed',
                 cluster_num: int = 6):
        """
        初始化组合模型
        """
        super(PrototypeFlowModel, self).__init__()
        
        # 创建cluster_num个模型
        self.cluster_num = cluster_num
        self.cluster_model = nn.ModuleList([
            cINN.ConditionalINN(
                input_dim=input_dim,
                condition_dim=condition_dim,
                num_coupling_layers=num_coupling_layers,
                hidden_dims=hidden_dims,
                use_permutation=use_permutation,
                permutation_type=permutation_type
            ) for _ in range(cluster_num)
        ])
        
        # 创建权重参数，用于对cINN模型的输出进行加权
        self.weights = nn.Parameter(torch.ones(cluster_num) / cluster_num)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                compute_jacobian: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播：从数据空间到隐变量空间
        """
        z_list = []
        log_det_list = []
        
        c_expand = c.repeat(x.shape[0], 1).to(x.device)
        for i in range(self.cluster_num):
            z, log_det = self.cluster_model[i](x, c_expand, compute_jacobian=compute_jacobian)
            z_list.append(z)
            log_det_list.append(log_det)
        
        z = torch.stack(z_list, dim=0)
        log_det = torch.stack(log_det_list, dim=0) if log_det_list[0] is not None else None
        
        z = torch.sum(z * self.weights.view(-1, 1, 1), dim=0)
        
        if log_det is not None:
            log_det = torch.sum(log_det * self.weights.view(-1, 1), dim=0)
        
        return z, log_det

    def inverse(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x_list = []
        c_expand = c.repeat(z.shape[0], 1).to(z.device)
        
        for i in range(self.cluster_num):
            x = self.cluster_model[i].inverse(z, c_expand)
            x_list.append(x)
        
        x = torch.stack(x_list, dim=0)
        x = torch.sum(x * self.weights.view(-1, 1, 1), dim=0)
        
        return x

    def sample(self, c: torch.Tensor, num_samples: int = 1, 
               prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        if c.dim() == 1:
            c_expand = c.unsqueeze(0)
                
        samples_list = []
        
        for i in range(self.cluster_num):
            samples = self.cluster_model[i].sample(c_expand, num_samples, prior_dist)
            samples_list.append(samples)
        
        samples = torch.stack(samples_list, dim=0)
        samples = torch.sum(samples * self.weights.view(-1, 1, 1, 1), dim=0)
        
        return samples

    def log_prob(self, x: torch.Tensor, c: torch.Tensor, 
                 prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        log_prob_list = []
        c_expand = c.repeat(x.shape[0], 1).to(x.device)
        
        for i in range(self.cluster_num):
            log_prob = self.cluster_model[i].log_prob(x, c)
            log_prob_list.append(log_prob)
        
        log_prob = torch.stack(log_prob_list, dim=0)
        log_prob = torch.sum(log_prob * self.weights.view(-1, 1), dim=0)
        
        return log_prob

class FlowBasedTissue(nn.Module):
    """
    由prototype_num个PrototypeFlowModel组成的原型模型，输出每个原型模型的对数似然
    
    简化建议：
    - 减少coupling_layers: 12 -> 3-4
    - 减少hidden_dims: [256,256] -> [64,64]
    - 减少cluster_num: 10 -> 4-6
    """
    def __init__(self, 
                 input_dim: int, 
                 prototype_num: int,
                 condition_dim: int=0, 
                 num_coupling_layers: int = 3,  # 减少: 12 -> 3
                 hidden_dims: List[int] = [64, 64],  # 减少: [256,256] -> [64,64]
                 use_permutation: bool = True,
                 permutation_type: str = 'fixed',
                 cluster_num: int = 4):  # 减少: 10 -> 4
        """
        初始化原型模型
        """
        super(FlowBasedTissue, self).__init__()
        
        self.prototype_num = prototype_num
        self.prototype_models = nn.ModuleList([
            PrototypeFlowModel(
                input_dim=input_dim,
                condition_dim=condition_dim,
                num_coupling_layers=num_coupling_layers,
                hidden_dims=hidden_dims,
                use_permutation=use_permutation,
                permutation_type=permutation_type,
                cluster_num=cluster_num
            ) for _ in range(prototype_num)
        ])
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        log_probs_list = []
        c_expand = c.repeat(x.shape[0], 1).to(x.device)
        
        for i in range(self.prototype_num):
            log_prob = self.prototype_models[i].log_prob(x, c)
            log_probs_list.append(log_prob)
        
        log_probs = torch.stack(log_probs_list, dim=0)
        
        return log_probs.t()


class FlowBasedCell(nn.Module):
    """
    基于标准化流的聚类器，用于未知类发现。
    """
    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 0,
        num_coupling_layers: int = 2,
        hidden_dims: List[int] = [32, 32],
        use_permutation: bool = True,
        permutation_type: str = 'fixed',
        cluster_num: int = 4,
    ):
        super(FlowBasedCell, self).__init__()
        self.input_dim = input_dim
        self.cluster_num = cluster_num
        self.init_noise_std = 0.01

        self.flows = nn.ModuleList([
            cINN.ConditionalINN(
                input_dim=input_dim,
                condition_dim=condition_dim,
                num_coupling_layers=num_coupling_layers,
                hidden_dims=hidden_dims,
                use_permutation=use_permutation,
                permutation_type=permutation_type
            ) for _ in range(cluster_num)
        ])

        initial_logits = torch.log(torch.ones(cluster_num) / cluster_num)
        noise = torch.randn_like(initial_logits) * self.init_noise_std
        self.mix_logits = nn.Parameter(initial_logits + noise)

        self._init_weights()

    def _init_weights(self):
        for flow in self.flows:
            if hasattr(flow, 'reset_parameters'):
                flow.reset_parameters()
        
        if self.training:
            with torch.no_grad():
                noise = torch.randn_like(self.mix_logits) * self.init_noise_std
                self.mix_logits.add_(noise)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                prior_dist: Optional[torch.distributions.Distribution] = None) -> torch.Tensor:
        device = x.device
        batch_size = x.shape[0]

        is_valid = ~torch.all(x == 0, dim=1)
        valid_indices = torch.where(is_valid)[0]
        x_valid = x[valid_indices]
        n_valid = x_valid.shape[0]

        if c.dim() == 1:
            c = c.unsqueeze(0)
        if c.shape[0] == 1:
            c_valid = c.expand(n_valid, -1)
        else:
            c_valid = c[valid_indices]

        gamma = torch.zeros(batch_size, self.cluster_num, device=device)
        full_log_probs = torch.full((batch_size, self.cluster_num), -10.0, device=device)

        if n_valid == 0:
            gamma[:] = 1.0 / self.cluster_num
            return gamma.detach(), full_log_probs

        log_probs_list = []
        for i in range(self.cluster_num):
            log_p = self.flows[i].log_prob(x_valid, c_valid)
            log_p = torch.clamp(log_p, min=-50.0, max=10.0)
            log_probs_list.append(log_p)

        log_probs = torch.stack(log_probs_list, dim=1)

        log_pi = F.log_softmax(self.mix_logits, dim=0)
        log_pi = log_pi.unsqueeze(0)

        weighted_log_prob = log_probs + log_pi
        gamma_valid = F.softmax(weighted_log_prob, dim=1)

        gamma[valid_indices] = gamma_valid
        full_log_probs[valid_indices] = log_probs

        invalid_mask = ~is_valid
        if invalid_mask.any():
            gamma[invalid_mask] = 1.0 / self.cluster_num
        
        return gamma, full_log_probs

    def get_mix_weights(self) -> torch.Tensor:
        return F.softmax(self.mix_logits, dim=0)
        
    def predict_clusters(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, _ = self.forward(x, c)
        return torch.argmax(gamma, dim=1)

# 示例
if __name__ == "__main__":
    input_dim = 32
    condition_dim = 0
    batch_size = 32
    prototype_num = 10
    
    classifier_model = FlowBasedTissue(
        input_dim=input_dim,
        prototype_num=prototype_num,
        condition_dim=condition_dim,
        num_coupling_layers=3,
        hidden_dims=[64, 64],
        use_permutation=True,
        permutation_type='fixed',
        cluster_num=4
    )

    clusterer_model = FlowBasedCell(
        input_dim = input_dim,
        condition_dim = 0,
        num_coupling_layers = 2,
        hidden_dims = [32, 32],
        use_permutation = True,
        permutation_type = 'fixed',
        cluster_num = 4
    )
    
    print(f"classifier_model parameters: {sum(p.numel() for p in classifier_model.parameters()):,}")
    print(f"clusterer_model parameters: {sum(p.numel() for p in clusterer_model.parameters()):,}")
    
    x = torch.randn(batch_size, input_dim)
    c = torch.zeros(batch_size, condition_dim)
    
    z1 = classifier_model.forward(x, c)
    z2 = clusterer_model.forward(x, c)
    print(f"\nForward pass:")
    print(f"Input x shape: {x.shape}")
    print(f"Log likelihood shape: {z1.shape}")
    print(f"Gamma shape: {z2[0].shape}")
