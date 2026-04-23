import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
    
    def forward(self, q, k, v):
        attn_score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn_weights = nn.Softmax(dim=-1)(attn_score)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model//n_heads

        self.WQ = nn.Linear(d_model, d_model)
        self.WK = nn.Linear(d_model, d_model)
        self.WV = nn.Linear(d_model, d_model)

        nn.init.normal_(self.WQ.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.WK.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_k)))
        nn.init.normal_(self.WV.weight, mean=0, std=np.sqrt(2.0 / (d_model + self.d_v)))

        self.scaled_dot_product_attn = ScaledDotProductAttention(self.d_k)
        self.linear = nn.Linear(n_heads * self.d_v, d_model)
        nn.init.xavier_normal_(self.linear.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        
        q_heads = self.WQ(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_heads = self.WK(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_heads = self.WV(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        attn, attn_weights = self.scaled_dot_product_attn(q_heads, k_heads, v_heads)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.dropout(self.linear(attn))
        return output, attn_weights



class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        output = self.relu(self.linear1(inputs))
        output = self.linear2(output)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_heads=1, p_drop=0.1, d_ff=2048):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, n_heads)
        self.dropout1 = nn.Dropout(p_drop)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        
        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.dropout2 = nn.Dropout(p_drop)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v):
        attn_outputs, attn_weights = self.mha(q, k, v)
        attn_outputs = self.dropout1(attn_outputs)
        attn_outputs = self.layernorm1(q + attn_outputs.squeeze())

        ffn_outputs = self.ffn(attn_outputs)
        ffn_outputs = self.dropout2(ffn_outputs)
        ffn_outputs = self.layernorm2(attn_outputs + ffn_outputs)
        
        return ffn_outputs, attn_weights


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, n_layers=1, n_heads=1, p_drop=0.1, d_ff=2048):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, p_drop, d_ff) for _ in range(n_layers)])

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

            if m.bias is not None:
                init.constant_(m.bias.data, 0)

    def forward(self, feature, center):
        for layer in self.layers:
            center, attn_weights = layer(center, feature, feature)
        return center.squeeze(0)


class OpenSetGenerater(nn.Module):
    def __init__(self,featdim, n_head=1, agg='mlp'):
        super(OpenSetGenerater, self).__init__()
        self.featdim = featdim
        self.agg = agg
        # Cross-attention projections: Q from support, K/V from base weights
        self.q_proj = nn.Linear(featdim, featdim)
        self.k_proj = nn.Linear(featdim, featdim)
        self.v_proj = nn.Linear(featdim, featdim)
        # Open-set attention projections
        self.open_q_proj = nn.Linear(featdim, featdim)
        self.open_k_proj = nn.Linear(featdim, featdim)
        self.open_v_proj = nn.Linear(featdim, featdim)
        self.norm1 = nn.LayerNorm
        self.norm2 = nn.LayerNorm
        if agg == 'mlp':
            self.agg_func = nn.Sequential(nn.Linear(featdim,featdim),nn.LeakyReLU(0.5),nn.Dropout(0.5),nn.Linear(featdim,featdim))
            self.agg_func1 = nn.Linear(featdim,featdim)

    def _cross_attention(self, query, key_value, q_proj, k_proj, v_proj):
        """Scaled dot-product cross-attention.
        query: [N_q, D],  key_value: [N_kv, D] or [B, N_kv, D]
        Returns: same shape as query
        """
        # Handle batched key_value (e.g. [1, N_kv, D])
        if key_value.dim() == 3:
            key_value = key_value.squeeze(0)  # [N_kv, D]

        Q = q_proj(query)           # [N_q, D]
        K = k_proj(key_value)       # [N_kv, D]
        V = v_proj(key_value)       # [N_kv, D]
        scale = np.sqrt(self.featdim)
        attn = torch.matmul(Q, K.t()) / scale   # [N_q, N_kv]
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)              # [N_q, D]
        return out

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

            if m.bias is not None:
                init.constant_(m.bias.data, 0)

    def forward(self, support_center, base_weight, base_open_weight):
        # support_center: [n_ways, D]
        # base_weight: [num_base, D],  base_open_weight: [num_base, D]

        # Step 1: cross-attend support prototypes to base class prototypes
        support_center = self._cross_attention(
            support_center, base_weight,
            self.q_proj, self.k_proj, self.v_proj)
        support_center = self.agg_func1(support_center)   # [n_ways, D]

        # Step 2: cross-attend to base open-set weights
        output = self._cross_attention(
            support_center, base_open_weight,
            self.open_q_proj, self.open_k_proj, self.open_v_proj)

        if self.agg == 'mlp':
            output = self.agg_func(output)  # [n_ways, D]

        # Aggregate into a single fake class center [1, 1, D]
        fakeclass_center = output.mean(dim=0, keepdim=True)  # [1, D]
        return output.unsqueeze(1), fakeclass_center.unsqueeze(0)
    


class SupportCalibrator(nn.Module):
    def __init__(self,feat_dim=512, n_head=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.calibrator = MultiHeadAttention(feat_dim,n_head)


    def forward(self, support_feat):
        ## support_feat: bs*nway*640, base_weights: bs*num_base*640, support_seman: bs*nway*300, base_seman:bs*num_base*300        
        base_weights = support_feat
        base_mem_vis = support_feat

        support_center,_ = self.calibrator(support_feat, base_weights, base_mem_vis)

        return support_center