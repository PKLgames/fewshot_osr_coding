# Flow OSR 理论分析：希尔伯特空间视角

## 1. 整体架构：条件神经样条流 (cNSF) 的核空间映射

代码中的核心是一个**条件归一化流 (Conditional Normalizing Flow)**，其数学本质是在**再生核希尔伯特空间 (RKHS)** 中构建一个参数化的测度变换。

### 1.1 特征空间 $\mathcal{H}$ 与核映射

特征提取器 (YAMNet/AST/PANNs → FC → 64-dim) 实现了一个核映射：

$$\phi: \mathcal{X} \to \mathcal{H} \subset \mathbb{R}^{64}$$

其中 $\mathcal{X}$ 是音频波形空间，$\mathcal{H}$ 是隐式定义的特征空间。SimCLR 对比预训练 (Phase 0a) 本质上在学习一个**自监督核函数**：

$$k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle_{\mathcal{H}}$$

使得同类样本在 RKHS 中内积大（近），异类内积小（远）。NT-Xent loss 即是在核空间中施加**谱聚类结构**。

### 1.2 条件流作为密度估计器

`ConditionalNSF` 实现了一个可逆变换 $f_\theta: \mathcal{Z} \leftrightarrow \mathcal{F}$：

$$\mathbf{z} = f_\theta(\mathbf{x} \mid \mathbf{c}), \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{d})$$

其中 $\mathbf{c}$ 是类别原型 (prototype)，$\mathcal{F} \subset \mathbb{R}^{d_{\text{flow}}}$ 是 flow 维度空间（$d_{\text{flow}}=48$）。

由变量替换公式，条件密度为：

$$\log p(\mathbf{x} \mid \mathbf{c}) = \log p_{\mathcal{N}}(f_\theta(\mathbf{x} \mid \mathbf{c})) + \log \left| \det \frac{\partial f_\theta}{\partial \mathbf{x}} \right|$$

这正是**在原型条件化的 RKHS 子空间中估计 Radon-Nikodym 导数**。

### 1.3 理性二次样条作为核函数的逼近

每个耦合层 (`SplineCouplingLayer`) 实现的理性二次样条变换：

$$y_i = \text{RQS}(x_i; \{w_k, h_k, d_k\}_{k=1}^K)$$

可以理解为**逐维度构造一个单调核函数**：
- 样条参数由网络 $\text{NN}(x_{2:d}, \mathbf{c})$ 预测，即**条件核** $k(x_i \mid x_{-i}, \mathbf{c})$
- `bound=5.0` 限定了核的支集 $[-B, B]$，在支集外退化为恒等映射
- `num_bins=8` 给出了核函数的分段有理逼近阶数

8 层耦合 + 固定置换 构成了一个**深度可逆合成核**：

$$f_\theta = f^{(8)} \circ P_8 \circ \cdots \circ f^{(1)} \circ P_1$$

每层 $f^{(l)}$ 是一个条件样条核变换，$P_l$ 是置换矩阵（对应不同坐标顺序），确保所有维度都被变换。

---

## 2. 原型 (Prototype) 作为 RKHS 中的再生核代表元

在元学习的 episode 中：

$$\mu_c = \frac{1}{K} \sum_{i=1}^{K} \phi(x_i^{(c)}) \in \mathcal{H}$$

$\mu_c$ 是类别 $c$ 在 RKHS 中的**核均值嵌入 (Kernel Mean Embedding)**。`feature_adapter` 对 $\mu_c$ 施加一个可学习的线性变换 $A: \mathcal{H} \to \mathcal{H}$（`nn.Linear + LayerNorm`），等价于在 RKHS 中学习一个新的内积核：

$$k'(x, y) = \langle A\phi(x), A\phi(y) \rangle = \phi(x)^\top A^\top A \phi(y)$$

---

## 3. 条件网络的核解释

`condition_network`（$\mathbb{R}^{48} \to \mathbb{R}^{96} \to \mathbb{R}^{48}$）将原型映射为更丰富的条件向量：

$$\tilde{\mathbf{c}} = g_\psi(\mu_c)$$

这等价于在**条件 RKHS** $\mathcal{H}_c$ 中寻找一个更好的核表示，使得：

$$p(\mathbf{x} \mid c) = p(f_\theta(\mathbf{x} \mid g_\psi(\mu_c))) \cdot \left| \det J_{f_\theta} \right|$$

的条件依赖更加显著。

---

## 4. 损失函数的希尔伯特空间结构

代码中的多阶段损失对应 RKHS 中的不同几何约束：

| 损失 | 希尔伯特空间含义 |
|------|------------------|
| $L_\text{CE}$ | 在 RKHS 中最小化**交叉熵距离**（分类） |
| $L_\text{density}$ | 最大化正确类的**对数似然** = 拉近 $f_\theta(\mathbf{x})$ 到 $\mathcal{N}(0, I)$ |
| $L_\text{gaussian}$ | 强制 $z_\text{correct} \sim \mathcal{N}(0, I)$，即**先验匹配**（RKHS 中的范数约束 $\|z\|^2 \approx d$，各维标准差 $\approx 1$） |
| $L_\text{non\_gaussian}$ | 强制 $z_\text{wrong}$ **偏离** $\mathcal{N}(0,I)$（$\|z\|^2 > 1.5d$），创造**核空间的排斥区** |
| $L_\text{latent\_repel}$ | 在 $z$-空间中用铰链损失推开 $z_\text{wrong}$ 与 $z_\text{correct}$，即 RKHS 中的**间距约束** |
| $L_\text{separation}$ | 最大化原型间的成对距离 = 最大化 RKHS 中均值嵌入的距离 |
| $L_\text{pseudo\_novel}$ | 对 GMM 边界样本施加**低密度约束** = 在 RKHS 的**低密度区域**创建排斥力 |

---

## 5. OOD 检测的似然比

`bg_flow`（无条件流）估计边缘密度 $p(\mathbf{x})$，`flow`（条件流）估计 $p(\mathbf{x} \mid c)$。似然比 OOD 分数：

$$\text{score}(\mathbf{x}) = \log p(\mathbf{x} \mid c) - \log p(\mathbf{x}) = \log \frac{p(\mathbf{x} \mid c)}{p(\mathbf{x})}$$

这在核空间中等价于检验：

$$H_0: \mathbf{x} \text{ 来自条件分布 } p(\cdot \mid c) \quad \text{vs.} \quad H_1: \mathbf{x} \text{ 来自边缘 } p(\cdot)$$

即 **RKHS 中的两样本检验**，而 `ood_head_v2`（10 维特征 → 二元分类）则是一个学习到的**核空间判别函数**。

---

## 6. GMM 边界采样器的核几何

`GMMBoundarySampler` 在特征空间 $\mathcal{H}$ 中拟合一个**高斯混合模型**：

$$p_\text{GMM}(\mathbf{x}) = \sum_{c=1}^{C} \pi_c \, \mathcal{N}(\mathbf{x}; \mu_c, \Sigma_c)$$

然后从 $p_\text{GMM}$ 的**低密度区域**（低于第 10 百分位）采样，这些样本位于 RKHS 中各类条件分布**支撑集的边界**——它们在流形附近但远离任何类中心，更接近真实的未知分布。

---

## 7. 总体数学本质

整个系统的数学本质可以概括为：

1. **核映射** $\phi$ 将音频映射到 64 维 RKHS $\mathcal{H}$
2. **条件神经样条流** $f_\theta$ 在 $\mathcal{H}$ 的 48 维子空间上构建条件密度 $p(\mathbf{x} \mid c)$
3. **样条耦合层**是逐维度的单调核变换，8 层叠加构成深度复合核
4. **训练目标**在 RKHS 中同时优化分类边界、先验匹配（$\mathcal{N}(0,I)$）、排斥区（非高斯惩罚）
5. **OOD 检测**通过似然比或判别函数在核空间中区分已知/未知样本

---

## 8. 代码结构对应

```
FeatureExtractor
    ↓ φ(x) ∈ ℝ⁶⁴
feature_adapter
    ↓ A·φ(x) ∈ ℝ⁶⁴
flow_dim_adapter
    ↓ ℝ⁶⁴ → ℝ⁴⁸
flow_pre_norm + projection
    ↓ z̃ ∈ ℝ⁴⁸
condition_network
    ↓ g_ψ(μ_c) ∈ ℝ⁴⁸
ConditionalNSF (8 layers)
    ↓ z = f_θ(z̃ | g_ψ(μ_c))
log p(x|c) = log N(z; 0, I) + log|det J_f|
```

---

## 参考文献

- Durkan et al., "Neural Spline Flows", NeurIPS 2019
- Gretton et al., "Kernel Mean Embedding of Distributions", 2012
- Arjovsky et al., "Wasserstein GAN", 2017
- Liu et al., "Energy-based Out-of-distribution Detection", 2020
