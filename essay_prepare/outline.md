# 基于标准化流与元学习的少样本声学场景开放集识别

---

## 摘要

本文提出了一种面向声学场景分类的少样本开放集识别（Few-Shot Open Set Recognition, FS-OSR）框架。该框架基于预训练音频神经网络（PANNs）骨干网络提取声学特征，通过标准化流（Normalizing Flow）建模类条件概率密度，结合原型网络（Prototypical Network）实现元学习分类。在训练阶段，采用分阶段的情节式元训练策略，通过高斯先验预热、多信号OOD检测头训练、GMM边界采样伪OOD增强等手段，使模型在已知类分类和未知类检测之间取得平衡。实验基于TAU Urban Acoustic Scenes 2022数据集，将10类声景分为6个基类和4个新类，评估了1-shot、5-shot、10-shot设定下的分类精度和多种OSR检测方法（特征马氏距离、隐空间一致性、流密度比、能量分数等）的性能。实验结果表明，所提方法在少样本分类和开放集识别两方面均取得了良好效果。

**关键词：** 开放集识别；少样本学习；标准化流；声学场景分类；元学习；原型网络

---

## 1 引言

### 1.1 研究背景

声学场景分类（Acoustic Scene Classification, ASC）是计算机听觉领域的核心任务之一，旨在将一段音频自动归类到预定义的声学环境类别中（如机场、有轨电车、公园等）。该技术在智能城市监控、环境感知、助听设备、无人驾驶等领域具有广泛应用前景。DCASE（Detection and Classification of Acoustic Scenes and Events）挑战赛自2013年起持续推动该领域的研究发展。

传统声学场景分类方法假设测试阶段出现的所有类别均已在训练阶段见过，即闭集识别（Closed-Set Recognition）。然而，在真实部署场景中，测试数据往往包含训练时未见过的新声学环境（如新的公共空间类型），这就要求系统不仅能正确分类已知类别，还能可靠地识别出未知类别——即开放集识别（Open Set Recognition, OSR）。

### 1.2 问题挑战

开放集声学场景识别面临以下核心挑战：

1. **特征空间中已知类与未知类的可分性**：未知类样本在特征空间中可能与某些已知类高度重叠，导致基于距离或密度的检测方法失效。
2. **少样本设定的数据稀缺性**：在实际应用中，难以获取大量标注数据。当只有少量新类别样本可用时（少样本学习），如何利用有限的支撑集（Support Set）快速建立对新类的认知是一个关键问题。
3. **分类与检测的平衡**：OSR需要在保持已知类分类精度的同时，尽可能准确地检测未知类。这两个目标之间往往存在冲突——过度增强未知类检测能力可能导致已知类的误判增加。
4. **声学数据的特殊性**：声学场景信号具有时变性和高噪声特性，且不同录音设备和环境条件引入域偏移（Domain Shift），给特征学习和泛化带来额外困难。

### 1.3 现有方法的局限性

当前FS-OSR领域的主要方法包括：

- **基于阈值的方法**：在分类器的softmax输出上设定阈值，低于阈值的判定为未知类。但softmax概率在OOD（Out-of-Distribution）样本上往往过度自信，导致可靠性不足。
- **基于距离的方法**：在特征空间中计算样本到各类原型的距离，距离过大则判定为未知类。但该方法依赖于特征空间的质量，对特征分布假设敏感。
- **基于生成模型的方法**：使用VAE、GAN或标准化流建模已知类分布，利用似然值进行OOD检测。但传统方法在少样本条件下难以有效训练。

这些方法通常独立处理分类和OOD检测，且在少样本场景下的泛化能力有限。

### 1.4 本文贡献

本文提出了一种融合标准化流密度估计与原型网络元学习的FS-OSR框架，主要贡献包括：

1. **多骨干特征提取器对比设计**：系统对比了YAMNet、Distil-AST和PANNs Cnn14三种预训练音频骨干网络，结合SE（Squeeze-and-Excitation）通道注意力机制，构建了从原始音频波形到64维判别性特征的提取管线。
2. **基于Neural Spline Flow的条件密度估计与OOD检测**：采用有理二次样条耦合层（Rational-Quadratic Spline Coupling）构建标准化流，将特征映射到标准正态隐空间，利用流的对数似然进行类条件密度估计；引入背景流（Background Flow）进行似然比检测，克服纯似然方法在OOD检测中的局限性。
3. **分阶段情节式元训练策略**：设计了四阶段训练管线——高斯先验预热→分类训练→辅助损失渐进→OSR结构化损失，通过温度调度的损失加权避免训练初期各损失项的冲突。
4. **GMM边界采样与多信号OOD检测**：提出了基于GMM低密度区域的伪OOD样本生成策略，替代传统随机噪声，使伪OOD样本更贴近真实未知类分布；设计了融合特征空间统计量、隐空间一致性、流密度比、OOD检测头等多信号的开放集检测方法。
5. **隐空间对比排斥损失（Latent Contrastive Repulsion, LCR）**：在流的隐空间中显式地将正确类的隐变量推向标准正态先验，同时将错误类的隐变量推离高斯分布，增强隐空间中已知类与未知类的可区分性。

---

## 2 相关工作

### 2.1 声学场景分类

- 传统方法：MFCC + GMM/SVM
- 深度学习方法：CNN、RNN、Transformer（AST、PANNs）
- DCASE挑战赛和TAU数据集的发展

### 2.2 开放集识别

- 基于阈值：OpenMax、ODIN
- 基于距离：马氏距离、原型距离
- 基于能量：Energy-based OOD
- 基于似然：标准化流似然比（LLR）、背景流校正

### 2.3 少样本学习

- 原型网络（Prototypical Network）
- 匹配网络（Matching Network）
- MAML
- 情节式训练（Episodic Training）

### 2.4 标准化流

- NICE、RealNVP
- Neural Spline Flow (Durkan et al., 2019)
- 条件流（cINN）
- 标准化流在OOD检测中的应用

### 2.5 少样本开放集识别

- 已有FS-OSR方法（PEELER、Transductive Finetuning等）
- 距离与密度融合的方法
- 元学习与OOD检测的结合

---

## 3 方法论

### 3.1 问题定义

给定：
- 基类集合 $\mathcal{C}_{base} = \{c_1, ..., c_K\}$，有充足的标注数据
- 新类集合 $\mathcal{C}_{novel} = \{c_{K+1}, ..., c_M\}$，仅有少量标注
- 支撑集 $\mathcal{S} = \{(x_i, y_i)\}$，其中 $y_i \in \mathcal{C}_{task}$，$\mathcal{C}_{task} \subseteq \mathcal{C}_{base} \cup \mathcal{C}_{novel}$
- 查询集 $\mathcal{Q} = \{x_j\}$

目标：对每个查询样本 $x_j$，若其属于 $\mathcal{C}_{task}$ 中的某个类，则正确分类；否则识别为"未知类"。

### 3.2 系统总体架构

整体流程分为四个阶段：

```
Phase 0: 特征提取器预训练（仅基类，防止数据泄露）
    ↓
Phase 1: 离线特征提取与缓存
    ↓
Phase 2: 情节式元训练（标准化流 + 原型网络）
    ↓
Phase 3: OSR校准与评估
```

### 3.3 特征提取器设计

#### 3.3.1 骨干网络

本文对比了三种AudioSet预训练骨干网络：

**（1）PANNs Cnn14_16k**（最终选用）

基于CNN架构，6个卷积块（通道数64→128→256→512→1024→2048），全局平均+最大池化输出2048维嵌入向量。

**（2）Distil-AST**

6层蒸馏版Audio Spectrogram Transformer，输出768维CLS嵌入。

**（3）YAMNet**

基于MobileNet的音频分类网络，输出521维嵌入。

#### 3.3.2 投影头与SE注意力

骨干网络输出经过三层全连接投影，每层融入SE（Squeeze-and-Excitation）通道注意力机制：

$$h_1 = \text{Dropout}(\text{SE}(\text{BN}(\text{GELU}(W_1 \cdot f))))$$

$$h_2 = \text{Dropout}(\text{SE}(\text{BN}(\text{GELU}(W_2 \cdot h_1))))$$

$$z = \text{BN}(W_3 \cdot h_2) \in \mathbb{R}^{64}$$

其中SE块的定义为：

$$\text{SE}(x) = x \cdot \sigma(W_{down} \cdot \text{ReLU}(W_{up} \cdot \text{AvgPool}(x)))$$

$\sigma$ 为Sigmoid函数，下采样和上采样层构成瓶颈结构（如256→64→256），实现通道维度的特征重标定。

#### 3.3.3 无数据泄露预训练

为防止未知类信息泄露到特征提取器，Phase 0仅在基类数据上训练。分类头采用两层MLP（64→32→K），训练使用交叉熵损失配合标签平滑（$\epsilon=0.1$）和30%概率的Mixup增强。

### 3.4 基于标准化流的密度建模

#### 3.4.1 Neural Spline Flow

核心的密度估计模块采用Neural Spline Flow（NSF），通过有理二次样条（Rational-Quadratic Spline）耦合层实现可逆变换。

**有理二次样条变换：** 给定 $K$ 个区间的节点位置 $\{x_k, y_k\}_{k=0}^K$ 和节点导数 $\{d_k\}_{k=0}^K$，输入 $x$ 在第 $k$ 个区间内的正向变换为：

$$y = y_k + h_k \cdot \frac{s_k \xi^2 + d_k \xi(1-\xi)}{s_k + (d_{k+1} + d_k - 2s_k)\xi(1-\xi)}$$

其中 $\xi = (x - x_k) / w_k \in [0,1]$ 为归一化坐标，$w_k = x_{k+1} - x_k$，$h_k = y_{k+1} - y_k$，$s_k = h_k / w_k$。

对数行列式为：

$$\log \left|\frac{\partial y}{\partial x}\right| = \log h_k - \log w_k + \log \alpha - 2\log(\text{denom})$$

其中 $\alpha = d_k(1-\xi)^2 + 2s_k\xi(1-\xi) + d_{k+1}\xi^2$。

**条件耦合层：** 将输入分为 $x_1, x_2$ 两部分，交替变换：

$$z_1 = \text{Spline}(x_1; \theta_1(x_2, c)), \quad z_2 = \text{Spline}(x_2; \theta_2(z_1, c))$$

其中 $\theta_1, \theta_2$ 为预测样条参数的神经网络，$c$ 为条件向量。层间采用固定排列（维度翻转）以增强维度间信息交换。

#### 3.4.2 流维度适配

为提高流模型的数值稳定性和表达能力，设计了维度适配管线：

$$f_{adapted} = \text{LayerNorm}(\text{Linear}(f_{64})) \in \mathbb{R}^{64}$$

$$f_{flow} = \text{LayerNorm}(\text{Linear}(f_{adapted})) \in \mathbb{R}^{48}$$

$$f_{proj} = W_2 \cdot \text{Dropout}(\text{ReLU}(W_1 \cdot f_{flow})) \in \mathbb{R}^{48}$$

条件网络将原型特征映射到与流输入相同的空间：

$$c_{proto} = \text{LayerNorm}(\text{Linear}(\text{ReLU}(\text{Linear}(p_{flow})))) \in \mathbb{R}^{48}$$

#### 3.4.3 类条件对数似然计算

对于 $N$-way分类，将每个查询样本 $x$ 与 $N$ 个原型 $\{p_c\}_{c=1}^N$ 配对，分别计算在对应类条件下的对数似然：

$$\log p(x | p_c) = \log p_{prior}(z_c) + \log \left|\det \frac{\partial z_c}{\partial x}\right|$$

其中 $z_c = f_\phi(x; c=p_c)$ 为条件流的前向映射，$p_{prior} = \mathcal{N}(0, I)$。

### 3.5 原型网络与流密度融合分类

#### 3.5.1 原型距离分数

在64维特征空间中计算查询样本到原型的加权负欧氏距离：

$$s_{proto}(x, p_c) = -\tau \cdot \|h(x) - h(p_c)\|_2^2$$

其中 $h(\cdot)$ 为可训练的距离投影头，$\tau$ 为可学习的温度参数（初始化为10）。

#### 3.5.2 分数融合

流密度分数与原型距离分数通过中心化后求和融合：

$$s_{combined}(x, c) = (s_{proto}(x, c) - \bar{s}_{proto}) + (s_{flow}(x, c) - \bar{s}_{flow})$$

中心化确保两个分数尺度可比，最后通过平滑截断 $\tanh(\cdot / 10) \times 10$ 防止极端分数值。

### 3.6 情节式元训练

#### 3.6.1 情节采样

每个训练情节（Episode）从基类中随机选取 $N$ 个类，每个类采样 $K$ 个支撑样本和 $Q$ 个查询样本。支撑集计算原型：

$$p_c = \frac{1}{K} \sum_{i: y_i = c} z_i$$

训练时对原型加入高斯噪声 $\epsilon \sim \mathcal{N}(0, 0.1^2 I)$ 以增强鲁棒性。

#### 3.6.2 任务增强

- **支撑集随机丢弃**：以概率 $p_{drop}=0.2$ 随机移除支撑样本，模拟更少样本的场景
- **特征级增强**：对查询特征施加随机缩放 $\alpha \in [0.8, 1.2]$ 和高斯噪声 $\epsilon \sim \mathcal{N}(0, 0.02^2 I)$

### 3.7 多组件损失函数

#### 3.7.1 分类损失

$$\mathcal{L}_{CE} = \text{CE\_cap} \cdot \tanh\left(\frac{-\frac{1}{B}\sum_{i=1}^B \log \frac{e^{s(x_i, y_i)}}{\sum_c e^{s(x_i, c)}}}{\text{CE\_cap}}\right)$$

使用 $\tanh$ 截断防止梯度爆炸（CE\_cap=10）。

#### 3.7.2 高斯先验匹配损失

推动正确类隐变量 $z_{correct}$ 服从标准正态分布：

$$\mathcal{L}_{Gaussian} = \frac{|\text{mean}(\|z_{correct}\|^2) - D|}{D} + \frac{1}{D}\sum_{j=1}^D (\text{std}(z_{correct}^{(j)}) - 1)^2$$

第一项确保隐变量模长符合 $\chi^2(D)$ 分布的期望（$E[\|z\|^2] = D$），第二项确保每个维度的方差为1。

#### 3.7.3 隐空间对比排斥损失（LCR）

在隐空间中将错误类的隐变量推离正确类隐变量：

$$\mathcal{L}_{LCR} = \frac{1}{|\mathcal{N}|}\sum_{c \neq y} \max(0, m - \|z_c - z_{correct}\|_2^2)$$

其中 $m$ 为间隔参数（margin=32）。

#### 3.7.4 非高斯排斥损失

将错误类隐变量推离标准正态分布：

$$\mathcal{L}_{NonGauss} = \frac{1}{|\mathcal{N}| \cdot D} \sum_{c \neq y} \max(0, 1.5D - \|z_c\|_2^2)$$

确保错误条件下的隐变量不落入高斯分布区域，从而增强已知类与未知类的隐空间可分性。

#### 3.7.5 伪OOD排斥损失

利用伪OOD样本（GMM边界采样+真实OOD）降低其对所有类的似然：

$$\mathcal{L}_{pseudo} = \text{mean}(\text{logsumexp}(\log p(x_{ood} | p_c)))$$

#### 3.7.6 密度最大化损失

$$\mathcal{L}_{density} = \text{clamp}(-\frac{1}{B}\sum_{i=1}^B \log p(x_i | p_{y_i}), -5, 5)$$

#### 3.7.7 熵最小化损失

$$\mathcal{L}_{entropy} = -\frac{1}{B}\sum_{i=1}^B \sum_c \text{softmax}(s_i)_c \cdot \log \text{softmax}(s_i)_c$$

#### 3.7.8 原型分离损失

$$\mathcal{L}_{sep} = \text{clamp}(-\text{mean}(\text{pdist}(\{p_c\})), -2, 0)$$

#### 3.7.9 总损失与分阶段调度

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \alpha_{cls}\mathcal{L}_{aux}^{cls} + \alpha_{gauss}\mathcal{L}_{gauss} + \alpha_{struct}\mathcal{L}_{struct}$$

各权重系数按训练阶段逐步提升：

| 阶段 | Episode范围 | $\alpha_{cls}$ | $\alpha_{gauss}$ | $\alpha_{struct}$ |
|------|-------------|----------------|-------------------|-------------------|
| Phase 0: 高斯预热 | 0~150 | 0 | 1.0 | 0 |
| Phase 1: CE+高斯 | 150~250 | 0 | 0.8 | 0 |
| Phase 2: 分类辅助 | 250~450 | 0→1 | 0.5→1 | 0 |
| Phase 3: 全部损失 | 450+ | 1.0 | 1.0 | 0→1 |

此设计的核心思想是：先让流收敛到高斯映射（Phase 0），再引入分类信号（Phase 1），逐步加入结构化约束（Phase 2-3），避免训练初期各损失项相互冲突。

### 3.8 OOD检测方法

#### 3.8.1 特征马氏距离

在特征空间中计算查询样本到各原型的马氏距离，取最小值作为已知度分数。

#### 3.8.2 隐空间一致性（z-consistency）

查询样本在不同类条件下的隐变量分布一致性。若 $z_c = f(x; p_c)$ 在不同 $c$ 下高度一致，说明样本不属于任何特定已知类。

#### 3.8.3 隐空间间隔（z-gap）

正确类与次优类的隐变量距离差异。

#### 3.8.4 流似然比（Likelihood Ratio）

引入无条件背景流 $f_{bg}$，利用似然比进行OOD检测：

$$\text{LR}(x) = \log p(x | p_c) - \log p_{bg}(x)$$

背景流在所有基类特征上训练，建模数据的边缘分布。似然比消除了"输入复杂度高→所有条件似然都高"的问题。

#### 3.8.5 能量分数（Energy-based）

$$E(x) = -T \cdot \log \sum_{c=1}^N \exp\left(\frac{\log p(x|p_c)}{T}\right)$$

能量越高越可能是未知类。

#### 3.8.6 OOD检测头

训练一个二分类MLP，输入为5维特征空间统计量：

$$\text{feat}_{ood} = [\min_c d(x, p_c), \frac{d_{min}}{d_{2nd\_min}}, \max_c \text{softmax}_c, H(\text{softmax}), \|x\|_2]$$

### 3.9 GMM边界采样

传统伪OOD生成方法（纯高斯噪声）脱离数据流形，不能有效训练OOD检测。本文提出GMM边界采样策略：

1. 用 $K$ 分量GMM拟合基类特征
2. 生成大量候选样本，计算GMM对数似然
3. 保留低于训练数据第10百分位数的低密度样本
4. 同时使用60%真实OOD样本 + 40% GMM边界样本

这种方法生成的伪OOD样本位于特征流形的边界区域，更接近真实未知类分布。

---

## 4 实验设置

### 4.1 数据集

**TAU Urban Acoustic Scenes 2022 Mobile Development**

- 10类声学场景：机场、有轨电车、公交、公共广场、购物中心、步行街、地铁站、交通干道、地铁、公园
- 基类（6类）：{0-5}，用于训练和少样本评估
- 新类（4类）：{6-9}，仅用于OSR评估
- 数据划分：train / calib / test（按CSV文件划分）
- 音频预处理：重采样至16kHz，分割/填充至1秒片段

### 4.2 训练配置

| 参数 | 值 |
|------|------|
| 骨干网络 | PANNs Cnn14_16k |
| 特征维度 | 64 |
| 流维度 | 48 |
| NSF耦合层数 | 12 |
| 样条区间数 | 8 |
| N-way / K-shot / Q-query | 5 / 5 / 30 |
| 训练Episodes | 5000 |
| 优化器 | AdamW (lr=1.5e-5, wd=0.01) |
| 学习率调度 | CosineAnnealing |
| 梯度累积 | 1步 |
| 原型噪声标准差 | 0.10 |
| OOD比率 | 0.4 |
| 早停patience | 800 episodes |

### 4.3 评估指标

**少样本分类：**
- 1-shot、5-shot、10-shot准确率
- 分别在基类和新类上评估

**开放集识别：**
- TNR（True Negative Rate）：已知类正确保留率
- TPR（True Positive Rate）：未知类正确检测率
- OSR Score：$(TNR + TPR) / 2$

### 4.4 对比方法

- 仅原型距离分类（无流）
- Softmax阈值法
- 特征马氏距离
- 标准化流似然
- 流似然比（背景流校正）
- 能量分数
- OOD检测头

---

## 5 项目实施中创新思维和创新实践方面的收获

### 5.1 问题驱动的系统设计思维

在项目初期，我们面对的是一个看似简单的"声学分类+异常检测"问题，但深入分析后发现需要同时解决数据泄露、特征空间质量、少样本泛化、分类与检测平衡等多个相互耦合的子问题。这促使我们从系统层面设计四阶段训练管线，每个阶段解决一个核心矛盾：

- Phase 0解决数据泄露问题（只在基类上训练特征提取器）
- Phase 1解决计算效率问题（离线提取特征，避免重复前向传播）
- Phase 2解决少样本泛化问题（情节式训练模拟部署时的低数据场景）
- Phase 3解决OSR阈值校准问题（在独立校准集上确定检测阈值）

这一经历让我们深刻认识到：复杂系统的设计不应追求一步到位的完美方案，而应识别各子问题的核心矛盾，通过分阶段策略逐步化解。

### 5.2 从失败中学习：数据泄露的发现与修复

在早期实验中，我们使用`osr_training_model.py`中的方法，FC层在全类别（含unknown类）上训练，导致FC权重编码了未知类信息。这一数据泄露问题使得few-shot阶段提取的特征不再是"纯净的"——模型已经"见过"未知类的统计信息。

修复方案是引入`BaseClassPretrainer`，仅在基类数据上训练特征提取器。这一教训让我们意识到：在few-shot和OSR的交叉领域，数据划分的纯净性至关重要，任何预训练阶段的标签泄露都会严重损害模型的泛化能力。

### 5.3 标准化流在OOD检测中的工程洞察

理论上，标准化流提供了精确的概率密度估计，可以直接用似然值进行OOD检测。但实践表明，流的似然值受到输入复杂度的影响——"复杂的输入即使不在训练分布内也可能获得高似然值"（类似图像领域的"Theory of Typicality"现象）。

为解决这一问题，我们引入了背景流（Background Flow）进行似然比校正：

$$\text{score}(x) = \log p_{class}(x) - \log p_{bg}(x)$$

背景流建模数据的边缘分布，似然比消除输入复杂度的影响，使OOD检测更加鲁棒。这一改进体现了"理论指导实践，实践修正理论"的工程研发思路。

### 5.4 损失函数设计的分阶段策略

在训练初期，我们尝试将所有损失项（分类、密度、高斯先验、排斥、伪OOD等）同时使用，结果发现模型训练极不稳定——各损失项的梯度方向相互矛盾，导致损失震荡甚至发散。

通过深入分析各损失项的依赖关系，我们设计了一个分阶段的损失调度策略：

1. **先让流收敛**（仅高斯先验损失，150 episodes）：流首先学会将特征映射到标准正态分布，建立良好的隐空间结构。
2. **再学习分类**（加入CE损失，100 episodes）：在流已收敛的基础上学习分类。
3. **逐步加入辅助损失**（分类辅助损失线性增加，200 episodes）：密度最大化、熵最小化等辅助分类损失逐步介入。
4. **最后加入结构化OSR损失**（伪OOD排斥、LCR等，后续所有episodes）：在分类稳定后，增强隐空间中已知类与未知类的可分性。

这种"先建基础再精装修"的策略，使模型训练的稳定性大幅提升。

### 5.5 伪OOD样本的生成策略演进

在OSR训练中，伪OOD样本的质量直接影响模型区分已知类与未知类的能力。我们的策略经历了以下演进：

1. **随机噪声**：直接使用高斯噪声作为伪OOD。效果差，因为随机噪声脱离数据流形，流模型可以轻松区分——但真实unknown类并非随机噪声。
2. **类间Mixup**：用不同类特征的凸组合作为伪OOD。比随机噪声更合理，但仍在数据流形上，可能误伤类间过渡区域的真实样本。
3. **GMM边界采样**（最终方案）：用GMM拟合基类分布后，从低密度区域采样。这些样本位于数据流形的边界附近——不像随机噪声那样远离流形，又不属于任何已知类的核心区域——因此更接近真实unknown类的分布特性。

这一演进过程让我们认识到：生成方法的有效性取决于生成样本与目标分布的匹配程度，而非生成方法的复杂度。

### 5.6 多信号融合的OOD检测

实验中发现，单一的OOD检测方法在不同数据分布下表现差异很大。例如：
- 特征马氏距离在特征空间质量好时表现优异，但对域偏移敏感
- 流似然比对输入复杂度鲁棒，但需要额外的背景流训练
- OOD检测头训练简单，但泛化能力依赖于训练数据的多样性

因此，我们设计了多信号融合策略，综合特征空间统计量（距离、距离比、范数）和流空间信息（似然、隐变量一致性）进行OOD判断，提高了检测的鲁棒性。

### 5.7 软件工程实践

在项目实施过程中，我们采用了多项软件工程最佳实践：

- **模块化设计**：将流模型（cINN、NSF）、特征提取器、损失函数、训练器分离为独立模块，便于替换和对比实验。
- **版本控制**：通过`CACHE_VERSION`和`_model_version`机制管理实验版本，自动检测配置变更并重新训练。
- **特征缓存**：预提取特征并缓存到磁盘，避免每个epoch重复计算，加速训练循环约10倍。
- **GPU优化**：使用混合精度训练、TF32加速、GPU端mel频谱提取，最大化GPU利用率。
- **梯度累积与NaN保护**：检测NaN/Inf损失并跳过该episode，防止模型参数被损坏。

### 5.8 对标准化流模型的深入理解

通过本项目，我们对标准化流的数学原理和工程实现有了深刻理解：

- **可逆性保证**：仿射耦合层和样条耦合层的可逆性是标准化流的核心——它保证了我们可以精确计算对数行列式 $\log|\det J|$，从而获得精确的概率密度。
- **条件流的设计**：条件信息的注入方式直接影响流的表达能力。我们通过条件网络对原型信息进行非线性变换后再输入流，比直接拼接原始原型获得更好的效果。
- **s_clamp机制**：仿射耦合中的 $s_{clamped} = s_{max} \cdot \tanh(s / s_{max})$ 限制了指数缩放的范围，防止数值不稳定。
- **样条导数L2正则化**：限制样条导数的幅度，避免极端非线性变换。

### 5.9 创新总结

本项目的创新实践可以总结为以下几点方法论启示：

1. **系统性思维**：面对多目标优化问题（分类+检测+少样本泛化），采用分阶段策略逐个击破，而非追求统一损失的一次性优化。
2. **失败驱动的迭代**：数据泄露的发现、训练不稳定性的分析、OOD样本质量的问题——每次失败都推动了对问题本质的更深层理解。
3. **理论与实践的闭环**：高斯先验的理论保证（$z \sim \mathcal{N}(0,I)$）指导了训练策略的设计，而实践中的数值稳定性问题又催生了s_clamp和自适应z范数惩罚等工程方案。
4. **对比实验驱动决策**：系统对比三种骨干网络、多种OOD检测方法，用实验数据而非直觉指导最终方案的选择。

---

## 6 实验结果与分析（待补充）

### 6.1 少样本分类结果

### 6.2 OSR检测对比

### 6.3 消融实验

- 流密度 vs 仅原型距离
- GMM边界采样 vs 随机噪声
- 分阶段训练 vs 一次性训练
- 各损失项的贡献

### 6.4 可视化分析

- t-SNE特征分布可视化（基类 vs 新类）
- 隐空间分布可视化
- 训练损失曲线

---

## 7 结论与展望

### 7.1 结论

### 7.2 局限性

### 7.3 未来工作

- 更大规模数据集验证
- 跨数据集泛化
- 在线增量学习
- 轻量化部署

---

## 参考文献

[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural spline flows. *NeurIPS*.

[2] Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *NeurIPS*.

[3] Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). PANNs: Large-scale pretrained audio neural networks for audio pattern recognition. *IEEE/ACM TASLP*.

[4] Mesaros, A., Heittola, T., & Virtanen, T. (2018). A multi-device dataset for urban acoustic scene classification. *DCASE Workshop*.

[5] Bendale, A., & Boult, T. E. (2016). Towards open set deep networks. *CVPR*.

[6] Liu, W., Wang, X., Owens, J., & Li, Y. (2020). Energy-based out-of-distribution detection. *NeurIPS*.

[7] Ren, J., et al. (2019). Likelihood ratios for out-of-distribution detection. *NeurIPS*.

[8] Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*.

[9] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML*.

[10] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *CVPR*.
