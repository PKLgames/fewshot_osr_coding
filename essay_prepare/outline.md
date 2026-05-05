# 基于原型学习与聚类的小样本未知环境声音发现方法

## 毕业论文大纲（约12000字）

> **图/表/公式/文献标注说明：**
> - `【图X.Y】` 表示第X章第Y张图
> - `【表X.Y】` 表示第X章第Y张表
> - `【公X.Y】` 表示第X章第Y个公式
> - `【引N】` 表示引用第N条参考文献（见末尾参考文献列表）
> - `已有图片` 表示实验中已生成的图片可直接使用

---

## 摘要（约500字）

本文研究小样本开放集声学场景识别（Few-Shot Open Set Acoustic Scene Recognition, FS-OSR）问题，即在每类仅有少量标注样本的条件下，系统需同时完成已知类的准确分类与未知类的可靠检测。针对传统方法在样本稀少且存在未知类别干扰时性能不足的缺陷，提出一种基于查询依赖反原型（Query-Dependent Anti-Prototype）与高斯混合模型（Gaussian Mixture Model, GMM）聚类增强的双信号融合框架。在几何层面，利用查询批次均值动态计算反原型中心，以零额外参数刻画查询样本在原型-反原型空间中的位置关系，克服传统固定中心在训练与测试阶段分布不一致的问题。在统计层面，构建包含GMM聚类马氏距离（Mahalanobis Distance）、最大后验概率（Maximum A Posteriori Probability）和后验分布信息熵（Posterior Entropy）三个新增特征的13维扩展特征向量，通过逻辑回归（Logistic Regression）输出分布外（Out-of-Distribution, OOD）检测评分。最终通过Z分数归一化（Z-score Normalization）与尤登J统计量（Youden's J Statistic）搜索最优融合权重，实现几何信号与统计信号的自适应融合。在TAU-2022声学场景数据集上的实验表明，本文方法在真阴性率（True Negative Rate, TNR）为95%的约束下，真阳性率（True Positive Rate, TPR）达到24.94%，开放集识别（Open Set Recognition, OSR）综合得分为59.97%，相比基线方法TPR相对提升41.6%。

**关键词：** 小样本学习；开放集识别；原型网络；高斯混合模型；声学场景分类

## Abstract

This paper addresses the problem of Few-Shot Open Set Acoustic Scene Recognition (FS-OSR), where the system must simultaneously achieve accurate classification of known classes and reliable detection of unknown classes with only a few labeled samples per category. To overcome the performance limitations of conventional methods under scarce samples and interference from unknown categories, we propose a dual-signal fusion framework based on Query-Dependent Anti-Prototype scoring and Gaussian Mixture Model (GMM) cluster-enhanced statistical detection. At the geometric level, the anti-prototype center is dynamically computed using the query batch mean, characterizing the position of query samples in the prototype–anti-prototype space with zero additional parameters and overcoming the distribution mismatch between training and testing that arises with fixed centers. At the statistical level, we construct a 13-dimensional extended feature vector that incorporates three novel GMM clustering features — cluster Mahalanobis distance, maximum a posteriori probability, and posterior entropy — and feed it into a logistic regression classifier to produce an out-of-distribution (OOD) detection score. The two signals are then normalized via Z-score normalization and adaptively fused using the optimal weight searched by maximizing Youden's J statistic on the validation set. Experiments on the TAU-2022 acoustic scene dataset demonstrate that the proposed method achieves a True Positive Rate (TPR) of 24.94% under the strict constraint of 95% True Negative Rate (TNR), yielding an Open Set Recognition (OSR) score of 59.97% and a relative TPR improvement of 41.6% over the baseline.

**Keywords:** Few-Shot Learning; Open Set Recognition; Prototypical Network; Gaussian Mixture Model; Acoustic Scene Classification

---

## 目录

- 第一章 绪论
- 第二章 相关数学与理论基础
- 第三章 基于原型学习与聚类的OSR算法设计
- 第四章 实验设计与结果分析
- 第五章 结论与展望
- 参考文献
- 致谢
- 附录

---

## 第一章 绪论（约2000字）

### 1.1 研究背景与意义（约400字）

- **应用背景**：我国已进入深度老龄化社会（65岁以上占15.4%），大量独居老人面临摔倒等意外风险。可穿戴设备采集环境音频进行行为识别，具有非侵入、低功耗、全天候适用等优点【引24】【引27】
- **需解决的核心问题**：该应用场景同时面临两个技术挑战——
  - （1）**小样本问题**：个性化需求不同，新增音频事件类型时难以获取大量标注样本，需从极少样本（5~10个）中快速学习新类别【引6】
  - （2）**开放集问题**：真实环境存在训练阶段未定义的未知声音事件（异常噪音等），系统必须能识别"未知"而非强制分类到已知类【引7】
- **本文解决方案**：将上述挑战统一为小样本开放集声学场景识别（FS-OSR）问题，提出查询依赖反原型与GMM聚类增强的双信号融合框架，在TAU-2022数据集上验证有效性

### 1.2 国内外研究现状（约800字）

#### 1.2.1 小样本学习方法

- 原型网络（Prototypical Network）【引2】：基于度量学习，将分类问题转化为嵌入空间中的最近邻问题
- 匹配网络（Matching Network）【引1】：基于注意力机制的少样本分类
- MAML【引3】：基于元学习的快速适应方法
- 关系网络（Relation Network）【引4】：学习样本间的可度量关系
- 小样本学习综述【引6】

#### 1.2.2 开放集识别方法

- 基于概率的方法：OpenMax【引8】，基于极值理论（EVT）校准softmax输出；基线方法MSP【引9】
- 基于距离的方法：Mahalanobis距离检测【引12】，利用类条件高斯假设；ODIN【引11】
- 反原型方法（Reciprocal Point / Anti-Prototype）：RPL【引13】通过学习已知类的对偶点界定未知空间；ARPL【引15】引入对抗训练增强开放空间边界
- 能量方法（Energy-based）【引14】：利用能量函数区分已知/未知
- OSR综述【引16】

#### 1.2.3 声学场景识别

- DCASE挑战赛综述【引24】【引28】
- TAU数据集系列【引25】【引29】
- 音频预训练模型：PANNs【引27】、VGGish【引23】、YAMNet
- 现有FS-OSR在音频领域的研究空白：绝大多数工作集中在视觉领域（CIFAR、miniImageNet）

> **【表1.1】** OSR方法分类对比表：按方法类型（概率/距离/反原型/能量）分类，列出各方法的代表工作、核心思想、优缺点，约5-6行。

### 1.3 本文工作与创新点（约300字）

- 提出查询依赖的反原型评分方法：反原型中心由查询样本动态计算，零额外参数，避免固定中心的泛化不足（vs RPL【引13】的固定可学习反原型）
- 提出GMM聚类增强的统计OOD检测：利用高斯混合模型的全局聚类结构（Mahalanobis距离、后验概率、后验熵）丰富OOD特征（vs Lee et al.【引12】的单一Mahalanobis距离）
- 设计几何-统计双信号融合框架：Z-score归一化 + Youden J最优权重搜索（vs Deep Ensemble【引22】的复杂集成策略）
- 在TAU-2022声学场景数据集的小样本开放集设定下验证方法有效性

### 1.4 论文组织结构（约300字）

- 简述各章节内容安排

---

## 第二章 相关数学与理论基础（约2500字）

### 2.1 小样本学习与原型网络（约500字）

- **问题形式化定义**：给定支持集 $\mathcal{S} = \{(x_i, y_i)\}_{i=1}^{N \times K}$（$N$ 个类别，每类 $K$ 个样本）和查询集 $\mathcal{Q} = \{x_j\}$，目标是学习映射 $f_\theta: \mathcal{X} \to \mathcal{Y} \cup \{\text{unknown}\}$

> **【公2.1】** 类原型计算：$\mathbf{p}_c = \frac{1}{|S_c|} \sum_{(x_i, y_i) \in S_c} f_\phi(x_i)$

> **【公2.2】** 原型网络分类概率：$P(y=c|x) = \frac{\exp(-d(f_\phi(x), \mathbf{p}_c))}{\sum_{c'} \exp(-d(f_\phi(x), \mathbf{p}_{c'}))}$ 【引2】

- 其中 $d(\cdot, \cdot)$ 为欧氏距离的平方：$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|_2^2$
- **度量学习的统计解释**：原型网络等价于假设各类特征服从各向同性高斯分布 $P(f_\phi(x)|y=c) = \mathcal{N}(\mathbf{p}_c, \sigma^2 I)$，分类等价于最大似然估计

> **【图2.1】** 原型网络示意图：在2D嵌入空间中展示支持集样本（少量点）、类原型（星号）、查询样本的分类过程。展示不同类别用不同颜色，决策边界用虚线。与图1.1（FS-OSR问题）形成前后呼应。

### 2.2 开放集识别的数学框架（约600字）

- **OSR问题形式化**【引7】【引16】：设已知类别集合 $\mathcal{C}_{\text{known}} = \{1, 2, \ldots, N\}$，未知类别集合 $\mathcal{C}_{\text{unknown}} = \{N+1, N+2, \ldots\}$。OSR目标：$g: \mathcal{X} \to \{1, 2, \ldots, N\} \cup \{\text{unknown}\}$

> **【公2.3】** OSR决策函数：
> $$g(x) = \begin{cases} \arg\max_c P(y=c|x) & \text{if } s(x) \geq \tau \\ \text{unknown} & \text{if } s(x) < \tau \end{cases}$$

- **评价指标**：

> **【公2.4】** 混淆矩阵定义TPR、TNR：
> $$\text{TPR} = \frac{TP}{TP + FN}, \quad \text{TNR} = \frac{TN}{TN + FP}$$

> **【公2.5】** OSR综合得分：$\text{OSR} = \frac{\text{TNR} + \text{TPR}}{2}$

> **【公2.6】** Youden J统计量【引35】：$J = \text{TNR} + \text{TPR} - 1$

> **【表2.1】** OSR混淆矩阵表：行=真实标签（已知/未知），列=预测标签（已知类c/未知），标注TP、FP、TN、FN的位置。

### 2.3 反原型方法（约400字）

- **核心思想**【引13】【引15】：对每个已知类原型 $\mathbf{p}_c$，定义其对偶点（反原型）$\mathbf{a}_c$，反原型应远离已知类特征分布
- **固定反原型 vs 查询依赖反原型**：

> **【公2.7】** 固定中心反原型：$\mathbf{a}_c = 2\bar{\mathbf{p}} - \mathbf{p}_c$，其中 $\bar{\mathbf{p}} = \frac{1}{N}\sum_{c=1}^N \mathbf{p}_c$

> **【公2.8】** 查询依赖反原型：$\mathbf{a}_c = 2\hat{\mathbf{c}} - \mathbf{p}_c$，其中 $\hat{\mathbf{c}} = \frac{1}{B}\sum_{i=1}^B f_\phi(x_i)$

> **【公2.9】** 反原型评分函数：$s(x) = -\min_c d(f_\phi(x), \mathbf{p}_c) + \min_c d(f_\phi(x), \mathbf{a}_c)$

- 直觉解释：已知类样本靠近原型、远离反原型（$s(x)$ 高）；未知类样本远离原型、靠近反原型（$s(x)$ 低）

> **【图2.2】** 反原型几何示意图：在2D特征空间中展示原型、固定中心反原型、查询依赖反原型的位置关系，标注已知/未知样本的评分差异。特别突出固定中心 vs 查询依赖中心的区别。

### 2.4 高斯混合模型与聚类分析（约500字）

- **GMM的概率模型**【引10】【引36】：

> **【公2.10】** GMM混合密度：$P(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$

- 参数估计：EM算法【引36】，E步计算后验 $P(z=k|\mathbf{x})$，M步更新参数
- 协方差矩阵 $\boldsymbol{\Sigma}_k$ 包含类内特征的二阶统计信息

> **【公2.11】** Mahalanobis距离【引12】：$d_M(\mathbf{x}, \boldsymbol{\mu}_k) = \sqrt{(\mathbf{x} - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k)}$

- 统计意义：在正态假设下，Mahalanobis距离的平方服从 $\chi^2$ 分布；当 $\boldsymbol{\Sigma}_k = \sigma^2 I$ 时退化为欧氏距离

> **【公2.12】** GMM后验概率（软分配）：$\gamma_k(\mathbf{x}) = P(z=k|\mathbf{x}) = \frac{\pi_k \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}|\boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$

> **【公2.13】** 后验分布信息熵：$H(\mathbf{x}) = -\sum_{k=1}^K \gamma_k(\mathbf{x}) \log \gamma_k(\mathbf{x})$

- 直觉：已知类样本的后验概率应集中于某个component（低熵），未知类样本的后验分布应较为分散（高熵）

### 2.5 逻辑回归与分类器融合（约500字）

- **逻辑回归的数学形式**【引10】：

> **【公2.14】** 逻辑回归概率输出：$P(y=1|\mathbf{h}) = \sigma(\mathbf{w}^\top \mathbf{h} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{h} + b)}}$

> **【公2.15】** 交叉熵损失：$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$

- 正则化：$L_2$正则化等价于参数的高斯先验（贝叶斯视角）

> **【公2.16】** Z-score标准化：$\hat{s} = \frac{s - \mu}{\sigma}$

> **【公2.17】** 加权线性融合：$s_{\text{ens}}(x) = \alpha \cdot \hat{s}_{\text{anti}}(x) + (1-\alpha) \cdot \hat{s}_{\text{ood}}(x)$

- 融合权重 $\alpha$ 通过在验证集上最大化Youden J统计量确定【引35】
- 集成学习的不确定性估计理论基础【引21】【引22】

---

## 第三章 基于原型学习与聚类的OSR算法设计（约3500字）

### 3.1 整体框架概述（约400字）

> **【图3.1】** 系统整体架构图（最重要的一张图，需精心绘制）：
> - 左侧：音频输入 → YAMNet特征提取 → 64维嵌入向量
> - 中间上：Episodic元训练（支持集→原型→分类损失）
> - 中间下：OSR校准（分两路）
>   - 路径A（几何信号）：查询依赖反原型评分 → s_anti
>   - 路径B（统计信号）：13维特征提取（含GMM聚类特征）→ 逻辑回归 → s_ood
> - 右侧：Z-score归一化 → 加权融合 → 已知/未知判定
> - 用不同颜色框标注各模块，箭头标注数据流向

### 3.2 特征提取与原型计算（约500字）

#### 3.2.1 音频特征提取

- YAMNet（Yet Another Mobile Network）架构【引23】：基于MobileNet v1的音频分类模型，训练于AudioSet数据集
- 输入：16kHz单声道音频 → log-mel频谱图 → YAMNet → 64维嵌入

> **【公3.1】** 特征适配器：$\mathbf{z} = \text{LayerNorm}(W\mathbf{x} + \mathbf{b})$

> **【公3.2】** 距离头：$\mathbf{z}' = W_2 \cdot \text{GELU}(\text{Dropout}(\text{LayerNorm}(W_1 \mathbf{z})))$

> **【图3.2】** 特征提取网络结构图：YAMNet（冻结）→ Feature Adapter（可训练）→ Distance Head（可训练）→ 64维输出，标注哪些层冻结、哪些层可训练。

#### 3.2.2 原型计算

> **【公3.3】** 原型计算：$\mathbf{p}_c = \frac{1}{K}\sum_{i=1}^K f_\phi(x_i^{(c)})$ 【引2】

> **【公3.4】** Episodic分类：$P(y=c|x) = \text{softmax}(-\|f_\phi(x) - \mathbf{p}_c\|_2^2 \cdot \tau)$

- 温度参数 $\tau$ 的作用：控制概率分布的锐度

### 3.3 查询依赖反原型评分（约700字）

#### 3.3.1 反原型的几何构造

- **问题分析**：固定中心 $\bar{\mathbf{p}} = \frac{1}{N}\sum_c \mathbf{p}_c$ 在训练集和测试集上不一致（训练时可见全部已知类，测试时仅可见当前episode中的已知类子集），导致反原型偏移
- 这与RPL【引13】的方法不同：RPL学习固定反原型参数，本文方法零参数动态计算

> **【公3.5】** 查询依赖中心：$\hat{\mathbf{c}}_Q = \frac{1}{|Q|}\sum_{x \in Q} f_\phi(x)$

> **【公3.6】** 反原型计算：$\mathbf{a}_c = 2\hat{\mathbf{c}}_Q - \mathbf{p}_c$

> **【图3.3】** 查询依赖反原型构造示意图：
> - 左子图：训练场景，展示全部6个已知类的原型、固定中心、反原型位置
> - 右子图：测试场景（episode中仅2个已知类），展示查询依赖中心与固定中心的差异
> - 标注：固定中心导致的反原型偏移 vs 查询依赖中心的正确位置

#### 3.3.2 评分函数

> **【公3.7】** 反原型评分：
> $$s_{\text{anti}}(x) = -\underbrace{\min_{c=1,\ldots,N} \|f_\phi(x) - \mathbf{p}_c\|_2}_{d_{\text{proto}}(x)} + \underbrace{\min_{c=1,\ldots,N} \|f_\phi(x) - \mathbf{a}_c\|_2}_{d_{\text{anti}}(x)}$$

- **决策规则**：$s_{\text{anti}}(x) \geq \tau_{\text{anti}} \Rightarrow \text{known}$，否则 $\Rightarrow \text{unknown}$
- **零额外参数**：反原型由查询样本和原型动态计算，无需额外训练参数

#### 3.3.3 查询依赖中心有效性的实验验证

> **【表3.1】** 查询依赖中心 vs 固定中心对比：
>
> | 中心类型 | TPR (%) | 说明 |
> |----------|---------|------|
> | 固定中心 (prototypes.mean()) | 8.32 | 训练-测试不一致导致性能下降 |
> | 查询依赖中心 (query.mean()) | 17.45 | 自适应调整参考坐标系 |

- 结论：查询依赖设计使TPR提升约9个百分点，验证了设计的必要性

### 3.4 GMM聚类增强的统计OOD检测（约800字）

#### 3.4.1 全局GMM建模

- **动机**：单一原型仅捕捉类中心（一阶统计量），忽略了类内分布的协方差结构（二阶统计量）和全局特征空间的聚类结构
- **GMM建模**：在训练集已知类特征上拟合高斯混合模型
  - 分量数：$K_{\text{GMM}} = 2N = 12$（$N=6$个已知类，每个类用2个component捕捉子结构）
  - 协方差类型：full（完全协方差矩阵，捕捉64维特征间的相关性）

> **【公3.8】** GMM对数似然优化目标：
> $$\log P(\mathbf{X}) = \sum_{i=1}^n \log \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_i|\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

- 参数估计：EM算法【引36】，实验中GMM在200轮迭代内收敛（converged=True）

#### 3.4.2 聚类特征提取

- 三个GMM聚类特征的数学定义：

> **【公3.9】** 聚类Mahalanobis距离：
> $$d_{\text{cluster}}(x) = \min_{k=1,\ldots,K} \sqrt{(f_\phi(x) - \boldsymbol{\mu}_k)^\top \boldsymbol{\Sigma}_k^{-1}(f_\phi(x) - \boldsymbol{\mu}_k)}$$

> **【公3.10】** 最大后验概率：$p_{\max}(x) = \max_{k=1,\ldots,K} \gamma_k(f_\phi(x))$

> **【公3.11】** 后验分布熵：$H_{\text{cluster}}(x) = -\sum_{k=1}^K \gamma_k(f_\phi(x)) \log \gamma_k(f_\phi(x))$

- 直觉解释：
  - $d_{\text{cluster}}$：已知类靠近某个cluster（距离小），未知类远离所有cluster（距离大）
  - $p_{\max}$：已知类最大后验高（高置信归属），未知类最大后验低
  - $H_{\text{cluster}}$：已知类后验集中（低熵），未知类后验分散（高熵）

> **【图3.4】** GMM聚类特征示意图：
> - 2D示意空间中展示GMM的12个component的等密度椭圆
> - 标注一个已知类样本（靠近某cluster中心，低熵）和一个未知类样本（远离所有中心，高熵）
> - 三个特征值的对比标注

#### 3.4.3 13维扩展OOD特征向量

> **【表3.2】** 13维OOD特征向量完整定义表：
>
> | 编号 | 特征名称 | 数学符号 | 维度 | 来源 |
> |------|----------|----------|------|------|
> | 1 | 到最近原型距离 | $d_{\min}$ | 距离 | 原型 |
> | 2 | 距离比 | $r_d = d_{\min}/d_{2\text{nd}}$ | 比率 | 原型 |
> | 3 | softmax最大概率 | $p_{\max}^{\text{softmax}}$ | 概率 | 分类器 |
> | 4 | softmax熵 | $H_{\text{softmax}}$ | 熵 | 分类器 |
> | 5 | 特征范数 | $\|\mathbf{z}\|_2$ | 范数 | 特征 |
> | 6 | 到第二近原型距离 | $d_{2\text{nd}}$ | 距离 | 原型 |
> | 7 | 距离差 | $\Delta d = d_{2\text{nd}} - d_{\min}$ | 距离 | 原型 |
> | 8 | 到全局中心距离 | $d_{\text{center}}$ | 距离 | 全局 |
> | 9 | 类间预测方差 | $\sigma_p^2$ | 方差 | 分类器 |
> | 10 | 归一化距离比 | $r_d' = d_{\min}/d_{\text{center}}$ | 比率 | 混合 |
> | **11** | **聚类Mahalanobis距离** | $d_{\text{cluster}}$ | **距离** | **GMM** |
> | **12** | **GMM最大后验概率** | $p_{\max}^{\text{GMM}}$ | **概率** | **GMM** |
> | **13** | **GMM后验熵** | $H_{\text{cluster}}$ | **熵** | **GMM** |
>
> 粗体 = 本文新增的GMM聚类特征（第11-13维）

#### 3.4.4 逻辑回归OOD分类器

- 训练：以 $\mathbf{h}_{13}$ 为输入，已知类标签1、未知类（校准集）标签0
- StandardScaler进行Z-score标准化，LogisticRegression（$L_2$正则化，$C=1.0$，class_weight='balanced'）

> **【公3.12】** OOD评分：$s_{\text{ood}}(x) = P(\text{known}|\mathbf{h}_{13}(x)) = \sigma(\mathbf{w}^\top \hat{\mathbf{h}}_{13}(x) + b)$

> **【表3.3】** 逻辑回归学习到的特征权重（来自实验日志）：
>
> | 特征 | 权重 | 解读 |
> |------|------|------|
> | feat_norm | +5.8004 | 特征范数越大越倾向已知类 |
> | dist_to_center | -4.2817 | 到中心距离越大越倾向未知类 |
> | min_dist | -1.1248 | 到最近原型距离越大越倾向未知类 |
> | 2nd_dist | -0.9276 | 第二近距离越大越倾向未知类 |
> | norm_dist_ratio | +0.8172 | 归一化距离比有判别力 |
> | cluster_ent | -0.4769 | GMM后验熵越高越倾向未知类 |
> | cluster_mahal | -0.3951 | GMM Mahalanobis距离越大越倾向未知类 |
> | cluster_post_max | -0.2084 | GMM最大后验越低越倾向未知类 |
>
> （展示全部13个特征的权重，按绝对值排序）

### 3.5 几何-统计双信号融合框架（约600字）

#### 3.5.1 Z-score归一化

> **【公3.13】** 双信号Z-score归一化：
> $$\hat{s}_{\text{anti}} = \frac{s_{\text{anti}} - \mu_{\text{anti}}}{\sigma_{\text{anti}}}, \quad \hat{s}_{\text{ood}} = \frac{s_{\text{ood}} - \mu_{\text{ood}}}{\sigma_{\text{ood}}}$$

- 其中 $\mu, \sigma$ 在校准集已知类样本上计算

#### 3.5.2 Youden J最优融合权重搜索

> **【公3.14】** 融合评分：$s_{\text{ens}}(x) = \alpha \cdot \hat{s}_{\text{anti}}(x) + (1-\alpha) \cdot \hat{s}_{\text{ood}}(x)$

> **【公3.15】** 最优权重搜索：$\alpha^* = \arg\max_{\alpha} J(\alpha) = \arg\max_{\alpha} (\text{TNR}_\alpha + \text{TPR}_\alpha - 1)$

- 搜索范围：$\alpha \in \{0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9\}$
- 阈值 $\tau$：校准集已知类 $s_{\text{ens}}$ 的第5百分位数，使 $\text{TNR} \approx 95\%$
- 实验结果：最优 $\alpha^* = 0.6$

> **【图3.5】** 融合权重搜索曲线：X轴为α（0.3~0.9），Y轴为Youden J值，标注最优点α=0.6。同时画出对应TPR和TNR随α变化的曲线。需根据实验数据绘制。

#### 3.5.3 互补性分析

- anti_prototype（几何信号）：捕捉样本在原型-反原型空间中的相对位置，关注局部距离关系
- ood_head_extended_v2（统计信号）：通过13维特征捕捉多层面的已知/未知差异，关注全局分布特征
- 两类信号的互补性：$\alpha=0.6$ 非极端值，说明两种信号确实提供了不同维度的信息

> **【图3.6】** 两种信号的互补性可视化（可选）：
> - 散点图：X轴=s_anti分数，Y轴=s_ood分数，颜色=已知/未知
> - 展示两种信号在已知/未知样本上的不同分布模式

### 3.6 Episodic元训练策略（约500字）

- Episode构建：从已知类中随机采样 $N_{\text{way}}$ 个类，每类 $K_{\text{shot}}$ 个样本作为支持集，$Q_{\text{query}}$ 个样本作为查询集【引1】【引2】

> **【公3.16】** 总损失函数：
> $$\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{density}} + \lambda_2 \mathcal{L}_{\text{entropy}} + \lambda_3 \mathcal{L}_{\text{sep}}$$

> **【公3.17】** 分类损失：$\mathcal{L}_{\text{CE}} = -\frac{1}{|Q|}\sum_{(x,y) \in Q} \log P(y|x)$

> **【公3.18】** 原型分离损失：$\mathcal{L}_{\text{sep}} = -\frac{1}{N(N-1)}\sum_{i \neq j}\|\mathbf{p}_i - \mathbf{p}_j\|_2^2$

> **【表3.4】** 元训练超参数配置表：
>
> | 参数 | 值 | 说明 |
> |------|-----|------|
> | N_way | 6 | 训练episode类别数 |
> | K_shot | 5 | 每类支持样本数 |
> | Q_query | 15 | 每类查询样本数 |
> | 训练episode数 | 4000（早停在3000） | |
> | 优化器 | AdamW | |
> | 温度参数τ | 可学习，初始10.0 | |
> | λ_density | 0.1 | 密度损失权重 |
> | λ_entropy | 0.05 | 熵正则化权重 |
> | λ_separation | 0.02 | 分离损失权重 |

---

## 第四章 实验设计与结果分析（约3000字）

### 4.1 实验设置（约600字）

#### 4.1.1 数据集

- TAU-2022声学场景数据集【引25】【引29】：10个城市环境声景类别（如机场、购物中心、公园、街道等），在多个欧洲城市录制
- 类别划分：6个已知类（base classes: 类0-5，每类5000个训练样本）用于元训练和原型计算；4个未知类（novel classes: 类6-9）仅用于测试开放集检测
- 音频预处理：使用YAMNet【引23】提取64维嵌入特征，LayerNorm归一化
- 校准集：10个类各1800个样本（用于OSR阈值校准）
- 测试集：10个类各1800个样本

> **【表4.1】** TAU-2022数据集统计信息表：
>
> | 划分 | 类别 | 每类样本数 | 用途 |
> |------|------|-----------|------|
> | 训练集 | 6个已知类 | 5000 | 元训练 |
> | 校准集 | 6已知+4未知 | 1800 | OSR阈值校准 |
> | 测试集 | 6已知+4未知 | 1800 | 评估 |

#### 4.1.2 小样本设定

- 训练episode：6-way 5-shot 15-query（已知类）
- 测试episode：2-way（已知）+ 2-open-way（未知）5-shot 15-query
- 训练episode数：4000（早停在3000），测试episode数：200
- 10轮独立测试取均值和标准差

#### 4.1.3 评价指标

- 闭集分类准确率（ACC）
- 真阳性率 TPR（未知类检测率，越高越好）
- 真阴性率 TNR（已知类正确识别率，固定为95%）
- OSR综合得分：$\text{OSR} = (\text{TNR} + \text{TPR})/2$

#### 4.1.4 对比方法

> **【表4.2】** 对比方法列表：
>
> | 方法简称 | 全称 | 信号类型 | 说明 |
> |----------|------|----------|------|
> | Mahalanobis | Feature Mahalanobis【引12】 | 统计 | 每类协方差Mahalanobis距离 |
> | Anti-Proto | Anti-Prototype | 几何 | 本文查询依赖反原型（单路） |
> | OOD-Ext | OOD Head Extended (11维) | 统计 | 11维特征+逻辑回归 |
> | OOD-Ext-V2 | OOD Head Extended V2 (13维) | 统计+聚类 | 13维特征（含GMM）+逻辑回归 |
> | Cluster-Bdy | Cluster Boundary | 几何 | K=2半径边界检测 |
> | Ensemble | Anti + OOD-Ext融合 | 几何+统计 | osr20b基线 |
> | **Ensemble-Clust** | **Anti + OOD-Ext-V2融合** | **几何+统计+聚类** | **本文最终方法** |

### 4.2 闭集分类性能（约300字）

> **已有图片** `phase3_fewshot_accuracy.png` → **【图4.1】** Few-shot分类准确率柱状图

- 闭集验证集准确率：70.07%（最佳模型在episode 3000处，早停）
- Base类Few-shot分类：1-shot 58.82±1.85%，5-shot 67.66±1.07%，10-shot 69.39±1.00%
- Novel类Few-shot分类：1-shot 40.37±1.79%，5-shot 45.33±1.62%，10-shot 47.00±1.32%

> **已有图片** `phase2_episodic_curves.png` → **【图4.2】** Episodic训练曲线（损失和准确率随episode变化）

- 分析：训练集到测试集的领域迁移是主要挑战；随着K-shot增加，分类准确率持续提升但趋于饱和

### 4.3 开放集检测性能对比（约800字）

#### 4.3.1 单方法对比（TNR=95%）

> **【表4.3】** 单方法OSR性能对比（TNR=95%，10轮均值）：
>
> | 方法 | TPR (%) | TNR (%) | OSR (%) | 信号类型 |
> |------|---------|---------|---------|----------|
> | Feature Mahalanobis【引12】 | 12.54 | 95.00 | 53.77 | 统计 |
> | Anti-Prototype (本文) | 18.92 | 95.00 | 56.96 | 几何 |
> | OOD Head Extended (11维) | 16.33 | 95.00 | 55.67 | 统计 |
> | **OOD Head Extended V2 (13维, +GMM)** | **21.76** | **95.00** | **58.38** | **统计+聚类** |
> | Cluster Boundary | 10.40 | 95.00 | 52.70 | 几何 |

- 关键发现：GMM聚类特征将统计方法从16.33%提升到21.76%（**+5.43%，相对提升33.3%**），验证了聚类结构信息的有效性

> **已有图片** `phase3_osr_analysis.png` → **【图4.3】** OSR各方法性能对比图

#### 4.3.2 融合方法对比（TNR=95%）

> **【表4.4】** 融合方法OSR性能对比（TNR=95%，10轮均值）：
>
> | 方法 | TPR (%) | OSR (%) | 融合权重 (α) | ΔTPR vs Ensemble |
> |------|---------|---------|-------------|------------------|
> | Ensemble (Anti + OOD-Ext) | 22.91 | 58.95 | 0.60 + 0.40 | — |
> | **Ensemble-Cluster (Anti + OOD-Ext-V2)** | **24.94** | **59.97** | **0.60 + 0.40** | **+2.03** |

- 分析：GMM聚类增强后，ensemble TPR从22.91%提升到24.94%（+2.03%）
- 融合权重最优值均为 $\alpha^*=0.6$，说明几何信号和统计信号在该比例下达到最佳互补

> **【图4.4】** 全部方法TPR对比柱状图：横轴为方法名，纵轴为TPR%，标注具体数值。用不同颜色区分单方法和融合方法。

### 4.4 消融实验（约600字）

#### 4.4.1 GMM聚类特征的消融

> **【表4.5】** GMM聚类特征消融（OOD Head性能变化）：
>
> | 特征配置 | 特征维度 | TPR (%) | ΔTPR |
> |----------|---------|---------|------|
> | 原始11维（无GMM） | 11 | 16.33 | — |
> | + cluster_mahal | 12 | 待测 | — |
> | + cluster_post_max | 12 | 待测 | — |
> | + cluster_ent | 12 | 待测 | — |
> | + 全部3个GMM特征 | **13** | **21.76** | **+5.43** |

- 分析各GMM特征的贡献度和互补性

#### 4.4.2 查询依赖中心的消融

> **【表4.6】** 反原型中心策略消融：
>
> | 中心策略 | TPR (%) | 分析 |
> |----------|---------|------|
> | 固定中心 (prototypes.mean()) | 8.32 | 训练-测试分布偏移 |
> | 查询依赖中心 (query.mean()) | 17.45/18.92 | 自适应坐标系 |
> | 置信加权中心 | ≈17 | 无显著改进 |

- 结论：查询依赖设计使TPR提升约10个百分点，是反原型方法有效性的关键

#### 4.4.3 失败方法分析

> **【表4.7】** 失败方法汇总：
>
> | 方法 | TPR (%) | ΔTPR vs 基线 | 失败原因 |
> |------|---------|-------------|----------|
> | Multi-Prototype (K=2) | 16.10 | -2.82 | 64维空间每类单模态，强制K=2破坏结构 |
> | Adaptive Ensemble | 20.11 | -2.80 | 局部密度与最优α相关性弱 |
> | Cluster Boundary | 10.40 | -8.51 | 欧氏距离阈值无法捕捉复杂边界 |

- 教训：在well-structured特征空间中，简单非参数方法优于复杂方法；强制引入额外结构（多原型、自适应权重）反而引入噪声

### 4.5 进化历程与总结（约400字）

> **【表4.8】** 实验进化历程：
>
> | 轮次 | 核心改动 | TPR (%) | ΔTPR |
> |------|----------|---------|------|
> | osr18a 基线 | Flow密度估计 | 17.62 | — |
> | osr19a | Flow辅助全面失败 | 17.69 | +0.07 |
> | osr20a | 去除Flow，纯原型 | 18.92 | +1.30 |
> | osr20b | 设计ensemble | 22.91 | +5.29 |
> | osr21a/b | Flow生成器/变换层证伪 | 22.29~23.07 | ≈0 |
> | **osr22a** | **GMM聚类增强** | **24.94** | **+2.03** |

> **【图4.5】** TPR进化历程折线图：X轴为实验轮次（osr18→osr22），Y轴为TPR%，标注每轮的关键改动。用箭头标注Flow证伪区间和聚类成功区间。需根据表4.8数据绘制。

- 总提升：17.62% → 24.94%（+7.32%，相对提升41.6%）

> **已有图片** `tsne_features.png` → **【图4.6】** t-SNE特征空间可视化：展示64维特征降维到2D后的分布，不同颜色代表不同类别，标注已知类和未知类的分布关系。

### 4.6 实验结果讨论（约300字）

- 本文方法在TPR绝对值上仍有提升空间（24.94%），这反映了FS-OSR问题的内在困难性——在仅5个已知样本的条件下，开放集检测的理论上界有限
- TNR=95%的严格约束下，TPR的每一点提升都具有实际应用价值（安防场景要求低误报率）
- 几何信号和统计信号的互补性在不同数据集上是否成立需进一步验证
- GMM聚类特征的有效性提示：在well-structured特征空间中，更好地描述已有结构比学习新变换更有效

---

## 第五章 结论与展望（约1000字）

### 5.1 主要工作与贡献（约500字）

1. 提出查询依赖的反原型评分方法，利用查询批次均值动态计算反原型中心（vs RPL【引13】的固定参数化反原型），消融实验验证其使TPR从8.32%提升到17.45%
2. 提出GMM聚类增强的统计OOD检测方法，通过Mahalanobis距离、后验概率和后验熵三个聚类特征（vs Lee et al.【引12】的单一Mahalanobis距离），将OOD Head从16.33%提升到21.76%
3. 设计了几何-统计双信号融合框架，通过Z-score归一化和Youden J【引35】准则实现自适应最优融合，ensemble从22.91%提升到24.94%
4. 在TAU-2022声学场景数据集【引25】上进行了系统性实验，包括5轮迭代实验，穷尽Flow路线并证伪，验证了非参数聚类增强路线的有效性
5. 最终方法在TNR=95%约束下达到TPR=24.94%、OSR=59.97%，相比基线相对提升41.6%

### 5.2 研究局限性（约200字）

- 方法仅在TAU-2022数据集上验证，泛化性需在CIFAR、miniImageNet等视觉数据集上进一步确认
- TPR绝对值仍有较大提升空间，反映FS-OSR问题的固有困难
- GMM的分量数（$2N$）为经验设定，缺乏BIC/AIC等信息准则的理论指导
- 训练阶段未显式优化开放集检测能力（仅依赖后处理），限制了性能上界

### 5.3 未来工作展望（约300字）

- 训练阶段改进：引入Margin Loss【引37】增大已知/未知特征空间边界，或利用SupCon对比学习【引38】增强特征判别力
- 多尺度特征融合：利用YAMNet中间层特征，不同层捕捉不同粒度的声景信息
- 更复杂的集成架构：用小型神经网络替代线性融合，学习非线性的评分组合
- 跨数据集验证：将方法应用于视觉领域（CIFAR、miniImageNet）验证通用性
- 理论分析：对查询依赖反原型中心的最优性进行数学证明，或推导TPR的概率下界

---

## 参考文献（40条）

### 小样本学习

[1] Vinyals O, Blundell C, Lillicrap T, et al. Matching networks for one shot learning[C]. NeurIPS, 2016.
[2] Snell J, Swersky K, Zemel R. Prototypical networks for few-shot learning[C]. NeurIPS, 2017.
[3] Finn C, Abbeel P, Levine S. Model-agnostic meta-learning for fast adaptation of deep networks[C]. ICML, 2017.
[4] Sung F, Yang Y, Zhang L, et al. Learning to compare: Relation network for few-shot learning[C]. CVPR, 2018.
[5] Nichol A, Achiam J, Schulman J. On first-order meta-learning algorithms[J]. arXiv:1803.02999, 2018.
[6] Wang Y, Yao Q, Kwok J T, et al. Generalizing from a few examples: A survey on few-shot learning[J]. ACM Computing Surveys, 2020, 53(3): 1-34.

### 开放集识别与OOD检测

[7] Scheirer W J, de Rezende Rocha A, Sapkota A, et al. Toward open set recognition[J]. IEEE TPAMI, 2013, 35(7): 1757-1772.
[8] Bendale A, Boult T E. Towards open set deep networks[C]. CVPR, 2016: 1563-1572.
[9] Hendrycks D, Gimpel K. A baseline for detecting misclassified and out-of-distribution examples in neural networks[C]. ICLR, 2017.
[10] Bishop C M. Pattern recognition and machine learning[M]. Springer, 2006.
[11] Liang S, Li Y, Srikant R. Enhancing the reliability of out-of-distribution image detection in neural networks (ODIN)[C]. ICLR, 2018.
[12] Lee K, Lee K, Lee H, et al. A simple unified framework for detecting out-of-distribution samples and adversarial attacks[C]. NeurIPS, 2018.
[13] Chen G, Peng P, Ma L, et al. Learning reciprocal points for open set recognition[C]. ECCV, 2020.
[14] Liu W, Wang X, Owens J, et al. Energy-based out-of-distribution detection[C]. NeurIPS, 2020.
[15] Chen G, Peng P, Ma L, et al. Adversarial reciprocal points learning for open set recognition[C]. NeurIPS, 2021.
[16] Geng C X, Huang S J, Chen C Y. Recent advances in open set recognition: A survey[J]. IEEE TPAMI, 2021, 43(10): 3614-3631.
[17] Lee K, Lee H, Lee K, et al. Training confidence-calibrated classifiers for detecting out-of-distribution samples[C]. ICLR, 2018.
[18] Ren J, Liu P J, Fertig E, et al. Likelihood ratios for out-of-distribution detection[C]. NeurIPS, 2019.

### 集成与不确定性

[19] Nalisnick E, Matsukawa A, Teh Y W, et al. Do deep generative models know what they don't know?[C]. ICLR, 2019.
[20] Nalisnick E, Matsukawa A, Teh Y W, et al. Detecting out-of-distribution inputs in deep generative models using typicality[J]. arXiv:1906.02994, 2019.
[21] Gal Y, Ghahramani Z. Dropout as a Bayesian approximation: Representing model uncertainty in deep learning[C]. ICML, 2016.
[22] Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles[C]. NeurIPS, 2017.

### 声学场景识别

[23] Hershey S, Chaudhuri S, Ellis D P W, et al. CNN architectures for large-scale audio classification[C]. ICASSP, 2017: 131-135.
[24] Mesaros A, Heittola T, Virtanen T. Acoustic scene classification in DCASE 2018 challenge[C]. DCASE Workshop, 2018.
[25] Mesaros A, Heittola T, Virtanen T. A multi-device dataset for urban acoustic scene classification[C]. DCASE Workshop, 2019.
[26] Park D S, Chan W, Zhang Y, et al. SpecAugment: A simple data augmentation method for ASR[C]. Interspeech, 2019.
[27] Kong Q, Cao Y, Iqbal T, et al. PANNs: Large-scale pretrained audio neural networks for audio pattern recognition[J]. IEEE/ACM TASLP, 2020, 28: 2880-2894.
[28] Mesaros A, Heittola T, Virtanen T. DCASE 2019 task 1: Acoustic scene classification with multiple devices[C]. DCASE Workshop, 2019.
[29] Mesaros A, Heittola T, Virtanen T. DCASE 2021 task 1: Acoustic scene classification with multiple devices[C]. DCASE Workshop, 2021.

### 数学基础

[30] Dempster A P, Laird N M, Rubin D B. Maximum likelihood from incomplete data via the EM algorithm[J]. Journal of the Royal Statistical Society: Series B, 1977, 39(1): 1-22.
[31] Cover T M, Hart P E. Nearest neighbor pattern classification[J]. IEEE Transactions on Information Theory, 1967, 13(1): 21-27.
[32] Friedman J, Hastie T, Tibshirani R. The elements of statistical learning[M]. Springer, 2009.
[33] Shannon C E. A mathematical theory of communication[J]. Bell System Technical Journal, 1948, 27(3): 379-423.
[34] Howard A G, Zhu M, Chen B, et al. MobileNets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv:1704.04861, 2017.

### 评估指标

[35] Youden W J. Index for rating diagnostic tests[J]. Cancer, 1950, 3(1): 32-35.

### 少样本开放集识别

[36] (小样本OSR方向，如Jeong et al.或Pal et al.相关工作，待补充)

### 对比学习（展望部分引用）

[37] Khosla P, Teterwak P, Wang C, et al. Supervised contrastive learning[C]. NeurIPS, 2020.
[38] Chen T, Kornblith S, Norouzi M, et al. A simple framework for contrastive learning of visual representations[C]. ICML, 2020.

### 补充参考文献

[39] 耿新, 黄圣君, 陈岑宇. 开放集识别研究进展[J]. 自动化学报, 2021, 47(9): 2063-2080.
[40] 王耀威, 姚强, 郭天佑, 等. 小样本学习研究综述[J]. 软件学报, 2020, 31(9): 2825-2844.
[41] 张志强, 张登银. 基于度量学习的小样本图像分类方法综述[J]. 计算机科学, 2022, 49(1): 185-195.
[42] 刘宏哲, 李凤宇, 袁磊, 等. 基于深度学习的异常检测综述[J]. 计算机研究与发展, 2022, 59(11): 2449-2470.
[43] 何清, 李宁, 罗文娟, 等. 大数据下的机器学习算法综述[J]. 模式识别与人工智能, 2014, 27(4): 327-336.
[44] 周志华. 机器学习[M]. 北京: 清华大学出版社, 2016.
[45] 李彦冬, 郝宗波, 雷航. 卷积神经网络研究综述[J]. 计算机应用, 2016, 36(9): 2508-2515.
[46] 张健, 陈云霁. 深度学习处理器研究综述[J]. 计算机学报, 2022, 45(1): 1-18.
[47] 余俊铭, 郭方方, 张超越, 等. 基于深度学习的环境声音识别综述[J]. 电子学报, 2021, 49(6): 1203-1218.
[48] 赵力. 语音信号处理[M]. 3版. 北京: 机械工业出版社, 2020.
[49] 吴培凯, 顾晶晶, 戴新宇, 等. 基于原型网络的小样本声学场景分类方法[J]. 信号处理, 2022, 38(5): 967-976.
[50] 郑烨辉, 郑智, 李宏亮, 等. 基于深度学习的开放集图像分类方法综述[J]. 电子与信息学报, 2023, 45(8): 2803-2818.
[51] 刘越, 赵巍, 刘昌平. 基于集成学习的分布外检测方法综述[J]. 计算机科学, 2024, 51(2): 38-49.
[52] 马勇, 丁勇, 周勇. 基于高斯混合模型的音频场景分类方法[J]. 声学技术, 2019, 38(4): 410-416.
[53] 邓欣, 陈恩庆, 孙钢, 等. 面向小样本学习的元学习方法综述[J]. 自动化学报, 2023, 49(7): 1405-1423.

### 英文补充参考文献

[54] Si Y, Li Y, Tan J, et al. Fully few-shot class-incremental audio classification using multi-level embedding extractor and ridge regression classifier[J]. arXiv preprint arXiv:2506.18406, 2025.
[55] Li Y, Tan J, Chen G, et al. Low-complexity acoustic scene classification using parallel attention-convolution network[C]. Proc. Interspeech, 2024: 567-571.

---

## 致谢（约200字）

- 感谢指导教师的悉心指导
- 感谢DCASE社区提供的TAU数据集【引25】【引29】
- 感谢同学和家人的支持

---

## 附录

### A 关键算法伪代码

> **【算法A.1】** 查询依赖反原型评分算法（Algorithm 1）
> ```
> 输入: 查询特征 Q∈R^{B×D}, 原型 P∈R^{N×D}
> 输出: 评分 s∈R^B
> 1. c ← mean(Q, dim=0)           // 查询依赖中心 (1,D)
> 2. A ← 2c - P                    // 反原型 (N,D)
> 3. d_proto ← min_cdist(Q, P)     // 到最近原型距离 (B,)
> 4. d_anti ← min_cdist(Q, A)      // 到最近反原型距离 (B,)
> 5. s ← -d_proto + d_anti         // 评分 (B,)
> 6. return s
> ```

> **【算法A.2】** GMM聚类特征提取算法（Algorithm 2）
> ```
> 输入: 特征 x∈R^D, GMM参数 {(π_k, μ_k, Σ_k)}
> 输出: 聚类特征 [d_cluster, p_max, H_cluster]
> 1. for k=1 to K:
> 2.   γ_k(x) ← π_k·N(x|μ_k,Σ_k) / Σ_j π_j·N(x|μ_j,Σ_j)
> 3.   d_M(x,k) ← √((x-μ_k)^T Σ_k^{-1} (x-μ_k))
> 4. d_cluster ← min_k d_M(x,k)
> 5. p_max ← max_k γ_k(x)
> 6. H_cluster ← -Σ_k γ_k(x)·log(γ_k(x))
> 7. return [d_cluster, p_max, H_cluster]
> ```

> **【算法A.3】** 双信号融合与阈值优化算法（Algorithm 3）
> ```
> 输入: 校准集 (已知X_kn, 未知X_uk), 原型P, 目标FPR
> 输出: 最优α*, 阈值τ*
> 1. // 步骤1: 计算两路评分
> 2. s_anti_kn ← score_anti(X_kn, P)
> 3. s_ood_kn ← score_ood(X_kn, P)  // 含GMM聚类特征
> 4. s_anti_uk ← score_anti(X_uk, P)
> 5. s_ood_uk ← score_ood(X_uk, P)
> 6. // 步骤2: Z-score归一化
> 7. μ_a, σ_a ← mean(s_anti_kn), std(s_anti_kn)
> 8. μ_o, σ_o ← mean(s_ood_kn), std(s_ood_kn)
> 9. // 步骤3: 搜索最优α
> 10. for α in {0.3, 0.4, ..., 0.9}:
> 11.   s_ens ← α·norm(s_anti) + (1-α)·norm(s_ood)
> 12.   τ ← percentile(s_ens_kn, FPR)
> 13.   TNR ← mean(s_ens_kn ≥ τ)
> 14.   TPR ← mean(s_ens_uk < τ)
> 15.   J ← TNR + TPR - 1
> 16.   if J > J_best: α* ← α, τ* ← τ
> 17. return α*, τ*
> ```

### B 实验超参数配置表

> **【表B.1】** 完整超参数配置（已包含在【表3.4】中）

### C 各方法的TPR/TNR详细结果

> **【表C.1】** 全部方法完整结果（来自实验日志，TNR=95%，10轮均值）：
>
> | 方法 | TNR (%) | TPR (%) | OSR (%) |
> |------|---------|---------|---------|
> | feature_mahalanobis | 95.00 | 12.54 | 53.77 |
> | anti_prototype | 95.00 | 18.92 | 56.96 |
> | geo_fusion | 95.00 | 14.06 | 54.53 |
> | ood_head | 95.00 | 9.01 | 52.00 |
> | ood_head_extended | 95.00 | 16.33 | 55.67 |
> | anti_prototype_per_class | 94.76 | 21.21 | 57.99 |
> | ood_head_extended_per_class | 95.68 | 12.27 | 53.97 |
> | ensemble_anti_oodext | 95.00 | 22.91 | 58.95 |
> | ensemble_anti_oodext_per_class | 95.08 | 20.68 | 57.88 |
> | anti_prototype_multi | 95.00 | 16.10 | 55.55 |
> | ood_head_extended_v2 | 95.00 | 21.76 | 58.38 |
> | ensemble_adaptive | 95.00 | 20.11 | 57.55 |
> | cluster_boundary | 95.00 | 10.40 | 52.70 |
> | **ensemble_cluster** | **95.00** | **24.94** | **59.97** |
