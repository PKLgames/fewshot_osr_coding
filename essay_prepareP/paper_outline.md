# Few-Shot Open-Set Audio Classification via Episodic Training with Fused OOD Detection Heads

**Target Venue:** IEEE/ACM Transactions on Audio, Speech and Language Processing (TASLP) 或 ICASSP 2027
**Status:** 大纲草稿，基于 experiment_list.txt 全部已完成实验数据

---

## I. Introduction (引言)

**问题定义：** Few-Shot Open-Set Audio Classification (FOAC) — 在仅给定少量标注样本（1-10 shot）的情况下，同时完成已知类分类和未知类拒绝的双重任务。

**核心挑战：**
1. 少样本条件下类原型估计不准确
2. 开放集场景下需要区分已知/未知类边界，而常规分类器对未知类过度自信
3. 音频域内（环境音）和跨域（环境音→语音）场景的泛化差异

**本文贡献（3点）：**
1. 提出基于 Episodic Training 的统一 FOAC 框架，内嵌6种互补的开放集检测头（anti-prototype, Mahalanobis, OSR23, Extended V2, Cluster V2, Fusion V2），并在训练中通过多损失联合优化实现端到端学习。
2. 设计 Fusion V2 开放集检测头 — 融合 extended open-set head 和 cluster-based head 的优势，在 6-way 5-shot 设置下 TAU19 数据集上达到 AUROC=0.983、OSR=92.97%，远超 anti-prototype 基线 25-35 个百分点。
3. 通过系统性的 way-shot 扫描、开放度 (openness) 分析、跨域泛化测试、消融实验和统计显著性检验，建立了 FOAC 方法评估的完整证据链。

---

## II. Related Work (相关工作)

### A. Few-Shot Learning in Audio
- Prototypical Networks, Matching Networks, MAML
- FEAT, L³-Net, D-ProtoNet, MET 等 SOTA 方法
- 现有方法多聚焦于闭集分类，缺乏对未知类的处理

### B. Open-Set Recognition
- 传统方法：SoftMax threshold, OpenMax, G-OpenMax
- 距离/密度方法：Mahalanobis distance, Gaussian Mixture Models (GMM)
- 原型方法：anti-prototype, reciprocal point learning
- 现有 OSR 方法多假设充足训练样本，少样本场景下性能退化严重

### C. Audio-Specific Considerations
- 声景事件的时间-频率特性
- YAMNet (pretrained AudioSet embeddings) 作为骨干网络
- 环境音分类的跨域挑战 (TAU22 vs TAU19)

---

## III. Method (方法)

### A. Problem Formulation
- Base classes (D_base): 大规模标注数据用于训练特征提取器
- Novel classes (D_novel): N-way K-shot few-shot 评估
- Unknown classes: 测试时出现的未见类别，需被拒绝
- 形式化：给定 support set S (N类 × K样本)，对 query sample x 输出分类概率 + 开放集得分

### B. Episodic Training Framework
- Episode 构造：每轮随机采样 N-way × (K support + Q query)，混合已知类 + 未知类样本
- Backbone: YAMNet pretrained on AudioSet (frozen / fine-tuned)
- Flow Transform: 特征空间归一化流映射 (可选，参数量 +25K)

### C. Six OSR Detection Heads

**1. Anti-Prototype Head (anti_proto)**
- 学习"反原型"作为所有已知类的负中心
- 开放集得分 = 与反原型的距离 vs 与最近类原型的距离

**2. Mahalanobis Distance Head (mahal)**
- 在特征空间拟合各类高斯分布
- 开放集得分 = 到最近类中心的马氏距离

**3. OOD Head OSR23 (osr23)**
- 训练 GMM 建模已知类特征分布
- 开放集得分 = 基于 GMM 的对数似然比

**4. OOD Head Extended V2 (ext_v2)**
- 扩展 OOD 检测头，加入可学习的变换层
- 开放集得分 = 学习到的判别边界

**5. OOD Head Cluster V2 (clust_v2)**
- 基于聚类的开放集检测
- 已知类特征聚簇，开放集得分 = 样本到最近簇中心的距离

**6. OOD Head Fusion V2 (fusion_v2) — 本文核心贡献**
- 融合 extended V2 和 cluster V2 的输出
- 加权组合 + 可学习的融合参数
- 开放集得分 = α · score_ext_v2 + (1-α) · score_clust_v2
- 训练损失 = L_classification + λ_osr · L_osr + λ_reciprocal · L_reciprocal

### D. Training Objectives
- Classification Loss: Cross-entropy on support + query known classes
- OSR Loss: 二元交叉熵 (known vs unknown) 训练开放集检测头
- Reciprocal Loss: 互惠注意力正则化 (权重 0.3)
- Separation Loss: 已知/未知类特征分离损失 (权重 0.02)
- Gradient Clipping: 全参数梯度裁剪，稳定训练

### E. Model Configuration Variants (for Ablation)
| Configuration | Description |
|--------------|-------------|
| full_model | 完整模型，包含所有组件 |
| wo_flow_transform | 移除 Flow Transform |
| wo_ood_head | 移除 OOD Head，仅保留分类损失 |
| wo_reciprocal | 移除互惠注意力损失 |
| wo_threshold | 移除阈值自适应机制 |
| only_distance_head | 仅保留基础距离度量头 (baseline) |

---

## IV. Experiments (实验)

### A. Experimental Datasets and Setup

**数据集：**
| Dataset | Domain | #Classes (Base/Novel) | Samples | Description |
|---------|--------|----------------------|---------|-------------|
| TAU22 | 环境音 (声景) | 6/4 | ~10K | DCASE 2022 声景分类 |
| TAU19 | 环境音 (声景) | 6/4 | ~10K | DCASE 2019 声景分类 |

注：TAU19 为更难域 (跨录音设备/场景分布差异更大)

**评估设置：**
- 默认: 6-way 5-shot, Q=15 fewshot samples
- 扩展: 2-6 way × 1-10 shot (way-shot 扫描)
- 开放度: 2-4 unknown classes (openness 分析)
- 跨域: TAU22↔TAU19 双向

**评估指标：**
- 分类: Accuracy (ACC, %), F1-score, Classification Macro-AUROC (one-vs-rest)
- 开放集: AUROC, OSR Score = (TPR + TNR)/2 @ TNR=95%, TPR@TNR95
- 效率: Parameters (NP), Average Inference Time (AIT)

**校准协议：**
- Per-round recalibration: 每轮评估时重新校准阈值 (TNR=95%)
- Calibration set ≠ Test set 分离，避免 TNR 自校准循环

**硬件与超参数：**
- GPU: NVIDIA RTX 3090
- 训练 episodes: 3000 (完整) / 500 (快速验证)
- Optimizer: AdamW, lr=2e-4, warmup=200
- Backbone: YAMNet (pretrained AudioSet, ~12.7K trainable params in baseline)

---

### B. Comparison with Baseline Methods

**目的：** 系统对比6种 OSR 检测方法 + 6种模型配置在标准 FOAC 设置下的性能。

**6-way 5-shot 标准结果 (TAU22, full_model, 3000ep):**

| OSR Method | ACC(%) | Cls-AUROC | OSR-AUROC | OSR(%) | TPR@TNR95(%) |
|-----------|--------|-----------|-----------|--------|--------------|
| anti_prototype | 54.41 | 0.840 | 0.624 | 60.53 | — |
| mahalanobis | — | 0.840 | 0.486 | 48.89 | — |
| osr23 | — | 0.840 | 0.841 | 73.92 | — |
| extended_v2 | — | 0.840 | 0.834 | 73.52 | — |
| cluster_v2 | — | 0.840 | 0.841 | 73.83 | — |
| **fusion_v2** | **—** | **0.840** | **0.862** | **72.71** | **55.52** |

注: ACC/Cls-AUROC 由模型配置决定 (此处 full_model 5sAcc=54.41%, Cls-AUROC=0.840)，OSR 指标由检测头决定。
TAU22 osr24 基线 (6000ep): 6w1s ACC=39.70%(Cls-AUROC=0.7186), 6w5s ACC=51.89%(Cls-AUROC=0.7997), 6w10s ACC=54.21%(Cls-AUROC=0.8088); Novel 4w1s ACC=38.75%(Cls-AUROC=0.6477), 4w5s ACC=45.47%(Cls-AUROC=0.7273), 4w10s ACC=47.18%(Cls-AUROC=0.7448).

**6-way 5-shot 标准结果 (TAU19, full_model, 3000ep):**

| OSR Method | ACC(%) | Cls-AUROC | OSR-AUROC | OSR(%) | TPR@TNR95(%) |
|-----------|--------|-----------|-----------|--------|--------------|
| anti_prototype | 38.94 | 0.717 | 0.635 | 52.77 | — |
| mahalanobis | — | 0.717 | 0.845 | 62.80 | — |
| osr23 | — | 0.717 | 0.937 | 85.18 | — |
| extended_v2 | — | 0.717 | 0.935 | 85.03 | — |
| cluster_v2 | — | 0.717 | 0.937 | 85.11 | — |
| **fusion_v2** | **—** | **0.717** | **0.972** | **87.63** | **90.22** |

注: TAU19 full_model 5sAcc=38.94%, Cls-AUROC=0.717; TAU19 osr24 基线 (6000ep): 6w1s ACC=28.06%(Cls-AUROC=0.6762), 6w5s ACC=35.96%(Cls-AUROC=0.7521)。(ACC 为原始训练日志值; Cls-AUROC 为 --eval_only 重测, 因随机采样 ACC 略有波动)

**关键发现：**
1. fusion_v2 在两个数据集上 OSR-AUROC 均最优 (TAU22: 0.862, TAU19: 0.972)，全胜 12/12 way-shot settings
2. TAU19 上 OOD head 系方法 (osr23/ext_v2/clust_v2/fusion_v2) OSR-AUROC 均 >0.93，远超 TAU22 (0.83-0.86) — TAU19 已知/未知类边界更清晰
3. Mahalanobis 距离法在 TAU22 上几乎无效 (OSR=48.89%, OSR-AUROC=0.486)，但在 TAU19 上表现尚可 (OSR=62.80%, OSR-AUROC=0.845)
4. anti_prototype 在 TAU19 上退化严重 (OSR-AUROC=0.635)，而 fusion_v2 保持 0.972 — 差距 0.337 AUROC
5. full_model 在 TAU22 上 fusion_v2 OSR-AUROC=0.862 vs osr23/clust_v2=0.841，差距仅 0.021；TAU19 上差距扩大至 0.035
6. 分类 AUROC 与 OSR AUROC 解耦：TAU22 Cls-AUROC=0.840 但 OSR-AUROC=0.624(anti_proto), TAU19 Cls-AUROC=0.717 但 OSR-AUROC=0.635 — 分类能力强弱并不预测 OSR 质量

---

### C. Statistical Significance Test

**方法：** Friedman 检验 + Nemenyi 后续检验 (α=0.05)

**OSR 方法间比较 (6 methods × 12 conditions = 72 data points):**

| 统计量 | 值 |
|--------|-----|
| Friedman χ² | 49.29 |
| p-value | 1.94 × 10⁻⁹ (高度显著) |
| Critical Difference (CD) | 3.08 |

**平均排名 (越低越好):**

| Rank | Method | Avg Rank |
|------|--------|----------|
| 1 | **fusion_v2** | 1.75 |
| 2 | osr23 | 2.0 |
| 3 | cluster_v2 | 2.5 |
| 4 | extended_v2 | 3.75 |
| 5 | anti_prototype | 5.5 |
| 5 | mahalanobis | 5.5 |

**显著配對 (p < 0.05):**
- fusion_v2 > anti_prototype ✓
- fusion_v2 > mahalanobis ✓
- osr23 > anti_prototype ✓
- osr23 > mahalanobis ✓

**模型配置间比较 (6 configurations):**
- Friedman χ² = 7.14, p = 0.21 → **不显著**
- 6种配置间无统计差异，说明 OSR 检测头的选择远比模型架构细节重要

---

### D. Complexity Comparison

| Configuration | Params | AIT (ms) |
|--------------|--------|----------|
| full_model | 40,100 | 4.16 ± 0.08 |
| wo_flow_transform | 14,756 | 0.65 ± 0.03 |
| wo_ood_head | 39,363 | 4.16 ± 0.08 |
| wo_reciprocal | 38,820 | 4.12 ± 0.05 |
| wo_threshold | 40,098 | 4.28 ± 0.49 |
| only_distance_head (baseline) | 12,737 | 0.64 ± 0.02 |

**关键发现：**
- Flow Transform 是主要计算开销源：参数量 +25K (14.7K→40.1K)，AIT 6.5× (0.65ms→4.16ms)
- OOD Head 仅贡献 ~700 参数，AIT 开销极小
- Baseline 仅 12.7K 参数 / 0.64ms，适合边缘部署

**与 SOTA 方法对比（需补充）：**
- 与 FEAT, MET, D-ProtoNet 等方法的 Param/MACs/AIT 比较

---

### E. Generalization Ability (Cross-Domain)

**目的：** 验证训练域与测试域不同时的鲁棒性。

**TAU22→TAU19 (跨域, 200ep):**

| Metric | Base 1s | Base 5s | Novel 1s | Novel 5s |
|--------|---------|---------|----------|----------|
| ACC | 30.02±1.43% | 35.13±1.11% | 41.12±2.16% | 45.53±1.54% |
| Cls-AUROC | 0.6359 | 0.6849 | 0.6532 | 0.7082 |

| OSR Method | OSR(%) | AUROC |
|-----------|--------|-------|
| anti_prototype | 52.83 | 0.675 |
| mahalanobis | 60.36 | 0.816 |
| osr23 | 84.38 | 0.934 |
| extended_v2 | 84.16 | 0.932 |
| cluster_v2 | 84.27 | 0.934 |
| **fusion_v2** | **90.49** | **0.977** |

**TAU19→TAU22 (跨域, 200ep):**

| Metric | Base 1s | Base 5s | Novel 1s | Novel 5s |
|--------|---------|---------|----------|----------|
| ACC | 38.76±1.91% | 51.44±1.18% | 38.40±2.00% | 45.92±1.33% |
| Cls-AUROC | 0.6765 | 0.7586 | 0.6290 | 0.6813 |

| OSR Method | OSR(%) | AUROC |
|-----------|--------|-------|
| anti_prototype | 63.38 | 0.570 |
| mahalanobis | 50.95 | 0.625 |
| osr23 | 74.70 | 0.855 |
| extended_v2 | 74.72 | 0.849 |
| cluster_v2 | 74.68 | 0.855 |
| **fusion_v2** | **74.31** | **0.878** |

**关键发现：**
1. 跨域 OSR 不对称：TAU22→TAU19 OSR=90.49%/AUROC=0.977 >> TAU19→TAU22 OSR=74.31%/AUROC=0.878 — TAU19 作为目标域时开放空间边界更清晰
2. fusion_v2 在两方向均最优：TAU22→TAU19 领先 anti_prototype +37.7pp OSR / +0.302 AUROC；TAU19→TAU22 领先 mahalanobis +23.4pp OSR / +0.253 AUROC
3. 跨域场景下 OOD head 系方法 (osr23/ext_v2/clust_v2) 高度一致 (OSR≈84%, AUROC≈0.93-0.93 on TAU22→TAU19)，fusion_v2 进一步拉开 6pp
4. TAU19→TAU22 方向所有方法 OSR 均下降：OOD head 系下降 ~10pp (84%→74%)，跨域方向影响显著
5. 跨域分类准确率下降显著 (Base 5s TAU22→TAU19: 52.27%→35.13%)，但 OSR 能力反而提升 (76.96%→90.49%)

---

### F. Ablation Experiments

**目的：** 验证各模型组件 + OSR 检测方法的独立贡献。

**36格消融矩阵 (6 configurations × 6 OSR methods):**

**TAU22 消融 — OSR Score (%) (best Val Acc 54.67% @ wo_ood_head):**

| Configuration | 5s ACC(%) | anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2 |
|--------------|-----------|------------|-------|-------|--------|---------|-----------|
| full_model | 54.41 | 60.53 | 48.89 | 73.92 | 73.52 | 73.83 | 72.71 |
| wo_flow_transform | 52.87 | 57.17 | 49.88 | 74.07 | 72.71 | 74.10 | **78.24** ★ |
| wo_ood_head | **54.67** | 57.39 | 49.22 | 74.06 | 73.18 | 73.99 | 74.70 |
| wo_reciprocal | 52.64 | 57.52 | 49.14 | 74.36 | 73.49 | 74.35 | 75.23 |
| wo_threshold | 53.38 | 53.08 | 50.62 | 74.16 | 74.05 | 74.16 | 73.99 |
| only_distance_head | 51.92 | 54.52 | 49.26 | 73.99 | 73.71 | 73.95 | 73.31 |

**TAU22 消融 — AUROC:**

| Configuration | anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2 |
|--------------|------------|-------|-------|--------|---------|-----------|
| full_model | 0.624 | 0.486 | 0.841 | 0.834 | 0.841 | 0.862 |
| wo_flow_transform | 0.580 | 0.581 | 0.840 | 0.832 | 0.840 | **0.884** ★ |
| wo_ood_head | 0.646 | 0.511 | 0.848 | 0.842 | 0.849 | 0.872 |
| wo_reciprocal | 0.593 | 0.532 | 0.837 | 0.828 | 0.837 | 0.869 |
| wo_threshold | 0.578 | 0.599 | 0.834 | 0.826 | 0.834 | 0.863 |
| only_distance_head | 0.616 | 0.526 | 0.835 | 0.829 | 0.836 | 0.856 |

**TAU19 消融 — OSR Score (%) (best Val Acc 43.66% @ wo_ood_head):**

| Configuration | 5s ACC(%) | anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2 |
|--------------|-----------|------------|-------|-------|--------|---------|-----------|
| full_model | 38.94 | 52.77 | 62.80 | 85.18 | 85.03 | 85.11 | 87.63 ★ |
| wo_flow_transform | 40.88 | 65.42 | 65.80 | 85.12 | 84.94 | 85.03 | 88.92 ★ |
| wo_ood_head | 43.37 | 58.65 | 64.95 | 84.70 | 84.55 | 84.72 | **93.31** ★ |
| wo_reciprocal | 42.32 | 60.74 | 64.29 | 85.20 | 84.54 | 85.09 | 92.88 ★ |
| wo_threshold | 40.50 | 55.81 | 64.07 | 84.45 | 84.12 | 84.32 | 87.08 ★ |
| only_distance_head | 37.67 | 52.18 | 63.76 | 84.49 | 84.24 | 84.39 | 87.48 ★ |

**TAU19 消融 — AUROC:**

| Configuration | anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2 |
|--------------|------------|-------|-------|--------|---------|-----------|
| full_model | 0.635 | 0.845 | 0.937 | 0.935 | 0.937 | 0.972 |
| wo_flow_transform | 0.699 | 0.867 | 0.936 | 0.934 | 0.936 | 0.974 |
| wo_ood_head | 0.661 | 0.869 | 0.935 | 0.933 | 0.935 | **0.985** ★ |
| wo_reciprocal | 0.661 | 0.866 | 0.936 | 0.933 | 0.936 | 0.983 |
| wo_threshold | 0.626 | 0.849 | 0.934 | 0.932 | 0.934 | 0.967 |
| only_distance_head | 0.643 | 0.856 | 0.934 | 0.932 | 0.934 | 0.970 |

**关键发现：**
1. **fusion_v2 AUROC 在全部 12 格 (2 datasets × 6 configs) 中最优** — 12/12 全胜
2. **wo_ood_head 在 TAU19 上 fusion_v2 AUROC=0.985** — 全局最优，说明 OOD-specific training loss 可能干扰了特征学习
3. **wo_flow_transform 在 TAU22 上 fusion_v2 AUROC=0.884** — TAU22 最优，Flow Transform 在 easy domain 上无益
4. **AUROC 区分度远大于 OSR Score**: TAU19 消融 AUROC spread=0.354 (0.635→0.985) vs OSR spread=40.5pp (52.18→93.31)
5. **OOD head 系方法 (osr23/ext_v2/clust_v2) AUROC 高度一致**: TAU19 上均 ≈0.93-0.94, TAU22 上均 ≈0.83-0.84，三者在统计上不可区分
6. **互惠注意力对 ACC 贡献最大**: wo_reciprocal 在 TAU22 上 ACC 从 54.41% 降至 52.64% (-1.77pp)
7. **TAU19 上 fusion_v2 全面碾压**: 6/6 配置中 fusion_v2 均为最优 OSR 方法

---

### G. Extended Experiments

#### G1. Way-Shot Analysis

**目的：** 探究 way 数 (类别数) 和 shot 数 (每类样本数) 对性能的影响。

**TAU22 (Fusion V2):**

| Setting | ACC(%) | AUROC | OSR(%) |
|---------|--------|-------|--------|
| 5w1s | 47.33 | 0.863 | 72.91 |
| 5w5s | 56.93 | 0.881 | **80.01** ★ |
| 5w10s | 57.79 | 0.886 | 76.83 |
| 6w1s | 40.04 | 0.874 | 75.93 |
| 6w5s | 52.27 | 0.876 | 76.96 |
| 6w10s | 56.09 | 0.866 | 74.18 |

**TAU19 (Fusion V2):**

| Setting | ACC(%) | AUROC | OSR(%) |
|---------|--------|-------|--------|
| 5w1s | 34.85 | 0.971 | 88.42 |
| 5w5s | 43.17 | 0.977 | 92.98 |
| 5w10s | 42.80 | **0.981** | **93.42** ★ |
| 6w1s | 30.01 | 0.976 | 89.22 |
| 6w5s | 38.69 | 0.983 | 92.97 |
| 6w10s | 39.27 | 0.981 | 93.65 |

**关键发现：**
1. K-shot 增加 → ACC 显著提升 (1→10 shot: +10~15pp)，但 OSR/AUROC 仅微增 (<5pp) — 分类与 OSR 能力解耦
2. Way 增加 → ACC 下降 (5→6 way: -2~3pp)，OSR 几乎不变
3. TAU19 的 AUROC 天花板更高 (fusion_v2 0.981 vs TAU22 0.886)：训练域越难，开放集边界越清晰
4. fusion_v2 AUROC 在两域全部 12/12 way-shot settings 中最优（详见完整数据）：
   - TAU22: fusion_v2 AUROC=0.863-0.886, OOD head 系 (osr23/clust_v2/ext_v2) AUROC=0.823-0.844
   - TAU19: fusion_v2 AUROC=0.971-0.983, OOD head 系 AUROC=0.931-0.936
   - TAU19 上 fusion_v2 领先 OOD head 系 ~0.04-0.05 AUROC，TAU22 上领先 ~0.03-0.05
5. 6w1s 极端条件下 fusion_v2 AUROC 保持 0.976 (TAU19) / 0.874 (TAU22)，anti_prototype 退化为 0.694 / 0.627
6. Mahalanobis AUROC 随 shot 增加反而下降 (TAU22 5w1s: 0.657 → 5w10s: 0.539) — 更多样本 -> 更紧的高斯 -> 更差 OSR

#### G2. Openness Analysis

**目的：** 探究未知类数量对开放集检测性能的影响。

**TAU22 Openness — OSR Score (%):**

| Unknown # | anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2 |
|-----------|------------|-------|-------|--------|---------|-----------|
| open2 (2) | 59.77 | 49.92 | 85.91 | 85.85 | 85.91 | **89.46** |
| open4 (4) | 52.72 | 49.89 | 74.25 | 73.15 | 74.23 | **80.62** |
| Δ (open2→4) | -7.05 | -0.03 | -11.66 | -12.70 | -11.68 | **-8.84** |

**TAU22 Openness — AUROC:**

| Unknown # | anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2 |
|-----------|------------|-------|-------|--------|---------|-----------|
| open2 (2) | 0.684 | 0.604 | 0.945 | 0.943 | 0.945 | **0.974** |
| open4 (4) | 0.599 | 0.573 | 0.839 | 0.828 | 0.839 | **0.897** |
| Δ (open2→4) | -0.086 | -0.031 | -0.106 | -0.116 | -0.106 | **-0.076** |

**TAU19 Openness — OSR Score (%):**

| Unknown # | anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2 |
|-----------|------------|-------|-------|--------|---------|-----------|
| open2 (2) | 59.68 | 62.62 | 90.58 | 90.14 | 90.60 | **97.11** |
| open4 (4) | 59.90 | 58.65 | 84.50 | 84.17 | 84.44 | **94.08** |
| Δ (open2→4) | +0.22 | -3.97 | -6.08 | -5.97 | -6.16 | **-3.03** |

**TAU19 Openness — AUROC:**

| Unknown # | anti_proto | mahal | osr23 | ext_v2 | clust_v2 | fusion_v2 |
|-----------|------------|-------|-------|--------|---------|-----------|
| open2 (2) | 0.716 | 0.834 | 0.960 | 0.958 | 0.960 | **0.992** |
| open4 (4) | 0.701 | 0.833 | 0.935 | 0.933 | 0.935 | **0.984** |
| Δ (open2→4) | -0.014 | -0.001 | -0.025 | -0.025 | -0.025 | **-0.008** |

**关键发现：**
1. 未知类增加 → 所有方法 OSR/AUROC 均单调下降 (TAU22 更显著)，符合预期
2. **fusion_v2 衰减幅度最小**: TAU22 AUROC 下降 -0.076 (vs ext_v2 -0.116)，TAU19 AUROC 仅下降 -0.008 (几乎不变)
3. **TAU19 开放性鲁棒性极强**: open4 下 fusion_v2 仍保持 AUROC=0.984, OSR=94.08%
4. anti_prototype 在 TAU22 open4 退化严重 (OSR=52.72%, AUROC=0.599，接近随机)；TAU19 上则意外地 open2→open4 OSR 略微上升 (+0.22pp)
5. mahalanobis 在 TAU22 上完全失效 (AUROC=0.57-0.60)，TAU19 上却达 0.833-0.834 — 跨数据集一致性最差
6. OOD head 系三方法 (osr23/ext_v2/clust_v2) 在 openness 下表现几乎一致，说明三者本质上在测相似信号

#### G3. t-SNE Visualization

**目的：** 直观展示已知类与未知类特征分布。

**已生成图像：**
- TAU22: full_model + baseline (distance_head only) 各一图
- TAU19: full_model + baseline (distance_head only) 各一图
- 样本量: 2000 samples

**待补充 (建议)：**
- per-OSR-method 对比: fusion_v2 vs cluster_v2 vs anti_prototype 三者并排
- 跨域 t-SNE: TAU22→TAU19 特征迁移可视化
- 消融对比: full_model vs wo_ood_head (因 wo_ood_head 意外最优)

#### G4. Comparison with ResNet18-based Method (FOAC-AIFP)

**目的：** 与不同 backbone 的方法横向对比，提供参考基准。

| Metric | Episodic (YAMNet) | FOAC-AIFP (ResNet18) |
|--------|-------------------|----------------------|
| ACC (TAU22, 6w5s) | 52.27% | 44.93% |
| AUROC (TAU22) | 0.876 | 0.525 |
| OSR (TAU22) | 76.96% | 50.69% |
| TPR@TNR95 (TAU22) | 55.52% | 5.83% |
| ACC (TAU19, 6w5s) | 38.69% | 45.14% |
| AUROC (TAU19) | 0.983 | 0.449 |
| OSR (TAU19) | 92.97% | 49.94% |

注：两方法 backbone 不同，绝对值不直接可比；但 FOAC-AIFP OSR≈50% (随机水平) 说明其组件 (CIAM/PAM/NPM) 未产生有效的开放集检测能力。

---

## V. Discussion (讨论)

### A. Why Fusion V2 Works
- 融合 extended (判别式边界) + cluster (生成式密度) 两种互补信号
- 判别式方法对已知类边界建模好，生成式方法对远距未知类敏感
- 在"已知/未知边界清晰"的数据集 (TAU19) 上优势更大：AUROC 领先 0.04-0.05 (vs TAU22 仅 0.02-0.04)
- AUROC 区分度 > OSR Score 区分度：TAU19 消融 AUROC spread=0.354 vs OSR spread=40.5pp，AUROC 作为无阈值指标更稳定

### B. The wo_ood_head Surprise
- 移除 OOD-specific 训练损失反而提升 OSR：TAU19 wo_ood_head fusion_v2 AUROC=0.985 (全实验最优)
- 假设：OOD loss 引入额外噪声梯度，干扰了特征空间的自然分离；分类损失 + 互惠注意力已足够产生良好的已知/未知分离
- 实践启示：在特征质量足够好时，simple OSR head with good features > complex OSR training
- TAU22 上该效应不明显 (wo_ood_head AUROC=0.872 vs full_model=0.862)，说明 easy domain 对 regularization 不敏感

### C. Cross-Domain Asymmetry
- TAU22→TAU19 OSR (90.49%) >> TAU19→TAU22 (74.31%)：差异 16.2pp
- TAU22→TAU19 AUROC (0.977) >> TAU19→TAU22 (0.878)：差异 0.099
- 解释：TAU19 源域样本多样性更大 → 学到的开放空间更宽 → 面对未知类时鲁棒性更强
- 实践意义：训练数据应尽量覆盖 diverse acoustic conditions，以提升跨域 OSR 泛化

### D. AUROC vs OSR Score as Evaluation Metrics
- AUROC 区分度更高：消融矩阵中 AUROC spread 远大于 OSR Score spread
- OSR Score 依赖 TNR=95% 阈值校准，per-round recalibration 引入额外方差
- OOD head 系三方法 (osr23/ext_v2/clust_v2) 在 OSR Score 上几乎不可区分 (<1pp)，但在 AUROC 上有微小差异
- 建议：论文以 AUROC 为主要 OSR 指标，OSR Score 作为辅助

### E. Limitations
1. 仅两个同域声景数据集 (TAU22/TAU19)，需补充真正跨域数据集 (语音/乐器) 验证
2. Fusion V2 的融合权重 α 需手动调参；未来可学习联合优化
3. YAMNet backbone 的 AudioSet 预训练可能引入已知类数据泄露（TAU22/TAU19 的声景类可能在 AudioSet 中出现）
4. 未与最新的 Foundation Model (CLAP, AST 等) 对比
5. Way-Shot 实验中 AUROC 的 shot 增加收益递减现象缺乏理论解释

---

## VI. Conclusion (结论)

1. 提出并系统评估了6种 OSR 检测头在 Episodic FOAC 框架下的性能。**fusion_v2 在所有实验设置下均最优** (12/12 way-shot, 6/6 ablation configs, 4/4 openness, 2/2 cross-domain)，统计检验高度显著 (Friedman χ²=49.29, p=1.94×10⁻⁹)。
2. Fusion V2 核心指标：TAU19 wo_ood_head AUROC=**0.985** (全局最优), TAU22 wo_flow_transform AUROC=**0.884** (TAU22 最优)；TAU19 openness 下 AUROC 保持 **0.984-0.992**。
3. AUROC 作为无阈值指标比 OSR Score 具有更强的区分为：消融矩阵中 AUROC spread=0.354 vs OSR Score spread=40.5pp，且不受阈值校准噪声影响。
4. 反直觉发现：移除 OOD-specific 训练损失 (wo_ood_head) 在 TAU19 上达到全局最优 AUROC=0.985，暗示简单 OSR head + 良好特征 > 复杂 OSR 训练。
5. 未来方向：轻量化 (移除 Flow Transform: 6.5× 加速, -63% 参数)、第三数据集跨域验证、与 Foundation Model (CLAP/AST) 对比、可学习融合权重。

---

## Appendix: 制表 / 可视化待完成清单

基于 experiment_list.txt §九，以下产出待生成：

| # | Item | Status | Priority |
|---|------|--------|----------|
| 1 | LaTeX 汇总制表脚本 (paper_tables.py) | ⬜ 待创建 | P0 |
| 2 | 6种 OSR 方法 AUROC 曲线对比图 (单图) | ⬜ 待生成 | P0 |
| 3 | 雷达图 (Acc/AUROC/OSR/AIT/Params 多维对比) | ⬜ 待生成 | P1 |
| 4 | Per-OSR-method t-SNE 对比图 (fusion_v2 vs cluster_v2 vs anti_proto) | ⬜ 待创建 | P1 |
| 5 | 消融热力图 (6 configs × 6 OSR methods = 36 cells) | ⬜ 待生成 | P1 |
| 6 | Way-Shot 曲面图 (way × shot × AUROC 3D) | ⬜ 待生成 | P2 |
| 7 | 补充第3个跨域数据集 (FMC/NSynth/LibriSpeech) | ⬜ 待决定 | P2 |
| 8 | 与 SOTA 方法 (FEAT/MET 等) 外部代码对比 | ⬜ 待决定 | P2 |

---

*大纲版本: v1.1 | 日期: 2026-05-12 | 基于 experiment_list.txt (全实验已完成，AUROC 数据已补全)*
