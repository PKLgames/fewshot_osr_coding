#!/bin/bash
cd /coding

# 全量: 6配置 × 6OSR × 4fold = 144次训练+评估 (每轮 ~3000ep)
python episodic_exp_ablation.py --dataset DCASE18 --fold all \
    2>&1 | tee experiment/episodic_exp/ablation/train_dcase18.log

cd /coding

# 全量:
python episodic_exp_way_shot.py --dataset DCASE18 --fold all \
    2>&1 | tee experiment/episodic_exp/wayshot/train_dcase18.log

cd /coding

# 全量:
python episodic_exp_openness.py --dataset DCASE18 --fold all \
    2>&1 | tee experiment/episodic_exp/openness/train_dcase18.log

cd /coding

# DCASE18 → TAU22  (source=DCASE18 fold1, target=TAU22)
python episodic_exp_cross_domain.py --source DCASE18 --target TAU22 \
    --source_fold 1 2>&1 | tee experiment/episodic_exp/cross_domain/train_dcase18_tau22.log

# DCASE18 → TAU19
python episodic_exp_cross_domain.py --source DCASE18 --target TAU19 \
    --source_fold 1 2>&1 | tee experiment/episodic_exp/cross_domain/train_dcase18_tau19.log

# TAU22 → DCASE18
python episodic_exp_cross_domain.py --source TAU22 --target DCASE18 \
    --target_fold 1 2>&1 | tee experiment/episodic_exp/cross_domain/train_tau22_dcase18.log

# TAU19 → DCASE18
python episodic_exp_cross_domain.py --source TAU19 --target DCASE18 \
    --target_fold 1 2>&1 | tee experiment/episodic_exp/cross_domain/train_tau19_dcase18.log

cd /coding

# config 模式 (比较6种消融配置)
python episodic_exp_statistical.py \
    --results experiment/episodic_exp/ablation/results.json \
    --mode config \
    2>&1 | tee experiment/episodic_exp/statistical/train_dcase18_config.log

# osr 模式 (比较6种OSR方法)
python episodic_exp_statistical.py \
    --results experiment/episodic_exp/ablation/results.json \
    --mode osr --metric osr_score \
    2>&1 | tee experiment/episodic_exp/statistical/train_dcase18_osr.log

注: results.json 中已包含所有 fold 数据; 统计脚本会识别 fold 维度.

cd /coding

# 4 fold 分别:
for fold in 1 2 3 4; do
    python episodic_exp_tsne.py --dataset DCASE18 --fold $fold \
    2>&1 | tee experiment/episodic_exp/tsne/train_dcase18_fold${fold}.log
done