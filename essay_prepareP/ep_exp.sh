#!/bin/bash
cd /coding && python episodic_exp_ablation.py 2>&1 | tee experiment/episodic_exp/ablation/train.log
cd /coding && python episodic_exp_way_shot.py 2>&1 | tee experiment/episodic_exp/wayshot/train.log
cd /coding && python episodic_exp_openness.py 2>&1 | tee experiment/episodic_exp/openness/train.log
cd /coding && python episodic_exp_cross_domain.py --all 2>&1 | tee experiment/episodic_exp/cross_domain/train.log
# cd /coding && python episodic_exp_complexity.py --all_ablation 2>&1 | tee experiment/episodic_exp/complexity/train.log
cd /coding && python episodic_exp_statistical.py --results experiment/episodic_exp/ablation/results.json 2>&1 | tee experiment/episodic_exp/statistical/train.log
cd /coding && python episodic_exp_tsne.py --compare_ablation 2>&1 | tee experiment/episodic_exp/tsne/train.log