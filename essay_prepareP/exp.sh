#!/bin/bash
cd /coding && python episodic_exp_way_shot.py --dataset TAU19 2>&1 | tee experiment/episodic_exp/wayshot/train_TAU19.log
cd /coding && python episodic_exp_openness.py --dataset TAU22 2>&1 | tee experiment/episodic_exp/openness/train_TAU22.log
cd /coding && python episodic_exp_openness.py --dataset TAU19 2>&1 | tee experiment/episodic_exp/openness/train_TAU19.log
cd /coding && python episodic_exp_cross_domain.py --all 2>&1 | tee experiment/episodic_exp/cross_domain/train.log
cd /coding && python episodic_exp_complexity.py --all_ablation 2>&1 | tee experiment/episodic_exp/complexity/train.log
cd /coding && python episodic_exp_statistical.py --results experiment/episodic_exp/ablation/results.json --mode config 2>&1 | tee experiment/episodic_exp/statistical/train_config.log
cd /coding && python episodic_exp_statistical.py --results experiment/episodic_exp/ablation/results.json --mode osr --metric osr_score 2>&1 | tee experiment/episodic_exp/statistical/train_osr.log
cd /coding && python episodic_exp_tsne.py --compare_ablation --dataset TAU22 2>&1 | tee experiment/episodic_exp/tsne/train_TAU22.log
cd /coding && python episodic_exp_tsne.py --compare_ablation --dataset TAU19 2>&1 | tee experiment/episodic_exp/tsne/train_TAU19.log
# FOAC-AIFP
cd /coding/FOAC-AIFP && python foac_exp_way_shot.py --dataset all 2>&1 | tee ../experiment/foac_exp/wayshot/train.log && mv foac_exp_way_shot_results.json ../experiment/foac_exp/wayshot/results.json
cd /coding/FOAC-AIFP && python foac_exp_openness.py --dataset all 2>&1 | tee ../experiment/foac_exp/openness/train.log && mv foac_exp_openness_results.json ../experiment/foac_exp/openness/results.json
cd /coding/FOAC-AIFP && python foac_exp_cross_domain.py --all 2>&1 | tee ../experiment/foac_exp/cross_domain/train.log && mv foac_exp_cross_domain_results.json ../experiment/foac_exp/cross_domain/results.json
cd /coding/FOAC-AIFP && python foac_exp_complexity.py --all_ablation 2>&1 | tee ../experiment/foac_exp/complexity/train.log && mv foac_exp_complexity_results.json ../experiment/foac_exp/complexity/results.json
cd /coding/FOAC-AIFP && python foac_exp_statistical.py --results ../experiment/foac_exp/ablation/results.json 2>&1 | tee ../experiment/foac_exp/statistical/train.log
cd /coding/FOAC-AIFP && python foac_exp_tsne.py --compare_ablation 2>&1 | tee ../experiment/foac_exp/tsne/train.log