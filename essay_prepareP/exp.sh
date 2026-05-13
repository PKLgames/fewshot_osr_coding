#!/bin/bash
  cd /coding && python episodic_trainer.py --dataset tau19 --eval_only

  cd /coding && python episodic_trainer.py --dataset tau19 --eval_only 2>&1 | tee experiment/yamnet_realfewshot_osr24_tau19/yamnet_realfewshot_osr24_tau19_eval.log