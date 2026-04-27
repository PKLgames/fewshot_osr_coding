import os
from functools import partial
from trainers import trainer, C2_Net_train
from models.Network import My_Net
from datasets import dataloaders
from utils.util import *
from models.Network import Backbone
import torch
import yaml
import argparse
from datasets.TAU22 import TAU22Pretrain



def dict2namespace(dicts):
    for i in dicts:
        if isinstance(dicts[i], dict):
            dicts[i] = dict2namespace(dicts[i]) 
    ns = argparse.Namespace(**dicts)
    return ns

args = trainer.train_parser()
with open(args.config) as f:           #training configuration file
        cfg = yaml.safe_load(f)
cfg = cfg['train']
cfg.update(vars(args))
args = dict2namespace(cfg)

train_loader = dataloaders.meta_train_dataloader(args)
eval_loader = dataloaders.meta_test_dataloader(args)
train_func = partial(C2_Net_train.default_train, train_loader=train_loader)
tm = trainer.Train_Manager(args, train_func=train_func)

if not args.pretrain:
    model = My_Net(args=args,mode='train')
    model = model.to('cuda')
    
    if args.test:
        model.eval()
        ckpt_names = ['model_TAU22_max_acc.pth', 'model_TAU22_max_osr.pth',
                       'model_TAU22_max_auroc.pth', 'model_TAU22_max_fscore.pth']
        test_log_path = os.path.join(args.save_folder, 'TAU22test.log')
        with open(test_log_path, 'w') as f:
            for ckpt_name in ckpt_names:
                ckpt_path = os.path.join(args.save_folder, ckpt_name)
                if not os.path.exists(ckpt_path):
                    print(f"Skip {ckpt_name}: not found")
                    continue
                state_dict = torch.load(ckpt_path)
                model.weight_base = state_dict['weight_base'].to('cuda')
                model.weight_base_open = state_dict['weight_base_open'].to('cuda')
                model.load_state_dict(state_dict, strict=False)
                result, loss = tm.run_test_fsl(model, eval_loader)
                acc, tnr, tpr, osr_score = result
                f.write(f"{'='*50}\n")
                f.write(f"{ckpt_name}\n")
                f.write(f"ACC:       {acc[0]:.3f} ± {acc[1]:.3f}\n")
                f.write(f"TNR:       {tnr[0]:.3f} ± {tnr[1]:.3f}\n")
                f.write(f"TPR:       {tpr[0]:.3f} ± {tpr[1]:.3f}\n")
                f.write(f"OSR Score: {osr_score[0]:.3f} ± {osr_score[1]:.3f}\n")
                f.write(f"Loss:      {loss:.5f}\n\n")
                print(f"{ckpt_name}: ACC={acc[0]:.3f}±{acc[1]:.3f}  TNR={tnr[0]:.3f}±{tnr[1]:.3f}  TPR={tpr[0]:.3f}±{tpr[1]:.3f}  OSR={osr_score[0]:.3f}±{osr_score[1]:.3f}")

        print(f"\nResults saved to {test_log_path}")
        exit()
    state_dict = torch.load(args.pretrained_model_path)['feature_params']
    full_params = torch.load(args.pretrained_model_path)

    model.load_state_dict(state_dict,strict=False)
    model.init_representation(full_params)
    tm.train(model,eval_loader)
else:
    model = Backbone(args)#My_Net(args=args,mode='pretrain')
    model.to('cuda')
    tm.Pretrain(model)

