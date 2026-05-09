import torch
from datasets.librispeech import Openlbrs
import numpy as np
from .nsynth import Opennds
from .FMC import Openfmc
from .TAU22 import OpenTAU22
from .TAU19 import OpenTAU19

def meta_train_dataloader(args):
    # Only load base classes from train CSV — avoid data leakage with eval CSV
    class_index = np.arange(args.train_classes)

    if args.dataset == 'librispeech':
        trainset = Openlbrs(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
    elif  'Nsynth' in args.dataset:
        trainset = Opennds(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
    elif args.dataset == 'FMC':
        trainset =  Openfmc(root=args.dataroot,index=class_index,args=args,partition='train', fix_seed=True)
    elif args.dataset == 'TAU22':
        trainset = OpenTAU22(args=args, index=class_index, root=args.dataroot, partition='train', fix_seed=True)
    elif args.dataset == 'TAU19':
        trainset = OpenTAU19(args=args, index=class_index, root=args.dataroot, partition='train', fix_seed=True)
    loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False,
                                         num_workers=8, pin_memory=True, persistent_workers=True)

    return loader



def meta_calib_dataloader(args):
    """Calibration/validation loader: all 10 classes from calib CSV."""
    if args.dataset == 'TAU22':
        calibset = OpenTAU22(args=args, index=np.arange(10), root=args.dataroot, partition='calib', fix_seed=True)
    elif args.dataset == 'TAU19':
        calibset = OpenTAU19(args=args, index=np.arange(10), root=args.dataroot, partition='calib', fix_seed=True)
    else:
        # Non-TAU datasets fall back to test loader
        return meta_test_dataloader(args)

    loader = torch.utils.data.DataLoader(calibset, batch_size=1, shuffle=False,
                                         num_workers=8, pin_memory=True, persistent_workers=True)
    return loader


def meta_test_dataloader(args):
    if args.dataset == 'librispeech':
        class_new = np.arange(args.train_classes,100)
        testset = Openlbrs(root=args.dataroot,index=class_new,args=args,partition='test', fix_seed=True)
    elif args.dataset == 'Nsynth':
        testset = Opennds(root=args.dataroot,index=np.arange(args.train_classes,100),args=args,partition='test', fix_seed=True)
    elif args.dataset == 'FMC':
        testset = Openfmc(root=args.dataroot,index=np.arange(args.train_classes,89),args=args,partition='test', fix_seed=True)
    elif args.dataset == 'TAU22':
        testset = OpenTAU22(args=args, index=np.arange(10), root=args.dataroot, partition='test', fix_seed=True)
    elif args.dataset == 'TAU19':
        testset = OpenTAU19(args=args, index=np.arange(10), root=args.dataroot, partition='test', fix_seed=True)

    loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=8, pin_memory=True, persistent_workers=True)

    return loader