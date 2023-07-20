import argparse

import torch
import pytorch_cifar_models
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torchmetrics.classification import Accuracy
from common import *
from collections import OrderedDict

def get_val_transforms(mean, std):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--mode', type=str)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):
    # CIFAR-100
    val_set = CIFAR100('./data', train=False, download=False, transform=get_val_transforms(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]))
    train_set = CIFAR100('./data', train=True, download=False)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False)
    
    # VGG-16 Model
    model = getattr(pytorch_cifar_models, opt.model)(pretrained=True)
    model.eval()
    
    # Setup
    dependencies = get_dependency_graph(model, opt.model)
    parameters_to_prune = get_parameters_to_prune(model, opt.model)
    parameters_to_prune = [
        (val, 'weight') for key, val in model.features.named_modules() if isinstance(val, torch.nn.Conv2d)
    ]
    name_to_module = get_name_to_module(model)
    
    # Run exploration
    delta = 0.005
    i = 0.005
    if opt.mode == 'mean':
        while i < 0.51:
            print(f'Sparsity={i}')
            global_smallest_filter_mean(parameters_to_prune, i)
            for key in dependencies:
                mod = name_to_module[key]
                mod_dep = name_to_module[dependencies[key]]
                prune_kernel2(mod, mod_dep)
            print(validate(val_loader, model))
            i += delta
    else:
        while i < 0.51:
            print(f'Sparsity={i}')
            global_smallest_filter_norm(parameters_to_prune, i, opt.mode)
            for key in dependencies:
                mod = name_to_module[key]
                mod_dep = name_to_module[dependencies[key]]
                prune_kernel2(mod, mod_dep)
            print(validate(val_loader, model))
            i += delta
        
        
if __name__ == '__main__':
    opt = parse_opt()
    main(opt) 