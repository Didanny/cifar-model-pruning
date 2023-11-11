import argparse
import yaml
from pathlib import Path
import csv
import os
import pandas as pd

import torch 
import torch.nn as nn

import pytorch_cifar_models
from models.common import DetectMultiBackend, AutoShape
import auto_nn_compression.torchslimkit as ts
from common import (init_dependency_graph, get_prunable_parameters, get_num_parameters, prune_filters, 
                    get_filter_indices, fuse_batchnorms, get_kernel_indices)
from finetune_yolo import prune_filters_
from measure_inference_times import prepare_model, name2attr, create_model_graph
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info, Profile,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
import val as validate
from utils.loss import ComputeLoss
from utils.callbacks import Callbacks

ROOT = Path('.')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparsities', type=str, default = 'checkpoints.sparsity.yaml')
    parser.add_argument('--checkpoints', type=str, default = 'checkpoints.files.yaml')
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):
    # Initialize results
    results = [[], [], [], [], []]
    
    # Directories
    save_dir = Path('.') / 'results'
    
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Load the checkpoint sparsities
    with open(ROOT / 'data' / 'model-runs' / opt.sparsities) as f:
        sparsity = yaml.safe_load(f)
    
    # Load the models
    models = {}
    all_weights = {}
    for model_name in sparsity.keys():
        results[0].append(model_name)
        
        if model_name.startswith('cifar100'):
            model = getattr(pytorch_cifar_models, model_name)(pretrained=True)
            models[model_name] = model
        else:
            model = DetectMultiBackend(model_name, device=device)                                
            model.model.fuse().eval()
            model = AutoShape(model)
            models[model_name] = model.model.model.model
    
    # Measure full-to-pruned swap time
    for model_name in sparsity.keys():
        model = models[model_name]
        
        # Measure full architecture reading time
        dt = Profile()
        with dt:
            orig = torch.load(save_dir / f'{model_name}.orig.pt')
        results[1].append(dt.t * 1E3)
        
        # Measure pruned architecture reading time
        dt = Profile()
        with dt:
            pruned = torch.load(save_dir / f'{model_name}.pruned.pt')
        results[2].append(dt.t * 1E3)
        
        all_weights = {}
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                # name = name2attr(name)
                all_weights[f'model.{name}.weight'] = module.weight.data
                if module.bias is not None:
                    all_weights[f'model.{name}.bias'] = module.bias.data
                else:
                    all_weights[f'model.{name}.bias'] = None

        # Measure swap time
        dt = Profile()
        st = Profile()
        end = False
        with dt:
            data = pruned['data']
            names = pruned['names']
            
            offset = 0
            while offset < data.shape[0]:
                # Get the args
                name = names[data[offset]]
                args = data[offset + 1: offset + 6]
                len_kernels = data[offset + 6]
                len_filters = data[offset + 7]
                
                # Get the kernel and filter indices
                kernel_indices = data[offset + 8: offset + 8 + len_kernels]
                filter_indices = data[offset + 8 + len_kernels: offset + 8 + len_kernels + len_filters]

                # Intiialize the new module
                with st:
                    new_mod = nn.Conv2d(args[0], args[1], (args[2], args[2]), stride=(args[3], args[3]), padding=(args[4], args[4]))
                weight = all_weights[f'{name}.weight']
                bias = all_weights[f'{name}.bias']
                
                if len_filters == 0 and len_kernels == 0:
                    raise ValueError
                elif len_filters == 0:
                    new_mod.weight.data = weight[:, kernel_indices, :, :]
                    new_mod.bias.data = bias if bias != None else None
                elif len_kernels == 0:
                    new_mod.weight.data = weight[filter_indices, :, :, :]
                    new_mod.bias.data = bias[filter_indices] if bias != None else None
                else:
                    # TODO: Fix
                    new_mod.weight.data = weight[:, filter_indices, :, :][kernel_indices, :, :, :]
                    new_mod.bias.data = bias[kernel_indices] if bias != None else torch.zeros(args[0])
                    
                new_mod.weight.requires_grad = False
                if new_mod.bias != None:
                    new_mod.bias.requires_grad = False
                
                # exec(f'{name} = new_mod')
                offset += 8 + len_kernels + len_filters
        results[3].append((dt.t * 1E3) - (st.t * 1E3))
        
        # Measure pruned-to-full swap time
        dt = Profile()
        st = Profile()
        end = False
        with dt:
            data = orig['data']
            names = orig['names']
            
            offset = 0
            while offset < data.shape[0]:
                # Get the args
                name = names[data[offset]]
                args = data[offset + 1: offset + 6]
                len_kernels = data[offset + 6]
                len_filters = data[offset + 7]
                
                # Get the kernel and filter indices
                kernel_indices = data[offset + 8: offset + 8 + len_kernels]
                filter_indices = data[offset + 8 + len_kernels: offset + 8 + len_kernels + len_filters]

                # Intiialize the new 
                with st :
                    new_mod = nn.Conv2d(args[0], args[1], (args[2], args[2]), stride=(args[3], args[3]), padding=(args[4], args[4]))
                weight = all_weights[f'{name}.weight']
                bias = all_weights[f'{name}.bias']
                
                if len_filters == 0 and len_kernels == 0:
                    new_mod.weight.data = weight[:, :, :, :]
                    new_mod.bias.data = bias[:] if bias != None else torch.zeros(args[0])
                elif len_filters == 0:
                    new_mod.weight.data = weight[:, kernel_indices, :, :]
                    new_mod.bias.data = bias if bias != None else None
                elif len_kernels == 0:
                    new_mod.weight.data = weight[filter_indices, :, :, :]
                    new_mod.bias.data = bias[filter_indices] if bias != None else None
                else:
                    new_mod.weight.data = weight[:, kernel_indices, :, :][filter_indices, :, :, :]
                    new_mod.bias.data = bias[filter_indices] if bias != None else None
                    
                new_mod.weight.requires_grad = False
                if new_mod.bias != None:
                    new_mod.bias.requires_grad = False
                
                # exec(f'{name} = new_mod')
                offset += 8 + len_kernels + len_filters
        results[4].append((dt.t * 1E3) - (st.t * 1E3))

        
        
    # Save results
    df = pd.DataFrame({
        'Model Name': results[0],
        'Orig Arch Load': results[1],
        'Pruned Arch Load': results[2],
        'Full-to-Pruned Swap Time': results[3],
        'Pruned-to-Full Swap Time': results[4]
    })
    df.to_csv(save_dir / 'overhead.swapping.csv', index=False)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
