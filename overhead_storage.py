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

def encode_architectures(model, model_name, graph):
    # Pruned architecture
    pruned_architecture = {}

    # Original architecture
    original_architecture = {}

    # Pruned architecture in tensor format
    pruned_architecture_pt = torch.tensor([], dtype=torch.int32)

    # Original architecture in tensor format
    original_architecture_pt = torch.tensor([], dtype=torch.int32)

    # Iterate over all conv layers
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        
        # Get the filter and kernel indices
        skip = False
        conv = module
        if model_name.startswith('cifar100'):
            parent = ts.get_all_parents(name, graph)
            if len(parent) != 0:
                parent_mod = eval(f'model.{name2attr(parent[0])}')
                kernel_indices = get_filter_indices(parent_mod, pruned=False)
            else:
                kernel_indices = get_kernel_indices(conv, pruned=False)
            filter_indices = get_filter_indices(conv, pruned=False)
        else:
            if name not in graph:
                skip = True
            else:
                parent = graph[name]
                offset = 0
                kernel_indices = torch.tensor([], dtype=torch.long)
                if isinstance(parent, tuple):
                    for mod in parent:
                        kernel_indices = torch.hstack([kernel_indices, (get_filter_indices(mod, pruned=False) + offset)])
                        offset += mod.out_channels
                else:
                    kernel_indices = get_filter_indices(parent, pruned=False)
            filter_indices = get_filter_indices(conv, pruned=False)
        
        if skip:
            continue
        
        # Get the params
        weight = conv.weight.data
        if conv.bias != None:
            bias = conv.bias.data
        else:
            bias = torch.zeros(conv.out_channels)
        stride = conv.stride
        padding = conv.padding
        kernel_size = conv.kernel_size
        
        # Create the new module
        new_mod = nn.Conv2d(kernel_indices.size()[0], filter_indices.size()[0], kernel_size, stride, padding)
        new_mod.weight.data = weight[filter_indices, :, :, :][:, kernel_indices, :, :]
        new_mod.bias.data = bias[filter_indices]
        
        # Store the original architecture of the layer
        layer = [
            f'model.{name}',
            module.in_channels,
            module.out_channels, 
            module.kernel_size[0],
            module.stride[0],
            module.padding[0],
            None,
            None,
        ]
        original_architecture[f'model.{name}'] = layer
        
        # Store the pruned architecture of the layer
        layer = [
            f'model.{name}',
            new_mod.in_channels,
            new_mod.out_channels,
            new_mod.kernel_size[0],
            new_mod.stride[0],
            new_mod.padding[0],
            filter_indices.tolist(),
            kernel_indices.tolist(),                
        ]
        pruned_architecture[f'model.{name}'] = layer     
            
    # Initialize the names list
    pruned_layer_names = []

    # Convert to a torch friendly format
    for i, p in enumerate(list(pruned_architecture.values())):
        # Insert the layer name into the names list
        pruned_layer_names.append(p[0])
        
        # Convert layer information into tensor
        args = torch.tensor([
            i,
            p[1], # in_channels
            p[2], # out_channels
            p[3], # kernel_size
            p[4], # stride
            p[5], # padding
            len(p[6]) if p[6] != None else 0, # Size of kernel indices list
            len(p[7]) if p[7]  != None else 0, # Size of filter indices list        
        ], dtype=torch.long)
        
        # Get the filter indices tensor
        kernel_indices = torch.tensor(p[6], dtype=torch.long) if p[6] != None else torch.tensor([], dtype=torch.long)
        
        # Get the kernel indices tensor
        filter_indices = torch.tensor(p[7], dtype=torch.long) if p[7] != None else torch.tensor([], dtype=torch.long)
        
        # Add the layer tensor to the architecture tensor
        p_pt = torch.hstack([args, kernel_indices, filter_indices])
        pruned_architecture_pt = torch.hstack([pruned_architecture_pt, p_pt])

    # Initialize the names list
    original_layer_names = []

    # Convert to a torch friendly format
    for i, p in enumerate(list(original_architecture.values())):
        # Insert the layer name into the names list
        original_layer_names.append(p[0])
        
        # Convert layer information into tensor
        args = torch.tensor([
            i,
            p[1], # in_channels
            p[2], # out_channels
            p[3], # kernel_size
            p[4], # stride
            p[5], # padding
            len(p[6]) if p[6] != None else 0, # Size of kernel indices list
            len(p[7]) if p[7]  != None else 0, # Size of filter indices list        
        ], dtype=torch.long)
        
        # Get the filter indices tensor
        kernel_indices = torch.tensor(p[6], dtype=torch.long) if p[6] != None else torch.tensor([], dtype=torch.long)
        
        # Get the kernel indices tensor
        filter_indices = torch.tensor(p[7], dtype=torch.long) if p[7] != None else torch.tensor([], dtype=torch.long)
        
        # Add the layer tensor to the architecture tensor
        p_pt = torch.hstack([args, kernel_indices, filter_indices])
        original_architecture_pt = torch.hstack([original_architecture_pt, p_pt])
        
    return ({'data': original_architecture_pt, 'names': original_layer_names}, 
            {'data': pruned_architecture_pt, 'names': pruned_layer_names})

def main(opt):
    # Initialize results
    results = [[], [], [], [], [], []]
    
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
    graphs = {}
    for model_name in sparsity.keys():
        results[0].append(model_name)
        
        # Measure loading time
        if model_name.startswith('cifar100'):
            dt = Profile()
            model = getattr(pytorch_cifar_models, model_name)(pretrained=True)                
            graph = create_model_graph(model)
            with dt:
                model = getattr(pytorch_cifar_models, model_name)(pretrained=True)
                model.eval()
                fuse_batchnorms(model, model_name, prune_=False)
            results[1].append(dt.t * 1E3)
            models[model_name] = model
            graphs[model_name] = graph
            
            # Get model size on disk
            torch.save(model, 'model.pt')
            model_stats = os.stat('model.pt')
            results[2].append(model_stats.st_size / 1024)
        else:
            dt = Profile()
            with dt:
                model = DetectMultiBackend(model_name, device=device)                                
                model.model.fuse().eval()
                model = AutoShape(model)
            results[1].append(dt.t * 1E3)
            models[model_name] = model
            
            # Get model size on disk
            torch.save(model, 'model.pt')
            model_stats = os.stat('model.pt')
            results[2].append(model_stats.st_size / (1024 * 1024))
                
    # Model partitioning
    for model_name in sparsity.keys():
        model = models[model_name]
        
        # Prune the model
        if model_name.startswith('cifar100'):
            prune_filters(model, model_name, sparsity[model_name])
            model(torch.rand(1,3,32,32))
        else:
            model = models[model_name].model.model
            graph = init_dependency_graph(model.model, 's')
            graphs[model_name] = graph
            prune_filters_(model, model_name, sparsity[model_name])
            model(torch.rand(1,3,640,640))
            model = model.model
        
        # Encode the pruned/full architectures
        orig, pruned = encode_architectures(model, model_name, graphs[model_name])
        
        # Save architecture files        
        torch.save(orig, 'orig.pt')
        torch.save(pruned, 'pruned.pt')
        torch.save(orig, save_dir / f'{model_name}.orig.pt')
        torch.save(pruned, save_dir / f'{model_name}.pruned.pt')
        
        # Get file sizes
        orig_stats = os.stat('orig.pt')
        pruned_stats = os.stat('pruned.pt')
        
        # Save file sizes
        results[3].append(orig_stats.st_size / 1024)
        results[4].append(pruned_stats.st_size / 1024)
        
    # Save results
    df = pd.DataFrame({
        'Model Name': results[0],
        'Loading Time': results[1],
        'Model Size': results[2],
        'Orig Arch Size': results[3],
        'Pruned Arch Size': results[4]
    })
    df.to_csv(save_dir / 'overhead.storage.csv', index=False)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
