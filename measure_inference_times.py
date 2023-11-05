import argparse
from pathlib import Path
import csv
from tqdm import tqdm
import re

import torch
import torch.nn as nn

import pytorch_cifar_models
from common import cifar100, Profile, dot_num_to_brack, dot_num_to_brack_end, get_num_parameters
import auto_nn_compression.torchslimkit as ts

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, nargs='+')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--increment', type=float)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    opt = parser.parse_args()
    print(vars(opt))
    return opt

@torch.no_grad()
def eval_latency(model, val_loader, device, epoch):
    # Eval mode
    model.eval()
    
    # Profiler
    dt = Profile()
    
    # Run 1 epoch
    seen = 0
    for i, data in enumerate(tqdm(val_loader, desc=f'Validation Epoch {epoch}'), 0):
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        with dt:
            outputs = model(inputs)
            
        seen += 1
    
    # Return result
    return dt.t * 1E3 / seen

def name2attr(name: str):
    attr = re.sub('\.[\d]+\.', dot_num_to_brack, name)
    attr = re.sub('\.[\d]+', dot_num_to_brack_end, attr)
    return attr

def prepare_model(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.orig_width = module.out_channels
            
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            exec(f'model.{name2attr(name)} = nn.Identity()')
            
def compress_model(model: nn.Module, graph: dict, amount: float):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.out_channels = module.orig_width - round(module.orig_width * amount)
            
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parent = ts.get_all_parents(name, graph)
            if len(parent) != 0:
                new_in = eval(f'model.{name2attr(parent[0])}.out_channels')
                module.in_channels = new_in
            
            # Update weight shape
            module.weight.data = module.weight.data[:module.out_channels, :module.in_channels, :, :]
            if module.bias != None:
                module.bias.data = module.bias.data[:module.out_channels]
        
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent = ts.get_all_parents(name, graph)
            parent_mod = eval(f'model.{name2attr(parent[0])}')
            if isinstance(parent_mod, nn.Conv2d):
                module.in_features = parent_mod.out_channels
                module.weight.data = module.weight.data[:, :module.in_features]
            break    
            
def create_model_graph(model):
    onnx_graph = ts.generate_onnx_graph(model, input_size=(1,3,32,32))
    graph = ts.generate_graph(onnx_graph=onnx_graph)
    return graph

def main(opt):
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Directories
    save_dir = Path('./results')
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    out_csv = save_dir / f'{device.type}.{opt.dataset}.csv'
    
    # Data loaders
    val_loaders = []
    for b in opt.batches: #[1, 2, 4, 8, 16, 32, 64, 128, 256]:
        _, val_loader = cifar100(batch_size=b)
        val_loaders.append(val_loader)
        
    # Initialize results
    results = []
        
    # Iterate over all models
    for model_name in opt.model:
        results.append([])
        
        # Add model name and pruning level to results
        results[-1].append(model_name)
        results[-1].append(0.0)
        results[-1].append(1.0)
        
        # Load intial model
        model = getattr(pytorch_cifar_models, model_name)(pretrained=True)
        graph = create_model_graph(model)
        model.to(device=device)
        
        # Warmup
        # for val_loader in val_loaders:
        #     eval_latency(model, val_loader, device, 0)
        
        # Get the initial latency
        for val_loader in val_loaders:
            latency = eval_latency(model, val_loader, device, 0)
            results[-1].append(latency)
        
        # Begin compression iterations
        prepare_model(model)
        total_params = get_num_parameters(model)
        i = 1
        while opt.increment * i < 0.99:
            results.append([])
            results[-1].append(model_name)
            results[-1].append(opt.increment * i)
            
            # Compress model
            compress_model(model, graph, opt.increment * i)
            num_params = get_num_parameters(model)
            results[-1].append(num_params/total_params)
            
            # Evaluate latency
            for val_loader in val_loaders:
                latency = eval_latency(model, val_loader, device, i)
                results[-1].append(latency)
            
            i += 1
        
    header = ['Model Name', 'Nominal Sparsity', 'Actual Density']
    header += [f'b{i}' for i in opt.batches]
    with open(out_csv, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)
    
        
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)