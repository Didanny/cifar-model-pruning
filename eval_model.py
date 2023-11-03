import csv
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn as nn

import pytorch_cifar_models
from common import cifar100, dummy_prune
from finetune import evaluate
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)

ROOT = Path('./runs')

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-runs', type=str, default='./data/model-runs/model-runs.cifar100.yaml')
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):        
    # Directories
    save_dir = Path('./results')
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    best_csv = save_dir / 'best.cifar100.csv'
    last_csv = save_dir / 'last.cifar100.csv'
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Load the model runs
    model_runs = check_yaml(opt.model_runs)
    if isinstance(model_runs, str):
        with open(model_runs, errors='ignore') as f:
            model_runs = yaml.safe_load(f)  # load hyps dict
            
    # Validate 
    best_results = []
    last_results = []
    for model_name, runs_dir in model_runs.items():
        # Add model name to result row
        best_results.append([])
        last_results.append([])
        best_results[-1].append(model_name)
        last_results[-1].append(model_name)
        
        # Weights directory
        weights = ROOT / runs_dir / 'weights'
        best_results[-1].append(weights)
        last_results[-1].append(weights)
        
        # Get the original metrics
        for weights_file in weights.glob('orig*'):
            # Load checkpoint
            checkpoint = torch.load(weights_file, map_location=torch.device('cpu'))
            
            best_results[-1].append(checkpoint['accuracy_top_1'])
            last_results[-1].append(checkpoint['accuracy_top_1'])
        
        # Loop over the best weights
        for weights_file in weights.glob('best*'):
            # Load checkpoint
            checkpoint = torch.load(weights_file, map_location=torch.device('cpu'))
            
            best_results[-1].append(checkpoint['accuracy_top_1'].item())
            
        # Loop over the best weights
        for weights_file in weights.glob('last*'):
            # Load checkpoint
            checkpoint = torch.load(weights_file, map_location=torch.device('cpu'))
            
            last_results[-1].append(checkpoint['accuracy_top_1'].item())
            
    # Write to the results file
    header = ['Model Name', 'Path', '0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '92%', '94%', '96%', '98%']
    with open(best_csv, 'w') as best_file:
        writer = csv.writer(best_file)
        writer.writerow(header)
        writer.writerows(best_results)
    with open(last_csv, 'w') as last_file:
        writer = csv.writer(last_file)
        writer.writerow(header)
        writer.writerows(last_results)           
            

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)