import argparse
from pathlib import Path
from argparse import Namespace
import csv

import torch 

import finetune_up

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-drop', type=float, default=0.05)
    parser.add_argument('--data-file', type=str, default='./results/best.cifar100.csv')
    parser.add_argument('--starting-state', type=str, default='orig')
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):
    # Get the desired checkpoints
    checkpoints = {}
    file_path = Path(opt.data_file)
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Skip header
            if row[0] == 'Model Name':
                continue            
            
            # Get attributes
            model, weights, target = row[0], row[1], torch.tensor(float(row[2]))
            metrics = torch.tensor([float(v) for v in row[3:]])
            
            # Determine the last checkpoint above the cutoff
            good_ckpts = (metrics >= target - opt.max_drop).to(dtype=torch.int32)
            good_ckpts = torch.cumsum(good_ckpts, dim=0)
            last_ckpt_idx = torch.argmax(good_ckpts)
            checkpoints[model] = Path(weights) / f'best_{last_ckpt_idx}.pt'
            
    # Fine tune checkpoints up
    for model_name, model_path in checkpoints.items():
        # Create args for finetuning routine
        new_opt = Namespace(
            model=model_name,
            checkpoint=model_path,
            starting_state=opt.starting_state
        )
        
        # Begin the fine-tuning experiment
        finetune_up.main(new_opt)        
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)