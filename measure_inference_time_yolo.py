import argparse
import yaml
from pathlib import Path
import csv

import torch 
import torch.nn as nn

from models.common import DetectMultiBackend, AutoShape
import auto_nn_compression.torchslimkit as ts
from common import init_dependency_graph, get_prunable_parameters, get_num_parameters
from measure_inference_times import prepare_model, name2attr
from utils.dataloaders import create_dataloader
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
import val as validate
from utils.loss import ComputeLoss
from utils.callbacks import Callbacks

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256]) 
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def compress_model(model: nn.Module, parameters, graph: dict, amount: float):            
    for module, _ in parameters:
        module.out_channels = module.orig_width - round(module.orig_width * amount)
            
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                print(name)
                parent = graph[name]
                if isinstance(parent, tuple):
                    s = 0
                    for m in parent:
                        s += m.out_channels
                    new_in = s
                    module.in_channels = s
                else:
                    new_in = parent.out_channels
                    module.in_channels = new_in
            except KeyError as e:
                print('KEY ERROR:', e)
            
            # Update weight shape
            module.weight.data = module.weight.data[:module.out_channels, :module.in_channels, :, :]
            if module.bias != None:
                module.bias.data = module.bias.data[:module.out_channels]

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
    
    # Nano
    yolov5n = DetectMultiBackend('yolov5n.pt', device=device)
    yolov5n.model.fuse().eval()
    yolov5n = AutoShape(yolov5n)

    # Small
    yolov5s = DetectMultiBackend('yolov5s.pt', device=device)
    yolov5s.model.fuse().eval()
    yolov5s = AutoShape(yolov5s)

    # Small Pruned
    yolov5s_pruned = DetectMultiBackend('yolov5s.pt', device=device)
    yolov5s_pruned.model.fuse().eval()
    yolov5s_pruned = AutoShape(yolov5s_pruned)
    model = yolov5s_pruned.model.model
    
    # Get dependency graph and prunable parameters
    graph = init_dependency_graph(model.model, 's')
    params = get_prunable_parameters(yolov5s_pruned)
    
    # Get number of parameters in nano and small models
    num_params_s = get_num_parameters(yolov5s)
    num_params_n = get_num_parameters(yolov5n)
    
    # Prepare dataset
    data = Path('.') / 'data/coco128.yaml'
    gs = 32
    hyp = Path('.') / 'data/hyps/hyp.scratch-low.yaml'
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict    
    data_dict = check_dataset(data)
    
    # Compress model
    prepare_model(yolov5s_pruned)
    compress_model(model, params.values(), graph, 0.02 * 27)
    
    # Get number of parameters after pruning
    num_params_pruned = get_num_parameters(yolov5s_pruned)
    
    # Initialize val_loaders
    val_loaders = []
    for batch_size in opt.batches:
        val_loader = create_dataloader(data_dict['val'], 640, batch_size, gs,
                                       False, hyp=hyp, cache='ram', rect=True,
                                       rank=-1, workers=0, pad=0.5,
                                       prefix=colorstr('val: '))[0]
        val_loaders.append(val_loader)
        
    # Begin experiment
    results = [['yolov5s', 1.0],['yolov5n', 1.0],['yolov5s_pruned', num_params_pruned/num_params_s]]
    for i, val_loader in enumerate(val_loaders):
        small_result = []
        nano_result = []
        pruned_result = []
        
        for j in range(10):
            # Nano
            _, _, timing = validate.run(data_dict, batch_size=opt.batches[i], imgsz=640, half=False,
                                        model=yolov5n, single_cls=False, dataloader=val_loader,
                                        save_dir=None, plots=False, callbacks=Callbacks(), compute_loss=None)
            nano_result.append(timing[1])
            
            # Small
            _, _, timing = validate.run(data_dict, batch_size=opt.batches[i], imgsz=640, half=False,
                                        model=yolov5s, single_cls=False, dataloader=val_loader,
                                        save_dir=None, plots=False, callbacks=Callbacks(), compute_loss=None)
            small_result.append(timing[1])
            
            # Pruned
            _, _, timing = validate.run(data_dict, batch_size=opt.batches[i], imgsz=640, half=False,
                                        model=yolov5s_pruned, single_cls=False, dataloader=val_loader,
                                        save_dir=None, plots=False, callbacks=Callbacks(), compute_loss=None)
            pruned_result.append(timing[1])
            
        # Record results
        results[0].append(sum(nano_result) / len(pruned_result))
        results[1].append(sum(small_result) / len(pruned_result))
        results[2].append(min(pruned_result))
        
    # Output to file
    header = ['Model Name', 'Actual Density']
    header += [f'b{i}' for i in opt.batches]
    with open(out_csv, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(results)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)