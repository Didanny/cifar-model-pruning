import argparse
import math
import os
import random
import subprocess
import sys
import re
import time
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
import torch.nn.utils.prune as prune
from tqdm import tqdm
from common import get_filter_indices, prune_filters, prune_structured, get_prunable_parameters, get_next

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    # parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    # parser.add_argument('--entity', default=None, help='Entity')
    # parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    # parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    # parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def log(writer: SummaryWriter, results, global_step: int, val_only=False):
    if val_only:
        mp, mr, map50, map, box_loss, cls_loss, obj_loss = results
        
        # Metrics
        writer.add_scalar('Metrics/Mean Presicion', mp, global_step)
        writer.add_scalar('Metrics/Mean Recall', mr, global_step)
        writer.add_scalar('Metrics/mAP 50', map50, global_step)
        writer.add_scalar('Metrics/mAP 50-95', map, global_step)
        
        # Loss
        writer.add_scalar('Val/Box loss', box_loss, global_step)
        writer.add_scalar('Val/Objectness loss', obj_loss, global_step)
        writer.add_scalar('Val/Classification loss', cls_loss, global_step)
    else:
        box_loss_t, cls_loss_t, obj_loss_t, mp, mr, map50, map, box_loss, cls_loss, obj_loss = results
        
        # Metrics
        writer.add_scalar('Metrics/Mean Presicion', mp, global_step)
        writer.add_scalar('Metrics/Mean Recall', mr, global_step)
        writer.add_scalar('Metrics/mAP 50', map50, global_step)
        writer.add_scalar('Metrics/mAP 50-95', map, global_step)
        
        # Loss
        writer.add_scalar('Val/Box loss', box_loss, global_step)
        writer.add_scalar('Val/Objectness loss', obj_loss, global_step)
        writer.add_scalar('Val/Classification loss', cls_loss, global_step)
        
        # Train Loss
        writer.add_scalar('Tra/Box loss', box_loss_t, global_step)
        writer.add_scalar('Tra/Objectness loss', obj_loss_t, global_step)
        writer.add_scalar('Tra/Classification loss', cls_loss_t, global_step)
    
def prune_filters_(model, weights, amount=0.1):
    # model.cpu()
    for name, module in model.named_modules():
        # Create list of all module names
        
        # The top level layers
        if re.search('model\.[\d]*$', name) != None: 
            # The first Conv layer--------------------------------------------------------------
            # mAP50-95: 0.466
            if name.endswith('model.0'):
                for n, m in module.named_modules():
                    if isinstance(m, nn.Conv2d):
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        # weight_mask = torch.ones_like(m.weight, device=next(model.parameters()).device)
                        # weight_mask[0,:,:,:] = 0
                        # prune.custom_from_mask(m, 'weight', weight_mask)
                        # breakpoint()
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The back-bone conv layers preceding C3 blocks-------------------------------------
            # mAP50-95: 0.00836
            if name.endswith('model.1') or name.endswith('model.3') or name.endswith('model.5') or name.endswith('model.7'):
                for n, m in module.named_modules():
                    if isinstance(m, nn.Conv2d):
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The first conv layer in the back-bone bottlenecks---------------------------------
            # mAP50-95: 0.0412
            if name.endswith('model.2') or name.endswith('model.4') or name.endswith('model.6') or name.endswith('model.8'):
                for n, m in module.named_modules():
                    if re.search('m\.[\d]*\.cv1.conv$', n) != None:
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The detour conv layer in the back-bone C3 blocks-----------------------------------
            # mAP50-95: 0.0566
            if name.endswith('model.2') or name.endswith('model.4') or name.endswith('model.6') or name.endswith('model.8'):
                for n, m in module.named_modules():
                    if n == 'cv2.conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # Last conv layer in the 1st back-bone C3 blocks------------------------------------
            # mAP50-95: 0.39
            if name.endswith('model.2'):
                for n, m in module.named_modules():
                    if n == 'cv3.conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # Last conv layer in the 2nd and 3rd back-bone C3 block-----------------------------
            # mAP50-95: 0.258
            if name.endswith('model.4') or name.endswith('model.6'): 
                for n, m in module.named_modules():
                    if n == 'cv3.conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The last conv layer in the 4th back-bone C3 block---------------------------------
            # mAP50-95: 0.428
            if name.endswith('model.8'):
                for n, m in module.named_modules():
                    if n == 'cv3.conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # TODO: Debug the issue with the maxpooling channel weights
            # The SPPF conv layers--------------------------------------------------------------
            # mAP50-95: ERROR 
            # if name.endswith('model.9'):
            #     for n, m in module.named_modules():
            #         if n == 'cv1.conv':
            #             parameters_to_prune.append((m, 'weight'))
            #             module_names.append(f'{name}.{n}')
            #             prune.ln_structured(m, 'weight', sl, 1, 0)
            # ----------------------------------------------------------------------------------
            
            # The first conv layer in the neck--------------------------------------------------
            # mAP50-95: 0.385
            if name.endswith('model.10'):
                for n, m in module.named_modules():
                    if n == 'conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The first conv layer in the neck--------------------------------------------------
            # mAP50-95: 0.385
            if name.endswith('model.10'):
                for n, m in module.named_modules():
                    if n == 'conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The 2nd conv layer in the neck----------------------------------------------------
            # mAP50-95: 0.187
            if name.endswith('model.14'):
                for n, m in module.named_modules():
                    if n == 'conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The 3rd and 4th conv layers in the neck-------------------------------------------
            # mAP50-95: 0.365
            if name.endswith('model.18') or name.endswith('model.21'):
                for n, m in module.named_modules():
                    if n == 'conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The external conv layers in the 1st C3 block in the neck--------------------------
            # mAP50-95: 0.125
            if name.endswith('model.13'):
                for n, m in module.named_modules():
                    if n == 'cv1.conv' or n == 'cv2.conv' or n == 'cv3.conv':
                        # parameters_to_prune.append((m, 'weight', f'{name}.{n}'))
                        # module_names.append(f'{name}.{n}')
                        prune_structured(m, 'weight', amount)
                        # prune.ln_structured(m, 'weight', amount, float('inf'), 0)
                        # if m.bias != None:
                        #     bias_mask = torch.ones_like(m.bias, device=next(model.parameters()).device)
                        #     filter_indices = get_filter_indices(m)
                        #     bias_mask[filter_indices] = 0
                        #     prune.custom_from_mask(m, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # TODO: See why the HEAD pruning is causing non-deterministic quality metrics
            # The external conv layers in the 2nd C3 block in the neck--------------------------
            # mAP50-95: 0.335
            # if name.endswith('model.17'):
            #     for n, m in module.named_modules():
            #         if n == 'cv1.conv' or n == 'cv2.conv': # or n == 'cv3.conv':
            #             parameters_to_prume.append((m, 'weight'))
            #             module_names.append(f'{name}.{n}')
            #             prune.ln_structured(m, 'weight', sl, 1, 0)
            # ----------------------------------------------------------------------------------
            
            # TODO: See why the HEAD pruning is causing non-deterministic quality metrics
            # The external conv layers in the 3nd C3 block in the neck--------------------------
            # mAP50-95: 0.268
            # if name.endswith('model.20'):
            #     for n, m in module.named_modules():
            #         if n == 'cv1.conv' or n == 'cv2.conv': # or n == 'cv3.conv':
            #             parameters_to_prume.append((m, 'weight'))
            #             module_names.append(f'{name}.{n}')
            #             prune.ln_structured(m, 'weight', sl, 1, 0)
            # ----------------------------------------------------------------------------------
            
            # TODO: See why the HEAD pruning is causing non-deterministic quality metrics
            # The external conv layers in the 4th C3 block in the neck--------------------------
            # mAP50-95: 0.338
            # if name.endswith('model.23'):
            #     for n, m in module.named_modules():
            #         if n == 'cv1.conv' or n == 'cv2.conv': # or n == 'cv3.conv':
            #             parameters_to_prume.append((m, 'weight'))
            #             module_names.append(f'{name}.{n}')
            #             prune.ln_structured(m, 'weight', sl, 1, 0)
            # ----------------------------------------------------------------------------------
            
            # The bottleneck conv layers in the 1st C3 block in the neck-------------------------
            # mAP50-95: 0.306
            if name.endswith('model.13'):
                for n, m in module.named_modules():
                    if n == 'm':
                        for n_, m_ in m.named_modules():
                            if n_.endswith('conv'):
                                # parameters_to_prune.append((m_, 'weight', f'{name}.{n}.{n_}'))
                                # module_names.append(f'{name}.{n}.{n_}')
                                prune_structured(m_, 'weight', amount)
                                # prune.ln_structured(m_, 'weight', amount, float('inf'), 0)
                                # if m_.bias != None:
                                #     bias_mask = torch.ones_like(m_.bias, device=next(model.parameters()).device)
                                #     filter_indices = get_filter_indices(m_)
                                #     bias_mask[filter_indices] = 0
                                #     prune.custom_from_mask(m_, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The bottleneck conv layers in the 2nd C3 block in the neck-------------------------
            # mAP50-95: 0.397
            if name.endswith('model.17'):
                for n, m in module.named_modules():
                    if n == 'm':
                        for n_, m_ in m.named_modules():
                            if n_.endswith('conv'):
                                # parameters_to_prune.append((m_, 'weight', f'{name}.{n}.{n_}'))
                                # module_names.append(f'{name}.{n}.{n_}')
                                prune_structured(m_, 'weight', amount)
                                # prune.ln_structured(m_, 'weight', amount, float('inf'), 0)
                                # if m_.bias != None:
                                #     bias_mask = torch.ones_like(m_.bias, device=next(model.parameters()).device)
                                #     filter_indices = get_filter_indices(m_)
                                #     bias_mask[filter_indices] = 0
                                #     prune.custom_from_mask(m_, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The bottleneck conv layers in the 3rd C3 block in the neck-------------------------
            # mAP50-95: 0.379
            if name.endswith('model.20'):
                for n, m in module.named_modules():
                    if n == 'm':
                        for n_, m_ in m.named_modules():
                            if n_.endswith('conv'):
                                # parameters_to_prune.append((m_, 'weight', f'{name}.{n}.{n_}'))
                                # module_names.append(f'{name}.{n}.{n_}')
                                prune_structured(m_, 'weight', amount)
                                # prune.ln_structured(m_, 'weight', amount, float('inf'), 0)
                                # if m_.bias != None:
                                #     bias_mask = torch.ones_like(m_.bias, device=next(model.parameters()).device)
                                #     filter_indices = get_filter_indices(m_)
                                #     bias_mask[filter_indices] = 0
                                #     prune.custom_from_mask(m_, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
            
            # The bottleneck conv layers in the 4th C3 block in the neck-------------------------
            # mAP50-95: 0.336
            if name.endswith('model.23'):
                for n, m in module.named_modules():
                    if n == 'm':
                        for n_, m_ in m.named_modules():
                            if n_.endswith('conv'):
                                # parameters_to_prune.append((m_, 'weight', f'{name}.{n}.{n_}'))
                                # module_names.append(f'{name}.{n}.{n_}')
                                prune_structured(m_, 'weight', amount)
                                # prune.ln_structured(m_, 'weight', amount, float('inf'), 0)
                                # if m_.bias != None:
                                #     bias_mask = torch.ones_like(m_.bias, device=next(model.parameters()).device)
                                #     filter_indices = get_filter_indices(m_)
                                #     bias_mask[filter_indices] = 0
                                #     prune.custom_from_mask(m_, 'bias', bias_mask)
            # ----------------------------------------------------------------------------------
    # model.cuda()

def main(opt):
    # Print args
    print_args(vars(opt))
    
    # Initialize Tensorboard writer
    writer = SummaryWriter(comment=f'_{opt.weights}')
    opt.save_dir = writer.log_dir
    
    # Prepare options
    opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
        check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    # opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    
    # Get current device
    device = select_device(opt.device, batch_size=opt.batch_size)
    
    # Begin Training
    save_dir, epochs, batch_size, weights, single_cls, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Hyperparameters
    hyp = opt.hyp
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints
    
    # Save run settings
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))

    # Config
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = check_dataset(data)  # check if None
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    # TODO: Make learning rate user-defined
    hyp['lr0'] = 0.0005
    # optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    
    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None
    
    # Resume
    best_fitness, start_epoch = 0.0, 0
    
    # Train loader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
    
    # Validation loader
    val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
    if not resume:
        if not opt.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
        model.half().float()  # pre-reduce anchor precision
        
    # Model attributes
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    global_step = 0
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    compute_loss = ComputeLoss(model)  # init loss class
    
    # Initial evaluation
    results, maps, _ = validate.run(data_dict,
                                        batch_size=batch_size // WORLD_SIZE * 2,
                                        imgsz=imgsz,
                                        half=amp,
                                        # model=ema.ema,
                                        model=model,
                                        single_cls=single_cls,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        plots=False,
                                        callbacks=Callbacks(),
                                        compute_loss=compute_loss)
    
    # Prune ema and model
    # TODO: Replace with a real solution to the ema key mismatch problem
    for name, module in ema.ema.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.identity(module, 'weight')
            if module.bias != None:
                prune.identity(module, 'bias')
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.identity(module, 'weight')
            if module.bias != None:
                prune.identity(module, 'bias')
                
    # Reinitialize optimizer (to include weight_orig instead of weight)
    # TODO: Find better place to do this
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
        
    # Tensorboard
    log(writer, results, global_step, val_only=True)
    global_step += 1
    
    # Get the parameters to prune
    # parameters_to_prune = get_prunable_parameters(model)
    
    # Get the pruning configurations
    pruning_configs = []
    file = Path(f'./data/{weights[:-3]}.log')
    for i, (score, density, config) in enumerate(get_next(file)):
        copy_config = {}
        for key in config:
            copy_config[key.replace('model.model.', '')] = config[key]
        if i % 9 == 0 or i == 299:
            pruning_configs.append(copy_config)
    
    # for pruning_step in range(7):
    total_epochs = 0
    from_config = False
    if from_config:
        g = enumerate(pruning_configs)
    else:
        g = range(30)
    for pruning_step in g: # range(7)
        if from_config:
            pruning_config = pruning_step[1]
            pruning_step = pruning_step[0] 
        
        # Initialize model checkpoints
        best_fitness = 0.0
        last, best = w / f'last_{pruning_step}.pt', w / f'best_{pruning_step}.pt'
        
        # Prune model iteratively
        if not from_config: 
            prune_filters_(model, opt.weights, 0.02 + (0.02 * pruning_step))
        else:
            for name, module in model.named_modules():
                if name in pruning_config:
                    prune_structured(module, 'weight', 5 * pruning_config[name])
            
        # Initial evalutation
        results, maps, _ = validate.run(data_dict,
                                        batch_size=batch_size // WORLD_SIZE * 2,
                                        imgsz=imgsz,
                                        half=amp,
                                        model=ema.ema,
                                        # model=model,
                                        single_cls=single_cls,
                                        dataloader=val_loader,
                                        save_dir=save_dir,
                                        plots=False,
                                        callbacks=Callbacks(),
                                        compute_loss=compute_loss)
        
        # Tensorboard
        log(writer, results, global_step, val_only=True)
        global_step += 1
        
        for epoch in range(epochs):
            model.train()
            
            mloss = torch.zeros(3, device=device)  # mean losses
            
            # Progress bar
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
            
            # Training batch
            optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:
                total_epochs += 1
                ni = i + nb * total_epochs  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

                # Forward
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)  # forward
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                
                # Backward
                scaler.scale(loss).backward()
                
                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                # print(ni - last_opt_step >= accumulate, ni, last_opt_step, accumulate)
                # if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni
                    
                # Log
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                
            # Validation
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = (epoch + 1 == epochs)
            
            results, maps, _ = validate.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            half=amp,
                                            model=ema.ema,
                                            # model = model,
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            plots=False,
                                            callbacks=Callbacks(),
                                            compute_loss=compute_loss)
            
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results)
            
            # Tensorboard
            log(writer, log_vals, global_step)
            global_step += 1
            
            # Save
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                # 'model': deepcopy(de_parallel(model)).half(),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict(),
                'opt': vars(opt),
                'date': datetime.now().isoformat()}
            
            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            del ckpt
            
            # End epoch
        # End training cycle
    # End iterative pruning
        
    
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)