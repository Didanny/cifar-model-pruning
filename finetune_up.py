import argparse

from tqdm import tqdm

import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy

import pytorch_cifar_models
from common import *

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--checkpoint', type=int)
    parser.add_argument('--starting-state', type=str)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):
    


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)