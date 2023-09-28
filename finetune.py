import argparse

from tqdm import tqdm

import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import pytorch_cifar_models

from common import *

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):    
    # torch.set_default_device('cuda')
    
    # Load the model
    model = getattr(pytorch_cifar_models, opt.model)(pretrained=True)
    
    # Get the data loaders
    train_loader, val_loader = cifar100()    
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()
    
    # Initialize optimizer
    optimizer = optim.SGD([v for n, v in model.named_parameters()], 0.1, 0.9, 0, 5e-4, True)
    
    # Intialize the scheduler
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=0)
    
    # Begin Training
    for epoch in range(30):
        
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader, 0), desc=f'Training Epoch {epoch}'):
            inputs, labels = data
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            running_loss += loss
            optimizer.step()
            
            scheduler.step()
            
        print(f'Loss = {running_loss}')   
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

