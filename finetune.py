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
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def prepare_for_training(device, model):
    # Load the model
    model = getattr(pytorch_cifar_models, model)(pretrained=True) 
    
    # Add the metrics to the model
    model.accuracy_top1 = Accuracy(task='multiclass', num_classes=100)
    model.accuracy_top5 = Accuracy(task='multiclass', num_classes=100, top_k=5)
    
    model.accuracy_top1.reset()
    
    # Get the data loaders
    train_loader, val_loader = cifar100()
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()
    
    # Move to cuda if available
    if torch.cuda.is_available():
        model.to(device=device)
        criterion.to(device=device)
        
    # Initialize optimizer
    # TODO: Make parameters user-defined
    optimizer = optim.SGD([v for n, v in model.named_parameters()], 0.1, 0.9, 0, 5e-4, True)
    
    # Intialize the scheduler
    # TODO: Make T_max a user-defined
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=0)
    
    # Return 
    return model, criterion, optimizer, scheduler, train_loader, val_loader

def train(model, criterion, optimizer, scheduler, train_loader, device, epoch):
    # Training mode
    model.train()
    
    # Keep track of loss
    running_loss = 0.0
    
    # Run 1 epoch
    for i, data in enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch}'), 0):
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        running_loss += loss
        optimizer.step()
        
        scheduler.step()
        
    # Record results
    running_loss = running_loss / len(train_loader)
    print(f'Training Loss = {running_loss}')  
    
    # Return results
    return running_loss
    
@torch.no_grad()
def evaluate(model, criterion, val_loader, device, epoch):
    # Eval mode
    model.eval()
    
    # Keep track of loss
    running_loss = 0.0
    
    # Run 1 epoch
    for i, data in enumerate(tqdm(val_loader, desc=f'Validation Epoch {epoch}'), 0):
        inputs, labels = data
        inputs = inputs.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        model.accuracy_top1.update(outputs, labels)
        model.accuracy_top5.update(outputs, labels)

        running_loss += loss
        
    # Record results
    running_loss = running_loss / len(val_loader)
    acc_1 = model.accuracy_top1.compute()
    acc_5 = model.accuracy_top5.compute()
    print(f'Validation Loss = {running_loss}, Accuracy (Top-1): {acc_1}, Accuracy (Top-5): {acc_5}')     
    
    # Reset the metrics 
    model.accuracy_top1.reset()
    model.accuracy_top5.reset()
    
    # Return results
    return running_loss, acc_1, acc_5
    
def main(opt):    
    # Get the current device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    
    # Set up tensorboard summary writer
    # TODO: Create more comprhensive automated commenting
    writer = SummaryWriter(comment=opt.model)
    
    # Set up training
    model, criterion, optimizer, scheduler, train_loader, val_loader = prepare_for_training(device, opt.model)
    
    # The initial evalutation
    evaluate(model, criterion, val_loader, device, 0)
    
    # Begine Fine-tuning
    global_step = 0
    for pruning_step in range(9):
        # Prune the model once
        prune_filters(model, opt.model)
    
        # The initial evaluation
        loss_eval, acc_1, acc_5 = evaluate(model, criterion, val_loader, device, pruning_step)
        global_step += 1
        
        # Tensorboard
        writer.add_scalar('Validation/Loss', loss_eval, global_step)
        writer.add_scalar('Validation/Accuracy (Top-1)', acc_1, global_step)
        writer.add_scalar('Validation/Accuracy (Top-5)', acc_5, global_step)

        # Begin Training
        # TODO: Make num_epochs user-defined (Make sure same arg is used for T_max in scheduler)
        for epoch in range(30):
            # Train
            loss_train = train(model, criterion, optimizer, scheduler, train_loader, device, epoch)
            
            # Eval
            loss_eval, acc_1, acc_5 = evaluate(model, criterion, val_loader, device, epoch)
            
            # Tensorboard
            writer.add_scalar('Training/Loss', loss_train, global_step)
            writer.add_scalar('Validation/Loss', loss_eval, global_step)
            writer.add_scalar('Validation/Accuracy (Top-1)', acc_1, global_step)
            writer.add_scalar('Validation/Accuracy (Top-5)', acc_5, global_step)
            
            # Increment global step
            global_step += 1        
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

