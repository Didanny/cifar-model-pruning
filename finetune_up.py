import argparse

from tqdm import tqdm

import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy

import pytorch_cifar_models
from common import *
from nn_utils import *
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--starting-state', type=str)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def prepare_for_training(device, model, starting_state, checkpoint):
    # Load the model
    # checkpoint = './runs/Oct09_04-08-33_poison.ics.uci.educifar100_vgg11_bn/cifar100_vgg11_bn_3_loss1.654237985610962_acco0.6675999760627747_accf0.8598999977111816.pt'
    model = load_checkpoint(model, checkpoint)
    # convert_model(model)
    
    # Add the metrics to the model
    model.accuracy_top1 = Accuracy(task='multiclass', num_classes=100)
    model.accuracy_top5 = Accuracy(task='multiclass', num_classes=100, top_k=5)
    
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
    optimizer = None
    # optimizer = optim.SGD([v for n, v in model.named_parameters()], 1, 0.9, 0, 5e-4, True)
    
    # Intialize the scheduler
    # TODO: Make T_max a user-defined
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=300, eta_min=0)
    scheduler = None
    
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
        
        # preserve_unpruned_weights(model)
        
        optimizer.step()
        
        # restore_unpruned_weights(model)
        
        # scheduler.step()
        
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
    writer = SummaryWriter(comment=f'_{opt.model}_{opt.starting_state}')
    save_dir = Path(writer.log_dir)
    
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Set up training
    model, criterion, optimizer, scheduler, train_loader, val_loader = prepare_for_training(device, opt.model, opt.starting_state, opt.checkpoint)
    
    # The initial evalutation
    evaluate(model, criterion, val_loader, device, 0)
    
    # Model starting state
    initialize_checkpoint(model, opt.starting_state)
    
    # Convert Conv2d layers to FrozenConv2d
    convert_model(model)
    
    # Move to cuda if available
    if torch.cuda.is_available():
        model.to(device=device)
        
    # EMA
    ema = ModelEMA(model)
        
    # Reinitialize optimizer
    p = []
    for _, mod in model.named_modules():
        for _, v in mod.named_parameters():
            p.append(v)            
    optimizer = optim.SGD(p, 0.001, 0.9, 0, 5e-4, True)
    
    # Reinitialize scheduler
    epochs = 300
    lf = lambda x: (1 - x / epochs) * (1.0 - 0.002) + 0.002 
    # def lf(x):
    #     if x <= 50:
    #         return (1 - x / 50) * (1.0 - 0.01) + 0.01
    #     else:
    #         return 0.01
    scheduler = LambdaLR(optimizer, lr_lambda=lf)
    
    # Initialize best and last model metrics
    best_dict = None
    last_dict = None
    best_fitness = 0.0
    last, best = w / 'last.pt', w / 'best.pt'
    
    # Begin Fine-tuning
    global_step = 0
    
    # The initial evaluation
    loss_eval, acc_1, acc_5 = evaluate(ema.ema, criterion, val_loader, device, 0)
    global_step += 1
    
    # Tensorboard
    writer.add_scalar('Validation/Loss', loss_eval, global_step)
    writer.add_scalar('Validation/Accuracy (Top-1)', acc_1, global_step)
    writer.add_scalar('Validation/Accuracy (Top-5)', acc_5, global_step)
    # writer.add_scalar('Debug/fixed_weight_1', model.features[11].weight_list[0].data[0,0,0,0], global_step)
    # writer.add_scalar('Debug/fixed_weight_2', model.features[11].weight_list[0].data[0,0,0,1], global_step)
    # writer.add_scalar('Debug/fixed_weight_3', model.features[11].weight_list[0].data[0,0,0,2], global_step)
    # writer.add_scalar('Debug/trainable_weight_1', model.features[11].weight_list[1].data[0,0,0,0], global_step)
    # writer.add_scalar('Debug/trainable_weight_2', model.features[11].weight_list[1].data[0,0,0,1], global_step)
    # writer.add_scalar('Debug/trainable_weight_3', model.features[11].weight_list[1].data[0,0,0,2], global_step)
    
    # Begin Training
    for epoch in range(epochs):
        # Train
        loss_train = train(model, criterion, optimizer, scheduler, train_loader, device, epoch)
        ema.update(model)
        scheduler.step()
        print(optimizer.param_groups[0]['lr'])
            
        # Eval
        loss_eval, acc_1, acc_5 = evaluate(ema.ema, criterion, val_loader, device, epoch)

        # Update best model metrics
        fitness = (0.1 * acc_5) + (0.9 * acc_1)
        if best_fitness < fitness:
            best_fitness = fitness
            best_dict = {'params': ema.ema.state_dict(), 'accuracy_top_5': acc_5, 'accuracy_top_1': acc_1}
            
        # Update last model metrics
        if epoch == 99:
            last_dict = {'params': ema.ema.state_dict(), 'accuracy_top_5': acc_5, 'accuracy_top_1': acc_1}
            
        # Tensorboard
        writer.add_scalar('Training/Loss', loss_train, global_step)
        writer.add_scalar('Validation/Loss', loss_eval, global_step)
        writer.add_scalar('Validation/Accuracy (Top-1)', acc_1, global_step)
        writer.add_scalar('Validation/Accuracy (Top-5)', acc_5, global_step)
        # writer.add_scalar('Debug/fixed_weight_1', model.features[11].weight_list[0].data[0,0,0,0], global_step)
        # writer.add_scalar('Debug/fixed_weight_2', model.features[11].weight_list[0].data[0,0,0,1], global_step)
        # writer.add_scalar('Debug/fixed_weight_3', model.features[11].weight_list[0].data[0,0,0,2], global_step)
        # writer.add_scalar('Debug/trainable_weight_1', model.features[11].weight_list[1].data[0,0,0,0], global_step)
        # writer.add_scalar('Debug/trainable_weight_2', model.features[11].weight_list[1].data[0,0,0,1], global_step)
        # writer.add_scalar('Debug/trainable_weight_3', model.features[11].weight_list[1].data[0,0,0,2], global_step)
        
        # Increment global step
        global_step += 1
        
    # Save the best and last
    torch.save(best_dict, best)
    torch.save(last_dict, last)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)