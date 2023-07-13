import contextlib
import time
import torch
import pytorch_cifar_models
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.utils.prune as prune
from torchvision.datasets import CIFAR10, CIFAR100
from torchmetrics.classification import Accuracy
from torch import nn
from pathlib import Path
from typing import Optional, Sequence, Union
from typing_extensions import Literal
from tqdm import tqdm

DATA_DIR = Path('./data')

def get_val_transforms(mean: Sequence[int], std: Sequence[int]):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

def get_dataset(dataset_name: Literal['cifar10', 'cifar100']):
    if dataset_name == 'cifar10':
        val_set = CIFAR10(DATA_DIR, train=False, download=False, transform=get_val_transforms(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
        train_set = CIFAR10(DATA_DIR, train=True, download=False)
        return train_set, val_set
    elif dataset_name == 'cifar100':
        val_set = CIFAR100('./data', train=False, download=False, transform=get_val_transforms(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]))
        train_set = CIFAR100('./data', train=True, download=False)
        return train_set, val_set
    
def load_model(dataset_name: Literal, architecture: Literal, pretrained: Optional[bool] = True):
    return getattr(pytorch_cifar_models, f'{dataset_name}_{architecture}')(pretrained=pretrained)

def validate(data: data.DataLoader, model: nn.Module):
    # Set up the quality metrics
    accuracy_top1 = Accuracy(task='multiclass', num_classes=100)
    accuracy_top5 = Accuracy(task='multiclass', num_classes=100, top_k=5)
    
    # Set up the profiler
    profile = Profile()
    
    # Set up progress bar
    pbar = tqdm(data)
    
    # Validate
    seen = 0
    for datum, target in pbar:
        with profile:
            predictions = model(datum)
        seen += datum.shape[0]
        accuracy_top1.update(predictions, target)
        accuracy_top5.update(predictions, target)
        
    # Get the inference times
    total_time_s = profile.t
    single_inference_time_ms = profile.t / seen * 1E3
    
    # Return results
    return  total_time_s, \
            single_inference_time_ms, \
            accuracy_top1.compute(), \
            accuracy_top5.compute()
    
class Profile(contextlib.ContextDecorator):
    def __init__(self, t: Optional[float] = 0.0):
        super().__init__()
        
        self.t = t 
        self.cuda = torch.cuda.is_available()
        
    def __enter__(self):
        self.start = self.time()
        return self
    
    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start
        self.t += self.dt
        
    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()
    
def prune_kernel(downstream_parameter: tuple(nn.Conv2d, Literal), upstream_parameter: tuple(nn.Conv2d, Literal)):
    # Get the pruned filter indices from the pruning mask of the upstream module
    up_module, up_name = upstream_parameter
    up_mask = getattr(up_module, up_name + '_mask')
    
    raise NotImplementedError

def global_smallest_filter_mean(parameters: prune.Iterable, amount: float, **kwargs):
    # Ensure parameters is a list or generator of tuples
    if not isinstance(parameters, prune.Iterable):
        raise TypeError('global_smallest_filter(): parameters is not an iterable')
    
    # Get the total number of parameters
    num_params = 0
    for module, name in parameters:
        num_params += module.weight.numel()
        prune.identity(module, name)
        
    # Determine the number of params to prune
    num_to_prune = int(num_params * amount)
    
    # TODO: Replace the hard-coded 'module.weight' and use getattr on param instead with some checks
    # Create importance scores
    importance_scores = [
        torch.vstack([
            torch.norm(module.weight, 1, (1, 2, 3)).to(module.weight.device) / (module.kernel_size[0]*module.kernel_size[1]*module.in_channels),
            torch.range(0, module.weight.shape[0] - 1).to(module.weight.device),
            (torch.ones(module.weight.shape[0]) * i).to(module.weight.device),
            torch.ones(module.weight.shape[0]).to(module.weight.device) * (module.kernel_size[0]*module.kernel_size[1]*module.in_channels)
        ])
        for i, (module, param) in enumerate(parameters)
    ]
    importance_scores = torch.hstack(importance_scores)
    
    # Sort the importance score matrix along the L1 norm row
    sorted_indices = importance_scores[0,:].sort()[1]
    importance_scores = importance_scores[:, sorted_indices]
    
    # Get the cumulative sum of the filter size row to determine when to stop pruning
    importance_scores[3,:] = torch.cumsum(importance_scores[3,:], -1)
    
    # Find the index of the last filter to be pruned
    importance_scores[3,:] > num_to_prune
    last_prune_idx = torch.argmax((importance_scores[3,:] > num_to_prune).to(dtype=torch.int))
    importance_scores = importance_scores[:, :last_prune_idx]
    
    # Get the unique module indices
    module_indices = torch.unique(importance_scores[2,:]).to(dtype=torch.int)
    
    # Iterate over the importance scores and prune the corresponding filter
    # for i in range(last_prune_idx):        
    for module_idx in module_indices:
        # Get the importance scores corresponding to this layer
        module_scores = importance_scores[:,(importance_scores[2,:] == module_idx).nonzero().squeeze(1)]
        
        # Get the kernel indices of all kernels in this layer
        filter_indices = module_scores[1,:].to(dtype=torch.long)
        
        # Get the module and name
        module, name = parameters[module_idx]
        
        # Get the default mask of the module
        mask = getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
        
        # Compute the new module mask
        mask[filter_indices, :, :, :] = 0
        
        # Get and modify the bias mask
        if module.bias != None:
            bias_mask = getattr(module, 'bias' + '_mask', torch.ones_like(getattr(module, 'bias')))
            bias_mask[filter_indices] = 0
            
        # Apply the mask
        prune.custom_from_mask(module, name, mask=mask)
        if module.bias != None:
            prune.custom_from_mask(module, 'bias', mask=bias_mask)
        #     prune.remove(module, 'bias')
        # prune.remove(module, name)
        
def global_smallest_filter_norm(parameters: prune.Iterable, amount: float, norm: Optional[Literal['1', '2', 'inf']] = 1, **kwargs):
    # Ensure parameters is a list or generator of tuples
    if not isinstance(parameters, prune.Iterable):
        raise TypeError('global_smallest_filter(): parameters is not an iterable')
    
    # Preprocess norm argument
    if norm == 'inf':
        norm = float(norm)
    else:
        norm = int(norm)
    
    # Get the total number of parameters
    num_params = 0
    for module, name in parameters:
        num_params += module.weight.numel()
        prune.identity(module, name)
        
    # Determine the number of params to prune
    num_to_prune = int(num_params * amount)
    
    # TODO: Replace the hard-coded 'module.weight' and use getattr on param instead with some checks
    # Create importance scores
    importance_scores = [
        torch.vstack([
            torch.norm(module.weight, norm, (1, 2, 3)).to(module.weight.device),
            torch.range(0, module.weight.shape[0] - 1).to(module.weight.device),
            (torch.ones(module.weight.shape[0]) * i).to(module.weight.device),
            torch.ones(module.weight.shape[0]).to(module.weight.device) * (module.kernel_size[0]*module.kernel_size[1]*module.in_channels)
        ])
        for i, (module, param) in enumerate(parameters)
    ]
    importance_scores = torch.hstack(importance_scores)
    
    # Sort the importance score matrix along the L-p norm row
    sorted_indices = importance_scores[0,:].sort()[1]
    importance_scores = importance_scores[:, sorted_indices]
    
    # Get the cumulative sum of the filter size row to determine when to stop pruning
    importance_scores[3,:] = torch.cumsum(importance_scores[3,:], -1)
    
    # Find the index of the last filter to be pruned
    importance_scores[3,:] > num_to_prune
    last_prune_idx = torch.argmax((importance_scores[3,:] > num_to_prune).to(dtype=torch.int))
    importance_scores = importance_scores[:, :last_prune_idx]
    
    # Get the unique module indices
    module_indices = torch.unique(importance_scores[2,:]).to(dtype=torch.int)
    
    # Iterate over the importance scores and prune the corresponding filter
    # for i in range(last_prune_idx):        
    for module_idx in module_indices:
        # Get the importance scores corresponding to this layer
        module_scores = importance_scores[:,(importance_scores[2,:] == module_idx).nonzero().squeeze(1)]
        
        # Get the kernel indices of all kernels in this layer
        filter_indices = module_scores[1,:].to(dtype=torch.long)
        
        # Get the module and name
        module, name = parameters[module_idx]
        
        # Get the default mask of the module
        mask = getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
        
        # Compute the new module mask
        mask[filter_indices, :, :, :] = 0
        
        # Get and modify the bias mask
        if module.bias != None:
            bias_mask = getattr(module, 'bias' + '_mask', torch.ones_like(getattr(module, 'bias')))
            bias_mask[filter_indices] = 0
            
        # Apply the mask
        prune.custom_from_mask(module, name, mask=mask)
        if module.bias != None:
            prune.custom_from_mask(module, 'bias', mask=bias_mask)
        #     prune.remove(module, 'bias')
        # prune.remove(module, name)