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
from typing import Optional, Sequence
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
            predictions = model(data)
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
    
    def __exit__(self):
        self.dt = self.time() - self.start
        self.t += self.dt
        
    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        else:
            time.time()