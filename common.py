import contextlib
import re
import time
import json
import torch
import pytorch_cifar_models
import torch.utils.data as data
import torchvision.transforms as T
import torch.nn.utils.prune as prune
from torchvision.datasets import CIFAR10, CIFAR100
from torchmetrics.classification import Accuracy
from torch import nn
from pathlib import Path
from typing import Optional, Sequence, Union, TypedDict
from typing_extensions import Literal, TypeAlias
from tqdm import tqdm

DATA_DIR = Path('./data')

def get_train_transforms(mean, std):
    return T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def get_val_transforms(mean, std):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def _cifar(root, image_size, mean, std, batch_size, num_workers, dataset_builder, **kwargs):
    train_transforms = get_train_transforms(mean, std)
    val_transforms = get_val_transforms(mean, std)

    trainset = dataset_builder(root, train=True, transform=train_transforms, download=True)
    valset = dataset_builder(root, train=False, transform=val_transforms, download=True)

    # TODO: Maybe add support for distributed runs
    train_sampler = None
    val_sampler = None

    train_loader = data.DataLoader(trainset, batch_size=batch_size,
                                   shuffle=(train_sampler is None),
                                   sampler=train_sampler,
                                   num_workers=num_workers,
                                   persistent_workers=True)
    val_loader = data.DataLoader(valset, batch_size=batch_size,
                                 shuffle=(val_sampler is None),
                                 sampler=val_sampler,
                                 num_workers=num_workers,
                                 persistent_workers=True)

    return train_loader, val_loader

def cifar100():
    return _cifar(
        root=DATA_DIR,
        image_size=32,
        mean=[0.5070, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2761],
        batch_size=256,
        num_workers=2,
        dataset_builder=CIFAR100,
    )
    
def dummy_prune(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.identity(module, 'weight')
            if module.bias != None:
                prune.identity(module, 'bias')    
    
def prune_filters(model: nn.Module, model_name: Literal, amount: float):
    if model_name.startswith('cifar100_vgg'):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, 'weight', amount, float('inf'), 0)
                
                if module.bias != None:
                    bias_mask = torch.ones_like(module.bias, device=next(model.parameters()).device)
                    filter_indices = get_filter_indices(module)
                    bias_mask[filter_indices] = 0
                    prune.custom_from_mask(module, 'bias', bias_mask)
    elif model_name.startswith('cifar100_resnet'):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, 'weight', amount, float('inf'), 0)
                
                if module.bias != None:
                    bias_mask = torch.ones_like(module.bias, device=next(model.parameters()).device)
                    filter_indices = get_filter_indices(module)
                    bias_mask[filter_indices] = 0
                    prune.custom_from_mask(module, 'bias', bias_mask)
                
def prune_structured(module: nn.Module, name: Literal, amount: float):
    # Get the number of filters to be pruned
    if amount < 1:
        k = round(amount * module.out_channels)
    else:
        k = int(amount)
    
    # Get the inf norms of the filters
    norms = torch.norm(module.weight.data, float('inf'), (1,2,3))
    
    # Get the indices of the bottom-k filters
    smallest_filters = torch.topk(norms, k, largest=False).indices 
    
    # Create and apply the mask
    mask = torch.ones_like(module.weight.data)
    mask[smallest_filters,:,:,:] *= 0
    prune.custom_from_mask(module, name, mask)
    
    # Apply the same to the bias if it exists
    if module.bias != None:
        bias_mask = torch.ones_like(module.bias.data)
        bias_mask[smallest_filters] = 0
        prune.custom_from_mask(module, 'bias', bias_mask)

def restore_unpruned_weights(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.data[module.filter_indices, :, :, :] = module.old_weight.data[module.filter_indices, :, :, :]
            if module.bias != None:
                module.bias.data[module.filter_indices] = module.old_bias.data[module.filter_indices]                    

def preserve_unpruned_weights(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            module.weight.grad[module.filter_indices, :, :, :] = 0
            if module.bias != None:
                module.bias.grad[module.filter_indices] = 0

def get_data_loaders(dataset_name: str):
    if dataset_name == 'cifar100':
        return cifar100()

def dot_num_to_brack(match):
    # print(match)
    return f'[{match.group()[1:-1]}].'

def dot_num_to_brack_end(match):
    # print(match)
    return f'[{match.group()[1:]}]'

# def get_val_transforms(mean: Sequence[int], std: Sequence[int]):
#     return T.Compose([
#         T.ToTensor(),
#         T.Normalize(mean, std)
#     ])
    
def get_dependency_graph(model: nn.Module, name: str):
    if name.startswith('cifar100_vgg'):
        return get_dependency_graph_vgg(model)
    elif name.startswith('cifar100_resnet'):
        return get_dependency_graph_resnet(model)
    elif name.startswith('cifar100_mobilenet'):
        return get_dependency_graph_mobilenet(model)
    
def get_residual_dependency(model: nn.Module, name: str):
    if name.startswith('cifar100_vgg'):
        return {}
    elif name.startswith('cifar100_resnet'):
        return get_residual_dependency_resnet(model)
    elif name.startswith('cifar100_mobilenet'):
        return {}
    
def fuse_batchnorms(model: nn.Module, name: str, **kwargs):
    if name.startswith('cifar100_vgg'):
        for i in range(len(model.features)):
            if isinstance(model.features[i], nn.Conv2d) and isinstance(model.features[i + 1], nn.BatchNorm2d):
                model.features[i] = fuse_conv_and_bn(model.features[i], model.features[i + 1], **kwargs)
                model.features[i].requires_grad = False
                model.features[i + 1] = nn.Identity()
    elif name.startswith('cifar100_resnet'):
        raise NotImplementedError
    elif name.startswith('cifar100_mobilenet'):
        raise NotImplementedError
    
def compress_model(model: nn.Module, name: str, **kwargs):
    if name.startswith('cifar100_vgg'):
        compress_model_vgg(model, **kwargs)
    elif name.startswith('cifar100_resnet'):
        raise NotImplementedError
    elif name.startswith('cifar100_mobilenet'):
        raise NotImplementedError
    
def transfer_masks(model_fuse: nn.Module, model: nn.Module, name: str, **kwargs):
    if name.startswith('cifar100_vgg'):
        for i in range(len(model_fuse.features)):
            if isinstance(model_fuse.features[i], nn.Conv2d):
                conv = model.features[i]
                prune.custom_from_mask(model_fuse.features[i], 'weight', mask=conv.weight_mask.data)
                bias_mask = getattr(conv, 'bias_mask', torch.ones_like(model_fuse.features[i].bias))
                prune.custom_from_mask(model_fuse.features[i], 'bias', mask=bias_mask)
    elif name.startswith('cifar100_resnet'):
        raise NotImplementedError
    elif name.startswith('cifar100_mobilenet'):
        raise NotImplementedError
    
def get_dependency_graph_vgg(vgg: nn.Module):
    dependencies = {}
    modules = [name for name, module in vgg.named_modules() if isinstance(module, nn.Conv2d)]
    for i, module in enumerate(modules):
        if i == 0:
            prev_module = module
            continue
        dependencies[module] = prev_module
        prev_module = module
    return dependencies

def get_dependency_graph_resnet(resnet: nn.Module):
    dependencies = {}
    for name, module in resnet.named_modules():
        if isinstance(module, nn.Conv2d) and name.endswith('conv2'):
            dependencies[name] = f'{name[:-5]}conv1'
    return dependencies
    
def get_residual_dependency_resnet(resnet: nn.Module):
    dependencies = {}
    for name, module in resnet.named_modules():
        if isinstance(module, nn.Conv2d) and 'downsample' in name:
            dependencies[f'{name[:-12]}conv2'] = name
    return dependencies

def get_dependency_graph_mobilenet(mobnet: nn.Module):
    dependencies = {}
    modules = [name for name, module in mobnet.named_modules() if isinstance(module, nn.Conv2d)]
    for i, module in enumerate(modules):
        if i == 0:
            prev_module = module
            continue
        dependencies[module] = prev_module
        prev_module = module
    return dependencies

def get_parameters_to_prune(model: nn.Module, name: str):
    if name.startswith('cifar100_vgg'):
        parameters_to_prune = [
            (val, 'weight') for key, val in model.features.named_modules() if isinstance(val, torch.nn.Conv2d)
        ]
        return parameters_to_prune
    
    elif name.startswith('cifar100_resnet'):
        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and name.endswith('conv1'):
                parameters_to_prune.append((module, 'weight'))
                
            if isinstance(module, nn.Conv2d) and 'downsample' in name:
                parameters_to_prune.append((module, 'weight'))
        
        return parameters_to_prune
    
    elif name.startswith('cifar100_mobilenet'):
        parameters_to_prune = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and not (name.endswith('2') and not name.startswith('features.17')):
                parameters_to_prune.append((module, 'weight'))
                
        return parameters_to_prune
    
def compress_model_vgg(vgg: nn.Module, **kwargs):
    i_s = []
    for i in range(len(vgg.features)):
        if isinstance(vgg.features[i], nn.Conv2d):
            i_s.append(i)
            
    profiler = Profile()
    indexing_time = Profile()
    with profiler:
        for i in i_s:
            conv = vgg.features[i]
            with indexing_time:
                kernel_indices = get_kernel_indices(conv, pruned=False)
                filter_indices = get_filter_indices(conv, pruned=False)
            
            weight = conv.weight.data
            bias = conv.bias.data
            stride = conv.stride
            padding = conv.padding
            kernel_size = conv.kernel_size
            
            vgg.features[i] = nn.Conv2d(kernel_indices.size()[0], filter_indices.size()[0], kernel_size, stride, padding)
            vgg.features[i].weight.data = weight[filter_indices, :, :, :][:, kernel_indices, :, :]
            vgg.features[i].bias.data = bias[filter_indices]
        
        # i = i_s[-1]
        # conv = model.features[i]
        # kernel_indices = get_kernel_indices(conv, pruned=False)
        # filter_indices = get_filter_indices(conv, pruned=False)

        # weight = conv.weight.data
        # bias = conv.bias.data
        # stride = conv.stride
        # padding = conv.padding
        # kernel_size = conv.kernel_size

        # model.features[i] = nn.Conv2d(kernel_indices.size()[0], conv.out_channels, kernel_size, stride, padding)
        # model.features[i].weight.data = weight[:, kernel_indices, :, :]
        # model.features[i].bias.data = bias
        
        fc = vgg.classifier[0]
        
        weight = fc.weight.data
        bias = fc.bias.data
        out_features = fc.out_features
        
        vgg.classifier[0] = nn.Linear(filter_indices.size()[0], fc.out_features, bias=True)
        vgg.classifier[0].weight.data = weight[:, filter_indices]
        vgg.classifier[0].bias.data = bias

def get_name_to_module(model: nn.Module):
    return dict((name, module) for name, module in model.named_modules()
                if isinstance(module, nn.Conv2d))
    
def get_num_parameters(model: nn.Module):
    num_params = 0
    for name, param in model.named_parameters():
        num_params += param.numel()
    return num_params

def get_num_pruned_parameters(params: list[tuple[nn.Module, Literal]]):
    num_pruned = 0
    for param, _ in params:
        num_pruned += (param.weight.view(-1) == 0).sum()
    return num_pruned

def get_num_masked_parameters(model):
    num_masked = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(name)
            num_masked += ((module.weight_mask == 0).view(-1).nonzero().numel())  
            if module.bias != None:
                num_masked += (module.bias_mask == 0).view(-1).nonzero().numel()     
    return num_masked   
    
def get_kernel_indices(module: nn.Module, pruned: Optional[bool] = True):
    check = (lambda x: x==0) if pruned else (lambda x: x!=0) 
    return (check(torch.norm(module.weight.data, 1, (0, 2, 3)))).nonzero().view(-1) 

def get_filter_indices(module: nn.Module, pruned: Optional[bool] = True):
    check = (lambda x: x==0) if pruned else (lambda x: x!=0)    
    return (check(torch.norm(module.weight.data, 1, (1, 2, 3)))).nonzero().view(-1)   

def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, prune_: Optional[bool] = True, remove: Optional[bool] = True):
    # Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # Prepare filters
    # print(f'before {fusedconv.weight.is_leaf}')
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.data = (torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    # print(f'after {fusedconv.weight.is_leaf}')

    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.data = (torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    
    # Prune fused layer
    if prune_:
        prune.custom_from_mask(fusedconv, 'weight', mask=conv.weight_mask.data)
        bias_mask = getattr(conv, 'bias_mask', torch.ones_like(fusedconv.bias))
        prune.custom_from_mask(fusedconv, 'bias', mask=bias_mask)
    if prune_ and remove:
        prune.remove(fusedconv, 'weight')
        prune.remove(fusedconv, 'bias')

    return fusedconv     
    
def embed_dependencies(model: nn.Module, 
                       dependencies: dict[Literal, Literal], 
                       name_to_module: dict[Literal, nn.Module]):
    for key, val in model.named_modules():
        if not isinstance(val, nn.Conv2d):
            continue
        if key not in dependencies:
            continue
        else:
            val.dependency = name_to_module[dependencies[key]]

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

def load_checkpoint(model: str, path: str):
    model = getattr(pytorch_cifar_models, model)(pretrained=True)
    state = torch.load(path, map_location=torch.device('cpu'))
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.identity(module, 'weight')
            if module.bias != None:
                prune.identity(module, 'bias')
    model.load_state_dict(state)
    
    return model

def initialize_checkpoint(model: nn.Module, starting_state: str):
    if starting_state == 'zero':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # TODO: Clean this up, for all if-elif blocks
                # Store the filter indices and old values
                module.trainable_indices = get_filter_indices(module, pruned=True)
                module.fixed_indices = get_filter_indices(module, pruned=False)
                module.filter_indices = get_filter_indices(module, False)
                module.old_weight = module.weight_orig.data
                
                # Remove the mask
                prune.remove(module, 'weight')
                
                # Repeat for bias
                if module.bias != None:
                    module.old_bias = module.bias_orig.data
                    prune.remove(module, 'bias')
    elif starting_state == 'orig':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Store the filter indices and old values
                module.trainable_indices = get_filter_indices(module, pruned=True)
                module.fixed_indices = get_filter_indices(module, pruned=False)
                module.filter_indices = get_filter_indices(module, False)
                module.old_weight = module.weight_orig.data
                
                # Remove the mask
                prune.remove(module, 'weight')
                
                # Return the pruned weights to their original values
                module.weight.data = module.old_weight.data
                
                # Repeat for bias
                if module.bias != None:
                    module.old_bias = module.bias_orig.data
                    prune.remove(module, 'bias')
                    module.bias.data = module.old_bias.data
    elif starting_state == 'rand':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Store the filter indices and old values
                module.trainable_indices = get_filter_indices(module, pruned=True)
                module.fixed_indices = get_filter_indices(module, pruned=False)
                module.filter_indices = get_filter_indices(module, False)
                module.old_weight = module.weight_orig.data
                
                # Remove the mask
                prune.remove(module, 'weight')

                # Repeat for bias
                if module.bias != None:
                    module.old_bias = module.bias_orig.data
                    prune.remove(module, 'bias')
                    module.bias.data = module.old_bias.data
                    
                # Reset parameters and restore the un-pruned weights
                module.reset_parameters()
                module.weight.data[module.filter_indices, :, :, :] = module.old_weight.data[module.filter_indices, :, :, :]
                if module.bias != None:
                    module.bias.data[module.filter_indices] = module.old_bias.data[module.filter_indices]
    else:
        raise NotImplementedError

def validate(data: data.DataLoader, model: nn.Module):
    # Set up the quality metrics
    # TODO: Remove hard-coded num_classes
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
            
def validate_onnx(data: data.DataLoader, model: any):
    # Set up the quality metrics
    # TODO: Remove hard-coded num_classes
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
            predictions = model.run(None, {'input': datum.numpy()})
        seen += datum.shape[0]
        predictions = torch.tensor(predictions[0])
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

def prune_kernel2(module: nn.Conv2d, dep: nn.Conv2d):
    # Get the indices of the pruned filters
    # dependency = getattr(module, 'dependency', None)
    # if dependency == None:
    #     return
    weight_mask = getattr(dep, 'weight_mask', torch.ones_like(dep.weight.data))
    pruned_indices = (torch.norm(weight_mask, 1, (1, 2, 3)) == 0).nonzero().view(-1)
    if pruned_indices.numel() == 0:
        return
    
    # Update the mask of the current layer
    if module.groups == 1:
        module.weight.data[:, pruned_indices, :, :] = 0
    else:
        module.weight.data[pruned_indices, :, :, :] = 0

def prune_kernel(module: nn.Conv2d):
    # Get the indices of the pruned filters
    dependency = getattr(module, 'dependency', None)
    if dependency == None:
        return
    pruned_indices = (torch.norm(module.dependency.weight.data, 1, (1, 2, 3)) == 0).nonzero().view(-1)
    if pruned_indices.numel() == 0:
        return
    
    # Update the mask of the current layer
    # module.weight.data[:, pruned_indices, :, :] = 0
    # module.weight = module.weight * module.weight_mask 
    # old_weight = module.weight_mask * module.weight_orig
    # old_weight[:, pruned_indices, :, :] = 0
    mask = getattr(module, 'weight_mask') #, torch.ones_like(module.weight.data))
    prune.remove(module, 'weight')
    mask[:, pruned_indices, :, :] = 0
    prune.custom_from_mask(module, 'weight', mask=mask)
    # new_weight = module.weight_mask * module.weight_orig
    # prune.remove(module, 'weight')
    return None

def prune_residual_filter(module: nn.Conv2d, dep: nn.Conv2d):
    weight_mask = getattr(dep, 'weight_mask', torch.ones_like(dep.weight.data))
    pruned_indices = (torch.norm(weight_mask, 1, (1, 2, 3)) == 0).nonzero().view(-1)
    if pruned_indices.numel() == 0:
        return
    
    # Update the mask of the current layer
    mask = getattr(module, 'weight_mask', torch.ones_like(module.weight.data))
    mask[pruned_indices, :, :, :] = 0
    prune.custom_from_mask(module, 'weight', mask=mask)    
    
def global_smallest_filter(parameters: prune.Iterable, amount: float, mode: Optional[Literal['mean', '1', '2', 'inf']]):
    if mode == 'mean':
        global_smallest_filter_mean(parameters, amount)
    else:
        global_smallest_filter_norm(parameters, amount, norm=mode)
    
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
            torch.norm(module.weight, 1, (1, 2, 3)).to(module.weight.device) / (module.kernel_size[0]*module.kernel_size[1]*module.weight.shape[1]),
            torch.range(0, module.weight.shape[0] - 1).to(module.weight.device),
            (torch.ones(module.weight.shape[0]) * i).to(module.weight.device),
            torch.ones(module.weight.shape[0]).to(module.weight.device) * (module.kernel_size[0]*module.kernel_size[1]*module.weight.shape[1])
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
            torch.norm(module.weight, norm, (1, 2, 3)).to(module.weight.device), # score
            torch.range(0, module.weight.shape[0] - 1).to(module.weight.device), # filter idx
            (torch.ones(module.weight.shape[0]) * i).to(module.weight.device), # module idx
            torch.ones(module.weight.shape[0]).to(module.weight.device) * (module.kernel_size[0]*module.kernel_size[1]*module.weight.shape[1]) # filter size
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
        
def remove_kernels_from_conv2d(conv, indices):
    # Make sure layer is a conv 2d layer
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f'Expected type nn.Conv2d, instead got {type(conv)}')
    
    # Get the parameters of the Conv2d layer
    weight = conv.weight
    bias = conv.bias
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    
    # Get the cropped weight and bias
    cropped_weight = weight.data[:, indices, :, :]
    
    # Re-initialize the module and set the weights and biases
    new_conv = nn.Conv2d(cropped_weight.shape[1], out_channels, kernel_size, stride=stride, padding=padding)
    new_conv.weight.data = cropped_weight
    new_conv.bias.data = bias.data if bias != None else torch.zeros_like(new_conv.bias.data)
    new_conv.weight.requires_grad = False
    new_conv.bias.requires_grad = False
    
    # Return the new module
    return new_conv

def remove_filters_from_conv2d(conv, indices):
    # Make sure layer is a conv 2d layer
    if not isinstance(conv, nn.Conv2d):
        raise TypeError(f'Expected type nn.Conv2d, instead got {type(conv)}')
    
    # Get the parameters of the Conv2d layer
    weight = conv.weight
    bias = conv.bias
    in_channels = conv.in_channels
    out_channels = conv.out_channels
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    
    # Get the cropped weight and bias
    cropped_weight = weight.data[indices, :, :, :]
    cropped_bias = bias.data[indices] if bias != None else None
    
    # Re-initialize the module and set the weights and biases
    new_conv = nn.Conv2d(in_channels, cropped_weight.shape[0], kernel_size, stride=stride, padding=padding)
    new_conv.weight.data = cropped_weight
    new_conv.bias.data = cropped_bias if cropped_bias != None else torch.zeros_like(new_conv.bias.data)
    new_conv.weight.requires_grad = False
    new_conv.bias.requires_grad = False
    
    # Return the new module
    return new_conv

def get_prunable_parameters(model: nn.Module):
    parameters_to_prune = {}
    for name, module in model.named_modules():
        # Create list of all module names
        
        # The top level layers
        if re.search('model\.[\d]*$', name) != None: 
            # The first Conv layer--------------------------------------------------------------
            # mAP50-95: 0.466
            if name.endswith('model.0'):
                for n, m in module.named_modules():
                    if isinstance(m, nn.Conv2d):
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The back-bone conv layers preceding C3 blocks-------------------------------------
            # mAP50-95: 0.00836
            if name.endswith('model.1') or name.endswith('model.3') or name.endswith('model.5') or name.endswith('model.7'):
                for n, m in module.named_modules():
                    if isinstance(m, nn.Conv2d):
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The first conv layer in the back-bone bottlenecks---------------------------------
            # mAP50-95: 0.0412
            if name.endswith('model.2') or name.endswith('model.4') or name.endswith('model.6') or name.endswith('model.8'):
                for n, m in module.named_modules():
                    if re.search('m\.[\d]*\.cv1.conv$', n) != None:
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The detour conv layer in the back-bone C3 blocks-----------------------------------
            # mAP50-95: 0.0566
            if name.endswith('model.2') or name.endswith('model.4') or name.endswith('model.6') or name.endswith('model.8'):
                for n, m in module.named_modules():
                    if n == 'cv2.conv':
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # Last conv layer in the 1st back-bone C3 blocks------------------------------------
            # mAP50-95: 0.39
            if name.endswith('model.2'):
                for n, m in module.named_modules():
                    if n == 'cv3.conv':
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # Last conv layer in the 2nd and 3rd back-bone C3 block-----------------------------
            # mAP50-95: 0.258
            if name.endswith('model.4') or name.endswith('model.6'): 
                for n, m in module.named_modules():
                    if n == 'cv3.conv':
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The last conv layer in the 4th back-bone C3 block---------------------------------
            # mAP50-95: 0.428
            if name.endswith('model.8'):
                for n, m in module.named_modules():
                    if n == 'cv3.conv':
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
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
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The first conv layer in the neck--------------------------------------------------
            # mAP50-95: 0.385
            if name.endswith('model.10'):
                for n, m in module.named_modules():
                    if n == 'conv':
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The 2nd conv layer in the neck----------------------------------------------------
            # mAP50-95: 0.187
            if name.endswith('model.14'):
                for n, m in module.named_modules():
                    if n == 'conv':
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The 3rd and 4th conv layers in the neck-------------------------------------------
            # mAP50-95: 0.365
            if name.endswith('model.18') or name.endswith('model.21'):
                for n, m in module.named_modules():
                    if n == 'conv':
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The external conv layers in the 1st C3 block in the neck--------------------------
            # mAP50-95: 0.125
            if name.endswith('model.13'):
                for n, m in module.named_modules():
                    if n == 'cv1.conv' or n == 'cv2.conv' or n == 'cv3.conv':
                        parameters_to_prune[f'{name}.{n}'] = (m, 'weight')
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
                                parameters_to_prune[f'{name}.{n}.{n_}'] = (m_, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The bottleneck conv layers in the 2nd C3 block in the neck-------------------------
            # mAP50-95: 0.397
            if name.endswith('model.17'):
                for n, m in module.named_modules():
                    if n == 'm':
                        for n_, m_ in m.named_modules():
                            if n_.endswith('conv'):
                                parameters_to_prune[f'{name}.{n}.{n_}'] = (m_, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The bottleneck conv layers in the 3rd C3 block in the neck-------------------------
            # mAP50-95: 0.379
            if name.endswith('model.20'):
                for n, m in module.named_modules():
                    if n == 'm':
                        for n_, m_ in m.named_modules():
                            if n_.endswith('conv'):
                                parameters_to_prune[f'{name}.{n}.{n_}'] = (m_, 'weight')
            # ----------------------------------------------------------------------------------
            
            # The bottleneck conv layers in the 4th C3 block in the neck-------------------------
            # mAP50-95: 0.336
            if name.endswith('model.23'):
                for n, m in module.named_modules():
                    if n == 'm':
                        for n_, m_ in m.named_modules():
                            if n_.endswith('conv'):
                                parameters_to_prune[f'{name}.{n}.{n_}'] = (m_, 'weight')
            # ----------------------------------------------------------------------------------
    # model.cuda()
    return parameters_to_prune

def get_next(file):
    score, density, config = 0, 0, {}
    is_conf = False
    for row in open(file):
        if row.startswith('Score:'):
            row_list = row.split(' ')
            score = float(row_list[1][:-1])
            density = float(row_list[3][:-1])
        elif row.startswith('Configuration:'):
            is_conf = True
        elif is_conf:
            is_conf = False
            config = json.loads(row.replace('\'', '"'))
            # print(score, density, config)
            yield score, density, config