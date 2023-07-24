import torch
import pytorch_cifar_models
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torchmetrics.classification import Accuracy
from common import *
from collections import OrderedDict

model_list = [
    'cifar100_vgg11_bn',
    'cifar100_vgg13_bn',
    'cifar100_vgg16_bn',
    'cifar100_vgg19_bn',
    # 'cifar100_resnet20',
    # 'cifar100_resnet32',
    # 'cifar100_resnet44',
    # 'cifar100_resnet56',
    # 'cifar100_mobilenetv2_x0_5',
    # 'cifar100_mobilenetv2_x0_75',
    # 'cifar100_mobilenetv2_x1_0',
    # 'cifar100_mobilenetv2_x1_4',
]

if __name__ == '__main__':
    # Set up the data loaders
    val_set = CIFAR100('./data', train=False, download=False, transform=get_val_transforms(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]))
    val_loader_256 = data.DataLoader(val_set, batch_size=256, shuffle=False)
    val_loader_32 = data.DataLoader(val_set, batch_size=32, shuffle=False)
    val_loader_1 = data.DataLoader(val_set, batch_size=1, shuffle=False)    
    
    # Measure onnx import time
    onnx_import = Profile()
    with onnx_import:
        import onnxruntime as ort
        
    # Set up model profilers
    profiler_dict = {}
    for model_name in model_list:
        profiler_dict[model_name] = Profile()
        
    # Measure the model load times
    load_time_dict = {}
    model_dict = {}
    # for model_name in model_list:
    #     with profiler_dict[model_name]:
    #         model_dict[model_name] = ort.InferenceSession(f'{model_name}.onnx', providers=['CPUExecutionProvider'])
    #     load_time_dict[model_name] = profiler_dict[model_name].t
        
    # Measure the Pytorch model load time
    pt_load_time_dict = {}
    for model_name in model_list:
        with profiler_dict[model_name]:
            model = getattr(pytorch_cifar_models, model_name)(pretrained=True)
            model.eval()
            fuse_batchnorms(model, model_name, prune_=False)
        pt_load_time_dict[model_name] = profiler_dict[model_name].t
        
    # # Measure inference time for batch=256/32/1
    # inference_time_dict = {}
    # for model_name in model_list:
    #     _, inference_time_256, _, _ = validate_onnx(val_loader_256, model_dict[model_name])
    #     _, inference_time_32, _, _ = validate_onnx(val_loader_32, model_dict[model_name])
    #     _, inference_time_1, _, _ = validate_onnx(val_loader_1, model_dict[model_name])
    #     inference_time_dict[model_name] = (inference_time_256, inference_time_32, inference_time_1)
        
    # Report results
    for model_name in model_list:
        print(f'{model_name}')
        # print(f'load time: {load_time_dict[model_name] + onnx_import.t}, inference_time: {inference_time_dict[model_name]}')
        print(f'load time: {pt_load_time_dict[model_name]}')
    