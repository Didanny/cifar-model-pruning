import torch
import pytorch_cifar_models
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torchmetrics.classification import Accuracy
from common import *
from collections import OrderedDict
import argparse
import onnxruntime as ort

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--thresh', type=float)
    opt = parser.parse_args()
    print(vars(opt))
    return opt

def main(opt):
    # CIFAR-100
    val_set = CIFAR100('./data', train=False, download=False, transform=get_val_transforms(mean=[0.5070, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]))
    val_loader_256 = data.DataLoader(val_set, batch_size=256, shuffle=False)
    val_loader_32 = data.DataLoader(val_set, batch_size=32, shuffle=False)
    val_loader_1 = data.DataLoader(val_set, batch_size=1, shuffle=False)    
    
    # VGG-16 Model
    model = getattr(pytorch_cifar_models, opt.model)(pretrained=True)
    model2 = getattr(pytorch_cifar_models, opt.model)(pretrained=True)
    # model.eval()
    total_params = get_num_parameters(model)
    
    # Setup
    dependencies = get_dependency_graph(model, opt.model)
    residual_dependencies = get_residual_dependency(model, opt.model)
    parameters_to_prune = get_parameters_to_prune(model, opt.model)
    name_to_module = get_name_to_module(model)
    
    # Get the pre-pruned inference times
    # ONNX
    # onnx_model = ort.InferenceSession(f'{opt.model}.onnx', providers=['CPUExecutionProvider'])
    # _, pre_inference_time_256, _, _ = validate_onnx(val_loader_256, onnx_model)
    # _, pre_inference_time_32, _, _ = validate_onnx(val_loader_32, onnx_model)
    # _, pre_inference_time_1, _, _ = validate_onnx(val_loader_1, onnx_model)
    # PyTorch
    _, pre_inference_time_256, _, _ = validate(val_loader_256, model)
    # _, pre_inference_time_32, _, _ = validate_onnx(val_loader_32, onnx_model)
    # _, pre_inference_time_1, _, _ = validate_onnx(val_loader_1, onnx_model)
    
    fuse_batchnorms(model2, opt.model, prune_ = False)
    _, pre2_inference_time_256, _, _ = validate(val_loader_256, model)
    
    # Prune
    delta = 0.005
    i = 0.005
    while i < opt.thresh:
        global_smallest_filter(parameters_to_prune, i, opt.mode)
        for key in dependencies:
            mod = name_to_module[key]
            mod_dep = name_to_module[dependencies[key]]
            prune_kernel2(mod, mod_dep)
        for key in residual_dependencies:
            mod = name_to_module[key] 
            mod_dep = name_to_module[residual_dependencies[key]]
            prune_residual_filter(mod, mod_dep)
        i += delta
        
    # Get a fused model for validation
    fuse_batchnorms(model, opt.model)
    # model_fuse = getattr(pytorch_cifar_models, opt.model)(pretrained=True)
    # model_fuse.eval()
    # transfer_masks(model_fuse, model, opt.model)
    # fuse_batchnorms(model_fuse, opt.model, prune_=True)
    # model_fuse.eval()
    model.eval()
    # compress_model(model_fuse, opt.model)
    compress_model(model, opt.model)
    model.eval()
    # pruned_params = get_num_parameters(model_fuse)
    pruned_params = get_num_parameters(model)
    print(f'Params before: {total_params}, After: {pruned_params}')
    # return
    
    # Get the fused onnx model
    dummy_input = torch.rand((1, 3, 32, 32))
    input_names = ['input']
    output_names = ['output']
    onnx_file = f'{opt.model}_fused.onnx'
    torch.onnx.export(model, dummy_input, onnx_file, input_names=input_names, output_names=output_names,
                      dynamic_axes={
                          'input': {0: 'batch_size'}
                      })
    
    # Get the inference times
    # ONNX
    # onnx_model_fuse = ort.InferenceSession(f'{opt.model}_fused.onnx', providers=['CPUExecutionProvider'])
    # _, inference_time_256, _, _ = validate_onnx(val_loader_256, onnx_model_fuse)
    # _, inference_time_32, _, _ = validate_onnx(val_loader_32, onnx_model_fuse)
    # _, inference_time_1, _, _ = validate_onnx(val_loader_1, onnx_model_fuse)
    # PyTorch
    _, inference_time_256, _, _ = validate(val_loader_256, model)
    # _, inference_time_32, _, _ = validate_onnx(val_loader_32, onnx_model_fuse)
    # _, inference_time_1, _, _ = validate_onnx(val_loader_1, onnx_model_fuse)
    
    # Report results
    # print(f'pre_inference time: {pre_inference_time_256, pre_inference_time_32, pre_inference_time_1}')   
    # print(f'inference time: {inference_time_256, inference_time_32, inference_time_1}')   
    # print(f'ratio: {inference_time_256/pre_inference_time_256, inference_time_32/pre_inference_time_32, inference_time_1/pre_inference_time_1}')   
    
    print(f'pre_inference time: {pre_inference_time_256}')#, pre_inference_time_32, pre_inference_time_1}')   
    print(f'inference time: {inference_time_256}')#, inference_time_32, inference_time_1}')   
    print(f'inference time: {pre2_inference_time_256}')
    print(f'ratio: {inference_time_256/pre_inference_time_256}')#, inference_time_32/pre_inference_time_32, inference_time_1/pre_inference_time_1}')   
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)