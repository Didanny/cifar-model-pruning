import torch
import pytorch_cifar_models

model = pytorch_cifar_models.cifar10_vgg11_bn(pretrained=True)
x = 1