import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Sequence, Union, TypedDict
from typing_extensions import Literal, TypeAlias
import re
from common import get_filter_indices, dot_num_to_brack, dot_num_to_brack_end

class FrozenConv2d(nn.Module):
    def __init__(self, trainable_indices: torch.Tensor, fixed_indices: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
                 stride: Optional[int] = 1, padding: Optional[int] = 0, dilation: Optional[int] = 1, groups: Optional[int] = 1):
        super().__init__()
        
        # Get the pruned and unpruned indices 
        self.trainable_indices = trainable_indices
        self.fixed_indices = fixed_indices
        
        # Create the list of parameter and non-parameter weight tensors
        self.weight_list = []
        for i in range(len(self.trainable_indices) + len(self.fixed_indices)):
            if i in self.trainable_indices:
                setattr(self, f'trainable_weight_{i}', nn.Parameter(weight[i,:,:,:].unsqueeze(0)))
                self.weight_list.append(f'trainable_weight_{i}')
            else:
                self.register_buffer(f'frozen_weight_{i}', weight[i,:,:,:].unsqueeze(0))
                self.weight_list.append(f'frozen_weight_{i}')
                
        # Create the list of parameter and non-paramter bias tensors
        self.bias_list = []
        if bias != None:
            for i in range(len(self.trainable_indices) + len(self.fixed_indices)):
                if i in self.trainable_indices:
                    setattr(self, f'trainable_bias_{i}', nn.Parameter(bias[i].unsqueeze(0)))
                    self.bias_list.append(f'trainable_bias_{i}')
                else:
                    self.register_buffer(f'frozen_bias_{i}', bias[i].unsqueeze(0))
                    self.bias_list.append(f'frozen_bias_{i}')
            
        # Copy the metadata
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
            
    def forward(self, x):            
        # Convolve
        weight = torch.cat([getattr(self, n) for n in self.weight_list])
        if len(self.bias_list) == 0:
            bias = None
        else:
            bias = torch.cat([getattr(self, n) for n in self.bias_list])
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    
@torch.no_grad()
def convert_model(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Create frozen conv layer
            if module.bias != None:
                b = module.bias.data.detach().clone()
            else:
                b = None
                
            frozen = FrozenConv2d(module.trainable_indices,
                                  module.fixed_indices,
                                  module.weight.data.detach().clone(),
                                  b,
                                  module.stride,
                                  module.padding,
                                  module.dilation,
                                  module.groups)
            
            # Sanitize name
            name = re.sub('\.[\d]+\.', dot_num_to_brack, name)
            name = re.sub('\.[\d]+', dot_num_to_brack_end, name)
            
            # Replace module
            exec(f'model.{name} = frozen')
            
    # for i in range(len(model.features)):
    #     if isinstance(model.features[i], nn.Conv2d):
    #         model.features[i] = FrozenConv2d(model.features[i].trainable_indices,
    #                                          model.features[i].fixed_indices,
    #                                          model.features[i].weight.data.detach().clone(),
    #                                          model.features[i].bias.data.detach().clone(),
    #                                          model.features[i].stride,
    #                                          model.features[i].padding,
    #                                          model.features[i].dilation,
    #                                          model.features[i].groups)