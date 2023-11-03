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
                self.weight_list.append(getattr(self, f'trainable_weight_{i}'))
            else:
                self.weight_list.append(weight[i,:,:,:].unsqueeze(0))
                
        # Create the list of parameter and non-paramter bias tensors
        self.bias_list = []
        if bias != None:
            for i in range(len(self.trainable_indices) + len(self.fixed_indices)):
                if i in self.trainable_indices:
                    setattr(self, f'trainable_bias_{i}', nn.Parameter(bias[i].unsqueeze(0)))
                    self.bias_list.append(getattr(self, f'trainable_bias_{i}'))
                else:
                    self.bias_list.append(bias[i].unsqueeze(0))
        
        # # Create the trainable parameters        
        # self.trainable_weight = nn.Parameter(weight[self.trainable_indices,:,:,:])
        # if bias != None:
        #     self.trainable_bias = nn.Parameter(bias[self.trainable_indices])
        # else:
        #     self.bias = None
            
        # # Create the fixed parameters
        # self.fixed_weight = weight[self.fixed_indices,:,:,:]
        # if bias != None:
        #     self.fixed_bias = bias[self.fixed_indices]
            
        # # Create the full intermediate parameter
        # self.weight = torch.zeros_like(weight)
        # if bias != None:
        #     self.bias = torch.zeros_like(bias)
            
        # # Set requires_grad
        # self.weight.requires_grad_()
        # self.trainable_weight.requires_grad_()
        # self.fixed_weight.requires_grad_()
        # if self.bias != None:
        #     self.bias.requires_grad_()
        #     self.trainable_bias.requires_grad_()
        #     self.fixed_bias.requires_grad_()
            
        # Copy the metadata
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
            
    def forward(self, x):
        # # Reinitialize weight
        # with torch.no_grad():
        #     self.weight.fill_(0.0)
        #     if self.bias != None:
        #         self.bias.fill_(0.0)
                
        # # Copy weight values into full weight tensor 
        # self.weight[self.trainable_indices,:,:,:] += self.trainable_weight
        # self.weight[self.fixed_indices,:,:,:] += self.fixed_weight
        # if self.bias != None:
        #     self.bias[self.trainable_indices] += self.trainable_bias
        #     self.bias[self.fixed_indices] += self.fixed_bias
            
        # Convolve
        weight = torch.cat(self.weight_list)
        if len(self.bias_list) == 0:
            bias = None
        else:
            bias = torch.cat(self.bias_list)
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