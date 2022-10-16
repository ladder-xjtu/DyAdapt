# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 18:09:23 2019

Reference:
https://github.com/NieXC/pytorch-pil/blob/master/nets/adaptive_conv.py

@author: chlian
"""

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple


class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
        
        
class AdaptiveConv3d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(AdaptiveConv3d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _triple(0), groups, bias)

    def forward(self, x, dynamic_weight):
        batch_num = x.size(0)
        # Tensor: (B, C, D, H, W) --> (1, B*C, D, H, W)
        x = x.view(1, -1, x.size(2), x.size(3), x.size(4))
        # Kernel: (B, C, K_d, K_h, K_w) --> (B*C, 1, K_d, K_h, K_w)
        dynamic_weight = dynamic_weight.view(-1, 1, 
                                             dynamic_weight.size(2), 
                                             dynamic_weight.size(3), 
                                             dynamic_weight.size(4))
        y = F.conv3d(x, dynamic_weight, self.bias, self.stride, 
                     self.padding, self.dilation, self.groups)
        # Tensor: (1, B*C, D, H, W) --> (B, C, D, H, W)
        y = y.view(batch_num, -1, y.size(2), y.size(3), y.size(4))
        return y



class AdaptiveConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(AdaptiveConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)

    def forward(self, input, dynamic_weight):
        # Get batch num
        batch_num = input.size(0)

        # Reshape input tensor from size (N, C, H, W) to (1, N*C, H, W)
        input = input.view(1, -1, input.size(2), input.size(3))

        # Reshape dynamic_weight tensor from size (N, C, H, W) to (1, N*C, H, W)
        dynamic_weight = dynamic_weight.view(-1, 1, dynamic_weight.size(2), dynamic_weight.size(3))

        # Do convolution
        conv_rlt = F.conv2d(input, dynamic_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Reshape conv_rlt tensor from (1, N*C, H, W) to (N, C, H, W)
        conv_rlt = conv_rlt.view(batch_num, -1, conv_rlt.size(2), conv_rlt.size(3))

        return conv_rlt


        
if __name__ == '__main__':
    
    x = torch.rand(2, 32, 6, 6)
    k = torch.rand(2, 32, 1, 1)
    
    aconv = AdaptiveConv2d(2*32, 2*32, 3, padding=0, groups=2*32, bias=False)
    y = aconv(x, k)
    print(y.size())
        
        
