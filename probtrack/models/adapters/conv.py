# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from mmengine.registry import MODELS

class Interpolate(nn.Module):
    def __init__(self, size=(1, 1)):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size)

@MODELS.register_module()
class ConvAdapter(nn.Module):
    def __init__(self,
                 dim=256,
                 interpolate_size=(1,1),
                 interpolate_fn='avgpool',
                 flatten=False,
                 transpose=False
        ):
        super().__init__()
        self.dim = dim
        if interpolate_fn == 'interpolate':
            interp_layer = Interpolate(interpolate_size) 
        elif interpolate_fn == 'avgpool':
            interp_layer = nn.AdaptiveAvgPool2d(interpolate_size) 
        self.layers = [
            nn.Conv2d(dim, dim, kernel_size=1, stride=(1,1), padding=(0,0)),
            nn.GELU(),
        ]
        if transpose:
            self.layers.append(Rearrange('b c h w -> b c w h')) 
        self.layers.append(interp_layer)
        self.layers.append(nn.Conv2d(dim, dim, kernel_size=1, stride=(1,1), padding=(0,0)))
        self.layers.append(nn.GELU())
        self.layers = nn.Sequential(*self.layers)
        self.flatten = flatten
    
    #x has shape B x D x H x W
    #output is B x 1 x D
    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(-1)
        x = x.permute(0, 2, 1)
        if self.flatten:
            x = x.flatten(1).unsqueeze(-2)
        return x
