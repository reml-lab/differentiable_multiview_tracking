import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights
from mmengine.registry import MODELS
from .image_norm import ImageNorm

@MODELS.register_module()
class ResNet50(nn.Module):
    def __init__(self, 
            frozen_stages=[1, 2, 3, 4],
            output_idx=-1):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.resnet = torchvision.models.resnet50(weights=weights)
        self.output_idx = output_idx
        self.img_norm = ImageNorm(
            mean=[123.675, 116.28, 103.53], 
            std=[58.395, 57.12, 57.375]
        )
        # self.out_norm = nn.GroupNorm(32, 2048)

        self.frozen_stages = frozen_stages

        for i in range(1, 5):
            layer = getattr(self.resnet, f'layer{i}')
            if i in self.frozen_stages:
                for param in layer.parameters():
                    param.requires_grad = False
                layer.eval()

        # for param in self.resnet.parameters():
            # param.requires_grad = not freeze

        # if freeze:
            # self.resnet.eval()
     
    def forward(self, x):
        x = self.img_norm(x)
        with torch.no_grad(): #stem
            x = self.resnet.conv1(x)
            x = self.resnet.bn1.eval()(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
        for i in range(1, 5):
            layer = getattr(self.resnet, f'layer{i}')
            x = layer(x)
            if i == self.output_idx:
                break
        # x = self.resnet.layer1(x)
        # x = self.resnet.layer2(x)
        # x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)
        # x = self.out_norm(x)
        return x

    #x is B x 3 x H x W
    #output is B x 2048 x H/32 x W/32
    def forward(self, x, return_all=False):
        out = []
        x = self.img_norm(x)
        with torch.no_grad(): #stem
            x = self.resnet.conv1(x)
            x = self.resnet.bn1.eval()(x)
            x = self.resnet.relu(x)
            x = self.resnet.maxpool(x)
        out.append(x)
        for i in range(1, 5):
            layer = getattr(self.resnet, f'layer{i}')
            x = layer(x)
            out.append(x)
            if i == self.output_idx:
                break
        # x = self.resnet.layer1(x)
        # x = self.resnet.layer2(x)
        # x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)
        # x = self.out_norm(x)
        if return_all:
            return out
        return out[-1]
