import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet50_Weights
from torchvision.models.resnet import resnet50
from mmengine.registry import MODELS
from .image_norm import ImageNorm
#from mmpretrain.models.backbones.convnext import ConvNeXtBlock

class ResNet50Stem(nn.Sequential):
    def __init__(self, frozen=True):
        r50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        super().__init__(
            r50.conv1,
            r50.bn1,
            r50.relu,
            r50.maxpool
        )
        self.frozen = frozen
        if self.frozen:
            self.forward = self._forward_frozen
    
    @torch.no_grad()
    def _forward_frozen(self, x):
        for layer in self:
            x = layer.eval()(x)
        return x

@MODELS.register_module()
class UAIConvNext(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        # weights = ResNet50_Weights.IMAGENET1K_V1
        # self.resnet = torchvision.models.resnet50(weights=weights)
        self.img_norm = ImageNorm(
            mean=[123.675, 116.28, 103.53], 
            std=[58.395, 57.12, 57.375]
        )
        self.stem = ResNet50Stem(frozen=True)
        self.layers = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
            ConvNeXtBlock(out_channels, layer_scale_init_value=0.0),
            nn.Conv2d(out_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
        )

    #x is B x 3 x H x W
    #output is B x 2048 x H/32 x W/32
    def forward(self, x):
        x = self.img_norm(x)
        x = self.stem(x)
        x = self.layers(x)
        return x
