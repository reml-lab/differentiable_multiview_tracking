import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

@MODELS.register_module()
class ImageNorm(nn.Module):
    def __init__(self, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
        super().__init__()
        mean = torch.tensor(mean).view(1, 3, 1, 1)
        std = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        return (x - self.mean) / self.std
