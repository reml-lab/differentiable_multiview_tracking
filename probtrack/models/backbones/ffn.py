import torch
import torch.nn as nn
from mmengine.registry import MODELS


@MODELS.register_module()
class IsoCNN(nn.Module):
    def __init__(self,
                 dim=256,
                 expansion_factor=4,
                 dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim * expansion_factor, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * expansion_factor, dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.layers(x) + x
        return x


#isotropic ffn with residual connection
#can drop this anywhere to add a blob of parameters
@MODELS.register_module()
class IsoFFN(nn.Module):
    def __init__(self,
                 dim=2048,
                 expansion_factor=4,
                 dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Linear(dim * expansion_factor, dim),
            # nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.out_norm = nn.LayerNorm(dim)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=1)

    #x is B x L x D
    #output is  B x L x D
    def forward(self, x):
        x = self.layers(x) + x
        x = self.out_norm(x)
        return x


@MODELS.register_module()
class BottleneckFFN(nn.Module):
    def __init__(self,
                 dim=256,
                 out_dim=8,
                 dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.layers(x)
        x = self.out_norm(x)
        return x
