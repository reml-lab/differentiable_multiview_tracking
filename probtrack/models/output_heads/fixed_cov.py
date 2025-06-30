import torch
import torch.nn as nn
import torch.distributions as D
from mmengine.registry import MODELS

@MODELS.register_module()
class FixedCovPredictor(nn.Module):
    def __init__(self, cov_add_eps=1e-4, dim=None):
        super().__init__()
        self.cov_add_eps = cov_add_eps

    def forward(self, x):
        B, N, C = x.shape
        cov = torch.eye(3).reshape(1, 1, 3, 3).cuda()
        cov = cov.expand(B, N, -1, -1) * self.cov_add_eps
        return cov
