import torch
import torch.nn as nn
import torch.distributions as D
from mmengine.registry import MODELS
from einops.layers.torch import Rearrange

@MODELS.register_module()
class FullCovConvPredictor(nn.Module):
    def __init__(self,
                 dim=256,
                 cov_add_eps=1e-8,
                 ): 
        super().__init__()
        self.eigenvalue_head = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=3, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Softplus()
        )

        self.eigenvector_head = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=3*3, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
        )
        self.register_buffer('cov_add_eps', torch.eye(3) * cov_add_eps)

    def forward(self, x):
        B = x.shape[0]
        N = 1 #TODO: ugly
        eigenvalues = self.eigenvalue_head(x)
        eigenvalues = torch.diag_embed(eigenvalues)
        eigenvalues = eigenvalues.reshape(B*N, 3, 3)
        
        eigenvectors = self.eigenvector_head(x)
        eigenvectors = eigenvectors.reshape(B*N, 3, 3)
        eigenvectors, _ = torch.linalg.qr(eigenvectors)

        cov = torch.bmm(eigenvectors, torch.bmm(eigenvalues, eigenvectors.transpose(-2,-1)))
        cov = cov.reshape(B, N, 3, 3)
        cov = (cov + cov.transpose(-2, -1)) / 2.0
        cov = cov + self.cov_add_eps
        return cov

@MODELS.register_module()
class FullCovPredictor(nn.Module):
    def __init__(self,
                 dim=256,
                 cov_add_eps=1e-8,
                 ffn_cfg=dict(type='IsoFFN', dim=256, expansion_factor=4, dropout_rate=0.1),
                 ): 
        super().__init__()
        self.ffn = MODELS.build(ffn_cfg)
        self.eigenvalue_head = nn.Sequential(
            nn.Linear(dim, 3),
            nn.Softplus()
        )
        self.eigenvector_head = nn.Linear(dim, 3*3)
        self.register_buffer('cov_add_eps', torch.eye(3) * cov_add_eps)
    
    #x has the shape B x num_object x D
    def forward(self, x):
        B, N, C = x.shape
        x = self.ffn(x)
        eigenvalues = self.eigenvalue_head(x)
        eigenvalues = torch.diag_embed(eigenvalues)
        eigenvalues = eigenvalues.reshape(B*N, 3, 3)
        
        eigenvectors = self.eigenvector_head(x)
        eigenvectors = eigenvectors.reshape(B*N, 3, 3)
        eigenvectors, _ = torch.linalg.qr(eigenvectors) #orthogonalize

        cov = torch.bmm(eigenvectors, torch.bmm(eigenvalues, eigenvectors.transpose(-2,-1)))
        cov = cov.reshape(B, N, 3, 3)
        cov = (cov + cov.transpose(-2, -1)) / 2.0 #symmetrize
        cov = cov + self.cov_add_eps
        return cov
