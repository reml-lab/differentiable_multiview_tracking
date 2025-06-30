import torch
import torch.nn as nn
import torch.distributions as D
from mmengine.registry import MODELS

@MODELS.register_module()
class DiagCovPredictor(nn.Module):
    def __init__(self,
                 dim=256,
                 cov_add_eps=1e-8): 
        super().__init__()
        self.cov_add_eps = cov_add_eps

        self.diag_head = nn.Sequential(
            nn.Linear(dim, 3),
            nn.Softplus()
        )
    
    #x has the shape B x num_object x D
    def forward(self, x):
        B, N, C = x.shape
        diag = self.diag_head(x)
        cov = torch.diag_embed(diag)
        eye = torch.eye(3).to(cov)
        cov = cov + self.cov_add_eps * eye
        return cov
