import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from mmengine.registry import MODELS
from einops.layers.torch import Rearrange


@MODELS.register_module()
class GaussianPredictor(nn.Module):
    def __init__(self,
                 dim=256,
                 cov_add_eps=1e-8,
                 freeze_ffn=False,
                 freeze_mean=False,
                 freeze_cov=False,
                 use_ffn=True,
                 scale_mean=True,
                 is_2d=True
                 ): 
        super().__init__()
        self.is_2d = is_2d
        self.scale_mean = scale_mean
        self.ffn = nn.GELU()
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )

        mean_head = [nn.Linear(dim, 3)]
        if scale_mean:
            mean_head.append(nn.Sigmoid())
        self.mean_head = nn.Sequential(*mean_head)

        #init mean head such that its around 0.5
        
        self.eigenvalue_head = nn.Sequential(
            nn.Linear(dim, 3),
            nn.Softplus()
        )
        self.eigenvector_head = nn.Linear(dim, 3*3)
        self.register_buffer('cov_add_eps', torch.eye(3) * cov_add_eps)
        self.init_weights()

        self.register_buffer('min', torch.tensor([-2776.4158, -2485.51226, 0.49277])/1000)
        self.register_buffer('max', torch.tensor([5021.02426, 3147.80981, 1621.69398])/1000)
        
        if self.ffn is not None:
            for param in self.ffn.parameters():
                param.requires_grad = not freeze_ffn

        for param in self.mean_head.parameters():
            param.requires_grad = not freeze_mean

        for param in self.eigenvalue_head.parameters():
            param.requires_grad = not freeze_cov

        for param in self.eigenvector_head.parameters():
            param.requires_grad = not freeze_cov

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=1)
        #init mean head such that its around 0.5, sigmoid(0) = 0.5
        nn.init.uniform_(self.mean_head[0].weight, -0.0001, 0.0001)
        nn.init.constant_(self.mean_head[0].bias, 0.0001)

        nn.init.uniform_(self.eigenvalue_head[0].weight, -0.0001, 0.0001)
        nn.init.constant_(self.eigenvalue_head[0].bias, 2.0)


    #x has the shape queries x D
    def forward(self, x):
        # x = self.shared_ffn(x) #N x D
        # x = self.dim_reducer(x.T) #D x 1
        # x = x.T #1 x D
        N, C = x.shape
        
        if self.ffn is not None:
            x = self.ffn(x)
        eigenvalues = self.eigenvalue_head(x)
        eigenvalues = torch.diag_embed(eigenvalues)
        eigenvalues = eigenvalues.reshape(N, 3, 3)
        # cov = eigenvalues
        
        eigenvectors = self.eigenvector_head(x)
        eigenvectors = eigenvectors.reshape(N, 3, 3)
        eigenvectors, _ = torch.linalg.qr(eigenvectors) #orthogonalize

        cov = torch.bmm(eigenvectors, torch.bmm(eigenvalues, eigenvectors.transpose(-2,-1)))
        cov = cov.reshape(N, 3, 3)
        cov = (cov + cov.transpose(-2, -1)) / 2.0 #symmetrize
        cov = cov + self.cov_add_eps

        mean = self.mean_head(x)

        if self.scale_mean:
            mean = mean * (self.max - self.min) + self.min
        if self.is_2d:
            mean = mean[..., 0:2]
            cov = cov[..., 0:2, 0:2]
        # eye = torch.eye(2).to(cov.device).unsqueeze(0)
        # cov = eye + 0 * cov
        # dist = D.MultivariateNormal(mean, cov)
        return mean, cov
