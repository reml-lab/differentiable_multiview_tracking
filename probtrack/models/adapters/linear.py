import torch
import torch.nn as nn
from mmengine.registry import MODELS
import torch.nn.functional as F
from probtrack.geometry.distributions import to_torch_dist

@MODELS.register_module()
class LinearAdapter(nn.Module):
    def __init__(self,
                 in_len=100,
                 out_len=1,
                 ffn_cfg=dict(type='IsoFFN', dim=256, expansion_factor=4, dropout_rate=0.1),
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lin_len = nn.Linear(in_len, out_len)
        self.ffn = MODELS.build(ffn_cfg)

        
    #x has shape B x in_len x D
    def forward(self, x, pos_embeds=None):
        if len(x.shape) == 4: #cov feat map
            x = x.flatten(2)
            x = x.permute(0, 2, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        x = self.lin_len(x)
        x = x.permute(0, 2, 1)
        return x
