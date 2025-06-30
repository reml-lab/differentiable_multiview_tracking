import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from mmengine.registry import MODELS


@MODELS.register_module()
class BackgroundOutputHead(nn.Module):
    def __init__(self, dim=256, loss_weight=1.0):
        super().__init__()
        # self.ffn = MODELS.build(ffn_cfg)
        self.logits_head = nn.Linear(dim, 1)
        self.loss_weight = loss_weight
    
    #x has the shape B x num_object x D
    def forward(self, x):
        B, N, C = x.shape
        # x = self.ffn(x)
        x = x.mean(dim=1)
        logits = self.logits_head(x)
        logits = logits.squeeze(-1)
        return logits
        
    def forward_train(self, x, gt):
        viewable = gt['viewable'] 
        target_probs = viewable.sum(dim=-1)
        target_probs = target_probs.clamp(0, 1)
        logits = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(logits, target_probs)
        return {'bg_loss': loss * self.loss_weight}

    def forward_test(self, x, gt=None):
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        probs = [p for p in probs.cpu().detach()] #TODO: hack
        return probs
