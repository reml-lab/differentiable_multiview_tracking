import torch
import torch.nn as nn
from mmengine.registry import MODELS

@MODELS.register_module()
class DecoderAdapter(nn.Module):
    def __init__(self,
                 dim=2048,
                 num_embeds=1,
                 attn_cfg=dict(type='QKVAttention'),
                 pos_embed_cfg=dict(type='SineEncoding2d', dim=2048, out_proj=True),
                 ):
        super().__init__()
        self.attn = MODELS.build(attn_cfg)
        self.dim = dim
        self.pos_embed = MODELS.build(pos_embed_cfg)
        self.embeds = nn.Embedding(num_embeds, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.init_weights()

    # follow the official DETR to init parameters
    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=1)
        # self._is_init = True
    
    #x has shape B x D x H x W
    #output is  B x num_embeds x D
    def forward(self, x):
        B, D, H, W = x.shape
        pos_embeds = self.pos_embed(x)
        embeds = self.embeds.weight.unsqueeze(0)
        embeds = embeds.expand(B, -1, -1)
        res = embeds
        
        x = x.reshape(B, D, H*W)
        x = x.permute(0, 2, 1) #B x H*W x D
        pos_embeds = pos_embeds.reshape(B, D, H*W)
        pos_embeds = pos_embeds.permute(0, 2, 1) #B x H*W x D
        
        #embeds = self.attn(embeds, x + pos_embeds, x) 
        embeds = self.attn(embeds, x + pos_embeds, x) 
        embeds = embeds + res
        embeds = self.out_norm(embeds)
        return embeds #B x num_embeds x D
