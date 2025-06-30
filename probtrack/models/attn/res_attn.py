import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

@MODELS.register_module()
class ResSelfAttn(nn.Module):
    def __init__(self,
                 attn_cfg=None,
                 out_norm_cfg=dict(type='LN'),
                 res_dropout_cfg=dict(type='DropPath', drop_prob=0.1),
        ):
        super().__init__()
        self.attn = MODELS.buid(attn_cfg)

        self.out_norm = nn.LayerNorm(attn_cfg['qk_dim'])
        self.res_dropout = None

        # if out_norm_cfg is not None:
            # self.out_norm = build_norm_layer(out_norm_cfg, attn_cfg['qk_dim'])[1]

        # self.res_dropout = build_from_cfg(res_dropout_cfg, DROPOUT_LAYERS)
    
    def forward(self, x, x_pos=None, offset=0):
        identity = x
        encoded_x = x if x_pos is None else x + x_pos
        x = self.attn(encoded_x, encoded_x, x, offset=offset)
        x = identity + self.res_dropout(x)
        if self.out_norm is not None:
            x = self.out_norm(x)
        return x

@MODELS.register_module()
class ResCrossAttn(nn.Module):
    def __init__(self,
                 attn_cfg=None,
                 out_norm_cfg=dict(type='LN'),
                 res_dropout_cfg=dict(type='DropPath', drop_prob=0.1),
        ):
        super().__init__()
        self.attn = MODELS.buid(attn_cfg)
        self.out_norm = nn.LayerNorm(attn_cfg['qk_dim'])
        self.res_dropout = None
        # self.out_norm = build_norm_layer(out_norm_cfg, attn_cfg['qk_dim'])[1]
        # self.res_dropout = build_from_cfg(res_dropout_cfg, DROPOUT_LAYERS)
    
    def forward(self, x, feats, x_pos=None, feats_pos=None, offset=0, return_weights=False):
        identity = x
        encoded_x = x if x_pos is None else x + x_pos
        encoded_feats = feats if feats_pos is None else feats + feats_pos
        out = self.attn(encoded_x, encoded_feats, feats, offset=offset, return_weights=return_weights)
        if return_weights:
            x, A = out
        else:
            x = out
        x = identity + self.res_dropout(x)
        x = self.out_norm(x)
        if return_weights:
            return (x, A)
        return x
