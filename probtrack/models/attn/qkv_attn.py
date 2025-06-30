import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

#basic scaled dot product attn
#adapted from torch impl
@MODELS.register_module()
class QKVAttention(nn.Module):
    def __init__(self, 
                 qk_dim=2048,
                 num_heads=8, 
                 in_proj=True, 
                 out_proj=True,
                 attn_drop=0.1, 
                 seq_drop=0.0,
                 v_dim=None
        ):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim if v_dim else qk_dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(attn_drop, inplace=False)
        self.seq_dropout = nn.Dropout2d(seq_drop, inplace=False)
        self.q_scaling = float(self.qk_dim // self.num_heads) ** -0.5
        
        self.in_proj = None
        if in_proj:
            self.in_proj = nn.ModuleDict({
                'q': nn.Linear(self.qk_dim, self.qk_dim),
                'k': nn.Linear(self.qk_dim, self.qk_dim),
                'v': nn.Linear(self.v_dim, self.v_dim)
            })
        
        self.out_proj = None
        if out_proj:
            self.out_proj = nn.Linear(self.v_dim, self.v_dim)

        self.init_weights()

    def init_weights(self):
        pass
    
    def split_heads(self, x):
        B, L, D = x.shape
        x = x.transpose(0, 1) #L B D
        x = x.reshape(L, B, self.num_heads, D // self.num_heads)
        x = x.permute(1, 2, 0, 3) #B nH L D/nH
        return x

    def merge_heads(self, x):
        B, nH, L, Dhead =  x.shape
        x = x.permute(2, 0, 1, 3) #L B nH Dhead
        x = x.reshape(L, B, nH*Dhead)
        x = x.transpose(0, 1)  #B L D
        return x
    
    #q,k,v are B ... D
    #offset must be broadcastable to add with B nh L L
    def forward(self, q, k, v, offset=0, return_weights=False):
        orig_shape = q.shape
        q, k, v = (x.reshape(x.shape[0], -1, x.shape[-1]) for x in (q, k, v))
        if self.in_proj:
            q = self.in_proj['q'](q)
            k = self.in_proj['k'](k)
            v = self.in_proj['v'](v)
        q = q * self.q_scaling
        q, k, v = (self.split_heads(x) for x in (q, k, v))
        
        A = q @ k.transpose(-2, -1) #B nH L L
        A = F.softmax(A + offset, dim=-1)
        A = self.attn_dropout(A)
        
        out = A @ v #B nH L D/nH
        out = self.merge_heads(out) #B L D

        if self.out_proj:
            out = self.out_proj(out)
        
        out = self.seq_dropout(out.unsqueeze(-1)).squeeze(-1)
        out = out.reshape(orig_shape)

        if return_weights:
            out = (out, A)
        return out
