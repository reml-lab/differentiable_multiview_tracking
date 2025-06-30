import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from mmengine.registry import MODELS
from probtrack.geometry.distributions import rotate_dist1d, shift_dist1d
import lap
from deformable_attention import DeformableAttention, DeformableAttention3D

from einops.layers.torch import Rearrange
from mmcv.ops import MultiScaleDeformableAttention

  
def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

@MODELS.register_module()
class AutoregressiveOutputHead(nn.Module):
    def __init__(self,
                 dim=256,
                 mean_min=[-1, -1, -1],
                 mean_max=[1, 1, 1],
                 attn_cfg=dict(type='QKVAttention', qk_dim=8, num_heads=4),
                 cov_predictor_cfg=dict(type='DiagCovPredictor'),
                 loss_weights={
                    'nll_loss': 0.1,
                    'ce_loss': 1.0
                 },
                 cov_dropout_cfg=dict(type='CovDropout', 
                     p=0.0, 
                     cov_eps=1e-4
                 ),
                 ffn_cfg=dict(type='IsoFFN',
                     dim=2048,
                     expansion_factor=1,
                 )
                 ): 
        super().__init__()
        self.register_buffer('mean_min', torch.tensor(mean_min))
        self.register_buffer('mean_max', torch.tensor(mean_max))
        self.cov_predictor = MODELS.build(cov_predictor_cfg)
        self.loss_weights = loss_weights

        deform_attn_cfg = dict(type='DeformableAttention2D',
            qk_dim=256,
            num_heads=8,
            num_levels=2,
            num_ref_points=4,
            attn_drop=0.1,
            im2col_step=64
        )
        self.deform_attn = MODELS.build(deform_attn_cfg).cuda()

        self.img_attn = DeformableAttention3D(
            dim=8,                   # feature dimensions
            dim_head=2,               # dimension per head
            heads=4,                   # attention heads
            dropout = 0.,                # dropout
            downsample_factor = (2,8,8),       # downsample factor (r in paper)
            offset_scale = (2,8,8),            # scale of offset, maximum offset
            offset_groups = None,        # number of offset groups, should be multiple of heads
            offset_kernel_size = (4,10,10),      # offset kernel size
        )

        self.context_attn = DeformableAttention3D(
            dim=8,                   # feature dimensions
            dim_head=2,               # dimension per head
            heads=4,                   # attention heads
            dropout = 0.,                # dropout
            downsample_factor = (2,8,8),       # downsample factor (r in paper)
            offset_scale = (2,8,8),            # scale of offset, maximum offset
            offset_groups = None,        # number of offset groups, should be multiple of heads
            offset_kernel_size = (4,10,10),      # offset kernel size
        )

        



        # self.img_attn = MODELS.build(attn_cfg)
        # self.context_attn = MODELS.build(attn_cfg)
        self.pos_embeds = MODELS.build(dict(type='SineEncoding2d', dim=8, out_proj=True))

        self.cov_dropout = MODELS.build(cov_dropout_cfg)

        self.mean_head = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)), #B x 3 x 1 x 1
            Rearrange('b c h w -> b (c h w)'), #flatten, B x 3
            # nn.Sigmoid()
        )

        self.eos_head = nn.Sequential(
            # MODELS.build(dict(type='IsoCNN', dim=2048, expansion_factor=1, dropout_rate=0.1)),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
        )
        # self.mix_head = nn.Linear(dim, 1)

        # self.eos_head = nn.Linear(dim, 1)


        self.ffn = MODELS.build(ffn_cfg)

        # self.init_h = nn.Parameter(torch.randn(1, 8, 34, 60))
        #self.init_context = nn.Parameter(torch.randn(1, 8, 1, 34, 60) * 1)
        self.init_context = nn.Parameter(torch.randn(1, 256, 1, 9, 15) * 1)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward_test(self, x, gt=None):
        B = x.shape[0]
        out = []
        for i in range(B):
            node_rot = gt['node_rot'][i][0]
            node_pos = gt['node_pos'][i][0]
            img_feats = x[i].unsqueeze(0)
            context = self.init_context[:, :, 0]
            out_i = []
            for j in range(10):
                context = self.deform_attn([img_feats, context])[1]
                dist, eos_logit = self(context)
                eos_prob = eos_logit.sigmoid()
                if eos_prob.item() < 0.5:
                    break
                dist = rotate_dist1d(dist, node_rot)
                dist = shift_dist1d(dist, node_pos)
                out_i.append(dist)
                # h = self.attn(feats, context, context)
                # context = torch.cat([context, h], dim=1)
            num_viewable = gt['viewable'][i].sum()
            mix_weights = x.new_ones(len(out_i))
            if len(mix_weights) > 0:
                mix = D.Categorical(logits=mix_weights)
                means = torch.stack([dist.mean for dist in out_i], dim=0)
                covs = torch.stack([dist.covariance_matrix for dist in out_i], dim=0)
                comp_dist = D.MultivariateNormal(means[:, 0], covs[:, 0])
                dist = D.MixtureSameFamily(mix, comp_dist)
                out.append(dist)
            else:
                out.append(None)

        return out

    def forward(self, x):
        # x = x.unsqueeze(1)
        mean = self.mean_head(x)
        cov = self.cov_predictor(x)
        cov = self.cov_dropout(cov)
        cov = cov[:, 0] #TODO: ugly
        eos_logit = self.eos_head(x)
        dist = D.MultivariateNormal(mean, cov)
        # mix_weights = x.new_ones(x.shape[0], 1)
        # mix = D.Categorical(mix_weights)
        # dist = D.MixtureSameFamily(mix, dist)
        return dist, eos_logit
        # return mean[:, 0], cov[:, 0], eos_logits[:, 0]

    #x is B x D
    def forward_train(self, x, gt):
        # pos_enc = self.pos_embeds(x).flatten(2).permute(0, 2, 1)
        # x = x.flatten(2).permute(0, 2, 1)
        B = x.shape[0]
        loss_dict = {'nll_loss': 0, 'ce_loss': 0}
        for i in range(B):
            #sort and filter out non-viewable objects
            gt_pos = gt['local_obj_pos'][i]
            sort_idx = torch.argsort(gt_pos[..., 0])
            gt_pos = gt_pos[sort_idx]
            is_viewable = gt['viewable'][i].bool() 
            gt_pos = gt_pos[is_viewable]
            N = len(gt_pos)
            
            img_feats = x[i].unsqueeze(0) #B x 256 x 9 x 15
            context = self.init_context[:,:,0] #B x 256 x 9 x 15
            
            nll_vals, ce_vals = [], []
            for j in range(N + 1): #num_objs + 1 for eos
                context = self.deform_attn([img_feats, context])[1]
                dist, eos_logit = self(context)
                if j < N:
                    nll_vals.append(
                        -dist.log_prob(gt_pos[j])
                    )
                    eos_target = x.new_ones(1)
                else:
                    eos_target = x.new_zeros(1)
                ce_vals.append(
                    F.binary_cross_entropy_with_logits(
                        eos_logit.flatten(), 
                        eos_target
                    )
                )

            if len(nll_vals) > 0:
                nll_vals = torch.cat(nll_vals, dim=0)
                loss_dict['nll_loss'] += nll_vals.mean()
            else:
                loss_dict['nll_loss'] += 0*x.mean()
            ce_vals = torch.stack(ce_vals, dim=0)
            loss_dict['ce_loss'] += ce_vals.mean()
            # loss_dict['ce_loss'] = x.mean()

        for k, v in loss_dict.items():
            loss_dict[k] = self.loss_weights[k] * v / B
        return loss_dict
