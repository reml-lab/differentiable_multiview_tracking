import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.registry import MODELS
from probtrack.models.pos.sine import AnchorEncoding

#copied from mmdetection impl of DeformableDETR
def get_reference_points(spatial_shapes, valid_ratios, device):
    """Get the reference points used in decoder.
    Args:
        spatial_shapes (Tensor): The shape of all
            feature maps, has shape (num_level, 2).
        valid_ratios (Tensor): The radios of valid
            points on the feature map, has shape
            (bs, num_levels, 2)
        device (obj:`device`): The device where
            reference_points should be.
    Returns:
        Tensor: reference points used in decoder, has \
            shape (bs, num_keys, num_levels, 2).
    """
    reference_points_list = []
    for lvl, (H, W) in enumerate(spatial_shapes):
        #  TODO  check this 0.5
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(
                0.5, H - 0.5, H, dtype=torch.float32, device=device),
            torch.linspace(
                0.5, W - 0.5, W, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / (
            valid_ratios[:, None, lvl, 1] * H)
        ref_x = ref_x.reshape(-1)[None] / (
            valid_ratios[:, None, lvl, 0] * W)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    return reference_points


#basically just a wrapper around mmcv MultiScaleDeformableAttention
#handles annoying pre and post processing of feats
#forward takes a list of conv feature maps of any size 
#dim must be the same tho
@MODELS.register_module()
class DeformableAttention2D(torch.nn.Module):
    def __init__(self,
                 qk_dim=256,
                 num_heads=8,
                 num_levels=4,
                 num_ref_points=4,
                 attn_drop=0.1,
                 im2col_step=64
        ):
        super().__init__()
        self.attn = MultiScaleDeformableAttention(
            embed_dims=qk_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_ref_points,
            im2col_step=im2col_step,
            dropout=attn_drop,
            batch_first=False
        )

        self.pos_encoding = AnchorEncoding(dim=qk_dim,
                out_proj=False, learned=False,
                grid_size=(100,100),
        )
    
    def forward(self, feats):
        spatial_shapes, flat_feats = [], []
        pos_embeds = self.pos_encoding(None)
        # import ipdb; ipdb.set_trace() # noqa
        query_pos = []
        for feat in feats:
            B, D, H, W = feat.shape
            shape = torch.tensor([H, W])
            spatial_shapes.append(shape)
            flat_feats.append(feat.flatten(2).permute(2, 0, 1))
            pos = F.interpolate(pos_embeds, (H, W))
            query_pos.append(pos.flatten(2).permute(2, 0, 1))

        flat_feats = torch.cat(flat_feats, dim=0)
        query_pos = torch.cat(query_pos, dim=0)
        
        spatial_shapes = torch.stack(spatial_shapes).to(flat_feats.device)
        
        lens = torch.tensor([0] + [h*w for h, w in spatial_shapes])
        level_start_index = torch.cumsum(lens, dim=0)[0:-1] #dont need last one
        level_start_index = level_start_index.to(flat_feats.device)
        
        valid_ratios = torch.ones(len(feats[0]), len(feats), 2).to(flat_feats.device)
        ref_points = get_reference_points(spatial_shapes, valid_ratios, flat_feats.device)
        result = self.attn(flat_feats, reference_points=ref_points, 
                      query_pos=query_pos,
                      spatial_shapes=spatial_shapes,
                      level_start_index=level_start_index)
        outputs = []
        i = 0
        for H, W in spatial_shapes:
            output = result[i:i+(H*W)]
            output = output.transpose(0,1)
            output = output.view(-1, H, W, output.shape[-1])
            output = output.permute(0,3,1,2)
            outputs.append(output)
        return outputs
