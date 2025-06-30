import torch
import torch.nn as nn
import sys
from mmengine.registry import MODELS
from mmengine.config import Config
# from mmdet.apis import init_detector
from .image_norm import ImageNorm
# from mmdet.registry import MODELS as MMDET_MODELS

def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the inverse.
        eps (float): EPS avoid numerical overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse function of sigmoid, has the same
        shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

@MODELS.register_module()
class DETRAdapter(nn.Module):
    def __init__(self, num_embeds=1):
        super().__init__()
        self.proj = nn.Linear(100, num_embeds)

    def forward(self, detr_output):
        embeds = detr_output['embeds']
        embeds = embeds[-1] #last layer #B x 100 x C
        embeds = embeds.permute(0, 2, 1) #B x C x 100
        embeds = self.proj(embeds)
        embeds = embeds.permute(0, 2, 1) #B x arg x C
        return embeds

@MODELS.register_module()
class PretrainedDETR(nn.Module):
    def __init__(self, 
            out_channels=256,
            return_embeds_only=False,
            mmdet_path='/home/csamplawski/src/mmdetection',
            freeze_bbox_head=True,
            freeze_cls_head=True,
        ):
        super().__init__()
        sys.path.append(mmdet_path)
        from mmdet.registry import MODELS as MMDET_MODELS
        config_file = '%s/configs/detr/detr_r50_8xb2-150e_coco.py' % mmdet_path
        config_file = '%s/configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py' % mmdet_path

        cfg = Config.fromfile(config_file)
        del cfg['model']['data_preprocessor']
        self.detr = MMDET_MODELS.build(cfg.model)
        # checkpoint_file = '%s/checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth' % mmdet_path
        checkpoint_file = '%s/checkpoints/dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth' % mmdet_path
        sd = torch.load(checkpoint_file)['state_dict']
        self.detr.load_state_dict(sd, strict=False)
        self.detr = self.detr.eval()
        self.img_norm = ImageNorm(
            mean=[123.675, 116.28, 103.53], 
            std=[58.395, 57.12, 57.375]
        )
        
        # for param in self.detr.bbox_head.reg_ffn.parameters():
            # param.requires_grad = not freeze_bbox_head
        # for param in self.detr.bbox_head.fc_reg.parameters():
            # param.requires_grad = not freeze_bbox_head
        # for param in self.detr.bbox_head.fc_cls.parameters():
            # param.requires_grad = not freeze_cls_head

    
    # @torch.no_grad()
    def forward(self, x):
        self.detr.eval()
        with torch.no_grad():
            query_pos = self.detr.query_embedding.weight #100 x C
            x = self.img_norm(x)
            feats = self.detr.backbone(x) #r50: B x 2048 x H/32 x W/32
            feats = self.detr.neck(feats)[0] #B x 256 x H/32 x W/32
            B, C, H, W = feats.shape
            attn_mask = feats.new_zeros((B, H, W))
            feats_pos_embed = self.detr.positional_encoding(attn_mask) #B x 256 x H/32 x W/32
            
            #flatten all the feats
            feats = feats.view(B, C, H*W).permute(0, 2, 1)
            feats_pos_embed = feats_pos_embed.view(B, C, H*W).permute(0, 2, 1)
            attn_mask = attn_mask.view(B, H*W).bool()

            encoder_output = self.detr.encoder( #6? x B x H*W x C
                query=feats,
                query_pos=feats_pos_embed,
                key_padding_mask=attn_mask,
            )
            # import ipdb; ipdb.set_trace() # noqa

            query_pos = query_pos.unsqueeze(0).expand(B, -1, -1) #B x 100 x C
            query = query_pos.new_zeros((B, self.detr.num_queries, C))
            

            hidden_states, references = self.detr.decoder( #6 x B x 100 x C
                query=query, #B x 100 x C
                key=encoder_output, #B x H*W x C
                # value=encoder_output, #B x H*W x C
                query_pos=query_pos, #B x 100 x C
                key_pos=feats_pos_embed, #B x H*W x C
                key_padding_mask=attn_mask,     
                reg_branches=self.detr.bbox_head.fc_reg
            )



        bbox_head = self.detr.bbox_head
        layers_cls_scores = bbox_head.fc_cls(hidden_states)
        references_before_sigmoid = inverse_sigmoid(references, eps=1e-3)
        tmp_reg_preds = bbox_head.fc_reg(hidden_states)
        tmp_reg_preds[..., :references_before_sigmoid.size(-1)] += references_before_sigmoid
        layers_bbox_preds = tmp_reg_preds.sigmoid()

        cls_logits = layers_cls_scores
        bboxes = layers_bbox_preds

        # cls_logits = bbox_head.fc_cls(decoder_output) #B x 100 x 80 + 1
        # bboxes = bbox_head.reg_ffn(decoder_output)
        # bboxes = bbox_head.activate(bboxes)
        # bboxes = bbox_head.fc_reg(bboxes)
        # bboxes = torch.sigmoid(bboxes) #B x 100 x 4
        
        output = {
            'cls_logits': cls_logits,
            'bboxes': bboxes,
            'embeds': hidden_states,
            'references': references,
        }
        return output
