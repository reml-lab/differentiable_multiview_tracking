import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import warnings
warnings.filterwarnings("ignore", module="mmengine")
from mmengine.registry import MODELS
from mmengine.config import Config
import torchvision
from .image_norm import ImageNorm
from mmdet.apis import DetInferencer
from probtrack.structs import BBoxDetections, cxcywh_to_bottom
import os
from copy import deepcopy

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
            freeze_bbox_head=True,
            freeze_cls_head=True,
            lru_cache_size=200000,
            cache_dir='/home/csamplawski/four/cache/detr',
            downsample_factor=1,
            use_cache=True,
            add_point_head=False,
            add_depth_head=False,
            freeze_point_head=True,
            freeze_depth_head=True,
        ):
        super().__init__()
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.downsample_factor = downsample_factor

        #checkpoint_path = '/home/csamplawski/src/mmdetection/checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
        #checkpoint_path = '/home/csamplawski/src/old_mmlab/mmdetection_fork/checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
        #checkpoint_data = torch.load(checkpoint_path)
        inferencer = DetInferencer(model='detr_r50_8xb2-150e_coco')
        self.detr = inferencer.model
        #self.detr.load_state_dict(checkpoint_data['state_dict'], strict=False)
        self.detr = self.detr.cuda().eval()
        # self.use_new_cls_head = use_new_cls_head
        self.img_norm = ImageNorm(
            mean=[123.675, 116.28, 103.53], 
            std=[58.395, 57.12, 57.375]
        )
        self.lru_cache_size = lru_cache_size
        self.disk_lookup = {}
        self.lru_cache = {}
        self.dets_cache = {}
        # self.sort_by_vehicle = sort_by_vehicle
        # self.cluster_iou_threshold = cluster_iou_threshold

        # self.vehicle_indices = [2, 5, 7]
        # self.vehicle_prob_threshold = vehicle_prob_threshold

        # self.precomputed = precomputed

        for param in self.detr.parameters():
            param.requires_grad = False

        # for param in self.detr.decoder.parameters():
            # param.requires_grad = True

        for param in self.detr.bbox_head.reg_ffn.parameters():
            param.requires_grad = not freeze_bbox_head
        for param in self.detr.bbox_head.fc_reg.parameters():
            param.requires_grad = not freeze_bbox_head
        for param in self.detr.bbox_head.fc_cls.parameters():
            param.requires_grad = not freeze_cls_head
        
        self.add_point_head = add_point_head
        if self.add_point_head:
            self.activate = nn.ReLU()
            self.fc_point = deepcopy(self.detr.bbox_head.fc_reg)
            self.point_ffn = deepcopy(self.detr.bbox_head.reg_ffn)

            for param in self.point_ffn.parameters():
                param.requires_grad = not freeze_point_head
            for param in self.fc_point.parameters():
                param.requires_grad = not freeze_point_head

        self.add_depth_head = add_depth_head
        if self.add_depth_head:
            self.activate = nn.ReLU()
            self.fc_depth = deepcopy(self.detr.bbox_head.fc_reg)
            self.depth_ffn = deepcopy(self.detr.bbox_head.reg_ffn)

            for param in self.depth_ffn.parameters():
                param.requires_grad = not freeze_depth_head
            for param in self.fc_depth.parameters():
                param.requires_grad = not freeze_depth_head



            
    def forward_output_heads(self, embeds):
        if len(embeds.shape) == 3:
            embeds = embeds.unsqueeze(0)
        num_layers, B, num_queries, C = embeds.shape
        bbox_head = self.detr.bbox_head
        cls_logits = bbox_head.fc_cls(embeds) #B x 100 x 80 + 1
        bboxes = bbox_head.reg_ffn(embeds)
        bboxes = bbox_head.activate(bboxes)
        bboxes = bbox_head.fc_reg(bboxes)
        bboxes = torch.sigmoid(bboxes) #B x 100 x 4

        if self.add_depth_head:
            import ipdb; ipdb.set_trace() # noqa
            depths = self.depth_ffn(embeds)
            depths = self.activate(depths)
            depths = self.fc_depth(depths)
            depths = torch.sigmoid(depths)
            depths = depths.mean(dim=-1)
            depths = depths * 5
        else:
            depths = torch.zeros(num_layers, B, num_queries).cuda() - 1

        if self.add_point_head:
            point_bboxes = self.point_ffn(embeds)
            point_bboxes = self.activate(point_bboxes)
            point_bboxes = self.fc_point(point_bboxes)
            point_bboxes = torch.sigmoid(point_bboxes)
            bottom_pixels = cxcywh_to_bottom(point_bboxes)
            output = []
            for b in range(B):
                dets = BBoxDetections(
                    bboxes[-1][b],
                    cls_logits[-1][b],
                    embeds[-1][b],
                    pixels=bottom_pixels[-1][b],
                    depths = depths[-1][b]
                )
                output.append(dets)
            return output
        
        bottom_pixels = cxcywh_to_bottom(bboxes)
        output = []
        for b in range(B):
            dets = BBoxDetections(
                bboxes[-1][b],
                cls_logits[-1][b],
                embeds[-1][b],
                pixels=bottom_pixels[-1][b],
                depths=depths[-1][b]
            )
            output.append(dets)
        return output

    def load_img(self, fname, target_H=1080, target_W=1920):
        target_H = target_H // self.downsample_factor
        target_W = target_W // self.downsample_factor
        img = cv2.imread(fname)
        if len(img.shape) == 2:  # grayscale to RGB
            img = np.stack([img] * 3, axis=-1)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scale = min(target_H / img.shape[0], target_W / img.shape[1]) 
        new_H, new_W = int(img.shape[0] * scale), int(img.shape[1] * scale)
        img = cv2.resize(img, (new_W, new_H))
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img

    
    def forward_transformer(self, x):
        self.detr.eval()
        self.detr.backbone.eval()
        x = x.cuda()
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
        attn_mask = attn_mask.view(B, H*W)

        encoder_output = self.detr.encoder( #6? x B x H*W x C
            query=feats,
            query_pos=feats_pos_embed,
            key_padding_mask=attn_mask,
        )

        query_pos = query_pos.unsqueeze(0).expand(B, -1, -1) #B x 100 x C
        query = torch.zeros_like(query_pos) #B x 100 x C

        embeds = self.detr.decoder( #6 x B x 100 x C
            query=query, #B x 100 x C
            key=encoder_output, #B x H*W x C
            value=encoder_output, #B x H*W x C
            query_pos=query_pos, #B x 100 x C
            key_pos=feats_pos_embed, #B x H*W x C
            key_padding_mask=attn_mask,     
        )
        return embeds

    def forward_batch(self, fnames):
        if self.use_cache:
            all_embeds = []
            for fname in fnames:
                if fname in self.lru_cache:
                    embeds = self.lru_cache[fname].cuda()
                else:
                    embed_fname = fname.replace('/', '_').replace('.', '_') + '.pt'
                    embed_path = os.path.join(self.cache_dir, embed_fname)
                    try:
                        embeds = torch.load(embed_path)[-1]
                    except:
                        frame = self.load_img(fname)
                        embeds = self.forward_transformer(frame)
                        torch.save(embeds, embed_path)
                        embeds = embeds[-1]
                    self.lru_cache[fname] = embeds.detach().cpu()
                    if len(self.lru_cache) > self.lru_cache_size:
                        self.lru_cache.popitem()

                all_embeds.append(embeds)
            embeds = torch.cat(all_embeds, dim=0)
        else:
            imgs = [self.load_img(fname) for fname in fnames]
            imgs = torch.cat(imgs, dim=0)
            embeds = self.forward_transformer(imgs)[-1] #last layer
        return embeds

    # @torch.no_grad()
    def forward(self, x):
        self.detr.eval()
        if isinstance(x, str):
            # if x in self.dets_cache:
                # output = self.dets_cache[x]
                # output = [o.to('cuda') for o in output]
                # return output
            embed_fname = x.replace('/', '_').replace('.', '_') + '.pt'
            embed_path = os.path.join(self.cache_dir, embed_fname)
            if os.path.exists(embed_path) and self.use_cache:
                embeds = torch.load(embed_path)
            else:
                frame = self.load_img(x)
                embeds = self.forward_transformer(frame)
                if self.use_cache:
                    torch.save(embeds, embed_path)

        elif 'embeds' in x: #precomputed embeddings
            embeds = x['embeds']
            import ipdb; ipdb.set_trace() # noqa
            if isinstance(embeds, list) and len(embeds) == 1:
                embeds = embeds[0].embeds.unsqueeze(0)
                print('embeds', embeds.shape)
            elif isinstance(embeds, list) and len(embeds) > 1:
                return embeds
        else:
            x = x['frame']
            embeds = self.forward_transformer(x)
                    
        output = self.forward_output_heads(embeds)

        # self.dets_cache[x] = [o.to('cpu') for o in output]
        return output

        # if self.precomputed and 'embeds' in x:
            # decoder_output = x['embeds'].unsqueeze(0)

        # if self.precomputed:
            # if self.compute_output_heads:
                # embeds = [det.embeds for det in x['embeds']]
                # embeds = torch.stack(embeds, dim=0)
                # embeds = embeds.unsqueeze(0) #num_layesr = 1
                # dets = self.forward_output_heads(embeds.cuda())
                # return dets
            # return x['embeds']
        

        # output = self.forward_output_heads(decoder_output)
        # num_layers, B, num_queries, C = decoder_output.shape
        # bbox_head = self.detr.bbox_head
        # cls_logits = bbox_head.fc_cls(decoder_output) #B x 100 x 80 + 1
        # bboxes = bbox_head.reg_ffn(decoder_output)
        # bboxes = bbox_head.activate(bboxes)
        # bboxes = bbox_head.fc_reg(bboxes)
        # bboxes = torch.sigmoid(bboxes) #B x 100 x 4

        # output = []
        # for b in range(B):
            # dets = BBoxDetections(
                # bboxes[-1][b],
                # cls_logits[-1][b],
                # decoder_output[-1][b],
            # )
            # output.append(dets)
        #return output

        # cls_probs = torch.softmax(cls_logits, dim=-1)

        # new_cls_logits = self.new_cls_head(decoder_output)
        # new_cls_probs = torch.softmax(new_cls_logits, dim=-1)

        # if self.use_new_cls_head:
            # cls_logits = new_cls_logits
            # cls_probs = new_cls_probs
        
        # if self.sort_by_vehicle:
            # sort_probs = -cls_probs[..., self.vehicle_indices].sum(dim=-1)
        # else:
            # sort_probs = cls_probs[..., -1] #background probs
        # sort_idx = torch.argsort(sort_probs, dim=-1)
        # sort_idx = sort_idx.unsqueeze(-1)#.expand(-1, -1, -1, 256)
        # cls_logits = torch.gather(cls_logits, 2, sort_idx.expand(-1, -1, -1, 81))[-1]
        # new_cls_logits = torch.gather(new_cls_logits, 2, sort_idx.expand(-1, -1, -1, 81))[-1]
        # cls_probs = torch.softmax(cls_logits, dim=-1)
        # bboxes = torch.gather(bboxes, 2, sort_idx.expand(-1, -1, -1, 4))[-1]
        # embeds = torch.gather(decoder_output, 2, sort_idx.expand(-1, -1, -1, 256))[-1]


        # vehicle_probs = cls_probs[..., self.vehicle_indices].sum(dim=-1)
        # is_vehicle = vehicle_probs > self.vehicle_prob_threshold

        # bboxes = bboxes[:, 0:self.top_k]
        # unnormed_bboxes = bboxes * torch.tensor([1920, 1080, 1920, 1080], device=bboxes.device)

        # bottom_pixels = cxcywh_to_bottom(bboxes)
        # unnormed_bottom_pixels = cxcywh_to_bottom(unnormed_bboxes)

        
        # cluster_assignments, cluster_reprs = [], []
        # if self.cluster_iou_threshold > 0:
            # for i in range(B):
                # a, r = cluster_bboxes(bboxes[i], iou_threshold=self.cluster_iou_threshold)
                # cluster_assignments.append(a)
                # cluster_reprs.append(r)


        # output = {
            # 'cls_logits': cls_logits[:, 0:self.top_k],
            # 'new_cls_logits': new_cls_logits[:, 0:self.top_k],
            # 'bboxes': bboxes[:, 0:self.top_k],
            # 'unnormed_bboxes': unnormed_bboxes[:, 0:self.top_k],
            # 'vehicle_probs': vehicle_probs[:, 0:self.top_k],
            # 'embeds': embeds[:, 0:self.top_k],
            # 'is_vehicle': is_vehicle[:, 0:self.top_k],
            # 'cluster_assignments': cluster_assignments,
            # 'cluster_reprs': cluster_reprs,
        # }
        # return output
