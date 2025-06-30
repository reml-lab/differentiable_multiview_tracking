import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from mmengine.registry import MODELS, DATASETS
from probtrack.geometry.distributions import rotate_dist, scale_dist, shift_dist
from mmengine.dataset import Compose
from datetime import datetime
from probtrack.structs import GeospatialDetections, BBoxDetections
sys.path.append('/home/csamplawski/src/iobtmax-data-tools')
from spatial_transform_utils import *
import json
import lap
import cv2
import time
from tqdm import tqdm
import torchvision
from probtrack.metrics import giou
from probtrack.datasets.coco import detr_coco_loss
from probtrack.models.output_heads.matching import linear_assignment, prune_pixels



@MODELS.register_module()
class LinearEncoder(nn.Module):
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

@MODELS.register_module()
class SimpleGeospatialDetector(nn.Module):
    def __init__(self, 
                 detector_cfg=None, 
                 bbox_pipeline=[],
                 scenarios=None,
                 server=None,
                 freeze_projs=False,
                 freeze_proj_stds=False,
        ):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.detector = MODELS.build(detector_cfg).cuda().eval()

        self.server = DATASETS.build(server)
        
        # self.shared_ffn = MODELS.build(ffn_cfg)
        #self.dim_reducer = nn.Linear(100, 1)
        output_head_cfg = dict(type='GaussianPredictor')
        self.output_head = MODELS.build(output_head_cfg)
        # self.clustering = MODELS.build(dict(type='ClusterBBoxes', iou_threshold=0.1, return_mode='assignments'))
        # self.refine_cls_head = nn.Linear(256, 81)

        # self.init_weights()
        self.projs = nn.ModuleDict()
        self.adapters = nn.ModuleDict()
        correction_date = datetime.strptime('2023-05-14', '%Y-%m-%d')
        for scenario in scenarios:
            env = self.server.envs[scenario]
            if 'date' in env.keys():
                date = datetime.strptime(env['date'], '%Y-%m-%d')
                apply_corrections=date < correction_date
            else:
                apply_corrections = False
             
            self.projs[scenario] = nn.ModuleDict()
            self.adapters[scenario] = nn.ModuleDict()
            for node_key, node_data in env['nodes'].items():
                node_key = node_key.replace('_', '')
                proj = point_projector(
                    node_data, 
                    node=node_key, 
                    mode=self.server.type,
                    apply_corrections=apply_corrections
                )
                self.projs[scenario][node_key] = proj
                self.adapters[scenario][node_key] = LinearEncoder(in_len=100, out_len=1)

        # self.proj = MultiViewProjector(scenarios, update_mode=gt_type)
        for name, param in self.projs.named_parameters():
            param.requires_grad = not freeze_projs

        for name, param in self.named_parameters():
            if 'std' in name:
                param.requires_grad = not freeze_proj_stds


        self.bbox_pipeline = []
        for module in bbox_pipeline:
            module = MODELS.build(module)
            self.bbox_pipeline.append(module)
        self.bbox_pipeline = nn.Sequential(*self.bbox_pipeline)

        # for m in self.adapters.modules():
            # if hasattr(m, 'weight') and m.weight.dim() > 1:
                # nn.init.xavier_uniform_(m.weight, gain=2)

        # for m in self.shared_ffn.modules():
            # if hasattr(m, 'weight') and m.weight.dim() > 1:
                # nn.init.xavier_uniform_(m.weight, gain=2)

        # for m in self.dim_reducer.modules():
            # if hasattr(m, 'weight') and m.weight.dim() > 1:
                # nn.init.xavier_uniform_(m.weight, gain=2)

        # for k, cfg in output_head_cfgs.items():
            # self.output_heads[k] = MODELS.build(cfg)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=2)

    def forward(self, x):
        embeds = self.detector(x)
        preds = {k: head(embeds) for k, head in self.output_heads.items()}
        return preds
    
    # def forward_test(self, sensor_data, gt=None, **kwargs):
        # bbox_dets = self.detector(sensor_data)
        # bbox_dets = [self.bbox_pipeline(d.to('cuda')) for d in bbox_dets]
        # preds = {}
        # for k, head in self.output_heads.items():
            # preds[k] = head.forward_test(bbox_dets, gt=gt, **kwargs)
        # return preds
    
    #def forward_test(self, bbox_dets, gt, scenario_idx=0, node_str='node_1'):
    @torch.no_grad()
    def forward_test(self, sensor_data, gt, scenario_idx=0, node_str='node_1'):
        fnames = sensor_data['fnames']
        bbox_dets = [self.detector(fname)[0] for fname in fnames]
        bbox_dets = [self.bbox_pipeline(d.to('cuda')) for d in bbox_dets]
        proj = self.projs[scenario_idx][node_str]
        bbox_dets = [dets.to('cpu') for dets in bbox_dets]
        return [bbox_dets, proj]


    def get_matches(self, sensor_data, gt=None, **kwargs):
        bbox_dets = self.detector(sensor_data)
        bbox_dets = [self.bbox_pipeline(d.to('cuda')) for d in bbox_dets]
        preds = {}
        for k, head in self.output_heads.items():
            preds[k] = head.get_matches(bbox_dets, gt=gt, **kwargs)
        return preds

    def forward_train_coco(self, sample):
        dets = self.detector(sample['img_path'])[0]
        # dets = dets.to('cuda')
        loss = detr_coco_loss(dets, sample['gt_bboxes'].cuda(), sample['gt_labels'].cuda())
        return loss

    def forward_train_geospatial(self, sensor_data, gt, scenario_idx=0, node_str='node_1'):
        proj = self.projs[scenario_idx][node_str]
        adapter = self.adapters[scenario_idx][node_str]
        gt_obj_pos = gt['obj_position'].cuda()
        gt_cls_idx = gt['obj_cls_idx'].cuda()
        obj_roll = gt['obj_roll'].cuda()
        obj_pitch = gt['obj_pitch'].cuda()
        obj_yaw = gt['obj_yaw'].cuda()
        obj_dims = gt['obj_dims'].cuda()
        device = gt_obj_pos.device
        B, N, _ = gt_obj_pos.shape
        output = []
        for i in range(B):
            dets = self.detector(sensor_data['fnames'][i])[0]
            embeds = dets.embeds
            # embeds = self.shared_ffn(embeds)
            embeds = adapter(embeds)[0]
            dist = self.output_head(embeds)
            nll = -dist.log_prob(gt_obj_pos[i, :, 0:2])
            nll_loss = 0.0 * nll.mean()

            mse_loss = F.mse_loss(dist.loc, gt_obj_pos[i, :, 0:2])
            loss = {'nll_loss': nll_loss, 'mse_loss': mse_loss, 'loss': nll_loss + mse_loss}
            output.append(loss)
        # print(dist.mean.detach().cpu().numpy(), gt_obj_pos[-1])
        return output

            ## means, _ = proj.image_to_world_ground_uq(
                # dets.bottoms,
                # z=torch.zeros(len(bboxes), 1).to(proj_pixels.device)
            # )


    def forward_train(self, sensor_data, gt, **kwargs):
        loss_dict = {}
        bbox_dets = self.detector(sensor_data)
        bbox_dets = [self.bbox_pipeline(d.to('cuda')) for d in bbox_dets]
        for k, head in self.output_heads.items():
            loss_k = head.forward_train(bbox_dets, gt, **kwargs)
            loss_dict.update(loss_k)

        final_loss = 0
        for k, v in loss_dict.items():
            if torch.isnan(v):
                print(f'Loss {k} is nan')
                v = 0
            final_loss += v + 0*self.dummy_param.mean()
        loss_dict['loss'] = final_loss
        return loss_dict
