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
import torch_tracking as tracking
import json
import lap
import cv2
import time
from tqdm import tqdm
import torchvision
from probtrack.metrics import giou
from probtrack.datasets.coco import detr_coco_loss
from probtrack.models.output_heads.matching import linear_assignment, prune_pixels
from probtrack.models.detectors.geospatial_detector import GeospatialDetector
from probtrack.geometry.distributions import to_torch_dist


@MODELS.register_module()
class UAIGeospatialDetector(GeospatialDetector):
    def __init__(self, 
                 detector_cfg=None, 
                 bbox_pipeline=[],
                 scenarios=None,
                 server=None,
                 freeze_projs=False,
                 freeze_proj_stds=False,
                 freeze_tracker=False,
                 dm='constant_velocity',
                 adapter_cfg=dict(type='LinearAdapter'),
                 freeze_adapters=False,
                 freeze_output_head=False,
        ):
        super().__init__(detector_cfg=detector_cfg, bbox_pipeline=bbox_pipeline, scenarios=scenarios, server=server,
                freeze_projs=freeze_projs, freeze_proj_stds=freeze_proj_stds, freeze_tracker=freeze_tracker, dm=dm)
        output_head_cfg = dict(type='GaussianPredictor')
        self.output_head = MODELS.build(output_head_cfg)
        self.adapters = nn.ModuleDict()
        for scenario in self.projs.keys():
            self.adapters[scenario] = nn.ModuleDict()
            for node in self.projs[scenario].keys():
                self.adapters[scenario][node] = MODELS.build(adapter_cfg)

        for param in self.adapters.parameters():
            param.requires_grad = not freeze_adapters
        for param in self.output_head.parameters():
            param.requires_grad = not freeze_output_head
       
        # for m in self.adapters.modules():
            # if hasattr(m, 'weight') and m.weight.dim() > 1:
                # nn.init.xavier_uniform_(m.weight, gain=2)

    def forward_adapter(self, embeds, scenario=0, node_str='node_1'):
        adapter = self.adapters[scenario][node_str]
        embeds = adapter(embeds)
        return embeds[:,0]

    def detection_loss(self, dets, gt, scenario=0, node_str='node_1'):
        num_objs = len(gt['obj_position'])
        adapter = self.adapters[scenario][node_str]
        view_embeds = adapter(dets.embeds)[0]
        det_means, det_covs = self.output_head(view_embeds)
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        assert len(det_normals) == num_objs
        
        det_nll_loss = torch.zeros(1).cuda()
        for j in range(num_objs):
            gt_pos = gt['obj_position'][j][0:2].cuda()
            nll_loss_j = -det_normals[j].log_prob(gt_pos).mean()
            det_nll_loss += nll_loss_j
        if num_objs > 0:
            det_nll_loss /= num_objs
        loss_dict = {
            'giou_loss': torch.zeros(1).cuda().mean() * self.dummy_param.mean(),
            'l1_loss': torch.zeros(1).cuda().mean() * self.dummy_param.mean(),
            'ce_loss': torch.zeros(1).cuda().mean() * self.dummy_param.mean(),
            'det_nll_loss': det_nll_loss
        }
        return loss_dict, det_normals, dets
    
    def detection_test(self, dets, bbox_pipeline=None, scenario=0, node_str='node_1'):
        adapter = self.adapters[scenario][node_str]
        view_embeds = adapter(dets.embeds)[0]
        det_means, det_covs = self.output_head(view_embeds)
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        return det_normals, dets
        import ipdb; ipdb.set_trace() # noqa
