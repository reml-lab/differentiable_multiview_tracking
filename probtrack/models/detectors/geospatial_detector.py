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
from probtrack.geometry.distributions import to_torch_dist

from scheduler import proj_sensor_model, Scheduler

@MODELS.register_module()
class GeospatialDetector(nn.Module):
    def __init__(self, 
                 detector_cfg=None, 
                 bbox_pipeline=[],
                 scenarios=None,
                 calibrated_scenarios=None,
                 server=None,
                 freeze_projs=False,
                 freeze_proj_stds=False,
                 freeze_tracker=False,
                 dm='constant_velocity',
                 output_head_cfg=None,
                 sched_max_sensors=4,
                 sched_mode='fixed',
                 sched_max_uncertainty=0.0,
                 sched_update_interval=1, 
                 sched_obj="eig"
        ):
        super().__init__()
        self.sched_max_sensors = sched_max_sensors
        self.sched_mode = sched_mode
        self.sched_max_uncertainty = sched_max_uncertainty
        self.sched_update_interval = sched_update_interval
        self.sched_obj = sched_obj
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.detector = MODELS.build(detector_cfg).cuda().eval()
        self.dm = dm
        if dm == 'constant_velocity':
            H = torch.FloatTensor([[1,0,0,0], [0,0,1,0]]).cuda()
            self.dms = nn.ModuleList([
                tracking.constant_velocity_model(std_acc=10, D=2), #car
                tracking.constant_velocity_model(std_acc=5, D=2), #bus
                tracking.constant_velocity_model(std_acc=8, D=2), #truck
            ])
            self.DZ = 4

        elif dm == 'steerable':
            H = torch.FloatTensor([[1,0,0,0,0], [0,1,0,0,0]]).cuda()
            self.dms = nn.ModuleList([
                tracking.steerable_model(std_accl=10, std_accy=1),
                tracking.steerable_model(std_accl=5, std_accy=1),
                tracking.steerable_model(std_accl=8, std_accy=1),
            ])
            self.DZ = 5
        else:
            raise ValueError(f"Unknown dynamical model {dm}")

        
        self.mm = tracking.linear_measurement_model(H)
        clutter_log_density = torch.log(torch.tensor(1/(5*7)))
        clutter_prob = torch.tensor(0.5)
        self.da = tracking.full_data_associator(clutter_prob=clutter_prob, 
                clutter_log_density=clutter_log_density, 
                Rscale=2,
                mm=self.mm)
        #self.um = tracking.full_updater(self.da)
        self.um = tracking.soft_map_updater(self.da)

        classes = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # classes = torch.tensor([0,1,2])
        self.register_buffer('classes', classes)

        for param in self.dms.parameters():
            param.requires_grad = not freeze_tracker

        for param in self.da.parameters():
            param.requires_grad = False
        
        self.da.clutter_prob_prime.requires_grad = not freeze_tracker
        self.da.clutter_log_density.requires_grad = not freeze_tracker
        self.da.Rscale_prime.requires_grad = not freeze_tracker
        # for i in range(len(self.dms)):
            # self.dms[i].std_acc_prime.requires_grad = not freeze_tracker
        print(self.da.clutter_prob_prime)
        print(self.da.clutter_log_density)
        print(self.da.Rscale_prime)
        # print(self.dms[0].std_acc_prime)
         

        self.server = DATASETS.build(server)
        self.projs = nn.ModuleDict()
        correction_date = datetime.strptime('2023-05-14', '%Y-%m-%d')

        if calibrated_scenarios is None:
            calibrated_scenarios = scenarios

        for scenario in calibrated_scenarios:
            env = self.server.envs[scenario]
            if 'date' in env.keys():
                date = datetime.strptime(env['date'], '%Y-%m-%d')
                apply_corrections=date < correction_date
            else:
                apply_corrections = False
             
            self.projs[scenario] = nn.ModuleDict()
            for node_key, node_data in env['nodes'].items():
                node_key = node_key.replace('_', '')
                proj = point_projector(
                    node_data, 
                    node=node_key, 
                    mode=self.server.type,
                    apply_corrections=apply_corrections,
                    scenario=scenario,
                    node_str=node_key
                )
                self.projs[scenario][node_key] = proj

        for name, param in self.projs.named_parameters():
            param.requires_grad = not freeze_projs


        for name, param in self.projs.named_parameters():
            param.requires_grad = not freeze_projs

        for name, param in self.projs.named_parameters():
            if 'std' in name:
                param.requires_grad = not freeze_proj_stds

        output_head_cfg.projs = self.projs
        self.output_head = MODELS.build(output_head_cfg)


    def init_tracker(self, gt, scenario=None):
        starting_pos = gt['obj_position'][..., 0:2].cuda() #num_objects x 2
        starting_cls = gt['obj_norm_cls_idx']
        # classes = self.classes[starting_cls]
        classes = starting_cls.cuda()
        K = len(starting_pos)
        P = torch.diag_embed(torch.ones(K, self.DZ).cuda())
        
        if self.dm == 'constant_velocity':
            P[..., 0] = P[..., 0] / 10
            P[..., 2] = P[..., 2] / 10
            Z = self.mm.project_state_X2Z(starting_pos)
        elif self.dm == 'steerable':
            psi = gt['obj_yaw'].cuda().unsqueeze(-1) 
            v = torch.zeros(K, 1).cuda()
            omega = torch.zeros(K, 1).cuda()
            Z = self.dms[0].state_from_components(starting_pos, psi, v, omega)
            P[..., 0] = P[..., 0] / 10
            P[..., 1] = P[..., 1] / 10
            P[..., 2] = P[..., 2] / 10


        sensor_models={}
        # scenario = '2023-04-05_09-38-33_09-43-33_bus_1'
        for node, proj in self.projs[scenario].items():
            #Initialize the snesor model for node k using the trained point projector object  
            node_idx = node[-1]
            node_name = f'zed_node_{node_idx}'
            sensor_models[node_name] = proj_sensor_model(dummy_cov_scale=1.0, proj=proj)

        sched = Scheduler(
            sensor_models, 
            max_sensors=self.sched_max_sensors,
            max_uncertainty=self.sched_max_uncertainty,
            update_interval=self.sched_update_interval,
            verbose=False,
            mode=self.sched_mode,
            obj=self.sched_obj
        )

        tracker = tracking.multi_object_tracker(Z, P, classes, self.mm, self.dms, self.da, self.um, const_dt=1/15, scheduler=sched)
        return tracker, Z, P

    # def detection_test(self, dets, bbox_pipeline, scenario=0, node_str='node_1'):
        # dets = bbox_pipeline(dets)
        # proj = self.projs[scenario][node_str]
        # det_means, det_covs = proj.image_to_world_ground_uq(
            # dets.as_scaled().bottoms,
            # z=torch.zeros(len(dets), 1).cuda()
        # )
        # det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        # return det_normals, dets

    def forward_train_coco(self, sample):
        dets = self.detector(sample['img_path'])[0]
        loss = detr_coco_loss(dets, sample['gt_bboxes'].cuda(), sample['gt_labels'].cuda())
        return loss

    def forward_train(self, dets, gt, scenario=0, node_str='node_1'):
        proj = self.projs[scenario][node_str]
        return self.output_head.forward_train(dets, proj, gt)


    def forward_train_no_matching(self, dets, gt, scenario=0, node_str='node_1', bbox_pipeline=None):
        proj = self.projs[scenario][node_str]
        return self.output_head.forward_train_no_matching(dets, proj, gt, bbox_pipeline)

    def forward_test(self, dets, scenario=0, node_str='node_1'):
        proj = self.projs[scenario][node_str]
        return self.output_head.forward_test(dets, proj)
    
    def detection_loss(self, dets, gt, scenario=0, node_str='node_1'):
        proj = self.projs[scenario][node_str]
        det_gt = compute_gt_bboxes(gt, proj)
        detr_loss, assign_idx = detr_coco_loss(
            dets,
            det_gt['gt_bboxes'],
            det_gt['gt_labels'],
            return_assignments=True,
            enforce_iou=True,
            bg_weight=1
        )
        det_mask = torch.zeros(len(dets), dtype=torch.bool).cuda()
        if len(assign_idx) > 0: #make sure something was viewable
            for pred_idx, gt_idx in assign_idx:
                det_mask[pred_idx] = True
        dets = dets.filter(det_mask)
        det_means, det_covs = proj.image_to_world_ground_uq(
            dets.as_scaled().bottoms,
            z=torch.zeros(len(dets), 1).cuda()
        )
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        det_nll_loss = torch.zeros(1).cuda()
        for j, (pred_idx, gt_idx) in enumerate(assign_idx):
            gt_pos = gt['obj_position'][gt_idx][0:2].cuda()
            nll_loss_j = -det_normals[j].log_prob(gt_pos).mean()
            det_nll_loss += nll_loss_j
        if len(assign_idx) > 0:
            det_nll_loss /= len(assign_idx)
        detr_loss['det_nll_loss'] = det_nll_loss
        return detr_loss, det_normals, dets
        # gt_labels = gt['obj_cls_idx'].cuda()
        # if model.dm == 'steerable' and args.track:
            # pos = torch.cat([
                # Z[..., 0:2], 
                # torch.zeros(len(Z), 1).cuda()
            # ], dim=-1)
            # track_bboxes = compute_proj_bboxes(
                # proj, 
                # pos=pos,
                # roll=torch.zeros(len(Z), 1).cuda(),
                # pitch=torch.zeros(len(Z), 1).cuda(),
                # yaw=Z[..., 2].unsqueeze(-1),
                # dims=gt['obj_dims'].cuda()
            # )
            # mask = prune_bboxes(track_bboxes)
            # gt_bboxes = track_bboxes[mask]
            # gt_labels = gt_labels[mask]
