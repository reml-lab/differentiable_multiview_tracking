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


@MODELS.register_module()
class SingleObjGeospatialDetector(nn.Module):
    def __init__(self, 
                 detector_cfg=None, 
                 bbox_pipeline=[],
                 scenarios=None,
                 server=None,
                 freeze_projs=False,
                 freeze_proj_stds=False,
                 freeze_tracker=False,
                 dm='constant_velocity',
        ):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.detector = MODELS.build(detector_cfg).cuda().eval()
        self.dm = dm
        if dm == 'constant_velocity':
            H = torch.FloatTensor([[1,0,0,0], [0,0,1,0]]).cuda()
            self.dms = nn.ModuleList([
                tracking.constant_velocity_model(std_acc=10, D=2), #car
                tracking.constant_velocity_model(std_acc=5, D=2), #bus
                tracking.constant_velocity_model(std_acc=2, D=2), #truck
            ])
            self.DZ = 4
        elif dm == 'steerable':
            H = torch.FloatTensor([[1,0,0,0,0], [0,1,0,0,0]]).cuda()
            self.dms = nn.ModuleList([
                tracking.simple_steerable_model(std_accl=10, std_accy=10),
                tracking.simple_steerable_model(std_accl=1, std_accy=1),
                tracking.simple_steerable_model(std_accl=1, std_accy=1),
            ])
            self.DZ = 5
        else:
            raise ValueError(f"Unknown dynamical model {dm}")

        
        self.mm = tracking.linear_measurement_model(H)
        clutter_log_density = torch.log(torch.tensor(1/(5*7)))
        clutter_prob = torch.tensor(0.01)
        self.da = tracking.full_data_associator(clutter_prob=clutter_prob, clutter_log_density=clutter_log_density, mm=self.mm)
        self.um = tracking.full_updater(self.da)

        classes = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.register_buffer('classes', classes)

        
        # self.dms = nn.ModuleList([
            # tracking.simple_steerable_model(std_accl=10, std_accy=10),
            # tracking.simple_steerable_model(std_accl=1, std_accy=1),
            # tracking.simple_steerable_model(std_accl=1, std_accy=1),
        # ])


        for param in self.dms.parameters():
            param.requires_grad = not freeze_tracker

        for param in self.da.parameters():
            param.requires_grad = not freeze_tracker
        

        # if K == 2:
            # classes = torch.FloatTensor([[1, 0], [0, 1]])
        # else:
            # classes = torch.FloatTensor([[1]])

        # K = len(starting_pos)
        # Z = mm.project_state_X2Z(starting_pos)
        # P = torch.diag_embed(torch.ones(K, DZ)) / 5
        # tracker = tracking.multi_object_tracker(Z, P, classes, mm, [dm]*K, da,um,const_dt=dt)


        self.server = DATASETS.build(server)
        self.projs = nn.ModuleDict()
        correction_date = datetime.strptime('2023-05-14', '%Y-%m-%d')
        for scenario in scenarios:
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
                    apply_corrections=apply_corrections
                )
                self.projs[scenario][node_key] = proj

        # self.proj = MultiViewProjector(scenarios, update_mode=gt_type)
        for name, param in self.projs.named_parameters():
            param.requires_grad = not freeze_projs

        for name, param in self.projs.named_parameters():
            if 'std' in name:
                param.requires_grad = not freeze_proj_stds


        self.bbox_pipeline = []
        for module in bbox_pipeline:
            module = MODELS.build(module)
            self.bbox_pipeline.append(module)
        self.bbox_pipeline = nn.Sequential(*self.bbox_pipeline)

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

    def compute_gt_bboxes(self, gt, proj):
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
            gt_obj_rot = euler_to_rot_torch(
                obj_roll[i],
                obj_pitch[i],
                obj_yaw[i],
            )
            points_3d = get_3d_box_points(gt_obj_pos[i], gt_obj_rot, obj_dims[i])
            No, Np, _ = points_3d.shape
            points_3d_img = proj.world_to_image(
                    points_3d.reshape(No * Np, 3), normalize=True)
            points_3d_img = points_3d_img.reshape(No, Np, 3)
            bboxes_projed = box3d_points_to_box2d_chw(points_3d_img)

            proj_pixels = proj.world_to_image(gt_obj_pos[i], normalize=True)
            is_viewable = prune_pixels(proj_pixels, return_mask=True)
            output_i = {'gt_bboxes': bboxes_projed[is_viewable], 'gt_labels': gt_cls_idx[i][is_viewable]}
            output.append(output_i)

        return output

    def forward_train_geospatial(self, sensor_data, gt, scenario_idx=0, node_str='node_1'):
        proj = self.projs[scenario_idx][node_str]
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
            gt_obj_rot = euler_to_rot_torch(
                obj_roll[i],
                obj_pitch[i],
                obj_yaw[i],
            )
            points_3d = get_3d_box_points(gt_obj_pos[i], gt_obj_rot, obj_dims[i])
            No, Np, _ = points_3d.shape
            points_3d_img = proj.world_to_image(
                    points_3d.reshape(No * Np, 3), normalize=True)
            points_3d_img = points_3d_img.reshape(No, Np, 3)
            bboxes_projed = box3d_points_to_box2d_chw(points_3d_img)

            proj_pixels = proj.world_to_image(gt_obj_pos[i], normalize=True)
            is_viewable = prune_pixels(proj_pixels, return_mask=True)

            loss = detr_coco_loss(
                dets,
                bboxes_projed[is_viewable],
                gt_cls_idx[i][is_viewable],
            )
            output.append(loss)
        return output
    

    def forward_test_tracker(self, seq_batch, scenario=0):
        gt = seq_batch['gt']
        num_frames = len(gt['obj_position'])
        sensor_keys = [k for k in seq_batch.keys() if k != 'gt']

        gt_bboxes = {} 
        for sensor_key in sensor_keys:
            node_idx = sensor_key.split('_')[-1]
            node_str = f'node{node_idx}'
            proj = self.projs[scenario][node_str]
            gt_bboxes[sensor_key] = self.compute_gt_bboxes(gt, proj)

        starting_pos = gt['obj_position'][0][..., 0:2].cuda() #num_objects x 2
        starting_cls = gt['obj_norm_cls_idx'][0]
        classes = self.classes[starting_cls]
        K = len(starting_pos)
        DZ = 4
        Z = self.mm.project_state_X2Z(starting_pos)
        P = torch.diag_embed(torch.ones(K, DZ).cuda())
        P[..., 0] = P[..., 0] / 10
        P[..., 2] = P[..., 2] / 10
        tracker = tracking.multi_object_tracker(Z, P, classes, self.mm, self.dms, self.da, self.um, const_dt=1/15)

        tracker_states = []
        detections = []
        for t in range(num_frames):
            Z, P = tracker.advance(Z, P)
            dets_t = {}
            for sensor_key in sensor_keys:
                node_idx = sensor_key.split('_')[-1]
                node_str = f'node{node_idx}'
                proj = self.projs[scenario][node_str]
                fname = seq_batch[sensor_key]['fnames'][t]
                dets = self.detector(fname)[0]
                dets = self.bbox_pipeline(dets)
                det_means, det_covs = proj.image_to_world_ground_uq(
                    dets.as_scaled().bottoms,
                    z=torch.zeros(len(dets), 1).cuda()
                )
                dets_t[sensor_key] = {
                    'X': det_means[..., 0:2], 
                    'R': det_covs[..., 0:2, 0:2],
                    'bboxes': dets,
                    'fname': fname,
                    'gt_bboxes': gt_bboxes[sensor_key][t]['gt_bboxes'],
                    'proj': proj
                }
            detections.append(dets_t)
            Z, P = tracker.um.update(dets_t, Z, P)
            normals = []
            for z, p in zip(Z, P):
                p = self.mm.project_cov_Z2X(p)
                p = (p + p.transpose(-1, -2)) / 2
                normal = D.MultivariateNormal(
                    self.mm.project_state_Z2X(z).unsqueeze(0),
                    p
                )
                normals.append(normal)
            tracker_states.append(normals)
        return detections, tracker_states

    def forward_train_tracker(self, seq_batch, scenario=0):
        gt = seq_batch['gt']
        num_frames = len(gt['obj_position'])
        sensor_keys = [k for k in seq_batch.keys() if k != 'gt']

        starting_pos = gt['obj_position'][0][..., 0:2].cuda() #num_objects x 2
        second_pos = gt['obj_position'][1][..., 0:2].cuda()
        dt = 1/15
        est_velo = (second_pos - starting_pos) / dt
        starting_cls = gt['obj_norm_cls_idx'][0]
        tracker_classes = self.classes[starting_cls]
        K = len(starting_pos)
        DZ = 4
        Z = self.mm.project_state_X2Z(starting_pos)
        # Z[..., 1] = est_velo[..., 0]
        # Z[..., 3] = est_velo[..., 1]
        P = torch.diag_embed(torch.ones(K, DZ).cuda())
        P[..., 0] = P[..., 0] / 10
        P[..., 2] = P[..., 2] / 10

        tracker = tracking.multi_object_tracker(Z, P, tracker_classes, self.mm, self.dms, self.da, self.um, const_dt=1/15)

        gt_bboxes = {} 
        for sensor_key in sensor_keys:
            node_idx = sensor_key.split('_')[-1]
            node_str = f'node{node_idx}'
            proj = self.projs[scenario][node_str]
            gt_bboxes[sensor_key] = self.compute_gt_bboxes(gt, proj)

        node2dets = {}
        Zs, Ps = [], []
        all_loss_dicts = []
        for t in range(num_frames):
            detr_loss_dicts = []
            Z, P = tracker.advance(Z, P)
            dets_t = {}
            for sensor_key in sensor_keys:
                node_idx = sensor_key.split('_')[-1]
                node_str = f'node{node_idx}'
                proj = self.projs[scenario][node_str]
                fname = seq_batch[sensor_key]['fnames'][t]
                dets = self.detector(fname)[0]
                loss, assign_idx = detr_coco_loss(
                    dets,
                    gt_bboxes[sensor_key][t]['gt_bboxes'],
                    gt_bboxes[sensor_key][t]['gt_labels'],
                    return_assignments=True
                )
                detr_loss_dicts.append(loss)
                
                det_mask = torch.zeros(len(dets), dtype=torch.bool).cuda()
                if len(assign_idx) > 0: #nothing was viewable
                    det_idx = assign_idx[:, 0]
                    det_mask[det_idx] = True
                dets = dets.filter(det_mask)
                # dets = self.bbox_pipeline(dets)
                det_means, det_covs = proj.image_to_world_ground_uq(
                    dets.as_scaled().bottoms,
                    z=torch.zeros(len(dets), 1).cuda()
                )
                dets_t[sensor_key] = {'X': det_means[..., 0:2], 'R': det_covs[..., 0:2, 0:2]}
            loss_keys = detr_loss_dicts[0].keys()
            loss_dict = {k: torch.stack([d[k] for d in detr_loss_dicts]).mean() for k in loss_keys}
            Z, P = tracker.um.update(dets_t, Z, P)
            # normal = D.MultivariateNormal(
                # self.mm.project_state_Z2X(Z),
                # self.mm.project_cov_Z2X(P)
            #)
            gt_pos = gt['obj_position'][t][..., 0:2].cuda()
            nll_loss = 0.0
            for i in range(len(gt_pos)):
                z = self.mm.project_state_Z2X(Z)[i]
                p = self.mm.project_cov_Z2X(P)[i]
                normal = D.MultivariateNormal(z.unsqueeze(0), p.unsqueeze(0))
                neg_log_prob = -normal.log_prob(gt_pos[i])
                nll_loss += neg_log_prob.mean()
            nll_loss = nll_loss / len(gt_pos)

            # neg_log_prob = -normal.log_prob(gt['obj_position'][t][..., 0:2].cuda())
            # nll_loss = neg_log_prob.mean() 
            loss_dict['track_nll'] = nll_loss
            final_loss = 0
            num_gt = K
            for k in loss_dict.keys():
                if num_gt != 0:
                    loss_dict[k] = loss_dict[k] / num_gt 
                final_loss += loss_dict[k]
            loss_dict['loss'] = final_loss
            all_loss_dicts.append(loss_dict)
        return all_loss_dicts


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
