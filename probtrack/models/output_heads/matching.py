import sys
import json
import numpy as np
import lap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from mmengine.registry import MODELS, DATASETS
sys.path.append('/home/csamplawski/src/iobtmax-data-tools')
from spatial_transform_utils import *
# from probtrack.models.backbones.detr import cxcywh_to_bottom
from datetime import datetime
from probtrack.structs import GeospatialDetections, BBoxDetections
from mmengine.dataset import Compose
import cv2
import time
from tqdm import tqdm
import torchvision
from probtrack.metrics import giou


class ScaledSigmoidParameter(nn.Module):
    def __init__(self, shape=(81,2), min_val=-0.01, max_val=0.01):
        super(ScaledSigmoidParameter, self).__init__()
        self.param = nn.Parameter(torch.zeros(shape))
        self.min_val = min_val
        self.max_val = max_val

    def forward(self):
        param = torch.sigmoid(self.param)
        scaled_param = self.min_val + (self.max_val - self.min_val) * param
        return scaled_param


def mse_matching_loss(pred, gt):
    all_pairs_mse = F.pairwise_distance(
        pred.unsqueeze(1),
        gt.unsqueeze(0),
    )
    assign_idx = linear_assignment(all_pairs_mse)
    loss = 0
    for x, y in assign_idx:
        loss += all_pairs_mse[x, y]
    return loss

def prune_pixels(pixels, return_mask=False):
    mask = pixels[..., 0] > 0
    mask = mask & (pixels[..., 0] < 1)
    mask = mask & (pixels[..., 1] > 0)
    mask = mask & (pixels[..., 1] < 1)
    mask = mask & (pixels[..., 2] > 0)
    if return_mask:
        return mask
    return pixels[mask]

def linear_assignment(cost_matrix):
    if cost_matrix.shape[0] == 0 or cost_matrix.shape[1] == 0:
        return []
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

@MODELS.register_module()
class MatchingOutputHead(nn.Module):
    def __init__(self,
                 dim=256,
                 loss_weights={
                    'pixel_mse_loss': 1.0,
                    'obj_mse_loss': 0.0,
                    'ce_loss': 0.0,
                 },
                 scenarios=None,
                 freeze_projs=False,
                 freeze_proj_stds=False,
                 cls_filter_during_training=False,
                 filter_by='vehicle_probs',
                 temp=1.0,
                 gt_type='mocap',
                 bg_threshold=0.5,
                 server=None,
                 ): 
        super().__init__()
        self.loss_weights = loss_weights
        self.temp = temp
        self.cls_filter_during_training = cls_filter_during_training
        self.bg_threshold = bg_threshold

        self.filter_by = filter_by

        self.server = DATASETS.build(server)
        self.clustering = MODELS.build(dict(type='ClusterBBoxes', iou_threshold=0.1, return_mode='assignments'))
        # self.refine_cls_head = nn.Linear(256, 81)

        self.init_weights()
        self.projs = nn.ModuleDict()
        correction_date = datetime.strptime('2023-05-14', '%Y-%m-%d')
        for scenario in tqdm(scenarios, desc='Initializing projectors'):
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

        for name, param in self.named_parameters():
            if 'std' in name:
                param.requires_grad = not freeze_proj_stds

        def inverse_sigmoid(x):
            return torch.log(x / (1 - x))
        
        #sigmoid(-18) = 1e-8
        # self.obj_dim_offsets = ScaledSigmoidParameter((81, 2), -0.05, 0.05)
        # self.obj_center_offsets = ScaledSigmoidParameter((81, 2), -0.05, 0.05)

    

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight, gain=2)
    
    def warmup(self):
        rand_pixels = torch.rand(100, 2).cuda()
        rand_z = torch.rand(100, 1).cuda()
        for scenario in tqdm(self.projs.keys(), desc='Warming up projectors'):
            for node in self.projs[scenario]:
                proj = self.projs[scenario][node]
                mean, cov = proj.image_to_world_ground_uq(
                    rand_pixels,
                    z=rand_z
                )

    @torch.no_grad()
    def forward_test(self, bbox_dets, gt, scenario_idx=0, node_str='node_1'):
        proj = self.projs[scenario_idx][node_str]
        bbox_dets = [dets.to('cpu') for dets in bbox_dets]
        return [bbox_dets, proj]

    def forward_train_coco(self, dets, gt_bbox, gt_labels):
        num_gt = len(gt_bbox)
        N, nC = dets.cls_logits.shape
        giou_cost = giou(dets.bboxes_cxcywh, gt_bbox)
        l1_cost = F.pairwise_distance(
            dets.bboxes_cxcywh.unsqueeze(1),
            gt_bbox.unsqueeze(0),
            p=1
        )

        expanded_labels = gt_labels.unsqueeze(-1).expand(-1, N) #num_gt x N
        true_cls_logits = dets.cls_logits.gather(1, expanded_labels)
    
        import ipdb; ipdb.set_trace() # noqa

    def get_matches(self, bbox_dets, gt, scenario_idx=0, node_str='node_1', bg_threshold=0.0):
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
            bboxes = bbox_dets[i]
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
            projed_area = bboxes_projed[..., 2] * bboxes_projed[..., 3]
            projed_area = projed_area.unsqueeze(-1)


            proj_pixels = proj.world_to_image(gt_obj_pos[i], normalize=True)
            is_viewable = prune_pixels(proj_pixels, return_mask=True)

            means, _ = proj.image_to_world_ground_uq(
                bboxes.bottoms,
                z=torch.zeros(len(bboxes), 1).to(proj_pixels.device)
            )

            h_abs = F.pairwise_distance(
                bboxes.bboxes_cxcywh[..., 2:3].unsqueeze(1),
                bboxes_projed[is_viewable, 2:3].unsqueeze(0),
                p=1
            )

            w_abs = F.pairwise_distance(
                bboxes.bboxes_cxcywh[..., 3:4].unsqueeze(1),
                bboxes_projed[is_viewable, 3:4].unsqueeze(0),
                p=1
            )

            pixel_mse = F.pairwise_distance(
                bboxes.bottoms.unsqueeze(1),
                proj_pixels[is_viewable, 0:2].unsqueeze(0),
            )
            world_mse = F.pairwise_distance(
                means.unsqueeze(1),
                gt_obj_pos[i][is_viewable].unsqueeze(0),
            )

            cls_probs = bboxes.cls_probs
            pred_labels = cls_probs.argmax(dim=-1)
            #bg_most_likely = pred_labels == 80
            mask = cls_probs[..., -1] < bg_threshold
            # mask = mask & ~bg_most_likely
            cost = self.loss_weights['pixel_mse_loss']  * pixel_mse
            cost += self.loss_weights['obj_mse_loss'] * world_mse
            cost += w_abs + h_abs
            assign_idx = linear_assignment(cost)
            assigned_mask = torch.zeros_like(mask)
            filter_assign_idx = []
            for pred_idx, gt_idx in assign_idx:
                if world_mse[pred_idx, gt_idx] > 1e8:
                    continue
                if pixel_mse[pred_idx, gt_idx] > 0.05:
                    continue
                if h_abs[pred_idx, gt_idx] > 0.01:
                    continue
                if w_abs[pred_idx, gt_idx] > 0.01:
                    continue
                filter_assign_idx.append([pred_idx, gt_idx])
                mask[pred_idx] = True
                pred_labels[pred_idx] = gt_cls_idx[i][is_viewable][gt_idx]
                assigned_mask[pred_idx] = 1

            
            #mask of idx that were assigned
            # assigned_mask[[x[0] for x in filter_assign_idx]] = 1

            # nothing matched so mask everything so we dont use this frame
            if len(filter_assign_idx) == 0:
                mask = torch.zeros_like(mask)
            
            num_viewable = is_viewable.sum().item()
            if len(filter_assign_idx) != num_viewable:
                mask = torch.zeros_like(mask)
            
            #filter bboxes but remember their original indices
            orig_indices = torch.arange(len(bboxes)).to(device)
            selected_bboxes = bboxes.filter(mask)
            selected_bboxes.labels = pred_labels[mask]
            selected_indices = orig_indices[mask]
            
            #no bboxes left, so stop here
            if len(selected_bboxes) == 0:
                output.append(None)
                continue

            #cluster remaining bboxes
            cluster_assignments = self.clustering(selected_bboxes)
            num_clusters = len(torch.unique(cluster_assignments))
            
            #get cluster representatives
            #if a cluster element has been assigned above then it is the repr
            #otherwise, the first element is the repr
            cluster_reprs = []
            for cluster_id in range(num_clusters):
                cluster_elements = (cluster_assignments == cluster_id).nonzero().squeeze(-1)
                cluster_elements_idx = selected_indices[cluster_elements]
                cluster_repr = cluster_elements_idx[0]
                for cei in cluster_elements_idx:
                    if assigned_mask[cei]:
                        cluster_repr = cei
                        break
                cluster_reprs.append(cluster_repr)
            cluster_reprs = torch.stack(cluster_reprs).squeeze(-1)
            final_mask = torch.zeros_like(mask)
            final_mask[cluster_reprs] = 1
            selected_bboxes = bboxes.filter(final_mask)
            selected_bboxes.labels = pred_labels[final_mask]
            
            filter_assign_idx = torch.tensor(filter_assign_idx).to(device)
            selected_proj_boxes = bboxes_projed[is_viewable][filter_assign_idx[:, 1]]
            fake_det = BBoxDetections(
                selected_proj_boxes,
                torch.zeros(len(selected_proj_boxes), 81).to(device),
                torch.zeros(len(selected_proj_boxes), 256).to(device),
                torch.tensor([1920, 1080]).to(device)
            )
            if len(fake_det) != len(selected_bboxes):
                output.append(None)
                continue
            output.append([selected_bboxes, fake_det])
        return output
                

    def forward_train(self, bbox_dets, gt, depth=None, scenario_idx=0, node_str='node_1'):
        # self.proj.setup(scenario_idx, node_str)
        device = gt['obj_position'].device
        B, N, _ = gt['obj_position'].shape
        loss_dict = {
            'ce_loss': torch.tensor(0).to(device).float(),
            'obj_mse_loss': torch.tensor(0).to(device).float(),
            'pixel_mse_loss': torch.tensor(0).to(device).float(),
            'nll_loss': torch.tensor(0).to(device).float()
        }
        gt_obj_pos = gt['obj_position'].cuda()
        gt_cls_idx = gt['obj_cls_idx'].cuda() #B x N

        #look up params by cls idx
        # obj_dim_offsets = self.obj_dim_offsets()[gt_cls_idx]
        # obj_center_offsets = self.obj_center_offsets()[gt_cls_idx]
        
        # gt['obj_dims'][..., 0:2] += obj_dim_offsets
        # gt['obj_position'][..., 0:2] += obj_center_offsets

        matches = self.get_matches(bbox_dets, gt, scenario_idx, node_str, bg_threshold=0.0)
        
        num_matches = 0
        for b in range(len(matches)):
            if matches[b] is None:
                continue
            selected_bboxes, proj_dets = matches[b]
            iou = selected_bboxes.iou(proj_dets)
            iou_loss = 1 - giou(selected_bboxes.bboxes_cxcywh, proj_dets.bboxes_cxcywh)
            loss_dict['pixel_mse_loss'] += iou_loss.mean()
            num_matches += 1
        if num_matches == 0:
            print('No matches found')
            return loss_dict
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / num_matches 
        return loss_dict

        

        # proj = self.projs[scenario_idx][node_str]

        gt_obj_pos = gt['obj_position'].cuda()
        gt_cls_idx = gt['obj_cls_idx'].cuda()

        # B, N, _ = gt_obj_pos.shape
        # gt_obj_pos = gt_obj_pos.reshape(B*N, 3)
        # gt_proj_pixels = proj.world_to_image(gt_obj_pos)
        # gt_obj_pos = gt_obj_pos.reshape(B, N, 3)
        # gt_proj_pixels = gt_proj_pixels.reshape(B, N, 3)
        # gt_norm_proj_pixels = gt_proj_pixels / torch.tensor([1920, 1080, 1]).to(gt_proj_pixels.device).unsqueeze(0).float()
        # gt_cls_idx = gt['obj_cls_idx']

        device = gt_obj_pos.device 
        B, N, _ = gt_obj_pos.shape
        for i in range(B):
            gdets = GeospatialDetections(bbox_dets[i], proj).to('cuda')
            is_viewable = gdets.is_viewable(gt_obj_pos[i])
            proj_pixels = gdets.world_to_image(gt_obj_pos[i], normalize=True)
            means = [dist.component_distribution.mean for dist in gdets.dists]
            if len(means) == 0:
                continue

            means = torch.cat(means, dim=0)

            # pixel_mse = F.pairwise_distance(
                # cxcywh_to_bottom(bboxes).unsqueeze(1),
                # gt_norm_proj_pixels[i][prune_mask][..., 0:2].unsqueeze(0),
            # )
            pixel_mse = F.pairwise_distance(
                gdets.bboxes.bottoms.unsqueeze(1),
                proj_pixels[is_viewable, 0:2].unsqueeze(0),
            )

            world_mse = F.pairwise_distance(
                means.unsqueeze(1),
                gt_obj_pos[i][is_viewable].unsqueeze(0),
            )

            cost = loss_dict['pixel_mse_loss']  * pixel_mse 
            cost += loss_dict['obj_mse_loss'] * world_mse
            # cost += (1 - x['vehicle_probs'][i][mask]).unsqueeze(-1)
            assign_idx = linear_assignment(cost)
            
            num_slots = len(gdets)
            ce_targets = torch.zeros(num_slots).to(device, dtype=torch.float)
            ce_targets += 80

            embeds = gdets.bboxes.embeds
            # cls_logits = self.refine_cls_head(embeds)
            cls_logits = gdets.bboxes.cls_logits
            for pred_idx, gt_idx in assign_idx:
                loss_dict['obj_mse_loss'] += world_mse[pred_idx, gt_idx]
                loss_dict['pixel_mse_loss'] += pixel_mse[pred_idx, gt_idx]

                if len(gt_cls_idx[i][is_viewable]) != 0:
                    ce_targets[pred_idx] = gt_cls_idx[i][is_viewable][gt_idx]

                if world_mse[pred_idx, gt_idx] > 0.5:
                    ce_targets[pred_idx] = 80

                if self.loss_weights['nll_loss'] > 0 and len(gt_cls_idx[i][is_viewable]) != 0:
                    import ipdb; ipdb.set_trace() # noqa
                    dist = D.MultivariateNormal(pred_mean[pred_idx], pred_covs[pred_idx])
                    nll = -dist.log_prob(gt_obj_pos[i][prune_mask][gt_idx])
                    loss_dict['nll_loss'] += nll

             

            # matched_to_obj = (ce_targets != 80).nonzero().squeeze(-1)

            # for pred_idx, gt_idx in assign_idx:
            # for idx in matched_to_obj:
                # import ipdb; ipdb.set_trace() # noqa
                # if world_mse[idx]
                    # ce_targets[idx] = 80
            
            cls_weights = torch.ones(81).to(device)
            cls_weights[-1] = 0.1
            if self.loss_weights['ce_loss'] > 0 and len(ce_targets) > 0:
                ce_loss = F.cross_entropy(cls_logits, ce_targets.long(), reduction='mean', weight=cls_weights)
                # Z = len(assign_idx) + 0.1 * (num_slots - len(assign_idx))
                # if Z != 0:
                    # loss_dict['ce_loss'] += ce_loss.sum() / Z
                loss_dict['ce_loss'] += ce_loss
        for k in loss_dict:
            loss_dict[k] = loss_dict[k] / B / N * self.loss_weights[k]
        return loss_dict
