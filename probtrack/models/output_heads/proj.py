import sys
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from mmengine.registry import MODELS
import torch.nn.functional as F
from probtrack.geometry.distributions import to_torch_dist
from probtrack.datasets.coco import detr_coco_loss
from probtrack.models.output_heads.matching import linear_assignment, prune_pixels
from spatial_transform_utils import *

def compute_gt_bboxes(gt, proj):
    gt_obj_pos = gt['obj_position'].cuda()
    gt_cls_idx = gt['obj_cls_idx'].cuda()
    obj_roll = gt['obj_roll'].cuda()
    obj_pitch = gt['obj_pitch'].cuda()
    obj_yaw = gt['obj_yaw'].cuda()
    obj_dims = gt['obj_dims'].cuda()
    device = gt_obj_pos.device
    N, _ = gt_obj_pos.shape
    
    gt_obj_rot = euler_to_rot_torch(
        obj_roll,
        obj_pitch,
        obj_yaw,
    )
    points_3d = get_3d_box_points(gt_obj_pos, gt_obj_rot, obj_dims)
    No, Np, _ = points_3d.shape
    points_3d_img = proj.world_to_image(
            points_3d.reshape(No * Np, 3), normalize=True)
    points_3d_img = points_3d_img.reshape(No, Np, 3)
    bboxes_projed = box3d_points_to_box2d_chw(points_3d_img)

    proj_pixels = proj.world_to_image(gt_obj_pos, normalize=True)
    is_viewable = prune_pixels(proj_pixels, return_mask=True)
    output_i = {'gt_bboxes': bboxes_projed[is_viewable], 'gt_labels': gt_cls_idx[is_viewable],
            'is_viewable': is_viewable, 'proj_pixels': proj_pixels.cpu()}
    return output_i

@MODELS.register_module()
class ProjOutputHead(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward_test(self, dets, proj, **kwargs):
        det_means, det_covs = proj.image_to_world_ground_uq(
            #dets.as_scaled().bottoms,
            dets.scaled_pixels,
            z=torch.zeros(len(dets), 1).cuda(),
            # d=dets.depths.unsqueeze(-1)
        )
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        return det_normals, dets

    def forward_train_no_matching(self, dets, proj, gt, bbox_pipeline, **kwargs):
        dets = bbox_pipeline(dets)
        det_means, det_covs = proj.image_to_world_ground_uq(
            dets.scaled_pixels,
            z=torch.zeros(len(dets), 1).cuda()
        )
        detection_loss = {
            'det_nll_loss': torch.zeros(1).cuda(),
            'giou_loss': torch.zeros(1).cuda(),
            'l1_loss': torch.zeros(1).cuda(),
            'ce_loss': torch.zeros(1).cuda(),
        }
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        return detection_loss, det_normals, dets


    def forward_train(self, dets, proj, gt, **kwargs):
        det_gt = compute_gt_bboxes(gt, proj)
        is_viewable = det_gt['is_viewable']
        detr_loss, assign_idx = detr_coco_loss(
            dets,
            det_gt['gt_bboxes'],
            det_gt['gt_labels'],
            return_assignments=True,
            enforce_iou=True,
            bg_weight=1,
            #vehicle_prob_threshold=0.5
        )
        pixels = dets.scaled_pixels #100x2
        assigned_pixels = []
        if len(assign_idx) > 0:
            for pred_idx, gt_idx in assign_idx:
                assigned_pixels.append(pixels[pred_idx])
            assigned_pixels = torch.stack(assigned_pixels)
        if len(assigned_pixels) == 0:
            assigned_pixels = torch.zeros(0, 2).cuda()

        det_mask = torch.zeros(len(dets), dtype=torch.bool).cuda()
        if len(assign_idx) > 0: #make sure something was viewable
            for pred_idx, gt_idx in assign_idx:
                det_mask[pred_idx] = True
        dets = dets.filter(det_mask)
        det_means, det_covs = proj.image_to_world_ground_uq(
            # dets.as_scaled().bottoms,
            # dets.scaled_pixels,
            assigned_pixels,
            z=torch.zeros(len(dets), 1).cuda(),
            # d=dets.depths.unsqueeze(-1)
        )
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        det_nll_loss = torch.zeros(1).cuda()
        for j, (pred_idx, gt_idx) in enumerate(assign_idx):
            gt_pos = gt['obj_position'].cuda()[is_viewable][gt_idx][0:2]
            nll_loss_j = -det_normals[j].log_prob(gt_pos).mean()
            mse_loss_j = F.mse_loss(det_means[j][0:2], gt_pos)
            det_nll_loss += nll_loss_j + mse_loss_j
        if len(assign_idx) > 0:
            det_nll_loss /= len(assign_idx)
        detr_loss['det_nll_loss'] = det_nll_loss
        return detr_loss, det_normals, dets
