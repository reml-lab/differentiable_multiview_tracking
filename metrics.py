import torch
import probtrack
import numpy as np
import json
from tqdm import tqdm, trange
from mmengine.config import Config 
from mmengine.registry import MODELS
import probtrack.datasets.utils as dutils
# from probtrack.geometry.distributions import reduce_dim, threshold_dist, scale_dist, shift_dist
from probtrack.models.output_heads.matching import linear_assignment

def rmse(x, y):
    return torch.sqrt(((x - y) ** 2).mean())

#dets list of detections (len need not be N)
#gt_obj_pos: N x 2
def compute_frame_metrics(dists, pred_labels, gt_obj_pos, gt_labels, is_viewable):
    nll_matched = torch.tensor(0.).cuda()
    rmse_matched = torch.tensor(0.).cuda()
    acc_matched = torch.tensor(0.).cuda()
    num_matched_dets = num_matched_gt = 0
    
    # is_viewable = dets.is_viewable(gt_obj_pos)
    num_gt = len(gt_obj_pos)
    gt_obj_pos = gt_obj_pos[is_viewable]
    # dists = [dist.component_distribution for dist in dets.dists]

    # num_matched_gt = num_unmatched_gt = 0
    # is_viewable = gt['is_viewable']
    # gt_obj_pos = dets['obj_pos'][is_viewable]
    # dists = [dist.component_distribution for dist in dets['dists']]
    
    nll_matrix = torch.zeros(len(dists), len(gt_obj_pos)).cuda()
    rmse_matrix = torch.zeros(len(dists), len(gt_obj_pos)).cuda()
    for d, dist in enumerate(dists):
        for o, obj_pos in enumerate(gt_obj_pos):
            nll_matrix[d, o] = -dist.log_prob(obj_pos)
            rmse_matrix[d, o] = rmse(dist.mean, obj_pos)

    assign_idx = linear_assignment(rmse_matrix)
    for a, b in assign_idx:
        # if rmse_matrix[a, b] > rmse_thresh:
            # continue
        nll_matched += nll_matrix[a, b]
        rmse_matched += rmse_matrix[a, b]
        num_matched_dets += 1
        num_matched_gt += 1
        
        acc_matched += int(pred_labels[a] == gt_labels[b])

    num_unmatched_dets = len(dists) - num_matched_dets
    num_unmatched_gt = len(gt_obj_pos) - num_matched_dets
    if num_matched_dets > 0:
        nll_matched /= num_matched_dets
        rmse_matched /= num_matched_dets
        acc_matched /= num_matched_dets
    num_viewable_gt = len(gt_obj_pos)
    if num_viewable_gt == 0:
        precision = 0 if num_matched_dets > 0 else 1 
        recall = 1
    else:
        if (num_matched_dets + num_unmatched_dets) > 0:
            precision = num_matched_dets / (num_matched_dets + num_unmatched_dets)
        else:
            precision = 0 # if no detections and something is viewable, then precision is 0
        recall = num_matched_gt / num_viewable_gt

    demon = precision + recall
    if demon == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / demon
    output = {
        'nll_matched': nll_matched.item() if num_matched_dets > 0 else np.nan,
        'rmse_matched': rmse_matched.item() if num_matched_dets > 0 else np.nan,
        'acc_matched': acc_matched.item() if num_matched_dets > 0 else np.nan,
        'num_unmatched_dets': num_unmatched_dets,
        'num_matched_dets': num_matched_dets,
        'num_unmatched_gt': num_unmatched_gt,
        'num_matched_gt': num_matched_gt,
        'num_viewable_gt': num_viewable_gt,
        'num_gt': num_gt,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return output
