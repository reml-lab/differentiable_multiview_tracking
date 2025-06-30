import cv2
import numpy as np
import torch
import json
from mmengine.registry import TRANSFORMS, MODELS
import torch.nn as nn
import copy
import torchvision
from probtrack.structs import BBoxDetections


@MODELS.register_module()
class SortDetections(nn.Module):
    def __init__(self, cls_indicies=[80], descending=True, top_k=None):
        super().__init__()
        self.cls_indicies = cls_indicies
        self.descending = descending
        self.top_k = top_k

    def __call__(self, dets):
        # if len(dets) == 0: return []
        cls_probs = dets.cls_probs
        cls_probs = cls_probs[..., self.cls_indicies]
        sort_probs = cls_probs.sum(dim=-1)
        sort_idx = torch.argsort(sort_probs, dim=-1, descending=self.descending)
        return BBoxDetections(
            bboxes_cxcywh=dets.bboxes_cxcywh[sort_idx][:self.top_k],
            cls_logits=dets.cls_logits[sort_idx][:self.top_k],
            embeds=dets.embeds[sort_idx][:self.top_k],
            image_size=dets.image_size
        )


@MODELS.register_module()
class TopK(nn.Module):
    def __init__(self, k=100):
        super().__init__()
        self.k = k

    def __call__(self, dets):
        return dets[0:self.k]

def cluster_bboxes(dets, iou_threshold=0.1):
    bboxes = dets.bboxes_cxcywh
    conf = dets.conf   
    bboxes_xyxy = torchvision.ops.box_convert(bboxes, 'cxcywh', 'xyxy')
    num_bboxes = bboxes.shape[0]
    iou_matrix = torchvision.ops.box_iou(bboxes_xyxy, bboxes_xyxy)
    cluster_assignment = torch.arange(num_bboxes).to(bboxes.device) #every bbox is its own cluster to start
    for i in range(num_bboxes):
        for j in range(i + 1, num_bboxes):
            if iou_matrix[i, j] > iou_threshold: #if iou is above threshold, merge clusters
                cluster_id = min(cluster_assignment[i], cluster_assignment[j])  #assign to the lower cluster id
                cluster_assignment[i] = cluster_assignment[j] = cluster_id #assign both to the same cluster
    cluster_ids = torch.unique(cluster_assignment) #get unique cluster ids
    for i, cluster_id in enumerate(cluster_ids): #reassign cluster ids to be renormalized to 0, 1, 2, ...
        cluster_assignment[cluster_assignment == cluster_id] = i 
    cluster_ids = torch.unique(cluster_assignment)
    cluster_reprs = []
    for cluster_id in cluster_ids:
        cluster_repr_idx = (cluster_assignment == cluster_id).nonzero()#first bbox in the cluster is the representative
        if len(cluster_repr_idx) == 1:
            cluster_repr_idx = cluster_repr_idx[0]
        else:
            #cluster repr is the bbox with the highest confidence
            cluster_repr_idx = torch.argmax(conf[cluster_assignment == cluster_id])
            cluster_repr_idx = cluster_repr_idx.unsqueeze(0)
        cluster_reprs.append(cluster_repr_idx)
        # cluster_mask = cluster_assignment == cluster_id
        # cluster_reprs.append(bboxes[cluster_mask][0]) #TODO: first bbox is the representative (works because sorting?)
    cluster_reprs = torch.stack(cluster_reprs).squeeze(-1)
    return cluster_assignment, cluster_reprs

@MODELS.register_module()
class RefinementClassifier(nn.Module):
    def __init__(self, dim, num_classes=81):
        super().__init__()
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, dets):
        embeds = dets.embeds
        cls_logits = self.classifier(embeds)
        dets.cls_logits = cls_logits
        return dets

@MODELS.register_module()
class ClusterBBoxes(nn.Module):
    def __init__(self, iou_threshold=0.1, return_mode='dets'):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.return_mode = return_mode

    def __call__(self, dets):
        if len(dets) == 0:
            if self.return_mode == 'mask':
                return torch.zeros(0, dtype=torch.bool)
            elif self.return_mode == 'dets':
                return dets
            elif self.return_mode == 'assignments':
                return torch.zeros(0, dtype=torch.long)
        # bboxes = torch.stack([det.bbox for det in dets], dim=0)
        assignment, reprs = cluster_bboxes(dets, iou_threshold=self.iou_threshold)
        
        mask = torch.zeros(len(dets), dtype=torch.bool)
        for r in reprs:
            mask[r] = True
        if self.return_mode == 'mask':
            return mask
        elif self.return_mode == 'dets':
            return dets.filter(mask)
        elif self.return_mode == 'assignments':
            return assignment

@MODELS.register_module()
class ClassThreshold(nn.Module):
    def __init__(self, cls_indicies=[81], threshold=0.5,
            mode='gt'):
        super().__init__()
        self.cls_indicies = cls_indicies
        self.threshold = threshold
        self.mode = mode

    def __call__(self, dets):
        cls_probs = dets.cls_probs
        cls_probs = cls_probs[..., self.cls_indicies]
        cls_probs = cls_probs.sum(dim=-1)
        if self.mode == 'gt':
            keep = cls_probs > self.threshold
        elif self.mode == 'lt':
            keep = cls_probs < self.threshold
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        return dets.filter(keep)
