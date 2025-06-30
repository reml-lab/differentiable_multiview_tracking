import json
import probtrack
from mmengine.registry import MODELS, DATASETS
import pycocotools.coco as coco
import pycocotools.cocoeval as cocoeval
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from tqdm import trange, tqdm
import torchvision
from probtrack.metrics import giou
import torch.nn.functional as F
from probtrack.models.output_heads.matching import linear_assignment

def detr_coco_loss(dets, gt_bboxes, gt_labels, proj=None, gt_pos=None, return_assignments=False,
        enforce_iou=False, bg_weight=0.1, vehicle_prob_threshold=0.0):
    loss_dict = {
        'giou_loss': torch.tensor(0.0).to(gt_bboxes.device),
        'l1_loss': torch.tensor(0.0).to(gt_bboxes.device),
        'ce_loss': torch.tensor(0.0).to(gt_bboxes.device),
    }
    num_gt = len(gt_bboxes)
    N, nC = dets.cls_logits.shape
    cls_logits = dets.cls_logits
    cls_probs = dets.cls_probs
    vehicle_prob = cls_probs[:, [2,5,7,8]].sum(-1)
    ce_targets = cls_logits.new_zeros(N).long() + 80 #default to background
    assign_idx = [] #passthrough
    new_assign_idx = []

    if num_gt > 0: # if nothing viewable, everything is background (mocap)
        giou_matrix, iou = giou(dets.bboxes_cxcywh, gt_bboxes)
        giou_matrix = 1 - giou_matrix
        l1_matrix = F.pairwise_distance(
            dets.bboxes_cxcywh.unsqueeze(1),
            gt_bboxes.unsqueeze(0),
            p=1
        )
        expanded_labels = gt_labels.unsqueeze(0).expand(N, -1) # N x num_gt
        true_cls_logits = torch.gather(cls_logits, 1, expanded_labels)

        ce_matrix = -true_cls_logits + torch.logsumexp(cls_logits, dim=-1, keepdim=True)

        cost_matrix = 2 * giou_matrix + 5 * l1_matrix + 1 * ce_matrix
        assign_idx = linear_assignment(cost_matrix)
        
        for pred_idx, gt_idx in assign_idx:
            if proj is not None:
                bottoms = dets.as_scaled().bottoms[pred_idx].unsqueeze(0)
                det_bboxes = dets.bboxes_cxcywh[pred_idx].unsqueeze(0)
                h_abs = F.pairwise_distance(
                    det_bboxes[..., 2:3].unsqueeze(1),
                    gt_bboxes[gt_idx, 2:3].unsqueeze(0).unsqueeze(0),
                )
                if h_abs.item() > 0.01:
                    continue
                w_abs = F.pairwise_distance(
                    det_bboxes[..., 3:4].unsqueeze(1),
                    gt_bboxes[gt_idx, 3:4].unsqueeze(0).unsqueeze(0),
                )
                if w_abs.item() > 0.01:
                    continue
                # det_means, det_covs = proj.image_to_world_ground_uq(
                    # bottoms,
                    # z=torch.zeros(len(bottoms), 1).to(bottoms.device)
                # )
                # mse = F.mse_loss(det_means.squeeze(0), gt_pos[gt_idx])
                # if mse > 0.3:
                    # continue
            if vehicle_prob[pred_idx] < vehicle_prob_threshold:
                continue
            if iou[pred_idx, gt_idx] == 0.0 and enforce_iou:
                continue
            ce_targets[pred_idx] = gt_labels[gt_idx]
            loss_dict['giou_loss'] += 2 * giou_matrix[pred_idx, gt_idx]
            loss_dict['l1_loss'] += 5 * l1_matrix[pred_idx, gt_idx]
            new_assign_idx.append((pred_idx, gt_idx))
    
    cls_weights = cls_logits.new_ones(nC)
    cls_weights[-1] = bg_weight
    loss_dict['ce_loss'] = F.cross_entropy(
        cls_logits, ce_targets, 
        reduction='mean',
        weight=cls_weights
    )
    
    # final_loss = 0
    for k in loss_dict.keys():
        if num_gt != 0:
            loss_dict[k] = loss_dict[k] / num_gt
        # final_loss += loss_dict[k]
    # loss_dict['loss'] = final_loss
    if return_assignments:
        return loss_dict, new_assign_idx
    return loss_dict



def stats2dict(stats):
    return {
        'AP': stats[0],
        'AP50': stats[1],
        'AP75': stats[2],
        'APs': stats[3],
        'APm': stats[4],
        'APl': stats[5],
        'AR1': stats[6],
        'AR10': stats[7],
        'AR100': stats[8],
        'ARs': stats[9],
        'ARm': stats[10],
        'ARl': stats[11],
    }


@DATASETS.register_module()
class COCODataset(Dataset):
    def __init__(self,
            data_root='/home/csamplawski/data/coco',
            split='val2017',
            **kwargs):
        self.data_root = data_root
        self.split = split
        self.annotation_file = f'{data_root}/annotations/instances_{split}.json'
        self.coco = coco.COCO(self.annotation_file)
        self.cat_ids = self.coco.getCatIds()
        self.reverse_cat_ids = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

    def evaluate(self, resFile):
        cocoDt = self.coco.loadRes(resFile)
        cocoEval = cocoeval.COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def __len__(self):
        return len(self.img_ids)

    def postprocess(self, dets, img_info):
        cls_probs = dets.cls_probs
        scores, det_labels = cls_probs[..., :-1].max(-1)
        pred_bbox = dets.bboxes_cxcywh #100 x 4
        H, W = img_info['height'], img_info['width']
        #img_shape = (H, W)
        det_bboxes = torchvision.ops.box_convert(dets.bboxes_cxcywh, 'cxcywh', 'xyxy')
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * W
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * H
        det_bboxes[:, 0::2].clamp_(min=0, max=W)
        det_bboxes[:, 1::2].clamp_(min=0, max=H)
        det_bboxes = torchvision.ops.box_convert(det_bboxes, 'xyxy', 'xywh')
        det_bboxes = det_bboxes.cpu().numpy()

        preds = []
        for i in range(len(dets)):
            label = det_labels[i].item()
            preds.append({
                'image_id': img_info['id'],
                'category_id': self.cat_ids[label],
                'bbox': det_bboxes[i].tolist(),
                'score': scores[i].item()
            })
        return preds


    def evaluate(self, detector):
        outputs = []
        for i in trange(len(self)):
            output = self[i]
            if output is None:
                continue
            img_path = output['img_path']
            img_info = output['img_info']
            dets = detector(img_path)[0]
            preds = self.postprocess(dets, img_info)
            outputs += preds

        resFile = '/tmp/coco_preds.json'
        with open(resFile, 'w') as f:
            json.dump(outputs, f)
        cocoDt = self.coco.loadRes(resFile)
        cocoEval = cocoeval.COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        results = {
            'all': stats2dict(cocoEval.stats),
        }
        for catid in self.cat_ids:
            cat_name = self.coco.cats[catid]['name']
            cocoEval.params.catIds = [catid]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            stats = cocoEval.stats
            results[cat_name] = stats2dict(stats)
        return results
        # cocoEval.summarize()

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        H, W = img_info['height'], img_info['width']
        anns = self.coco.imgToAnns[img_id]
        if len(anns) == 0:
            return None
        img_path = os.path.join(self.data_root, self.split, img_info['file_name'])
        

        gt_bboxes = [ann['bbox'] for ann in anns]
        gt_labels = [ann['category_id'] for ann in anns]
        gt_labels = [self.reverse_cat_ids[label] for label in gt_labels]
        

        gt_bboxes = torch.tensor(gt_bboxes).float()
        gt_bboxes = torchvision.ops.box_convert(gt_bboxes, 'xywh', 'cxcywh')
        gt_bboxes[:, 0::2] = gt_bboxes[:, 0::2] / W
        gt_bboxes[:, 1::2] = gt_bboxes[:, 1::2] / H

        gt_labels = torch.tensor(gt_labels).long()
        output = {
            'gt_bboxes': gt_bboxes,
            'gt_labels': gt_labels,
            'img_info': img_info,
            'img_path': img_path
        }
        return output
