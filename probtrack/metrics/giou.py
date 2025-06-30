import torch
def giou(boxes1, boxes2):
    # Convert boxes from cx, cy, w, h to x1, y1, x2, y2
    boxes1 = torch.cat((boxes1[:, :2] - boxes1[:, 2:] / 2, boxes1[:, :2] + boxes1[:, 2:] / 2), dim=1)
    boxes2 = torch.cat((boxes2[:, :2] - boxes2[:, 2:] / 2, boxes2[:, :2] + boxes2[:, 2:] / 2), dim=1)

    # Intersection
    inter_min = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_max = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter = (inter_max - inter_min).clamp(0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]

    # Union
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1[:, None] + area2 - inter_area

    # IoU
    iou = inter_area / union_area

    # Enclosing box
    enc_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enc_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enc = (enc_max - enc_min).clamp(0)
    enc_area = enc[:, :, 0] * enc[:, :, 1]

    # GIoU
    giou = iou - (enc_area - union_area) / enc_area

    return giou, iou
