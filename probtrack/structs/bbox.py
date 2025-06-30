import torch
import cv2
import torchvision

def cxcywh_to_bottom(bboxes):
    return torch.stack([
        bboxes[..., 0],
        bboxes[..., 1] + bboxes[..., 3] / 2,
    ], dim=-1)

def scale_cxcywh(bboxes, H=1080, W=1920):
    return torch.stack([
        bboxes[..., 0] * W,
        bboxes[..., 1] * H,
        bboxes[..., 2] * W,
        bboxes[..., 3] * H,
    ], dim=-1)

def cxcywh_to_xywh(bboxes):
    return torch.stack([
        bboxes[..., 0] - bboxes[..., 2] / 2,
        bboxes[..., 1] - bboxes[..., 3] / 2,
        bboxes[..., 2],
        bboxes[..., 3],
    ], dim=-1)
    # cx, cy, w, h = bbox #in [0,1]
    # x = int((cx - w / 2) * image_size[0])
    # y = int((cy - h / 2) * image_size[1])
    # w = int(w * image_size[0])
    # h = int(h * image_size[1])
    # return x, y, w, h


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush', 'bg'
]

class BBoxDetections(torch.nn.Module):
    def __init__(self, bboxes_cxcywh, cls_logits, embeds, 
            labels=None, pixels=None, depths=None,
            image_size=(1920, 1080)):
        super().__init__()
        self.bboxes_cxcywh = bboxes_cxcywh
        # if len(bbox_cxcywh.shape) == 0:
            # self.bbox_cxcywh = bbox_cxcywh.unsqueeze(0)
        self.cls_logits = cls_logits
        if labels is not None:
            self.labels = labels
        self.embeds = embeds
        self.image_size = image_size

        self.pixels = pixels
        self.depths = depths

        

    def __len__(self):
        return len(self.bboxes_cxcywh)

    def infer_labels(self):
        self.labels = torch.argmax(self.cls_logits, dim=-1)

    def allpairs_iou(self):
        bboxes_xyxy = torchvision.ops.box_convert(self.bboxes_cxcywh, 'cxcywh', 'xyxy')
        iou = torchvision.ops.box_iou(bboxes_xyxy, bboxes_xyxy)
        return iou

    def iou(self, other_dets):
        bboxes_xyxy = torchvision.ops.box_convert(self.bboxes_cxcywh, 'cxcywh', 'xyxy')
        other_bboxes_xyxy = torchvision.ops.box_convert(other_dets.bboxes_cxcywh, 'cxcywh', 'xyxy')
        iou = torchvision.ops.box_iou(bboxes_xyxy, other_bboxes_xyxy)
        return iou

    @property
    def conf(self): #highest non-bg class prob
        non_bg_probs = self.cls_probs[..., 0:80]
        return torch.max(non_bg_probs, dim=-1).values

    def filter(self, mask):
        #if doesnt have labels attr then add it
        if not hasattr(self, 'labels'):
            self.infer_labels()
        return BBoxDetections(
            self.bboxes_cxcywh[mask],
            self.cls_logits[mask],
            self.embeds[mask],
            self.labels[mask],
            self.pixels[mask],
            self.depths[mask],
            self.image_size
        )

    @property
    def bboxes(self):
        return self.bboxes_cxcywh

    @property
    def cls_probs(self):
        return torch.softmax(self.cls_logits, dim=-1)

    @property
    def centers(self):
        return self.bboxes_cxcywh[..., 0:2]

    def as_scaled(self):
        H, W = self.image_size[1], self.image_size[0]
        if not hasattr(self, 'labels'):
            self.infer_labels()
        return BBoxDetections(scale_cxcywh(self.bboxes_cxcywh, H, W), self.cls_logits, self.embeds, self.labels)
    
    @property
    def bottoms(self):
        if self.external_bottom_pixels is not None:
            return self.external_bottom_pixels
        return cxcywh_to_bottom(self.bboxes_cxcywh)

    @property
    def scaled_pixels(self):
        H, W = self.image_size[1], self.image_size[0]
        return self.pixels * torch.tensor([W, H]).to(self.pixels.device)

    def to(self, device):
        if not hasattr(self, 'labels'):
            self.infer_labels()
        return BBoxDetections(
            self.bboxes_cxcywh.to(device),
            self.cls_logits.to(device),
            self.embeds.to(device),
            self.labels.to(device),
            self.pixels.to(device),
            self.depths.to(device),
            self.image_size
        )

    @property
    def area(self):
        area = self.bboxes_cxcywh[..., 2] * self.bboxes_cxcywh[..., 3]
        return area.unsqueeze(-1)
    
    @property
    def coco_annotations(self):
        if not hasattr(self, 'labels'):
            self.infer_labels()
        annos = []
        scaled_bboxes = self.as_scaled().bboxes_cxcywh
        for i, bbox in enumerate(scaled_bboxes):
            x, y, w, h = cxcywh_to_xywh(bbox)
            x, y, w, h = x.item(), y.item(), w.item(), h.item()
            label = self.labels[i].item()
            score = self.cls_probs[i, label].item()
            annos.append({
                'category_id': label,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'score': score,
            })
        return annos

    #tostring?
    def __repr__(self):
        return f"BBoxDetection({self.bboxes_cxcywh})"

    def plot(self, img, color=(0, 255, 0)):
        if not hasattr(self, 'labels'):
            self.infer_labels()
        for i, bbox in enumerate(self.bboxes_cxcywh):
            H, W = img.shape[:2]
            scaled_bbox = scale_cxcywh(bbox, H, W)
            label = self.labels[i].item()
            cx, cy, w, h = scaled_bbox
            x = int(cx - w / 2)
            y = int(cy - h / 2)
            w = int(w)
            h = int(h)
            cls_name = COCO_CLASSES[label]
            img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)
            #add solid color rectangle at top of bbox
            img = cv2.rectangle(img, (x, y-30), (x + w, y), color, -1)
            img = cv2.putText(img, cls_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3, cv2.LINE_AA)
            #plot scaled pixel
            img = cv2.circle(img, (int(self.scaled_pixels[i, 0].item()), int(self.scaled_pixels[i, 1].item())), 5, color, -1)
        return img
