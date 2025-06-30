import cv2
import numpy as np
import torch
import json
from mmengine.registry import TRANSFORMS
import copy


name2cocoid = {
    'car': 2,
    'bus': 5,
    'truck': 7,
}

cocoid2normid = {
    2: 0,
    5: 1,
    7: 2,
}

id2dims = {
    #2: torch.tensor([0.23, 0.12, 0.07]), #car
    2: torch.tensor([0.295, 0.13, 0.08]), #car
    5: torch.tensor([0.30, 0.075, 0.08]), #bus
    7: torch.tensor([0.28, 0.14, 0.15]), #truck
}

id2center_offsets = {
    #2: torch.tensor([-0.295/2+0.04, -0.13/2+0.035, 0.0]), #car
    2: torch.tensor([0.0, 0.0, 0.0]), #car
    5: torch.tensor([0.0, 0.0, 0.0]), #bus
    7: torch.tensor([0.0, 0.0, 0.0]), #truck
}

@TRANSFORMS.register_module()
class ParseMocapGT:
    def __init__(self):
        pass

    def __call__(self, gt):
        output = {
            'obj_position': [],
            'obj_rotation': [],
            'obj_cls_idx': [],
            'obj_norm_cls_idx': [],
            'obj_roll': [],
            'obj_pitch': [],
            'obj_yaw': [],
            'obj_dims': [],
            'obj_center_offsets': [],
            'timestamp': [],
            'obj_id': [],
        }
        for obj_id, obj_data in gt.items():
            try:
                cls_type = obj_data['type']
            except:
                import ipdb; ipdb.set_trace() # noqa
            obj_cls_idx = name2cocoid[cls_type]
            dims = id2dims[obj_cls_idx]

            output['obj_cls_idx'].append(obj_cls_idx)
            output['obj_norm_cls_idx'].append(cocoid2normid[obj_cls_idx])
            output['obj_dims'].append(dims)
            output['obj_id'].append(obj_data['id'] - 1)

            center_offsets = id2center_offsets[obj_cls_idx]
            output['obj_center_offsets'].append(center_offsets)
            position = torch.tensor(obj_data['position']).float()
            position = position / 1000
            position[-1] = 0
            output['obj_position'].append(position)

            rotation = torch.tensor(obj_data['rotation']).float()
            rotation = rotation.reshape(3, 3)
            output['obj_rotation'].append(rotation)

            output['obj_roll'].append(np.deg2rad(obj_data['roll']))
            output['obj_pitch'].append(np.deg2rad(obj_data['pitch']))
            output['obj_yaw'].append(np.deg2rad(obj_data['yaw']))

            output['timestamp'].append(obj_data['time'])
        output['obj_position'] = torch.stack(output['obj_position'], dim=0)
        output['obj_rotation'] = torch.stack(output['obj_rotation'], dim=0)
        output['obj_roll'] = torch.tensor(output['obj_roll']).float()
        output['obj_pitch'] = torch.tensor(output['obj_pitch']).float()
        output['obj_yaw'] = torch.tensor(output['obj_yaw']).float()
        output['obj_cls_idx'] = torch.tensor(output['obj_cls_idx']).long()
        output['obj_norm_cls_idx'] = torch.tensor(output['obj_norm_cls_idx']).long()
        output['obj_dims'] = torch.stack(output['obj_dims'], dim=0)
        output['obj_center_offsets'] = torch.stack(output['obj_center_offsets'], dim=0)
        output['obj_id'] = torch.tensor(output['obj_id']).long()
        output['timestamp'] = torch.tensor(output['timestamp']).long()
        return output

@TRANSFORMS.register_module()
class ParseGQGT:
    def __init__(self):
        pass

    def __call__(self, gt):
        position = [gt['x'], gt['y'], gt['z']]
        position = torch.tensor(position).float().unsqueeze(0)
        obj_cls_idx = torch.tensor([2])
        timestamp = 0
        output = {
            'obj_position': position,
            'obj_cls_idx': torch.tensor([2]),
            'timestamp': torch.tensor([timestamp]),
        }
        return output
