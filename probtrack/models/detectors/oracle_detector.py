import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from mmengine.registry import MODELS
from probtrack.geometry.distributions import rotate_dist, scale_dist, shift_dist

@MODELS.register_module()
class OracleDetector(nn.Module):
    def __init__(self, cov_scale=1e-4):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        cov = torch.eye(3) * cov_scale
        cov = cov.unsqueeze(0)
        self.register_buffer('cov', cov)

    def forward_train(self, data, node_idx=0):
        return {'loss': self.dummy_param.mean()}

    def forward_test(self, data, node_idx=0):
        
        mocap_data = data['mocap']
        pos = mocap_data['normalized_local_location']['obj_position'][:, node_idx]
        
        node_pos = mocap_data['normalized_location']['node_position'][:, node_idx]
        node_rot = mocap_data['location']['node_rotation'][:, node_idx]

        
        out = []
        for obj_pos in pos:
            O, _ = obj_pos.shape
            cov = self.cov.expand(O, 3, 3)
            probs = cov.new_ones(O)
            normals = D.MultivariateNormal(obj_pos, cov)
            mix = D.Categorical(probs=probs)
            dist = D.MixtureSameFamily(mix, normals)
            dist = rotate_dist(dist, node_rot[0][0])
            dist = shift_dist(dist, node_pos[0][0])
            out.append(dist)
        return out
        # sensor_data = data['sensor_data']
        # dists = self.forward(sensor_data)

        # node_pos = mocap_data['normalized_location']['node_position'][:, node_idx]
        # node_rot = mocap_data['location']['node_rotation'][:, node_idx]
        # global_dists = []
        # for i, dist in enumerate(dists):
            # dist = rotate_dist(dist, node_rot[i][0])
            # dist = shift_dist(dist, node_pos[i][0])
            # global_dists.append(dist)
        # return global_dists
