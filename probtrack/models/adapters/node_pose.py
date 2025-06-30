import torch
import torch.nn as nn
from mmengine.registry import MODELS
import torch.nn.functional as F
from probtrack.geometry.distributions import to_torch_dist
import sys
sys.path.append('/home/csamplawski/src/iobtmax-data-tools')
from spatial_transform_utils import *
from scipy.spatial.transform import Rotation as R

@MODELS.register_module()
class NodePoseAdapter(nn.Module):
    def __init__(self,
                 in_len=100,
                 out_len=1,
                 ffn_cfg=dict(type='IsoFFN', dim=256, expansion_factor=1, dropout_rate=0.1),
                 node_pos_ffn_dim=16,
                 pose_dropout_rate=0.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = nn.Sequential(
            nn.Linear(in_len, out_len),
            nn.GELU(),
        )

        self.ffn = nn.GELU()
        if ffn_cfg is not None:
            self.ffn = MODELS.build(ffn_cfg)


        self.pose_ffn = nn.Sequential(
            nn.Linear(7, node_pos_ffn_dim),
            nn.GELU(),
            nn.Linear(node_pos_ffn_dim, node_pos_ffn_dim),
            nn.GELU(),
        )
        self.w_lin = nn.Linear(node_pos_ffn_dim, 256*16)
        self.b_lin = nn.Linear(node_pos_ffn_dim, 16)
        self.register_buffer('min', torch.tensor([-2776.4158, -2485.51226, 0.49277])/1000)
        self.register_buffer('max', torch.tensor([5021.02426, 3147.80981, 1621.69398])/1000)

        self.pose_dropout = nn.Dropout(pose_dropout_rate)

        
    def forward(self, x, node_pos=None):
        if len(x.shape) == 4: #cov feat map
            x = x.flatten(2)
            x = x.permute(0, 2, 1)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        # if self.ffn is not None:
            # x = self.ffn(x)
        x = x.permute(0, 2, 1)
        x = self.selector(x)
        x = x.permute(0, 2, 1) # 1 x 256
        B, N, C = x.shape
        x = x.reshape(B*N, C)
        
        X, Y, Z, dX, dY, dZ, roll, pitch, yaw = node_pos.unbind(dim=-1)
        X = (X - self.min[0]) / (self.max[0] - self.min[0])
        Y = (Y - self.min[1]) / (self.max[1] - self.min[1])
        Z = (Z - self.min[2]) / (self.max[2] - self.min[2])
        rot_mat = euler_to_rot_torch(
            roll,
            pitch,
            yaw
        )[0]
        rot_mat = rot_mat.detach().cpu().numpy()
        rot_mat = R.from_matrix(rot_mat)
        quat = rot_mat.as_quat()
        quat = torch.tensor(quat, device=X.device).float()
        node_pos = torch.cat([X, Y, Z, quat], dim=-1).unsqueeze(0)
        # node_pos = self.pose_dropout(node_pos)
        # node_pos = torch.cat([X, Y, Z, roll, pitch, yaw], dim=-1).unsqueeze(0)
        node_pos = self.pose_ffn(node_pos)
        W = self.w_lin(node_pos).reshape(16, 256)
        b = self.b_lin(node_pos)
        x = F.linear(x, W, b)
        x = x.unsqueeze(1) # B x 1 x 16
        #x = x.permute(0, 2, 1)
        return x
