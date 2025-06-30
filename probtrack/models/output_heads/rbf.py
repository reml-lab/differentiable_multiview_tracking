import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS

def generate_grid(H, W):
    x = torch.linspace(0, 1, steps=W, dtype=torch.float32)
    y = torch.linspace(0, 1, steps=H, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(x, y), dim=-1)
    return grid

@MODELS.register_module()
class GridRBF(nn.Module):
    def __init__(self, grid_size=(20, 20), temp=1, prior=1, predict_from_state=False):
        super().__init__()
        self.grid_size = grid_size
        grid = generate_grid(*grid_size)
        self.register_buffer("grid", grid)
        if predict_from_state:
            self.depth_map_predictor = nn.Sequential(
                nn.Linear(6, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, grid_size[0] * grid_size[1]),
            )
            self.depth_map = None
        else:
            depth_map = torch.zeros(grid_size) + prior
            self.depth_map = nn.Parameter(depth_map.float())
        self.temp = temp

    def forward(self, points_image, state=None):
        N = points_image.shape[0]

        flat_grid = self.grid.reshape(-1, 2)
        all_pairs_mse = F.pairwise_distance(
            flat_grid.unsqueeze(0), 
            points_image.unsqueeze(1)
        )
        probs = torch.softmax(-all_pairs_mse / self.temp, dim=-1)
        probs = probs.reshape(N, self.grid_size[0], self.grid_size[1])

        if state is not None and self.depth_map_predictor is not None:
            depth_map = self.depth_map_predictor(state)
            depth_map = depth_map.reshape(N, self.grid_size[0], self.grid_size[1])
        else:
            depth_map = self.depth_map.unsqueeze(0).expand(N, -1, -1)

        pred_depth = probs * depth_map
        pred_depth = pred_depth.sum(dim=(1, 2))
        pred_depth = pred_depth.unsqueeze(-1)
        return pred_depth
