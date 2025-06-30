import probtrack
from mmengine.config import read_base

with read_base():
    from configs.models.proj import *
    from configs.optim import *

num_iters = 10000
subseq_len = 200

loss_weights.update(
    track_nll_loss=0,
    giou_loss=0,
    l1_loss=0,
    ce_loss=0,
)

model.update(
    freeze_proj_stds=True
)


model.detector_cfg.update(
    freeze_depth_head=False,
    freeze_point_head=True,
    freeze_bbox_head=True,
)

