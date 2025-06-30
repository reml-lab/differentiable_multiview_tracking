import probtrack
from mmengine.config import read_base

with read_base():
    from configs.models.uai import *
    from configs.optim import *

num_iters = 50

loss_weights.update(
    track_nll_loss=0,
    giou_loss=0,
    l1_loss=0,
    ce_loss=0,
)

model.output_head_cfg.update(
    freeze_adapter=False,
)

model.output_head_cfg.predictor_cfg.update(
    freeze_ffn=False,
    freeze_mean=False,
    freeze_cov=False,
)
