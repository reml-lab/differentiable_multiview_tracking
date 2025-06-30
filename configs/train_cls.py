import probtrack
from mmengine.config import read_base

with read_base():
    from configs.models.proj import *
    from configs.optim import *

coco_batch_size = 64
num_iters = 10000
subseq_len = 200

optimizer.update(
    lr=1e-3,
)

loss_weights.update(
    det_nll_loss=0,
    track_nll_loss=0,
)

model.detector_cfg.update(
    freeze_cls_head=False
)
