import probtrack
from mmengine.config import read_base

with read_base():
    from configs.models.proj import *
    from configs.optim import *

coco_batch_size = 0
num_iters = 1
subseq_len = -1

loss_weights.update(
    det_nll_loss=0,
    track_nll_loss=0,
)

model.update(
    freeze_projs=False,
)

track = False
