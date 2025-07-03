import probtrack
from mmengine.config import read_base

with read_base():
    from configs.datasets.full import *
    from configs.datasets.coco import *
    from configs.datasets.server import *
    from configs.eval import *
    from configs.models.proj import *

subset = 'val'
thresholds = [0.3]
viz = True
model.dm = 'steerable'
