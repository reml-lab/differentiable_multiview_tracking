from mmengine.config import read_base

with read_base():
    from configs.models.uai import *

model.output_head_cfg.update(type='FlexLocOutputHead')
