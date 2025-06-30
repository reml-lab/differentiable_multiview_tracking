from mmengine.config import read_base

with read_base():
    from configs.models.proj import *

model.update(
    output_head_cfg=dict(type='UAIOutputHead',
        freeze_adapter=True,
        predictor_cfg=dict(type='GaussianPredictor', 
            freeze_ffn=True,
            freeze_mean=True,
            freeze_cov=True
        ),
    ),
)
