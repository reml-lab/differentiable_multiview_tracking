from mmengine.config import read_base

with read_base():
    from configs.models.proj import *

model.update(
    output_head_cfg=dict(type='UAIOutputHead',
        freeze_adapter=True,
        adapter_cfg=dict(type='LinearAdapter',
            ffn_cfg=dict(type='BottleneckFFN', out_dim=8),
        ),
        predictor_cfg=dict(type='GaussianPredictor', 
            dim=8,
            freeze_ffn=True,
            freeze_mean=True,
            freeze_cov=True
        ),
    ),
)
