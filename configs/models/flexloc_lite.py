from mmengine.config import read_base

with read_base():
    from configs.models.proj import *

model.update(
    output_head_cfg=dict(type='FlexLocOutputHead',
        freeze_adapter=True,
        adapter_cfg=dict(type='NodePoseAdapter',
            # ffn_cfg=dict(type='BottleneckFFN', out_dim=8),
            node_pos_ffn_dim=16,
            pose_dropout_rate=0.0,
        ),
        predictor_cfg=dict(type='GaussianPredictor', 
            dim=16,
            freeze_ffn=True,
            freeze_mean=True,
            freeze_cov=True,
            use_ffn=False
        ),
    ),
)
