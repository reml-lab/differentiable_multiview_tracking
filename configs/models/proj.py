model = dict(type='GeospatialDetector',
    detector_cfg=dict(type='PretrainedDETR',
        freeze_bbox_head=True,
        freeze_cls_head=True,
        add_point_head=True,
        add_depth_head=False,
        freeze_point_head=True,
    ),
    freeze_projs=True,
    freeze_proj_stds=True,
    freeze_tracker=True,
    dm='constant_velocity',
    output_head_cfg=dict(type='ProjOutputHead')
)
