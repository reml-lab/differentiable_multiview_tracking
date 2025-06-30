import probtrack
from mmengine.config import read_base

with read_base():
    from configs.optim import *


subset = 'train'
num_iters = 2000
subseq_len = 200

track = True
loss_weights.update(
    det_nll_loss=0,
    giou_loss=0,
    l1_loss=0,
    ce_loss=0
)

# model.update(
    # freeze_tracker=False,
# )




# scenario_names = [
    # '2022-09-01_12-59-54_13-04-54_truck_1',
# ]


# scenarios = {}
# for scenario_name in scenario_names:
    # scenarios[scenario_name] = {
        # 'sensor_keys': {
            # 'train': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
            # 'val': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
            # 'test': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
        # },
        # 'subsets': {
            # 'test': [0.0, 0.2],
            # 'val': [0.2, 0.3],
            # 'train': [0.3, 1.0],
        # }
    # }
