from copy import deepcopy

scenario_names = [
    '2022-09-01_13-07-39_13-12-39_truck_1_truck_2',
    '2023-08-23_12-23-19_12-28-19_bus_1_car_1',
]


scenarios = {}
for scenario_name in scenario_names:
    scenarios[scenario_name] = {
        'sensor_keys': {
            'train': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
            'val': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
            'test': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
        },
        'subsets': {
            'test': [0.2, 0.3],
            'val': [0.0, 0.2],
            'train': [0.3, 1.0],
        }
    }

calibrated_scenarios = deepcopy(scenarios)
