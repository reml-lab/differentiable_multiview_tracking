import probtrack

scenario_names = [
    '2023-04-05_09-38-33_09-43-33_bus_1',
    '2023-04-05_09-51-03_09-56-03_car_1',
    '2022-09-01_12-59-54_13-04-54_truck_1',
]


scenarios = {
    '2023-04-05_09-38-33_09-43-33_bus_1': {
        'sensor_keys': {
            'train': ['zed_node_1', 'zed_node_3', 'zed_node_4'],
            'val': ['zed_node_1', 'zed_node_3', 'zed_node_4'],
            'test': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
        },
        'subsets': {
            'test': [0.0, 0.2],
            'val': [0.2, 0.3],
            'train': [0.3, 1.0],
        }
    },
    '2023-04-05_09-51-03_09-56-03_car_1': {
        'sensor_keys': {
            'train': ['zed_node_1', 'zed_node_2', 'zed_node_4'],
            'val': ['zed_node_1', 'zed_node_2', 'zed_node_4'],
            'test': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
        },
        'subsets': {
            'test': [0.0, 0.2],
            'val': [0.2, 0.3],
            'train': [0.6, 1.0],
        }
    },
    '2022-09-01_12-59-54_13-04-54_truck_1': {
        'sensor_keys': {
            'train': ['zed_node_1', 'zed_node_2', 'zed_node_3'],
            'val': ['zed_node_1', 'zed_node_2', 'zed_node_3'],
            'test': ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4'],
        },
        'subsets': {
            'test': [0.0, 0.2],
            'val': [0.2, 0.3],
            'train': [0.3, 1.0],
        }
    }
}
