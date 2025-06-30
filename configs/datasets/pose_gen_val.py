# '2023-04-05_09-38-33_09-43-33_bus_1',
# '2023-04-05_09-51-03_09-56-03_car_1',
# '2022-09-01_12-59-54_13-04-54_truck_1',

from copy import deepcopy



scenario_names = [
    '2022-09-01_12-59-54_13-04-54_truck_1',
    '2023-04-05_09-38-33_09-43-33_bus_1',
    '2023-04-05_09-51-03_09-56-03_car_1',
    '2023-04-05_11-14-58_11-19-58_truck_1',
    '2023-08-23_09-54-13_09-59-13_truck_2',
    
    '2023-08-23_09-39-46_09-44-46_bus_1',
    '2023-08-23_09-47-07_09-52-07_car_1',
    
    '2023-08-23_10-55-04_11-00-04_bus_1',
    '2023-08-23_11-01-24_11-06-24_car_1',

    
    # '2022-09-01_13-07-39_13-12-39_truck_1_truck_2',
    # '2023-08-23_12-23-19_12-28-19_bus_1_car_1',
    # '2023-08-24_12-59-02_13-04-02_bus_1_car_1', #tarp
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
            'test': [0.0, 0.2],
            'val': [0.2, 0.3],
            'train': [0.3, 1.0],
        }
    }

#acount for missing data

scenarios['2023-04-05_09-51-03_09-56-03_car_1']['subsets']['train'] = [0.6, 1.0] 
scenarios['2023-08-23_09-39-46_09-44-46_bus_1']['subsets']['test'] = [0.1, 0.2]
calibrated_scenarios = deepcopy(scenarios)

# del scenarios['2023-08-23_09-39-46_09-44-46_bus_1']
# del scenarios['2023-08-23_09-47-07_09-52-07_car_1']
del scenarios['2023-08-23_10-55-04_11-00-04_bus_1']
del scenarios['2023-08-23_11-01-24_11-06-24_car_1']
