import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm, trange
import glob
import argparse
import datetime
import copy
import pandas as pd
from probtrack.datasets.utils import buffer_series
# from fov import *

map = {
        "type": "indoor",
        "map_width": 800,
        "map_height": 800,
        "map_file": "mocap_grid.png",
        "xlim": [
            -4,
            4
        ],
        "ylim": [
            -2.5,
            2.5
        ]
}

fov_db = {"zed_left": {"horizontal": 84, "vertical": 53, "horizontal_err": 6, "vertical_err": 6, "offset_x": -30, "offset_y": -200, "offset_z": 150},
               "zed_camera_right": {"horizontal": 84, "vertical": 53, "horizontal_err": 6, "vertical_err": 6, "offset_x": -30, "offset_y": -80, "offset_z": 150},
                "mmwave": {"horizontal": 120, "vertical": 30, "horizontal_err": 4, "vertical_err": 4, "offset_x": 25, "offset_y": -60, "offset_z": 55},
                "realsense_camera_img": {"horizontal": 71, "vertical": 44, "horizontal_err": 4, "vertical_err": 4, "offset_x": 5, "offset_y": -210, "offset_z": 55},
                "realsense_camera_depth": {"horizontal": 70, "vertical": 55, "horizontal_err": 4, "vertical_err": 4, "offset_x": 5, "offset_y": -210, "offset_z": 70}}


node_cali={"zed_left":{"node_1": {"fx": 1092.17, "fy": 1092.17, "cx": 978.29, "cy": 522.96},
                            "node_2": {"fx": 1076.19, "fy": 1076.19, "cx": 1001.32, "cy": 542.01},
                            "node_3": {"fx": 1070.52, "fy": 1070.52, "cx": 929.94, "cy": 517.58},
                            "node_4": {"fx": 1052.25, "fy": 1052.25, "cx": 989.61, "cy": 526.76}},
        "zed_camera_right":{"node_1": {"fx": 1092.17, "fy": 1092.17, "cx": 978.29, "cy": 522.96},
                            "node_2": {"fx": 1076.19, "fy": 1076.19, "cx": 1001.32, "cy": 542.01},
                            "node_3": {"fx": 1070.52, "fy": 1070.52, "cx": 929.94, "cy": 517.58},
                            "node_4": {"fx": 1052.25, "fy": 1052.25, "cx": 989.61, "cy": 526.76}},
        "realsense_camera_img":{"node_1": {"fx": 1343.85, "fy": 1344.06, "cx": 983.07, "cy": 519.88},
                            "node_2": {"fx": 1350.66, "fy": 1351.58, "cx": 979.96, "cy": 529.18},
                            "node_3": {"fx": 1346.93, "fy": 1348.03, "cx": 982.04, "cy": 524.66},
                            "node_4": {"fx": 1355.56, "fy": 1356.70, "cx": 972.23, "cy": 546.43}},
        "realsense_camera_depth":{"node_1": {"fx": 461.56, "fy": 461.05, "cx": 309.96, "cy": 247.26},
                            "node_2": {"fx": 458.71, "fy": 458.72, "cx": 330.11, "cy": 246.64},
                            "node_3": {"fx": 465.54, "fy": 465.84, "cx": 288.22, "cy": 248.91},
                            "node_4": {"fx": 457.46, "fy": 457.98, "cx": 310.13, "cy": 253.33}}}

def find_scenarios(data, min_length=1000, min_diff=1000, fps=100):
    all_positions = []
    uniq_ids = set()
    time_series = {}
    for point in data:
        uniq_id = '%s_%d' % (point['type'], point['id'])
        uniq_ids.add(uniq_id)
        timestamp = point['time']
        if timestamp not in time_series:
            time_series[timestamp] = {}
        time_series[timestamp][uniq_id] = point #{'position': point['position'], 'rotation': point['rotation']}
        all_positions.append(point['position'])

    df = []
    all_positions = np.array(all_positions)
    min_vals = np.min(all_positions, axis=0)
    max_vals = np.max(all_positions, axis=0)
    bounds = {'min': min_vals.tolist(), 'max': max_vals.tolist()}

    uniq_ids = set(uniq_ids)
    timestamps = list(time_series.keys())
    timestamps = np.array(timestamps).astype(int)
    timestamps = np.unique(timestamps) #sorts

    diffs = np.diff(timestamps)
    assert diffs.min() >= 0
    gaps = np.where(diffs > min_diff)[0]

    chunks = []
    start_index = 0
    for index in gaps:
        end_index = index + 1
        chunks.append(timestamps[start_index:end_index])
        start_index = end_index
    chunks.append(timestamps[start_index:])

    chunk_lengths = [chunk[-1] - chunk[0] for chunk in chunks]
    chunks = [chunk for chunk, length in zip(chunks, chunk_lengths) if length > min_length]
        
    scenarios = {}
    for chunk in chunks:
        chunk_time_series = {timestamp: time_series[timestamp] for timestamp in chunk}
        # nodes = {'node_%d' % (i+1): {} for i in range(4)}
        nodes = defaultdict(dict)
        objs = defaultdict(dict)
        obj_types = set()
        for timestamp, data in chunk_time_series.items():
            for uniq_id, pos_rot in data.items():
                if 'node' in uniq_id:
                    nodes[uniq_id][timestamp] = pos_rot
                else:
                    objs[timestamp][uniq_id] = pos_rot
                    # obj[uniq_id][timestamp] = pos_rot
                    obj_types.add(uniq_id)

        if len(obj_types) == 0 or 'drone_1' in obj_types:
            continue
        
        if len(obj_types) != 0:
            buff_objs = buffer_series(objs, fps=fps)
        else:
            buff_objs = {}
        env = {'nodes': {}}
        for node_id, data in nodes.items():
            position = np.array([pos_rot['position'] for pos_rot in data.values()]).mean(axis=0)
            position_std = np.array([pos_rot['position'] for pos_rot in data.values()]).std(axis=0)
            rotation = np.array([pos_rot['rotation'] for pos_rot in data.values()]).mean(axis=0)
            rotation_std = np.array([pos_rot['rotation'] for pos_rot in data.values()]).std(axis=0)
            roll = np.array([pos_rot['roll'] for pos_rot in data.values()]).mean()
            roll_std = np.array([pos_rot['roll'] for pos_rot in data.values()]).std()
            pitch = np.array([pos_rot['pitch'] for pos_rot in data.values()]).mean()
            pitch_std = np.array([pos_rot['pitch'] for pos_rot in data.values()]).std()
            yaw = np.array([pos_rot['yaw'] for pos_rot in data.values()]).mean()
            yaw_std = np.array([pos_rot['yaw'] for pos_rot in data.values()]).std()
            env['nodes'][node_id] = {
                'location': {
                    'X': position[0],
                    'Y': position[1],
                    'Z': position[2],
                    'X_std': position_std[0],
                    'Y_std': position_std[1],
                    'Z_std': position_std[2]
                },
                'zed': {
                    'fx': node_cali['zed_left'][node_id]['fx'],
                    'fy': node_cali['zed_left'][node_id]['fy'],
                    'cx': node_cali['zed_left'][node_id]['cx'],
                    'cy': node_cali['zed_left'][node_id]['cy'],
                    'roll': roll,
                    'roll_std': roll_std,
                    'pitch': pitch,
                    'pitch_std': pitch_std,
                    'yaw': yaw,
                    'yaw_std': yaw_std,
                    'rotation': rotation.tolist(),
                    'rotation_std': rotation_std.tolist(),
                    'pixel_size_mm': 0.02,
                    'pixel_size_m': 2e-06,
                    'fov_h_rad': 1.4660765716752369,
                    'fov_v_rad': 0.9250245035569946,
                    'location_offset_X': 0, #fov_db['zed_left']['offset_x'],
                    'location_offset_Y': 0,# fov_db['zed_left']['offset_y'],
                    'location_offset_Z': 0,# fov_db['zed_left']['offset_z'],
                    'height': 1080,
                    'width': 1920,
                },
                'name': node_id,
                'id': int(node_id.split('_')[-1])
            }
        env['map'] = map
        env['type'] = 'mocap' 
        env['object_types'] = sorted(list(obj_types))
        env['start_time'] = int(chunk[0])
        env['end_time'] = int(chunk[-1])
        env['duration'] = env['end_time'] - env['start_time']
        date = datetime.datetime.fromtimestamp(chunk[0] / 1000).strftime('%Y-%m-%d')
        start_date = datetime.datetime.fromtimestamp(chunk[0] / 1000).strftime('%Y-%m-%d-%H-%M-%S')
        start_date = '-'.join(start_date.split('-')[3:])
        end_date = datetime.datetime.fromtimestamp(chunk[-1] / 1000).strftime('%Y-%m-%d-%H-%M-%S')
        end_date = '-'.join(end_date.split('-')[3:])
        env['date'] = date
        env['bounds'] = bounds

       
        for t, data in objs.items():
            for obj_id, obj_data in data.items():
                row = {'t': t, 'id': obj_id, 'type': 'mocap',
                        'X': obj_data['position'][0],
                        'Y': obj_data['position'][1],
                        'Z': obj_data['position'][2],
                        'roll': obj_data['roll'], # / 180 * np.pi,
                        'pitch': obj_data['pitch'], # / 180 * np.pi,
                        'yaw': obj_data['yaw'] #/ 180 * np.pi - np.pi/2
                }
                df.append(row)
        
        objs_str = '_'.join(env['object_types'])

        sname = '%s_%s_%s_%s' % (date, start_date, end_date, objs_str)
        scenarios[sname] = {'env': env, 'objs': buff_objs}
        # if not os.path.exists(output_dir):
            # os.makedirs(output_dir)
        # with open(output_dir + 'env.json', 'w') as f:
            # json.dump(env, f, indent=4)
        # with open(output_dir + 'objects.json', 'w') as f:
            # json.dump(buff_objs, f, indent=4)
    return scenarios
