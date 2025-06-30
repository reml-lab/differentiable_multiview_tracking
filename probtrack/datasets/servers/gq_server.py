import os
import glob
from tqdm import tqdm, trange
import torch
import probtrack.datasets.utils as dutils
import numpy as np
import json
from torch.utils.data import DataLoader
from mmengine.registry import DATASETS
import sys
sys.path.append('/home/csamplawski/src/iobtmax-data-tools')
import data_classes as dc
from datetime import datetime, timezone
from .server import DataServer

def parse_gt_gq(gt):
    position = [gt['x'], gt['y'], gt['z']]
    position = torch.tensor(position).float().unsqueeze(0)
    obj_cls_idx = torch.tensor([2])
    timestamp = 0
    output = {
        'obj_position': position,
        'obj_cls_idx': torch.tensor([2]),
        'timestamp': torch.tensor([timestamp]),
    }
    return output


def prepare_gps_data(obj_data):
    obj_data.gps_to_metric()
    obj_data.threshold_accuracy(5)
    obj_data.geofence_outside(30, 100, -100, -25)
    obj_data.geofence_inside(-50, 50, -50, 50, thresh=20)
    obj_data.set_altitude(8)
    return obj_data


@DATASETS.register_module()
class GQDataServer(DataServer):
    def __init__(self, raw_root='/home/csamplawski/eight/data/raw/gq', 
            processed_root='/tmp/gq', 
            scenarios_path='/home/csamplawski/src/iobtmax-data-tools/environments/gq-small/scenarios',
            purge=False, **kwargs):
        pipeline = [dict(type='ParseGQGT')]
        super().__init__(raw_root, processed_root, purge, pipeline=pipeline, **kwargs)
        self.type = 'gq'
        if not os.path.exists(self.zed_root):
            os.makedirs(self.zed_root)
            self.create_dirs_from_bags(topic_key='frame_timestamp')

        self.scenarios_path = scenarios_path
        # self.load_scenarios()
        env = dc.environment("gq-small", base_path="/home/csamplawski/src/iobtmax-data-tools/")
        gps_data = dc.gps_loader(env)
        objects = gps_data.get_objects()
        obj_data = prepare_gps_data(objects['Wanda'])
        df = obj_data.data['world']
        obj_time_series = df.set_index('t').to_dict(orient='index')
        self.buffer_scenarios(obj_time_series)
        self.scenario_root = os.path.join(self.processed_root, 'scenarios')
        self.load_scenarios()
        

    def buffer_scenarios(self, obj_time_series):
        self.scenario_root = os.path.join(self.processed_root, 'scenarios')
        if not os.path.exists(self.scenario_root):
            envs = {}
            os.makedirs(self.scenario_root, exist_ok=True)
            self.zed_time_series = self.load_zed_timestamps(mode='gq')
            scenario_json_files = sorted(glob.glob(os.path.join(self.scenarios_path, '*.json')))
            for scenario_json_file in scenario_json_files:
                scenario_name = scenario_json_file.split('/')[-1].replace('.json', '')
                with open(scenario_json_file, 'r') as f:
                    scenario = json.load(f)
                envs[scenario_name] = scenario
                scenario_id = scenario_json_file.split('/')[-1].replace('.json', '')
                start_timestamp = datetime.strptime(scenario['start_time'], '%Y-%m-%d %H:%M:%S')
                start_timestamp = start_timestamp.replace(tzinfo=timezone.utc).timestamp() * 1000
                start_timestamp = int(start_timestamp)
                end_timestamp = datetime.strptime(scenario['end_time'], '%Y-%m-%d %H:%M:%S')
                end_timestamp = end_timestamp.replace(tzinfo=timezone.utc).timestamp() * 1000
                end_timestamp = int(end_timestamp)
                
                merged_series = self.zed_time_series.copy()
                merged_series['gt'] = obj_time_series
                merged_series = dutils.merge_series(merged_series)
                merged_series = {k: v for k, v in merged_series.items() if k >= start_timestamp and k <= end_timestamp}
                buffered_series = dutils.buffer_series(merged_series, fps=15, string_keys=True)
                json_fname = os.path.join(self.scenario_root, scenario_id + '.json')
                with open(json_fname, 'w') as f:
                    json.dump(buffered_series, f)
            with open(os.path.join(self.scenario_root, 'envs.json'), 'w') as f:
                json.dump(envs, f)
            self.envs = envs
        else:
            with open(os.path.join(self.scenario_root, 'envs.json'), 'r') as f:
                self.envs = json.load(f)

    def load_scenarios(self):
        scenario_json_files = sorted(glob.glob(os.path.join(self.scenario_root, '*.json')))
        self.scenarios = {}
        for scenario_json_file in scenario_json_files:
            with open(scenario_json_file, 'r') as f:
                scenario = json.load(f)
            scenario_id = scenario_json_file.split('/')[-1].replace('.json', '')
            self.scenarios[scenario_id] = scenario
