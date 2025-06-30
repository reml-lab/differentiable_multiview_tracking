import os
import json
import glob
import numpy as np
import torch
from tqdm import tqdm, trange
from dataproc.parse_tsv import parse_tsv
from dataproc.find_mocap_scenarios import find_scenarios
import probtrack.datasets.utils as dutils
from mmengine.registry import DATASETS
from .server import DataServer

@DATASETS.register_module()
class MocapDataServer(DataServer):
    def __init__(self, raw_root='/home/csamplawski/eight/data/raw/umass', 
            processed_root='/tmp/mocap', purge=False, **kwargs):
        pipeline = [dict(type='ParseMocapGT')]
        super().__init__(raw_root, processed_root, purge, pipeline=pipeline, **kwargs)
        self.type = 'mocap'
        self.tsv_files = sorted(glob.glob(os.path.join(self.raw_root, 'mocap', '*.tsv')))
        if not os.path.exists(self.zed_root):
            os.makedirs(self.zed_root)
            self.create_dirs_from_bags(topic_key='zed_timestamp')
        self.buffer_scenarios()

    def buffer_scenarios(self):
        scenario_dir = os.path.join(self.processed_root, 'scenarios')
        env_fname = os.path.join(scenario_dir, 'envs.json')
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir)
            self.zed_time_series = self.load_zed_timestamps()
            zed_keys = sorted(list(self.zed_time_series.keys()))
            scenarios = self.load_scenarios()
            self.buffered_scenarios = {}
            self.envs = {}
            for scenario_id, scenario in tqdm(scenarios.items(), desc='Buffering scenarios'):
                env = scenario['env']
                self.envs[scenario_id] = env
                start_timestamp = env['start_time']
                end_timestamp = env['end_time']
                obj_data = {int(k): v for k, v in scenario['objs'].items()}
                obj_timestamps = dutils.keys2timestamps(obj_data.keys())
                final_scenario = {'gt': obj_data.copy()}
                for zed_id in self.zed_time_series.keys():
                    zed_node_series = self.zed_time_series[zed_id].copy()
                    zed_node_series = {k: v for k, v in zed_node_series.items() if k >= start_timestamp and k <= end_timestamp}
                    zed_timestamps = dutils.keys2timestamps(zed_node_series.keys())
                    if len(zed_timestamps) == 0: # no data for this zed node, why?
                        print('No data for zed node %s in scenario %s' % (zed_id, scenario_id))
                        continue
                    nearest_idx = dutils.find_nearest_idx(obj_timestamps, zed_timestamps)
                    nearest_timestamps = obj_timestamps[nearest_idx]
                    buffered_series = {}
                    for i, zed_timestamp in enumerate(zed_timestamps):
                        t_key = str(zed_timestamp)
                        if zed_timestamp not in buffered_series:
                            buffered_series[t_key] = {}
                        buffered_series[t_key][zed_id] = zed_node_series[zed_timestamp]
                        t = nearest_timestamps[i]
                        buffered_series[t_key]['gt'] = obj_data[t]
                    final_scenario[zed_id] = buffered_series
                with open(os.path.join(scenario_dir, scenario_id + '.json'), 'w') as f:
                    json.dump(final_scenario, f)
            with open(env_fname, 'w') as f:
                json.dump(self.envs, f)
        with open(env_fname, 'r') as f:
            self.envs = json.load(f)
                    
                    # merged_series['gt'] = obj_data
                    # merged_series = dutils.merge_series(merged_series)
                    # merged_series = {k: v for k, v in merged_series.items() if k >= start_timestamp and k <= end_timestamp}
                    # buffered_series = dutils.buffer_series(merged_series, fps=15, string_keys=True)
                    # scenario_fname = os.path.join(scenario_dir, scenario_id + '.json')


    def buffer_scenarios_(self):
        scenario_dir = os.path.join(self.processed_root, 'scenarios')
        env_fname = os.path.join(scenario_dir, 'envs.json')
        if not os.path.exists(scenario_dir):
            os.makedirs(scenario_dir)
            self.zed_time_series = self.load_zed_timestamps()
            scenarios = self.load_scenarios()
            self.buffered_scenarios = {}
            self.envs = {}
            for scenario_id, scenario in scenarios.items():
                # if '2022' not in scenario_id:
                    # continue
                env = scenario['env']
                self.envs[scenario_id] = env
                start_timestamp = env['start_time']
                end_timestamp = env['end_time']
                obj_data = {int(k): v for k, v in scenario['objs'].items()}
                merged_series = self.zed_time_series.copy()
                merged_series['gt'] = obj_data
                merged_series = dutils.merge_series(merged_series)
                merged_series = {k: v for k, v in merged_series.items() if k >= start_timestamp and k <= end_timestamp}
                buffered_series = dutils.buffer_series(merged_series, fps=15, string_keys=True)
                scenario_fname = os.path.join(scenario_dir, scenario_id + '.json')
                with open(scenario_fname, 'w') as f:
                    json.dump(buffered_series, f)

            with open(env_fname, 'w') as f:
                json.dump(self.envs, f)

        with open(env_fname, 'r') as f:
            self.envs = json.load(f)

    def load_scenarios(self):
        scenario_json_fname = os.path.join(self.processed_root, 'scenarios.json')
        if not os.path.exists(scenario_json_fname):
            data = []
            for tsv_file in tqdm(self.tsv_files, desc='Parsing tsv files'):
                data_i = parse_tsv(tsv_file)
                data.extend(data_i)
            scenarios = find_scenarios(data, min_length=1000, min_diff=1000, fps=100)
            with open(scenario_json_fname, 'w') as f:
                json.dump(scenarios, f)
        else:
            print('Using cached scenarios from %s' % scenario_json_fname)
            with open(scenario_json_fname, 'r') as f:
                scenarios = json.load(f)
        return scenarios
