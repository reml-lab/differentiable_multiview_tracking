import numpy as np
import torch
import copy
from tqdm import tqdm

#move nested dicts to gpu
def move_to_gpu(data, device='cuda'):
    output = {}
    for k, v in data.items():
        if isinstance(v, dict):
            output[k] = move_to_gpu(v)
        else:
            output[k] = v.to(device)
    return output

def subset_dataset(cfg, dataset, subset='val'):
    subset2frac= {
        'train': cfg.train_frac,
        'val': cfg.val_frac,
        'test': cfg.test_frac
    }
    frac = subset2frac[subset]
    return make_subset(dataset, frac)

class SubsetDataset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.timestamps = dataset.timestamps[indices]
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def __len__(self):
        return len(self.indices)

#frac is [a, b] in [0, 1]
def make_subset(dataset, frac):
    start_idx = int(len(dataset) * frac[0])
    end_idx   = int(len(dataset) * frac[1])
    subset = SubsetDataset(dataset, np.arange(start_idx, end_idx))
    return subset

def find_nearest_idx(sorted_list, queries):
    sorted_array = np.array(sorted_list)
    queries_array = np.array(queries)
    indices = np.searchsorted(sorted_array, queries_array, side="left")

    closest_idx = []
    for idx, query in zip(indices, queries_array):
        if idx > 0 and (idx == len(sorted_array) or abs(query - sorted_array[idx-1]) < abs(query - sorted_array[idx])):
            closest_idx.append(idx-1)
        else:
            closest_idx.append(idx)
    closest_idx = np.array(closest_idx)
    return closest_idx

def keys2timestamps(keys):
    timestamps = list(keys)
    timestamps = np.array(timestamps).astype(int)
    timestamps = np.unique(timestamps) #sorts
    return timestamps

#series is a dict of time series
def merge_series(series):
    merged_time_series = {}
    for id, time_series in series.items():
        for timestamp, data in time_series.items():
            if timestamp not in merged_time_series.keys():
                merged_time_series[timestamp] = {}
            merged_time_series[timestamp][id] = data
    return merged_time_series

def invert_series(multi_time_series):
    keys = list(multi_time_series.keys())
    time_series = {}
    for key in keys:
        sub_series = multi_time_series[key]
        for timestamp, data in sub_series.items():
            if timestamp not in time_series.keys():
                time_series[timestamp] = {}
            time_series[timestamp][key] = data
    return time_series


def trunc_first_endtime(time_series):
    first_endtime = None
    for key, series in time_series.items():
        timestamps = keys2timestamps(series.keys())
        end_time = timestamps[-1]
        if first_endtime is None or end_time < first_endtime:
            first_endtime = end_time
    new_time_series = {}
    for key, series in time_series.items():
        new_time_series[key] = {k: v for k, v in series.items() if k <= first_endtime}
    return new_time_series


#times_series is a dict with timestamps as keys (ms)
#each value is a dict with uniq_ids as keys (nodes, objects, sensor data, etc)
#returns a new time_series with the same keys, but the values are dicts with all uniq_ids present
def buffer_series(time_series, fps=100, string_keys=True):
    timestamps = keys2timestamps(time_series.keys()) #sorts
    start_time = timestamps[0]
    end_time = timestamps[-1]
    save_timestamps = np.arange(start_time, end_time, 1000 / fps).astype(int)
    all_timestamps = np.concatenate([timestamps, save_timestamps])
    all_timestamps = np.unique(all_timestamps).astype(int)

    buffer, buff_series = {}, {}
    all_ids = set()
    for timestamp in tqdm(all_timestamps, desc='Buffering times series'):
        # is_save_slow = bool(np.isin(timestamp, save_timestamps))
        is_save = np.searchsorted(save_timestamps, timestamp, side='right') - 1
        is_save = (is_save >= 0) and (save_timestamps[is_save] == timestamp)

        is_data = np.searchsorted(timestamps, timestamp, side='right') - 1
        is_data = (is_data >= 0) and (timestamps[is_data] == timestamp)

        

        # is_data = bool(np.isin(timestamp, timestamps))
        if is_data:
            data = time_series[timestamp]
            for uniq_id, obj_data in data.items():
                buffer[uniq_id] = obj_data
                all_ids.add(uniq_id)
        if is_save:
            buff_series[timestamp] = copy.deepcopy(buffer)

    #things start a slighly different times,
    #so we need to make sure that all ids are present
    #remove the handleful of frames that are missing data
    new_buff_series = {}
    for timestamp, data in buff_series.items():
        missing = False
        for uniq_id in all_ids:
            if uniq_id not in data.keys():
                missing = True
        if not missing:
            key = str(timestamp) if string_keys else timestamp
            new_buff_series[key] = data
    return new_buff_series
