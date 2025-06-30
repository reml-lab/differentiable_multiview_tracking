import numpy as np
import argparse
import rosbag
from tqdm import tqdm, trange
import os
import json
from datetime import datetime, timezone

def str2ms(s):
    timestamp = datetime.strptime(s, '%Y-%m-%d %H:%M:%S.%f')
    timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = int(timestamp.timestamp() * 1000)
    return timestamp

def parse_msg(msg):
    tokens = msg.split(' ')
    frame_idx = int(tokens[2])
    path = tokens[4]
    base_fname = os.path.basename(path)
    date = tokens[-2]
    time = tokens[-1]
    timestamp = str2ms(f'{date} {time}')
    return frame_idx, base_fname, timestamp

def bag2timestamps(bag_file, topic_key):
    base_bag = bag_file.split('/')[-1]
    try:
        bag = rosbag.Bag(bag_file)
    except:
        print(f'Failed to open {base_bag}')
        return None
    info = bag.get_type_and_topic_info()
    msgs = bag.read_messages()
    msgs = list(msgs)
    timestamps = []
    for topic, msg, t in msgs:
        if topic_key not in topic:
            continue
        if topic_key == 'frame_timestamp':
            frame_idx, svo_fname, timestamp = parse_msg(msg.data)
            dirname = svo_fname.replace('.svo', '')
        elif topic_key in ['zed_timestamp', 'rgb_timestamp', 'depth_timestamp']:
            timestamp = str2ms(msg.data)
            dirname = base_bag.replace('.bag', '') #assuming bag name == svo name, MOCAP ONLY
        else:
            raise ValueError(f'Unknown topic key: {topic_key}')
        
        timestamps.append(timestamp)
    if len(timestamps) == 0:
        print(f'No timestamps found in {base_bag}')
        return None
    timestamps = np.array(timestamps).astype(int)
    timestamps = np.sort(timestamps)
    return dirname, timestamps
