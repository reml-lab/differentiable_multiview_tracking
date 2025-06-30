import numpy as np
import argparse
from tqdm import tqdm
import csv
import os
import pickle
import json
import scipy.interpolate as spi
from datetime import datetime
import cv2
import rosbag
#from std_msgs.msg import Int32, String
#from msg_pkg.msg import RadarData
import mmwave_utils as mf
import math
import ffmpeg
from dataproc.parse_mmwave import parse_mmwave
min = math.inf
max = -math.inf
import glob 

path = '/work/pi_marlin_umass_edu/20230918/'
# output_root = '/work/pi_marlin_umass_edu/tmp'
output_root = '/project/pi_marlin_umass_edu/mmwave_processed'
#bag_files = glob.glob(path + '*/*mmwave.bag', recursive=True)
bag_files = glob.glob('/work/pi_marlin_umass_edu/20230918/orin*/*mmwave.bag')
#bag_files = glob.glob('/work/pi_marlin_umass_edu/20230918/orin*/20230919_1025*mmwave.bag')
#bag_files = glob.glob('/work/pi_marlin_umass_edu/20230918/orin*/20230920_1025*mmwave.bag')
#bag_files = [bf for bf in bag_files if 'orin_6' not in bf and 'orin_7' not in bf and 'orin_10' not in bf]
bag_files = sorted(bag_files)
# bag_files = [bf for bf in bag_files if '20230919' in bf]


vids = []
for bag_file in tqdm(bag_files):
    print(bag_file)
    # tokens = bag_file.split('/')[-1].split('_')
    # date, time = tokens[0], int(tokens[1])
    data = parse_mmwave(bag_file)
    if data is None:
        continue
    timestamps = sorted(list(data.keys()))
    range_dopplers = [data[t]['range_doppler'] for t in timestamps]
    range_dopplers = [rd.T for rd in range_dopplers]
    range_dopplers = np.stack(range_dopplers, axis=0)
    vids.append((range_dopplers, timestamps))

# max_val = np.max([np.max(frame) for frame in vids])
# min_val = np.min([np.min(frame) for frame in vids])
max_val = 9000
min_val = 2000
num_processed = 0

for i in range(len(vids)):
    vid, timestamps = vids[i]
    bag_file = bag_files[i]
    basename = os.path.basename(bag_file).replace('.bag', '')
    vid_path = os.path.join(output_root, basename, 'range_doppler.mp4')
    if not os.path.exists(os.path.dirname(vid_path)):
        os.makedirs(os.path.dirname(vid_path), exist_ok=True)
    
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # writer = cv2.VideoWriter(vid_path, fourcc, fps=4, (vid.shape[2], vid.shape[1]), isColor=True)

    size = vid.shape[2], vid.shape[1]
    writer = (ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='%sx%s' % size, r='4')
        .output(vid_path, vcodec='libx264', an=None, pix_fmt='yuv420p')
        .global_args('-y')
        .global_args('-loglevel', 'error')
        .run_async(pipe_stdin=True)
    )

    for frame in tqdm(vid):
        frame = (frame - min_val) / (max_val - min_val) * 255
        frame = frame.astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        writer.stdin.write(frame)
    writer.stdin.close()

    #timestamps to json
    timestamps = [int(t) for t in timestamps]
    with open(os.path.join(output_root, basename, 'timestamps.json'), 'w') as f:
        json.dump(timestamps, f)
