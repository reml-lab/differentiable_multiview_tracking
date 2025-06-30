import numpy as np
import argparse
from tqdm import tqdm
import csv
import os
import pickle
import json
import scipy.interpolate as spi
from datetime import datetime
import rosbag
#from std_msgs.msg import Int32, String
#from msg_pkg.msg import RadarData
import mmwave_utils as mf


# parameters: source, destination, boolean to add azimuth to the pickle file
def parse_mmwave(src, read_azimuth=False):
    parser = argparse.ArgumentParser()
    # parser.add_argument('src', type=str, help='bag file')
    # parser.add_argument('dst', type=str, help='output pickle file')
    args = parser.parse_args()
    # args.src= src
    # args.dst= dst

    try:
        bag = rosbag.Bag(src)
    except:
        print("Error opening bag file, %s" % src)
        return None
    info = bag.get_type_and_topic_info()
    topics = info.topics
    msgs = bag.read_messages()
    data = {}

    #for topic, msg, T in tqdm(msgs, total=bag.get_message_count()):
    for topic, msg, T in msgs:
        if not 'radar' in topic and 'RadarData' not in topic:
            continue

        cfg_str = msg.radar_cfg
        cfg = json.loads(cfg_str)
        cfg, par = mf.get_conf(cfg)


        buff = {}
        if msg.range_profile_valid:
            y = msg.range_profile
            # bin = mf.range_maximum(cfg) / len(y)
            # x = [i*bin for i in range(len(y))]
            # x = [v - par['range_bias'] for v in x]
            buff["range_profile"] = y
        else:
            buff["range_profile"] = None
            
        if msg.noise_profile_valid:
            buff["noise_profile"] = msg.noise_profile
        else:
            buff["noise_profile"] = None

        if msg.azimuth_static_valid and read_azimuth:
            a = msg.azimuth_static

            if len(a) != mf.num_range_bin(cfg) * mf.num_tx_azim_antenna(cfg) * mf.num_rx_antenna(cfg) * 2:
                continue

            a = np.array([a[i] + 1j * a[i+1] for i in range(0, len(a), 2)])
            a = np.reshape(a, (mf.num_range_bin(cfg), mf.num_tx_azim_antenna(cfg) * mf.num_rx_antenna(cfg)))
            a = np.fft.fft(a, mf.num_angular_bin(cfg))
            a = np.abs(a)

            a = np.fft.fftshift(a, axes=(1,))  # put left to center, put center to right       

            a = a[:,1:]  # cut off first angle bin

            #buff['azimuth_static_raw'] = a

            t = np.array(range(-mf.num_angular_bin(cfg)//2 + 1, mf.num_angular_bin(cfg)//2)) * (2 / mf.num_angular_bin(cfg))
            t = np.arcsin(t) # t * ((1 + np.sqrt(5)) / 2)
            r = np.array(range(mf.num_range_bin(cfg))) * mf.range_resolution(cfg)

            range_depth = mf.num_range_bin(cfg) * mf.range_resolution(cfg)
            range_width, grid_res = range_depth / 2, 400
            buff["mmwave_cfg"] = {"range_depth": range_depth, "range_resolution": mf.range_resolution(cfg), 
                    "num_range_bin": mf.num_range_bin(cfg), "num_tx_azim_antenna": mf.num_tx_azim_antenna(cfg), "num_rx_antenna": mf.num_rx_antenna(cfg), "num_angular_bin": mf.num_angular_bin(cfg)}

            
            xi = np.linspace(-range_width, range_width, grid_res)
            yi = np.linspace(0, range_depth, grid_res)
            xi, yi = np.meshgrid(xi, yi)

            x = np.array([r]).T * np.sin(t)
            y = np.array([r]).T * np.cos(t)
            y = y - par['range_bias']
            
            zi = spi.griddata((x.ravel(), y.ravel()), a.ravel(), (xi, yi), method='linear')
            zi = zi[:-1,:-1]

            buff["azimuth_static"] = zi[::-1,::-1]
        else:
            buff["azimuth_static"] = None
            #buff['azimuth_static_raw'] = None
            buff['mmwave_cfg'] = None

        if msg.range_doppler_valid:
            if len(msg.range_doppler) != mf.num_range_bin(cfg) * mf.num_doppler_bin(cfg):
                continue

            a = np.array(msg.range_doppler)
            b = np.reshape(a, (mf.num_range_bin(cfg), mf.num_doppler_bin(cfg)))
            c = np.fft.fftshift(b, axes=(1,))  # put left to center, put center to right
            buff["range_doppler"] = c
        else:
            buff["range_doppler"] = None
        
        points = msg.points

        # print(points)

        p_all = {}
        ii = 0
        for p in points:
            p_tmp = {}
            p_tmp["x"] = float(p.x)
            p_tmp["y"] = float(p.y)
            p_tmp["z"] = float(p.z)
            p_tmp["v"] = float(p.intensity)
            # p_all['"{}"'.format(ii)] = p_tmp
            p_all["{},{}".format(int(p.range), int(p.doppler))] = p_tmp
            
            ii += 1
        
        buff["detected_points"] = p_all

        h_t = {}
        h_t["time"] = 2021
        h_t["number"] = 0
        buff["header"] = h_t

        if buff['range_doppler'] is None:
            continue
        
        if read_azimuth:
            buff['azimuth_static'] = buff['azimuth_static'].astype(np.float32)
        
        buff['range_doppler'] = buff['range_doppler'].astype(np.float32)

        #timestamp to ms from string
        timestamp = datetime.strptime(msg.timestamp, "%Y-%m-%d %H:%M:%S.%f").timestamp()
        timestamp = int(timestamp * 1000)
        data[timestamp] = buff

    bag.close()

    # with open(args.dst, 'wb') as f:
        # pickle.dump(data, f)

    return data
