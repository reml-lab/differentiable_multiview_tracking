import pyzed.sl as sl
import cv2
import numpy as np
import argparse
import rosbag
from tqdm import tqdm, trange
import os
import h5py
import ffmpeg
import json
from datetime import datetime

def parse_img(img, div_factor=1):
    img = img.get_data()
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    H, W, _ = img.shape
    dsize = int(W//div_factor), int(H//div_factor)
    img = cv2.resize(img, dsize=dsize)
    #code = cv2.imencode('.jpg', img)[1]
    return img

def parse_dmap_lossless(dmap, div_factor=1):
    dmap = dmap.get_data()
    dmap = dmap.astype(np.uint16)
    H, W = dmap.shape
    dsize = int(W//div_factor), int(H//div_factor)
    dmap = cv2.resize(dmap, dsize=dsize)
    return dmap

def parse_dmap(dmap, div_factor=1):
    dmap = dmap.get_data()
    dmap = dmap.astype(np.float32)
    dmap = dmap / 20000
    inf_mask = np.isinf(dmap)
    dmap[inf_mask] = 0
    dmap = np.nan_to_num(dmap)
    dmap = (dmap * 255).astype(np.uint8)
    H, W = dmap.shape
    dsize = int(W//div_factor), int(H//div_factor)
    dmap = cv2.resize(dmap, dsize=dsize)
    dmap = cv2.cvtColor(dmap, cv2.COLOR_GRAY2RGB)
    return dmap


def svo_to_vids(svo_file, dst, 
               depth_mode='ultra', 
               sensing_mode='standard', 
               div=1, outputs=['left', 'right', 'depth'], pbar=True):
    if not os.path.exists(dst):
        os.makedirs(dst, exist_ok=True)
    
    # Set SVO file for playback
    init_parameters = sl.InitParameters()
    if depth_mode == 'ultra':
        init_parameters.depth_mode = sl.DEPTH_MODE.ULTRA
    elif depth_mode == 'quality':
        init_parameters.depth_mode = sl.DEPTH_MODE.QUALITY
    elif depth_mode == 'performance':
        init_parameters.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    elif depth_mode == 'neural':
        init_parameters.depth_mode = sl.DEPTH_MODE.NEURAL
    else:
        print("Depth Mode Error! [ultra, performance, quality] only.")
        exit(1)
    
    init_parameters.svo_real_time_mode = False # Don't convert in realtime
    init_parameters.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)
    init_parameters.set_from_svo_file(svo_file)

    # Open the ZED
    zed = sl.Camera()
    status = zed.open(init_parameters)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        zed.close()
        return

    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height
    width_sbs = width * 2
    
    # Prepare image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_measure = sl.Mat()
    
    runtime_parameter = sl.RuntimeParameters()
    if sensing_mode == 'standard':
        runtime_parameter.sensing_mode = sl.SENSING_MODE.STANDARD
    elif sensing_mode == 'fill':
        runtime_parameter.sensing_mode = sl.SENSING_MODE.FILL
    else:
        print("Sensing Mode Error! [standard, fill] only.")
        exit(1)

    nb_frames = zed.get_svo_number_of_frames()
    size = (1920 // div, 1080 // div)
    size = (int(size[0]), int(size[1]))

    vids, images = {}, {}
    for name in outputs:
        vids[name] = cv2.VideoWriter(f'{dst}/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
        images[name] = sl.Mat()
    # left_vid = cv2.VideoWriter(f'{dst}/left.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    # right_vid = cv2.VideoWriter(f'{dst}/right.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    # depth_vid = cv2.VideoWriter(f'{dst}/depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    num_frames_written = 0
    loop_range = trange(nb_frames+1, desc='converting SVO file') if pbar else range(nb_frames+1)
    for i in loop_range:
        state = zed.grab(runtime_parameter)
        svo_position = zed.get_svo_position()
        #assert svo_position == i
        if state == sl.ERROR_CODE.SUCCESS:
            for name in outputs:
                if name == 'depth':
                    zed.retrieve_measure(images[name], sl.MEASURE.DEPTH)
                    img = parse_dmap(images[name], div)
                elif name == 'left':
                    zed.retrieve_image(images[name], sl.VIEW.LEFT)
                    img = parse_img(images[name], div)
                elif name == 'right':
                    zed.retrieve_image(images[name], sl.VIEW.RIGHT)
                    img = parse_img(images[name], div)
                vids[name].write(img)
            num_frames_written += 1

            # zed.retrieve_image(left_image, sl.VIEW.LEFT)
            # zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            # zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH)
            # left_img = parse_img(left_image, div)
            # right_img = parse_img(right_image, div)
            # depth_map = parse_dmap(depth_measure, div)

            # left_vid.write(left_img)
            # right_vid.write(right_img)
            # depth_vid.write(depth_map)
    
    for name in outputs:
        vids[name].release()
    # left_vid.release()
    # right_vid.release()
    # depth_vid.release()
    zed.close()
    return num_frames_written

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('svo_file', type=str, help='Path to svo file')
    parser.add_argument('dst', type=str, help='Path to output directory')
    parser.add_argument('-m', action="store", choices=['ultra', 'performance', 'quality', 'neural'], default='quality', help="Specify depth mode. [ultra, performance, quality, neural], default quality")
    parser.add_argument('-s', action="store", choices=['standard', 'fill'], default='standard', help="Specify depth sensing mode. [standard, fill], default standard")
    parser.add_argument('-div', action="store", type=float, default=1.0, help="Image height/width will be divided by this value. default 1.0")
    parser.add_argument('-left_only', action="store_true", help="Only save left image")
    args = parser.parse_args()
    
    #svo_to_vids(args.svo_file, args.dst, args.m, args.s, args.div, args.left_only)
    svo_to_vids(args.svo_file, args.dst, args.m, args.s, args.div, pbar=True)
