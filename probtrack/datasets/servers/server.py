import os
import glob
import json
import numpy as np
import torch
import cv2
import ffmpeg
from tqdm import tqdm, trange
from dataproc.parse_svo import svo_to_vids
from dataproc.parse_bag import bag2timestamps
import probtrack.datasets.utils as dutils
from torch.utils.data import DataLoader
from mmengine.registry import MODELS, DATASETS, TRANSFORMS
from mmengine.dataset import Compose

import os
def count_jpg_files(directory):
    return len([name for name in os.listdir(directory) if name.endswith('.jpg')])


def embed_video(vid_fname, detr):
    cap = cv2.VideoCapture(vid_fname)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    outputs = []
    for i in trange(num_frames, desc='Embedding video with detector'):
        ret, frame = cap.read()
        if not ret:
            break  # If no more frames are left, exit the loop

        frame = torch.from_numpy(frame).float()
        frame = frame.cuda()
        frame = frame.permute(2,0,1)
        frame = frame.unsqueeze(0)
        with torch.no_grad():
            detr_output = detr({'frame': frame})
        assert len(detr_output) == 1
        detr_output = detr_output[0].to('cpu')
        outputs.append(detr_output)
        # embecds.append(detr_output['embeds'][-1].cpu())
    cap.release()
    # embeds = torch.cat(embeds, dim=0)
    # embeds = embeds.reshape(-1, 100, 256)
    return outputs

@DATASETS.register_module()
class ServerDataset:
    def __init__(self, data, type='mocap', pipeline=None):
        self.data = data
        self.timestamps = dutils.keys2timestamps(self.data.keys())
        self.timestamps = np.array(self.timestamps).astype(int)
        self.timestamps = np.unique(self.timestamps)
        self.time_series = {int(t): idx for idx, t in enumerate(self.timestamps)}
        self.inv_time_series = {idx: int(t) for idx, t in enumerate(self.timestamps)}
        self.type = type
        self.pipeline = None
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        
    def timestamp2idx(self, timestamp):
        return self.data[str(timestamp)]

    def __len__(self):
        return len(self.time_series)

    def getitem_by_timestamp(self, timestamp):
        idx = self.time_series[timestamp]
        return self[idx]

    def __getitem__(self, idx):
        timestamp = self.inv_time_series[idx]
        data = self.data[str(timestamp)]
        gt = data['gt'].copy()
        if self.pipeline is not None:
            gt = self.pipeline(gt)
        output = {k: v for k, v in data.items() if k != 'gt'}
        output['gt'] = gt
        return output


class DataServer:
    def __init__(self, raw_root='/home/csamplawski/eight/data/raw/umass', 
            processed_root='/tmp/mocap', purge=False, lru_cache_size=100,
            pipeline=None, detector_cfg=dict(type='PretrainedDETR'), load_detector=True,
            frames_dir_name='frames'):
        self.raw_root = raw_root
        self.processed_root = processed_root
        os.makedirs(self.processed_root, exist_ok=True)
        if purge:
            os.system('rm -rf %s' % self.processed_root)
        self.svo_files = sorted(glob.glob(os.path.join(self.raw_root, 'sensors/zed', '*.svo')))
        self.bag_files = sorted(glob.glob(os.path.join(self.raw_root, 'sensors/zed', '*.bag')))
        self.zed_root = os.path.join(self.processed_root, 'zed')
        self.type = 'abstract'
        self.pipeline = pipeline
        self.frames_dir_name = frames_dir_name

                
        # self.load_scenarios()
        # self.detector = None
        # if load_detector:
            # self.detector = MODELS.build(detector_cfg)
            # self.detector = self.detector.cuda().eval()
            # self.detector_name = detector_cfg['type']
        self.lru_cache = {}
        self.lru_cache_size = lru_cache_size
    
    def create_dirs_from_bags(self, topic_key='zed_timestamp'):
        for bag_file in tqdm(self.bag_files, desc='parsing bag files'):
            if 'orin_10' in bag_file:
                output = bag2timestamps(bag_file, 'zed_timestamp')
            else:
                output = bag2timestamps(bag_file, topic_key)
            # if '20230919_102000_dvpg_gq_orin_2_zed' in bag_file:
                # import ipdb; ipdb.set_trace() # noqa
            if output is None:
                continue
            dirname, timestamps = output
            os.makedirs(os.path.join(self.zed_root, dirname), exist_ok=True)
            pt_fname = os.path.join(self.zed_root, dirname, 'timestamps.pt')
            timestamps = torch.tensor(timestamps)
            torch.save(timestamps, pt_fname)

    def get_scenario(self, scenario_id):
        scenario_fname = os.path.join(self.processed_root, 'scenarios', scenario_id + '.json')
        with open(scenario_fname, 'r') as f:
            data = json.load(f)
        return data

    def get_dataset(self, scenario_id, sensor_id='zed_node_1'):
        scenario_fname = os.path.join(self.processed_root, 'scenarios', scenario_id + '.json')
        with open(scenario_fname, 'r') as f:
            data = json.load(f)
        return ServerDataset(data[sensor_id], type=self.type, pipeline=self.pipeline)

    def get_scenario_gt(self, scenario_id, frac=[0.0, 1.0]):
        scenario_fname = os.path.join(self.processed_root, 'scenarios', scenario_id + '.json')
        with open(scenario_fname, 'r') as f:
            data = json.load(f)
        gt = data['gt']
        gt = {int(k): v for k, v in gt.items()}
        gt_transform = TRANSFORMS.build(dict(type='ParseMocapGT'))
        gt_timestamps = dutils.keys2timestamps(gt.keys())
        full_length = gt_timestamps[-1] - gt_timestamps[0]
        start_time = gt_timestamps[0] + frac[0] * full_length
        end_time = gt_timestamps[0] + frac[1] * full_length
        gt = {k: gt_transform(v) for k, v in gt.items() if k >= start_time and k <= end_time}
        return gt

    def process_svo(self, dirname):
        basename = dirname.split('/')[-1]
        left_vid_fname = os.path.join(self.zed_root, dirname, 'left.mp4')
        # embeds_fname = left_vid_fname.replace('.mp4', '_detr_embeds.pt')
        # frames_dir = os.path.join(self.zed_root, dirname, 'frames')
        if not os.path.exists(left_vid_fname):
            svo_fname = os.path.join(self.raw_root, 'sensors/zed', basename + '.svo')
            svo_to_vids(svo_fname, dirname)
                        # embeds = embed_video(os.path.join(self.zed_root, dirname, 'left.mp4'), self.detr)
            # torch.save(embeds, os.path.join(self.zed_root, dirname, 'left_detr_embeds.pt'))

    def create_frames(self, dirname):
        full_frames_path = os.path.join(self.zed_root, dirname, self.frames_dir_name)
        if not os.path.exists(full_frames_path):
            os.makedirs(full_frames_path, exist_ok=True)
            if self.frames_dir_name == 'frames': #low res frames
                output_pattern = os.path.join(self.zed_root, dirname, 'frames', 'left_%04d.jpg')
                ffmpeg.input(os.path.join(self.zed_root, dirname, 'left.mp4')).output(output_pattern, vf='scale=-1:480,fps=15').run()
            elif self.frames_dir_name == 'frames_hq':
                output_pattern = os.path.join(self.zed_root, dirname, 'frames_hq', 'left_%04d.jpg')
                ffmpeg.input(os.path.join(self.zed_root, dirname, 'left.mp4')).output(output_pattern, vf='scale=-1:1080,fps=15', 
                        **{'qscale:v': 2}).run()
            else:
                raise ValueError(f'Unknown frames_dir_name: {self.frames_dir_name}')



    def process_all_svos(self):
        for svo_file in tqdm(self.svo_files, desc='Processing SVO files'):
            basename = os.path.basename(svo_file)
            basename = basename.replace('.svo', '')
            dirname = os.path.join(self.zed_root, basename)
            self.process_svo(dirname)

    def process_instance(self, inst):
        inst = {k: [v] for k, v in inst.items()}
        output = self.process_batch(inst)
        return output


    def process_batch(self, batch, load_jpg=False):
        all_embeds, all_frames, all_timestamps, all_fnames = [], [], [], []
        dirnames = batch['dirname']
        for i, dirname in enumerate(dirnames):
            self.process_svo(dirname)
            self.create_frames(dirname)
            # embeds_fname = os.path.join(self.zed_root, dirname, f'left_{self.detector_name}_embeds.pt')
            # if not os.path.exists(embeds_fname):
                # embeds = embed_video(os.path.join(self.zed_root, dirname, 'left.mp4'), self.detector)
                # torch.save(embeds, embeds_fname)
            
            frames_dir = os.path.join(self.zed_root, dirname, self.frames_dir_name)
            # if embeds_fname in self.lru_cache:
                # embeds = self.lru_cache[embeds_fname]
            # else:
                # embeds = torch.load(embeds_fname)
                # self.lru_cache[embeds_fname] = embeds
                # if len(self.lru_cache) > self.lru_cache_size:
                    # self.lru_cache.popitem(last=False)

            timestamp = batch['frame_timestamp'][i]
            all_timestamps.append(timestamp)
        


            frame_idx = batch['frame'][i]


            num_frames_file = os.path.join(self.zed_root, dirname, 'num_frames.txt')
            if not os.path.exists(num_frames_file):
                num_frames = count_jpg_files(frames_dir)
                with open(num_frames_file, 'w') as f:
                    f.write(str(num_frames))
            else:
                with open(num_frames_file, 'r') as f:
                    num_frames = int(f.read())
            
            #num_frames = count_jpg_files(frames_dir)
            #num_frames = len(glob.glob(os.path.join(frames_dir, '*.jpg')))
            
            if frame_idx >= num_frames:
                frame_idx = num_frames - 1


            # frame_embed = embeds[frame_idx]
            # all_embeds.append(frame_embed)
                
            frame_fname = os.path.join(frames_dir, 'left_%04d.jpg' % (frame_idx + 1))

            all_fnames.append(frame_fname)
            if load_jpg:
                frame = cv2.imread(frame_fname)
                if frame is None:
                    import ipdb; ipdb.set_trace() # noqa
                frame = torch.from_numpy(frame).float()
                all_frames.append(frame)
        # all_embeds = torch.stack(all_embeds, dim=0)
        all_timestamps = torch.tensor(all_timestamps)
        output = {'timestamp': all_timestamps}
        output['fnames'] = all_fnames
        if load_jpg:
            all_frames = torch.stack(all_frames, dim=0)
            output['frames'] = all_frames
        return output
    
    def load_zed_timestamps(self, mode='mocap'):
        zed_dirs = sorted(glob.glob(os.path.join(self.zed_root, '*')))
        time_series = {}
        uniq_ids = set()
        for zed_dir in tqdm(zed_dirs, desc='Loading zed timestamps'):
            timestamps = torch.load(os.path.join(zed_dir, 'timestamps.pt'))
            if mode == 'mocap':
                node_id = zed_dir.split('/')[-1].split('_')[2]
                zed_id = 'zed_' + 'node_' + node_id
            elif mode == 'gq':
                node_id = zed_dir.split('/')[-1].split('_')[-2]
                zed_id = 'zed_' + 'node_' + node_id
            uniq_ids.add(zed_id)
            for t, timestamp in enumerate(timestamps):
                timestamp = int(timestamp)
                if timestamp not in time_series:
                    time_series[timestamp] = {}
                time_series[timestamp][zed_id] = {
                    'frame': t, 'dirname': zed_dir,
                    'frame_timestamp': timestamp, 
                    'node_id': node_id
                }
        all_node_time_series = {}
        for uid in uniq_ids:
            all_node_time_series[uid] = {k: v[uid] for k, v in time_series.items() if uid in v}
        return all_node_time_series


    def load_time_series(self, scenario, sensor_keys, frac=(0,1), save_fps=15, train_fps=1):
        full_gt = self.get_scenario_gt(scenario, frac=frac)
        all_time_series = {'gt': full_gt}
        for sensor_key in sensor_keys:
            time_series = {}
            dataset = self.get_dataset(scenario, sensor_key)
            #dataset = dutils.subset_dataset(cfg, dataset, subset=cfg.subset)
            dataset = dutils.make_subset(dataset, frac)
            for ele in dataset:
                sensor_data = ele[sensor_key]
                timestamp = sensor_data['frame_timestamp']
                time_series[timestamp] = sensor_data
            all_time_series[sensor_key] = time_series
        all_time_series = dutils.trunc_first_endtime(all_time_series)
        all_time_series = dutils.invert_series(all_time_series)
        time_series = all_time_series

        scenario_timestamps = dutils.keys2timestamps(time_series.keys())
        duration = scenario_timestamps[-1] - scenario_timestamps[0]
        
        if save_fps != 0:
            dt_ms = (1 / save_fps) * 1000
            save_timestamps = range(scenario_timestamps[0], scenario_timestamps[-1], int(dt_ms))
            for t in save_timestamps:
                if t not in time_series:
                    time_series[t] = {}
                time_series[t]['save'] = True
            scenario_timestamps = dutils.keys2timestamps(time_series.keys())

        if train_fps != 0:
            dt_ms = (1 / train_fps) * 1000
            train_timestamps = range(scenario_timestamps[0], scenario_timestamps[-1], int(dt_ms))
            for t in train_timestamps:
                if t not in time_series:
                    time_series[t] = {}
                time_series[t]['train'] = True
            scenario_timestamps = dutils.keys2timestamps(time_series.keys())
        return time_series, scenario_timestamps

