import os
import argparse
import torch
import probtrack
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS, TRANSFORMS, OPTIMIZERS, PARAM_SCHEDULERS
from tqdm import tqdm, trange
import probtrack.datasets.utils as dutils
import sys
import json
import numpy as np
# sys.path.append('/home/csamplawski/src/iobtmax-data-tools')
#from spatial_transform_utils import *
# from probtrack.geometry.distributions import reduce_dim, threshold_dist, scale_dist, shift_dist, transform_dist_for_viz
# import torch.distributions as D
# import numpy as np
# import ffmpeg
# from probtrack.models.output_heads.matching import linear_assignment, prune_pixels
# import collections
# import copy
from metrics import compute_frame_metrics
from loop import tracking_loop

parser = argparse.ArgumentParser()
parser.add_argument('config', help='path to folder with config.py')
parser.add_argument('expdir', help='path to folder with config.py')
parser.add_argument('--first_n', type=float, default=1.0, help='first n percent of data to use')
parser.add_argument('--train_threshold', type=float, default=-1.0, help='threshold for training')
args = parser.parse_args()

cfg = Config.fromfile(args.config)

if cfg.coco_batch_size > 0:
    coco_trainset = DATASETS.build(cfg.coco_trainset)
else:
    coco_trainset = None

cfg.model.server = cfg.server
cfg.model.scenarios = cfg.scenarios
cfg.model.calibrated_scenarios = cfg.calibrated_scenarios
model = MODELS.build(cfg.model)
if os.path.exists(f'{args.expdir}/checkpoint.pt'):
    print(f'Loading model from {args.expdir}/checkpoint.pt')
    state_dict = torch.load(f'{args.expdir}/checkpoint.pt')
    model.load_state_dict(state_dict, strict=False)
model = model.eval().cuda()


# da_state_dict = model.da.state_dict()
# da_state_dict['clutter_prob_prime'] = torch.log(torch.tensor(0.9))
# model.da.load_state_dict(da_state_dict)

cfg.optimizer.params = [
    {'params': model.detector.parameters(), 'lr': 1e-4},
    {'params': model.projs.parameters(), 'lr': 1e-4},
    {'params': model.da.parameters(), 'lr': cfg.optimizer.lr},
    {'params': model.dms.parameters(), 'lr': cfg.optimizer.lr},
]
optimizer = OPTIMIZERS.build(cfg.optimizer)




# cfg.scheduler.optimizer = optimizer
# scheduler = PARAM_SCHEDULERS.build(cfg.scheduler)

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# bbox_pipeline=[
    # dict(type='ClassThreshold', cls_indicies=[2,5,7,8], threshold=args.det_threshold),
    # dict(type='ClusterBBoxes', iou_threshold=0.1)
# ]
# bbox_pipeline = nn.Sequential(*[MODELS.build(p) for p in bbox_pipeline])

# datasets = dutils.build_datasets(cfg, subset=args.subset, use_cache=False)

# subset2frac = {
    # 'train': cfg.train_frac,
    # 'val': cfg.val_frac,
    # 'test': cfg.test_frac
# }

server = DATASETS.build(cfg.server)

# frac = subset2frac[cfg.subset]
total_duration = 0
scenarios = {}
for scenario, scenario_data in cfg.scenarios.items():
    time_series, timestamps = server.load_time_series(
        scenario,
        sensor_keys=scenario_data['sensor_keys'][cfg.subset],
        frac=scenario_data['subsets'][cfg.subset],
        save_fps=cfg.save_fps, 
        train_fps=cfg.train_fps
    )
    # rate = 100
    # dt = 1 / rate
    # gt = [v for k, v in time_series.items() if 'gt' in v]
    # gt_pos = [x['gt']['obj_position'] for x in gt]
    # gt_pos = torch.stack(gt_pos, dim=0)
    # velocity = (gt_pos[1:] - gt_pos[:-1]) / dt
    # accel = (velocity[1:] - velocity[:-1]) / dt
    # std_accel = accel.std(dim=0)
    # print(scenario)
    # print(std_accel)
    # print(std_accel.sum() / 2)
        
    timestamps = timestamps[0:int(len(timestamps) * args.first_n)]
    duration = timestamps[-1] - timestamps[0]
    print(scenario, duration)
    total_duration += duration
    time_series = {t: time_series[t] for t in timestamps}
    scenarios[scenario] = (time_series, timestamps)


bbox_pipeline = None
args.train_threshold = 0.3
if args.train_threshold != -1.0:
    bbox_pipeline = [
        dict(type='ClassThreshold', cls_indicies=[2,5,7,8], threshold=args.train_threshold),
        dict(type='ClusterBBoxes', iou_threshold=0.1)
    ]
    bbox_pipeline = [MODELS.build(p) for p in bbox_pipeline]
    bbox_pipeline = torch.nn.Sequential(*bbox_pipeline)


print(f'Total duration: {total_duration}')
scenario_names = list(scenarios.keys())
# print(cfg.scenarios)
if cfg.backprop:
    for i in trange(cfg.num_iters):
        if i == int(cfg.num_iters * 0.8) and i > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        rand_scenario = np.random.choice(scenario_names)
        sensor_keys = cfg.scenarios[rand_scenario]['sensor_keys'][cfg.subset]
        time_series, timestamps = scenarios[rand_scenario]
        if cfg.subseq_len > 0:
            rand_start = np.random.randint(0, len(timestamps) - cfg.subseq_len)
            sub_timestamps = timestamps[rand_start:rand_start+cfg.subseq_len]
            sub_time_series = {k: time_series[k] for k in sub_timestamps}
            model, optimizer, history = tracking_loop(rand_scenario, sub_time_series,
                    model, optimizer, server,
                    cfg, coco_trainset=coco_trainset, pbar=False, sensor_keys=sensor_keys,
                    bbox_pipeline=bbox_pipeline)
        else:
            model, optimizer, history = tracking_loop(scenario, time_series, model, optimizer, server, 
                    cfg, coco_trainset=coco_trainset, pbar=True, sensor_keys=sensor_keys,
                    bbox_pipeline=bbox_pipeline)

    checkpoint_fname = f'{args.expdir}/checkpoint.pt'
    torch.save(model.state_dict(), checkpoint_fname)


# if cfg.viz:
    # vid_fname = f'{args.expdir}/{cfg.subset}_{scenario}.mp4'
    # model, optimizer, history = tracking_loop(
        # scenario, time_series, model, optimizer, server, 
        # cfg, coco_trainset=None, vid_fname=vid_fname, pbar=True)



if not cfg.backprop:
    for scenario in cfg.scenarios.keys():
        time_series, timestamps = scenarios[scenario]
        sensor_keys = cfg.scenarios[scenario]['sensor_keys'][cfg.subset]
        for threshold in cfg.thresholds:
            print(threshold, scenario)
            pipeline = [
                dict(type='ClassThreshold', cls_indicies=[2,5,7,8], threshold=threshold),
            ]
            if threshold != 0.0:
                pipeline.append(dict(type='ClusterBBoxes', iou_threshold=0.1))
            bbox_pipeline = [MODELS.build(p) for p in pipeline]
            bbox_pipeline = torch.nn.Sequential(*bbox_pipeline)
            
            vid_fname = f'{args.expdir}/{cfg.subset}_{threshold}_{scenario}.mp4'

            model, optimizer, history = tracking_loop(scenario, time_series, model, optimizer, 
                server, cfg, bbox_pipeline=bbox_pipeline, coco_trainset=None, 
                vid_fname=vid_fname, pbar=True, sensor_keys=sensor_keys)
            results_fname = f'{args.expdir}/results_{cfg.subset}_{threshold}_{scenario}.pt'
            torch.save(history, results_fname)
