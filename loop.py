import os
import torch
import probtrack
from mmengine.config import Config
from mmengine.registry import DATASETS, MODELS, TRANSFORMS,OPTIMIZERS, PARAM_SCHEDULERS
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import probtrack.datasets.utils as dutils
from probtrack.datasets.coco import detr_coco_loss
from torch.utils.data.dataloader import default_collate
import sys
from spatial_transform_utils import *
import torch_tracking as tracking
import cv2
import probtrack.viz.utils as vutils 
from probtrack.geometry.distributions import reduce_dim, threshold_dist, scale_dist, shift_dist, transform_dist_for_viz
import torch.distributions as D
import numpy as np
import ffmpeg
from probtrack.models.output_heads.matching import linear_assignment, prune_pixels
from metrics import compute_frame_metrics
from probtrack.models.output_heads.proj import compute_gt_bboxes

#orange (255, 165, 0)
#purple (255, 0, 255)

colors_dict = {
    'zed_node_1': (0, 255, 0), #green
    'zed_node_2': (255, 0, 255), #pink
    'zed_node_3': (0, 165, 255), #orange
    'zed_node_4': (255, 165, 0), #teal
}

track_colors = [(0, 0, 255), (255, 0, 0)]
max_vals = vutils.mocap_info['bounds']['max']
max_vals = torch.tensor(max_vals).float() / 1000
max_vals = max_vals[0:2].cuda()
min_vals = vutils.mocap_info['bounds']['min']
min_vals = torch.tensor(min_vals).float() / 1000
min_vals = min_vals[0:2].cuda()

def tracking_loop(scenario, time_series, model, optimizer, server, cfg, 
        sensor_keys=None,
        coco_trainset=None, bbox_pipeline=None, vid_fname=None, pbar=True):
    num_swaps = 0
    timestamps = dutils.keys2timestamps(time_series.keys())
    history = {k: [] for k in sensor_keys}
    history['track'] = []
    track_history_for_plot = []
    gt_history_for_plot = []

    curr_dets = {k: None for k in sensor_keys}

    loss_dict = {k: [] for k in cfg.loss_weights.keys()}
    map_img = vutils.init_map(H=1080, W=1920)
    gt, vid = None, None
    frames = {}
    num_frames_written = 0
    active_sensors = ['zed_node_1', 'zed_node_2', 'zed_node_3', 'zed_node_4']
    if pbar:
        timestamps = tqdm(timestamps, desc='Detecting')
    for timestamp in timestamps:
        data = time_series[timestamp]
        if 'gt' in data and gt is None:
            gt = data['gt']
            tracker, Z, P = model.init_tracker(gt, scenario=scenario)
            tracker_timestamp = timestamp

        if 'gt' in data: #and args.track:
            gt = data['gt']
            
            if cfg.track:
                dt = (timestamp - tracker_timestamp) / 1000
                if dt > 0: #if we just initialized tracker, don't need to advance
                    Z, P = tracker.advance(Z, P, dt=dt)
                    active_sensors = tracker.scheduler.update(tracker,Z,P,timestamp/1000,dt)

                tracker_timestamp = timestamp
                
                track_nll_loss = 0
                track_normals = model.mm.project_Z2X_dist(Z, P)
                means = torch.cat([normal.mean for normal in track_normals], dim=0)
                gt_pos = gt['obj_position'][..., 0:2].cuda()
                cost = torch.cdist(means, gt_pos)
                assign_idx = linear_assignment(cost)
                if torch.all(assign_idx == torch.eye(2)):
                    swap_idx = torch.tensor([1, 0]).long()
                    Z = Z[swap_idx]
                    P = P[swap_idx]
                    track_normals = model.mm.project_Z2X_dist(Z, P)
                    print('SWAP OCCURRED')
                    num_swaps += 1
                
                nll_vals, rmse_vals = [], []
                for j, normal in enumerate(track_normals):
                    gt_pos = gt['obj_position'][j, 0:2].cuda()
                    nll_loss_j = -normal.log_prob(gt_pos).mean()
                    rmse_loss_j = torch.sqrt(((normal.mean - gt_pos) ** 2).sum())
                    if torch.isnan(nll_loss_j):
                        import ipdb; ipdb.set_trace() # noqa
                        nll_loss_j = torch.zeros(1).cuda().mean()
                        print('track_nll_loss is nan')
                    nll_vals.append(nll_loss_j.item())
                    rmse_vals.append(rmse_loss_j.item())
                    track_nll_loss += nll_loss_j
                track_nll_loss /= len(track_normals)
                loss_dict['track_nll_loss'].append(track_nll_loss)
                history['track'].append({
                    'nll': nll_vals,
                    'rmse': rmse_vals,
                    'means': torch.cat([normal.mean.detach().cpu() for normal in track_normals], dim=0),
                    'covs': torch.cat([normal.covariance_matrix.detach().cpu() for normal in track_normals],dim=0),
                    'gt': gt,
                    'timestamp': timestamp,
                    'active_sensors': active_sensors,
                })

        if gt is None:
            continue
        
        curr_sensor_keys = [k for k in data if k != 'gt' and k != 'save' and k != 'train']
        curr_sensor_keys = [k for k in curr_sensor_keys if k in sensor_keys]
        if len(curr_sensor_keys) > 1 and vid is None and cfg.viz and vid_fname is not None:
            vid_h = 1080
            vid_w = 1920
            vid_size = f'{vid_w}x{vid_h}'
            vid = (ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=vid_size, r=str(cfg.save_fps))
                .output(vid_fname, vcodec='libx264', an=None, pix_fmt='yuv420p')
                .global_args('-y')
                .global_args('-loglevel', 'error')
                .run_async(pipe_stdin=True)
            )
        
        tracker_input = {}
        for sensor_key in curr_sensor_keys:
            node_idx = sensor_key.split('_')[-1]
            node_str = f'node{node_idx}'
            proj = model.projs[scenario][node_str]

            det_gt = compute_gt_bboxes(gt, proj)
            model_input = server.process_instance(data[sensor_key])
            assert model_input['timestamp'].item() == timestamp
            fname = model_input['fnames'][0]
            dets = model.detector(fname)[0]
            dets.infer_labels()
            
            if cfg.backprop:
                if bbox_pipeline is not None:
                    detection_loss, det_normals, dets = model.forward_train_no_matching(
                            dets, gt, scenario, node_str, 
                            bbox_pipeline)
                else:
                    detection_loss, det_normals, dets = model.forward_train(dets, gt, scenario, node_str)
                for k, v in detection_loss.items():
                    if k in loss_dict:
                        loss_dict[k].append(v)
            else:
                if bbox_pipeline is not None:
                    dets = bbox_pipeline(dets)
                det_normals, dets = model.forward_test(dets, scenario, node_str)
            
            means = [normal.mean for normal in det_normals]
            covs = [normal.covariance_matrix for normal in det_normals]
            if len(means) > 0:
                tracker_input[sensor_key] = {
                    'X': torch.cat(means, dim=0),
                    'R': torch.cat(covs, dim=0),
                }
            
            frame_metrics = compute_frame_metrics(
                det_normals,
                dets.labels,
                gt['obj_position'][..., 0:2].cuda(),
                gt['obj_cls_idx'].cuda(),
                det_gt['is_viewable'],
            )
            frame_metrics['timestamp'] = timestamp
            frame_metrics['fname'] = fname
            if len(means) == 0:
                frame_metrics['means'] = torch.empty(0,2).float().cpu()
                frame_metrics['covs'] = torch.empty(0,2,2).float().cpu()
            else:
                frame_metrics['means'] = torch.cat(means, dim=0).detach().cpu()
                frame_metrics['covs'] = torch.cat(covs, dim=0).detach().cpu()
            frame_metrics['is_viewable'] = det_gt['is_viewable'].cpu()
            frame_metrics['bboxes'] = dets.bboxes_cxcywh.detach().cpu()
            history[sensor_key].append(frame_metrics)

            curr_dets[sensor_key] = {
                'dets': dets,
                'normals': det_normals,
                'proj': proj,
                'fname': fname
            }

                    
            if vid is not None:
                frames[sensor_key] = cv2.imread(fname)
                # if len(dets) != 100: #hack for baselines which always have 100 dets
                    # frames[sensor_key] = dets.plot(frames[sensor_key], colors_dict[sensor_key])
                # for bbox in det_gt['gt_bboxes']:
                    # cx, cy, w, h = bbox
                    # H, W = 1080, 1920
                    # cx, cy, w, h = cx * W, cy * H, w * W, h * H
                    # x = int(cx - w / 2)
                    # y = int(cy - h / 2)
                    # w = int(w)
                    # h = int(h)
                    # frames[sensor_key] = cv2.rectangle(
                        # frames[sensor_key],
                        # (x, y),
                        # (x + w, y + h),
                        # colors_dict[sensor_key],
                        # 4,
                    # )

        if len(tracker_input) > 0 and cfg.track:
            dt = (timestamp - tracker_timestamp) / 1000
            if dt > 0: 
                Z, P = tracker.advance(Z, P, dt=dt)
                active_sensors = tracker.scheduler.update(tracker,Z,P,timestamp/1000,dt)
            tracker_input = {k: tracker_input[k] for k in tracker_input if k in active_sensors}
            Z, P = tracker.um.update(tracker_input, Z, P)
            tracker_timestamp = timestamp
                    
        if 'train' in data:
            if cfg.backprop and cfg.coco_batch_size > 0 and coco_trainset is not None:
                random_coco_idx = np.random.choice(len(coco_trainset), size=cfg.coco_batch_size)
                coco_batch = [coco_trainset[i] for i in random_coco_idx]
                coco_batch = [ele for ele in coco_batch if ele is not None]
                for sample in coco_batch:
                    model_output_i = model.forward_train_coco(sample)
                    for k, v in model_output_i.items():
                        if k in loss_dict:
                            loss_dict[k].append(v)

            for k, v in loss_dict.items():
                if len(v) == 0:
                    v = torch.zeros(1).cuda() * model.dummy_param
                    v = v.mean()
                else:
                    v = torch.stack(v).mean()
                loss_dict[k] = v * cfg.loss_weights[k]
            
            final_loss = (0 * model.dummy_param).mean()
            for k, v in loss_dict.items(): 
                final_loss += v 

            if cfg.backprop:
                final_loss.backward(retain_graph=True)
                for name, param in model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        param.grad = torch.zeros_like(param.grad)
                        print(f'grad is nan for {name}')
                        # import ipdb; ipdb.set_trace() # noqa
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()
            print(model.da)
            print(model.da.Rscale)
            # print([dm.std_acc for dm in model.dms])
            print({k: v.item() for k, v in loss_dict.items()})

            loss_dict = {k: [] for k in cfg.loss_weights.keys()}
            Z = Z.detach()
            P = P.detach()
            

        if 'save' in data and vid is not None:
            map_img = vutils.draw_mocap(map_img, gt, timestamp=None, H=1080, W=1920)

            if cfg.track:
                track_normals = model.mm.project_Z2X_dist(Z, P)
                for i, normal in enumerate(track_normals):
                    map_img = vutils.plot_dist(map_img,
                        dist=transform_dist_for_viz(
                            normal, min_vals, max_vals, 
                            img_scale=torch.tensor([1920, 1080]).float().cuda()
                        ),
                        color=track_colors[i],
                    )

                # track_history_for_plot.append(track_normals)
                # gt_history_for_plot.append(gt)
                # r = np.arange(len(track_history_for_plot))
                # r = r[-32:-1:8]
                # for idx in r:
                    # normals = track_history_for_plot[idx]
                    # gt_for_plot = gt_history_for_plot[idx]
                    # map_img = vutils.draw_mocap(map_img, gt_for_plot, timestamp=None, H=1080, W=1920, point_only=False)
                    # for i, normal in enumerate(normals):
                        # map_img = vutils.plot_dist(map_img,
                            # dist=transform_dist_for_viz(
                                # normal, min_vals, max_vals, 
                                # img_scale=torch.tensor([1920, 1080]).float().cuda()
                            # ),
                            # color=track_colors[i],
                        # )


            for sensor_key in sensor_keys:
                if curr_dets[sensor_key] is None or sensor_key not in frames or sensor_key not in active_sensors:
                    continue
                det_data = curr_dets[sensor_key]
                proj = det_data['proj']
                if len(det_data['dets']) != 100:
                    frames[sensor_key] = det_data['dets'].plot(frames[sensor_key], colors_dict[sensor_key])
                for normal in det_data['normals']:
                    map_img = vutils.plot_dist(map_img,
                        dist=transform_dist_for_viz(
                            normal, min_vals, max_vals, 
                            img_scale=torch.tensor([1920, 1080]).float().cuda()
                        ),
                        color=colors_dict[sensor_key],
                        cov_line=False,
                        alpha=0.3
                    )
                map_img = vutils.draw_node_from_proj(
                    map_img, proj, color=colors_dict[sensor_key]
                )

            img_grid = np.zeros((3, 3, 1080, 1920, 3), dtype=np.uint8)
            if 'zed_node_1' in frames:
                img_grid[1][2] = frames['zed_node_1']
            if 'zed_node_2' in frames:
                img_grid[0][1] = frames['zed_node_2']
            if 'zed_node_3' in frames:
                img_grid[1][0] = frames['zed_node_3']
            if 'zed_node_4' in frames:
                img_grid[2][1] = frames['zed_node_4']
            img_grid[1][1] = map_img
            img_grid = np.vstack([np.hstack(row) for row in img_grid])
            img_grid = cv2.resize(img_grid, (1920, 1080))
            vid.stdin.write(img_grid)
            num_frames_written += 1
            map_img = vutils.init_map(H=1080, W=1920)

    if vid is not None:
        vid.stdin.close()
    history['num_swaps'] = num_swaps
    return model, optimizer, history
