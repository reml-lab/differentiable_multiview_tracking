import torch
import numpy as np
import cv2
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import torch.distributions as D
from probtrack.geometry.distributions import dist_to_np
from datetime import datetime
import sys
sys.path.append('/home/csamplawski/src/iobtmax-data-tools')
from spatial_transform_utils import *

mocap_info = {
    'objects': {
        'node': {'size': [150, 300, 0], 'id': 0, 'color': (0, 0, 0)},
        'truck': {'size': [300, 150, 0.1452*100], 'id': 1, 'color': (0, 165, 255)}, # orange
        'bus': {'size': [310, 80, 80], 'id': 2, 'color': (0, 0, 255)}, # red
        'car': {'size': [290, 130, 70], 'id': 3, 'color': (255, 0, 0)}, # blue
        'drone': {'size': [320, 270, 80], 'id': 4, 'color': (128, 128, 128)}, # gray
        'tunnel': {'size': [300, 300, 0], 'id': 5, 'color': (42, 42, 165)} # brown
    },
    'bounds': {
        'min': np.array([-2776.4158, -2485.51226, 0.49277]),
        'max': np.array([5021.02426, 3147.80981, 1621.69398])
    }
}

def plot_pixel(img, gt):
    # node_ids = imgs.keys()
    viewable = gt['viewable']
    pixels = gt['pixels']
    for pixel in pixels:
        if torch.any(pixel.isnan()):
            continue
        pixel[0] *= 1920
        pixel[1] *= 1080
        pixel = [pixel[0].item(), pixel[1].item()] #TODO: hack
        img = cv2.circle(img, 
            (int(pixel[0]), int(pixel[1])), 
            10, (0, 0, 255), -1
        )
    score = viewable[0].item()
    img = cv2.putText(img, 
        f'{score:.2f}', 
        (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 
        2, (0, 0, 255), 2, cv2.LINE_AA
    )
    # imgs[node_id] = img
    return img

#dist should already scaled to pixel space
def plot_dist(img, dist, chi_sq=5.991, color=(0, 0, 255), alpha=0.1, cov_line=True):
    means, covs, probs = dist_to_np(dist)

    for i in range(len(means)):
        mean = means[i]
        cov = covs[i]
        
        try: 
            eigvals, eigvecs = np.linalg.eig(cov)
        except:
            import ipdb; ipdb.set_trace() # noqa
        ind = np.argsort(-1*eigvals)
        eigvals = eigvals[ind]
        eigvecs = eigvecs[:,ind]

        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0]) #TODO: y, x order?
        angle *= (180 / np.pi)
        axes_lengths = np.sqrt(eigvals * chi_sq)
        # try:
            # img = cv2.circle(img, 
                # (int(mean[0]), int(mean[1])), 
                # 5, color, -1
            # )
        # except:
            # import ipdb; ipdb.set_trace() # noqa


        if np.any(np.isnan(axes_lengths)):
            continue

        # img = cv2.ellipse(
            # img,
            # (int(mean[0]), int(mean[1])),
            # (int(axes_lengths[0]), int(axes_lengths[1])), 
            # angle, 0, 360, 
            # color, 2
        # )

        if cov_line:
            try:
                img = cv2.ellipse(
                    img,
                    center=(int(mean[0]), int(mean[1])),
                    axes=(int(axes_lengths[0]), int(axes_lengths[1])),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=color,
                    thickness=4,
                    lineType=cv2.LINE_AA
                )
            except:
                print('failed to draw cov line')
                continue
        try:
            overlay = img.copy()
            cv2.ellipse(
                overlay,
                center=(int(mean[0]), int(mean[1])),
                axes=(int(axes_lengths[0]), int(axes_lengths[1])),
                angle=angle,
                startAngle=0,
                endAngle=360,
                color=color,
                thickness=-1
            )
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        except:
            print('failed to draw cov fill')
            continue
    return img


def plot_grid(img, grid_size=100, color=(0,0,0), thickness=1):
    for x in range(0, img.shape[1], grid_size):
        cv2.line(img, (x, 0), (x, img.shape[0]), color, thickness)
    for y in range(0, img.shape[0], grid_size):
        cv2.line(img, (0, y), (img.shape[1], y), color, thickness)
    return img

def init_map(H=1080, W=1920):
    map_img = np.zeros((int(H), int(W), 3), dtype=np.uint8) + 255
    gray = (128, 128, 128)
    map_img = plot_grid(map_img, thickness=1, color=gray)
    return map_img

def draw_node_from_proj(map_img, proj, color=(0, 0, 0)):
    params = proj.get_parameter_vector().squeeze()
    X, Y, Z, _, _, _, roll, pitch, yaw = params
    X, Y, Z = X*1000, Y*1000, Z*1000
    min_vals = mocap_info['bounds']['min']
    max_vals = mocap_info['bounds']['max']
    rot_matrix = euler_to_rot_torch(
        roll.unsqueeze(0),
        pitch.unsqueeze(0),
        yaw.unsqueeze(0)
    ).squeeze(0)
    rot_matrix = rot_matrix[0:2, 0:2]
    pos = torch.tensor([X, Y, Z])
    pos = pos[0:2]
    pos = (pos - min_vals[0:2]) / (max_vals[0:2] - min_vals[0:2])
    pos[0] *= map_img.shape[1]
    pos[1] *= map_img.shape[0]
    node_w, node_h, _ = mocap_info['objects']['node']['size']
    node_w = node_w / (max_vals[0] - min_vals[0])
    node_h = node_h / (max_vals[1] - min_vals[1])
    map_img = draw_rotated_box(map_img, pos.cpu(), node_w, node_h, 
            rot_matrix.cpu(), color=color, arrow_length=40)
    return map_img


def draw_mocap(map_img, gt, timestamp=None, H=1080, W=1920, point_only=False,
        color=(0, 0, 0)):
    min_vals = mocap_info['bounds']['min']
    max_vals = mocap_info['bounds']['max']
    
    # node_pos = mocap_data['normalized_location']['node_position'][:, 0]
    #node_rot = mocap_data['location']['node_rotation'][:, 0]
    
    #timestamp in ms -> YYYY-MM-DD HH:MM:SS:MS
    # timestamp = int(gt['timestamp'][0].item())
    # timestamp = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S.%f')
    # map_img = cv2.putText(map_img,
        # timestamp, (10, 30), 
        # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
    # )

    # node_pos = gt['node_position'][0]
    # num_nodes = len(node_pos)
    # for i in range(num_nodes):
        # type = 'node'
        # color = mocap_info['objects'][type]['color']
        # node_w, node_h, _ = mocap_info['objects'][type]['size']
        # node_w = node_w / (max_vals[0] - min_vals[0])
        # node_h = node_h / (max_vals[1] - min_vals[1])
        # pos = node_pos[i][0:2]
        # pos = (pos - min_vals[0:2]) / (max_vals[0:2] - min_vals[0:2])
        # pos[0] *= map_img.shape[1]
        # pos[1] *= map_img.shape[0]
        # node_id = str(i+1)
        # map_img = cv2.putText(map_img,
            # node_id, 
            # (int(pos[0]), int(pos[1])), 
            # cv2.FONT_HERSHEY_SIMPLEX, 
            # 2, (0, 0, 0), 4, cv2.LINE_AA
        # )
        # rot = node_rot[i][0:2, 0:2].T
        # map_img = draw_rotated_box(map_img, pos, node_w, node_h, rot, color=color)
    
    # obj_pos = mocap_data['normalized_location']['obj_position'][0] #TODO: batched
    # obj_rot = mocap_data['location']['obj_rotation'][0]
    obj_pos = gt['obj_position'] * 1000
    obj_rot = gt['obj_rotation']
    obj_dims = gt['obj_dims'] * 1000
    obj_ids = gt['obj_id']
    # obj_rot = gt['obj_rot']
    num_objs = len(obj_pos)
    
    for i in range(num_objs):
        type = 'truck'
        # color = mocap_info['objects'][type]['color']
        obj_w, obj_h, _ = obj_dims[i]
        # obj_w, obj_h, _ = mocap_info['objects'][type]['size']
        obj_w = obj_w / (max_vals[0] - min_vals[0])
        obj_h = obj_h / (max_vals[1] - min_vals[1])
        pos = obj_pos[i][0:2]
        pos = (pos - min_vals[0:2]) / (max_vals[0:2] - min_vals[0:2])
        pos[0] *= map_img.shape[1]
        pos[1] *= map_img.shape[0]
        rot = obj_rot[i][0:2, 0:2].T
        color = (0, 0, 0)

        if point_only:
            map_img = cv2.circle(map_img,
                (int(pos[0]), int(pos[1])),
                5, color, -1
            )
        else:
            map_img = draw_rotated_box(map_img, pos, obj_w, obj_h, rot, color=color)
    
    # if timestamp is not None:
        # time_str = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S.%f')
        # map_img = cv2.putText(map_img, 
            # time_str, (10, 30), 
            # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        # )
    return map_img

def draw_rotated_box(image, center_normalized, width, height, rotation_matrix,
                     color=(0, 255, 0), arrow_length=50):
    # Convert normalized coordinates to image coordinates
    center_x, center_y = center_normalized
    # center_x *= image.shape[1]
    # center_y *= image.shape[0]

    height *= image.shape[0]
    width *= image.shape[1]

    # Half dimensions of the box
    half_width = width / 2
    half_height = height / 2

    # Coordinates of the box corners before rotation
    box_corners = np.array([
        [center_x - half_width, center_y - half_height],
        [center_x + half_width, center_y - half_height],
        [center_x + half_width, center_y + half_height],
        [center_x - half_width, center_y + half_height]
    ])

    # Apply rotation to each corner
    rotated_corners = []
    for corner in box_corners:
        # Shift to origin for rotation
        shifted_corner = np.array([corner[0] - center_x, corner[1] - center_y])
        # Apply rotation
        rotated_corner = np.dot(rotation_matrix, shifted_corner)
        # Shift back
        rotated_corner += np.array([center_x, center_y])
        rotated_corners.append(rotated_corner)

    # Convert float to int for drawing
    rotated_corners = np.int32(rotated_corners)

    # Draw the box
    cv2.polylines(image, [rotated_corners], isClosed=True, color=color, thickness=4)

    direction = rotation_matrix[:, 0]  # Using the first column of the rotation matrix
    arrow_head_x = int(center_x + arrow_length * direction[0])
    arrow_head_y = int(center_y + arrow_length * direction[1])

    # Draw the arrow
    cv2.arrowedLine(image, (int(center_x), int(center_y)), (arrow_head_x, arrow_head_y), color, 5, tipLength=0.3)
    return image

#https://gamedev.stackexchange.com/questions/86755/how-to-calculate-corner-positions-marks-of-a-rotated-tilted-rectangle
def is_on_right_side(points, v1, v2):
    x0, y0 = v1
    x1, y1 = v2
    a = y1 - y0
    b = x0 - x1
    c = - a*x0 - b*y0
    return a*points[:,0] + b*points[:,1] + c >= 0

def points_in_rec(points, rec):
    corners = rec.get_corners()
    num_corners = len(corners)
    is_right = [is_on_right_side(points, corners[i], corners[(i + 1) % num_corners]) for i in range(num_corners)]
    is_right = np.stack(is_right, axis=1)
    all_left = ~np.any(is_right, axis=1)
    all_right = np.all(is_right, axis=1)
    final = all_left | all_right
    return final

def rot_matrix(angle):
    rad = 2*np.pi * (angle/360)
    R = [np.cos(rad), np.sin(rad),-np.sin(rad), np.cos(rad)]
    R = np.array(R).reshape(2,2)
    R = torch.from_numpy(R).float()
    return R


#https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals/12321306#12321306
def gen_ellipse(pos, cov, nstd=np.sqrt(5.991), **kwargs):
    if len(pos) > 2:
        pos = pos[0:2]
        cov = cov[0:2, 0:2]
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    return ellip

def rot2angle(rot, return_rads=True):
    if rot[4] <= 0:
        rads = np.arcsin(rot[3]) / (2*np.pi)
    else:
        rads = np.arcsin(rot[1]) / (2*np.pi)
    if not return_rads:
        rads *= 360
    return rads


def gen_rectange(pos, angle, w, h, color='black'):
    # angle = rot2angle(rot, return_rads=False)
    rec = Rectangle(xy=([pos[0]-w/2, pos[1]-h/2]), width=w, height=h, angle=angle, rotation_point='center',
                        edgecolor=color, fc='None', lw=5)
    corners = rec.get_corners()

    x = np.arange(0.5,30,1) / 100.0
    y = np.arange(0.5,15,1) / 100.0
    X, Y = np.meshgrid(x,y)
    grid = np.stack([X,Y])
    grid = torch.from_numpy(grid).float()
    grid = grid.permute(1,2,0)
    grid = grid.reshape(-1,2)
    R = rot_matrix(angle)
    grid = torch.mm(grid, R)
    grid[:,0] += corners[0][0]
    grid[:,1] += corners[0][1]
    return rec, grid

def init_fig_(valid_mods, num_cols=4, colspan=1):
    #assert ('mocap', 'mocap') in valid_mods

    mods = [vk[0] for vk in valid_mods if vk != ('mocap', 'mocap')]
    mods = sorted(list(set(mods)))
    num_mods = len(mods)
    num_cols = num_mods + 1
    num_rows = 4
    
    fig = plt.figure(figsize=(num_cols*10, num_rows*10))
    axes = {}
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (1, 0), 
            rowspan=1, colspan=1)

    axes[('mocap', 'mocap')].linewidth = 5
    axes[('mocap', 'mocap')].node_size = 20*4**2

    valid_mods = [vk for vk in valid_mods if vk != ('mocap', 'mocap')]
    for i, key in enumerate(valid_mods):
        col = mods.index(key[0])
        row = int(key[1].split('_')[-1]) - 1
        # row += 2
        col += 1

        # x, y = i % num_mods, num + num_mods
        print(row, col, key)
        axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes

def init_fig_vert(valid_mods, num_cols=4, colspan=1):
    #assert ('mocap', 'mocap') in valid_mods

    mods = [vk[0] for vk in valid_mods if vk != ('mocap', 'mocap')]
    mods = sorted(list(set(mods)))
    num_mods = len(mods)
    num_cols = 2 + 4
    num_rows = num_mods
    
    fig = plt.figure(figsize=(num_cols*10, num_rows*10))
    axes = {}
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (0, 0), 
            rowspan=2, colspan=2)

    valid_mods = [vk for vk in valid_mods if vk != ('mocap', 'mocap')]
    for i, key in enumerate(valid_mods):
        row = mods.index(key[0])
        col = int(key[1].split('_')[-1]) - 1
        col += 2
        # x, y = i % num_mods, num + num_mods
        print(row, col, key)
        axes[key] = plt.subplot2grid((num_rows, num_cols), (row, col))
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes

def init_fig(valid_mods, num_cols=4, colspan=1):
    #assert ('mocap', 'mocap') in valid_mods

    mods = [vk[0] for vk in valid_mods if vk != ('mocap', 'mocap')]
    num_mods = len(set(mods))
    num_cols = num_mods + 2 + 1
    num_rows = num_mods + 2 + 1
    
    fig = plt.figure(figsize=(num_cols*16, num_rows*9))
    axes = {}
    axes[('mocap', 'mocap')] = plt.subplot2grid((num_rows, num_cols), (1, 1), rowspan=num_mods + 1, colspan=num_mods+1)

    axes[('mocap', 'mocap')].linewidth = 20
    axes[('mocap', 'mocap')].node_size = 20*16**2

    #row, col = 0, colspan
    node2row = {'node_2': num_rows-1, 'node_4': 0}
    node2col = {'node_3': 0, 'node_1': num_cols-1}
   
    valid_mods = [vk for vk in valid_mods if vk != ('mocap', 'mocap')]
    for node_num, col_num in node2col.items():
        count = 1
        for i, key in enumerate(valid_mods):
            if key[1] != node_num:
                continue
            axes[key] = plt.subplot2grid((num_rows, num_cols), (count, col_num))
            count += 1

    
    for node_num, row_num in node2row.items():
        count = 1
        for i, key in enumerate(valid_mods):
            if key[1] != node_num:
                continue
            axes[key] = plt.subplot2grid((num_rows, num_cols), (row_num, count))
            count += 1
             
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    return fig, axes
