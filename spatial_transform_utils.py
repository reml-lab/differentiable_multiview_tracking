import os
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.spatial.transform import Rotation
import torch.nn as nn
import copy
import json
import numpy as np
import sys
# sys.path.append('/home/csamplawski/src/iobt-tracker')
# from probtrack.models.attn import QKVAttention
# from spatial_transform_utils import point_projector, MultiViewProjector

def generate_grid(H, W):
    x = torch.linspace(0, 1, steps=W, dtype=torch.float32)
    y = torch.linspace(0, 1, steps=H, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(x, y), dim=-1)
    return grid

def meters_per_pixel(lat, zoom):
    #reference: https://medium.com/techtrument/how-many-miles-are-in-a-pixel-a0baf4611fff
    #Reference does not seem to account for scale=2 when getting maps
    #Needs zoom+1, no zomm
    return (156543.03392 * torch.cos(torch.tensor(lat * torch.pi / 180.0)) / (torch.pow(torch.tensor(2.0), 1+zoom)))

def euler_to_rot(roll, pitch, yaw, order="XYZ"):
    rot = Rotation.from_euler(order, [roll.item(), pitch.item(), yaw.item()])
    return torch.tensor(rot.as_matrix()).float()

#roll, pitch, yaw tensors of shape (N,) in radians
def euler_to_rot_torch(roll, pitch, yaw, instrinsic=True):
    N = roll.shape[0]
    ones = torch.ones_like(roll)
    zeros = torch.zeros_like(roll)
    
    Rx = torch.cat([
        torch.cat([ones, zeros, zeros]),
        torch.cat([zeros, torch.cos(roll), -torch.sin(roll)]),
        torch.cat([zeros, torch.sin(roll), torch.cos(roll)])
    ]).reshape(3, 3, N).permute(2,0,1)

    Ry = torch.cat([
        torch.cat([torch.cos(pitch), zeros, torch.sin(pitch)]),
        torch.cat([zeros, ones, zeros]),
        torch.cat([-torch.sin(pitch), zeros, torch.cos(pitch)])
    ]).reshape(3, 3, N).permute(2,0,1)

    Rz = torch.cat([
        torch.cat([torch.cos(yaw), -torch.sin(yaw), zeros]),
        torch.cat([torch.sin(yaw), torch.cos(yaw), zeros]),
        torch.cat([zeros, zeros, ones])
    ]).reshape(3, 3, N).permute(2,0,1)

    if instrinsic:
        R = torch.bmm(Rx, torch.bmm(Ry, Rz))
    else:
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))

    # R = torch.bmm(Rx, torch.bmm(Ry, Rz))
    
    #TODO: this is the only way to match scipy, why?
    #R = R.permute(0, 2, 1)
    #index = torch.tensor([0, 2, 1]).long().to(R.device)
    #R = R[:, index]
    #R = R[:, :, index]
    return R

def to_implicit(lat, lon, zoom):
    implicit_height = (2*256)*(2**zoom)
    implicit_width  = (2*256)*(2**zoom)
    
    R = implicit_width/(2*torch.pi)
    FE = 180.0
    lonRad = torch.deg2rad(lon + FE)
    implicit_x = lonRad * R
    
    latRad = torch.deg2rad(lat)
    verticalOffsetFromEquator = R * torch.log(torch.tan(torch.pi / 4 + latRad / 2))
    implicit_y = implicit_height / 2 - verticalOffsetFromEquator
    
    return implicit_x, implicit_y

def to_pixel(lat, lon, center_lat, center_lon, zoom, width, height):
    cix, ciy = to_implicit(center_lat, center_lon, zoom)
    ix, iy = to_implicit(lat, lon, zoom)
    x = (ix - cix) + width/2.0
    y = (iy - ciy) + height/2.0
    return x, y

def gps_to_meters(points,center_lat=39.351,center_lon=-76.345,zoom=18,map_width=800,map_height=800,type="",xlim=[],ylim=[]):
    mppx   = meters_per_pixel(center_lat, zoom)
    x, y   = to_pixel(points[:,[0]], points[:,[1]], torch.tensor(center_lat), torch.tensor(center_lon), zoom=zoom, width=map_width, height=map_height)
    output = torch.hstack([(x-map_width/2)*mppx,-1*(y-map_height/2)*mppx,points[:,[2]]])
    return output.float()

def parse_gps_log(fname):
    with open(fname) as f:
        lines = f.readlines()
    data = [eval(l.strip()) for l in lines]
    df   = pd.DataFrame(data)
    pos  = torch.tensor([[df['lt'].mean(), df['ln'].mean(), df['al'].mean()]],dtype=torch.float64)
    std  = torch.tensor([[df['lt'].std(), df['ln'].std(), df['al'].std()]])
    return pos, std

def get_3d_box_points(pos,rot,dims, center_offsets=None, force_ground_plane=True):
    obj_dims = dims.unsqueeze(1)
    centers  = pos.clone().unsqueeze(1)

    
    if force_ground_plane:
        centers[...,2] = 0

    
    points_bot = torch.tensor([
        [-1,1,0],
        [1,1,0],
        [1,-1,0],
        [-1,-1,0]
    ]).float()
    points_bot /= 2
    points_bot = points_bot.to(pos.device)
    points_top = points_bot.clone()
    points_top[:,2] = 1

    points = torch.cat([points_bot,points_top]).unsqueeze(0)
    points = points*obj_dims

    points = (rot@points.transpose(-1,-2)).transpose(-1,-2)

    if center_offsets is not None:
        rotated_offset = (rot @ center_offsets.T).squeeze(1).squeeze(-1)
        centers += rotated_offset.unsqueeze(1)

    points += centers
    return(points)

def box3d_points_to_box2d_chw(points):

    minx = torch.min(points[...,0],axis=1)[0]
    maxx = torch.max(points[...,0],axis=1)[0]
    miny = torch.min(points[...,1],axis=1)[0]
    maxy = torch.max(points[...,1],axis=1)[0]

    h = maxy-miny
    w = maxx-minx
    cx = (maxx + minx)/2
    cy = (maxy + miny)/2

    return torch.vstack([cx,cy,w,h]).T

def get_boxes_from_track(obj_pos,obj_dims,proj):

    num_objs = obj_pos[0].shape[0]
    old_yaw  = torch.zeros(num_objs)
    roll     = torch.tensor([0,0])
    pitch    = torch.tensor([0,0])
    old_p    = obj_pos[0]

    box2d      = []
    box3d_proj = []

    for p in obj_pos:
        
        #Estimate yaw from velocity vector
        dvec     = p-old_p
        ind      = torch.norm(dvec[:,:2],dim=1)<=0.02
        this_yaw = torch.atan2(dvec[:,1],dvec[:,0])
        yaw      = this_yaw
        yaw[ind] = old_yaw[ind]
        old_yaw  = yaw.clone()
        rot      = euler_to_rot_torch(roll, pitch,yaw)
        old_p    = p 

        box3d    = get_3d_box_points(p,rot,obj_dims)
        box3d_proj.append(proj.world_to_image(box3d.reshape(-1,3)).detach().reshape(-1,8,3))
        box2d.append(box3d_points_to_box2d_chw(box3d_proj[-1]))

    return(box2d,box3d_proj)

def update_info(node_info,node,pitch=0,roll=0,yaw=0, mode='mocap'):
    
    if mode=='mocap':
        zed_offsets={}
        zed_offsets[1] = {"x":-0.25,"y":0.2,"z":-0.15}
        zed_offsets[2] = {"x":-0.2,"y":0.2,"z":-0.15}
        zed_offsets[3] = {"x":-0.05,"y":0.2,"z":-0.25}
        zed_offsets[4] = {"x":-0.2,"y":0.2,"z":-0.15}

        # node_info['zed']['location_offset_X'] = zed_offsets[node]['x'] 
        # node_info['zed']['location_offset_Y'] = zed_offsets[node]['y'] 
        # node_info['zed']['location_offset_Z'] = zed_offsets[node]['z'] 

        node_info['location']['X'] /= 1000
        node_info['location']['Y'] /= 1000
        node_info['location']['Z'] /= 1000
        
        # if(node==2 or node==4):
            # r=node_info['zed']['roll']
            # p=node_info['zed']['pitch']
            # node_info['zed']['roll']=p
            # node_info['zed']['pitch']=r

        node_info['zed']['roll']  = np.deg2rad(node_info['zed']['roll'] + roll)   
        node_info['zed']['pitch'] = -1*np.abs(np.deg2rad(node_info['zed']['pitch'] + pitch)) 
        node_info['zed']['yaw']   = np.deg2rad(node_info['zed']['yaw'] + yaw) - (np.pi / 2)
    
    
    node_info = {
        'fx':node_info['zed']['fx'],
        'fy':node_info['zed']['fy'],
        'cx':node_info['zed']['cx'],
        'cy':node_info['zed']['cy'],
        'roll': node_info['zed']['roll'],
        'pitch': node_info['zed']['pitch'],
        'yaw': node_info['zed']['yaw'],
        'dX': node_info['zed']['location_offset_X'],
        'dY': node_info['zed']['location_offset_Y'],
        'dZ': node_info['zed']['location_offset_Z'],
        't': torch.tensor([
            node_info['location']['X'],
            node_info['location']['Y'],
            node_info['location']['Z']
        ]).unsqueeze(-1)
    }
    for key, val in node_info.items():
        node_info[key] = torch.tensor(val, dtype=torch.float32)
    return node_info


def inverse_softplus(x):
    return torch.log(torch.exp(x) - 1)

class unscented_transform(nn.Module):

    def __init__(self,f,dim,alpha=1e-2,beta=2.0,diag_cov=False,lmbda=1/100, mode="advanced",diff_func=None, wmean_func=None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.f=f
        self.dim=dim
        self.alpha=alpha
        self.beta=beta
        self.lambd = self.compute_sigma_points_lambda()
        weights_m, weights_c = self.compute_sigma_weights()
        self.weights_m = weights_m.to(self.device)
        self.weights_c = weights_c.to(self.device)
        self.diag_cov = diag_cov
        self.lmbda = lmbda

        if diff_func is None:
            self.diff_func = lambda x,y: x-y
        else: 
            self.diff_func = diff_func

        if wmean_func is None:
            self.wmean_func = lambda w,x,d: torch.sum(x*w,dim=d,keepdim=True)  
        else: 
            self.wmean_func = wmean_func       
    
    def compute_sigma_points_lambda(self):
        """Compute sigma point scaling parameter.
        Args:
            dim (int): Dimensionality of input vectors.
        Returns:
            float: Lambda scaling parameter.
        # Code based on: https://github.com/stanford-iprl-lab/torchfilter/blob/master/torchfilter/utils/_sigma_points.py
        # Method based on: http://www.gatsby.ucl.ac.uk/~byron/nlds/merwe2003a.pdf
        """

        if(self.mode=="advanced"):
            kappa = 3.0 - self.dim
            out = (self.alpha ** 2) * (self.dim + kappa) - self.dim
        else:
            out=1

        return out

    def compute_sigma_weights(self): 
        """Helper for computing sigma weights.
        Args:
            dim (int): Dimensionality of input vectors.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Covariance and mean weights. We expect 1D
            float32 tensors on the CPU.
        # Code based on: https://github.com/stanford-iprl-lab/torchfilter/blob/master/torchfilter/utils/_sigma_points.py
        """

        # Create covariance weights
        weights_c = torch.full(
            size=(2 * self.dim + 1,),
            fill_value=1.0 / (2.0 * (self.dim + self.lambd)),
            dtype=torch.float32,
        )
        if(self.mode=="advanced"):
            weights_c[0] = self.lambd / (self.dim + self.lambd) + (1.0 - self.alpha ** 2 + self.beta)
        else:
            weights_c[0] = self.lambd / (self.dim + self.lambd)

        # Mean weights should be identical, except for the first weight
        weights_m = weights_c.clone()
        weights_m[0] = self.lambd / (self.dim + self.lambd)

        return weights_m.reshape(-1,1,1), weights_c.reshape(-1,1,1)

    def compute_sigma_points(self,input_mean, input_covariance):
        #Code based on: https://github.com/stanford-iprl-lab/torchfilter/blob/master/torchfilter/utils/_unscented_transform.py

        if(self.diag_cov):
            # input_covariance = torch.clamp(input_covariance,min=1e-6)
            input_scale_tril = torch.sqrt(input_covariance)
            if torch.isnan(input_scale_tril).any():
                import ipdb; ipdb.set_trace()
        else:
            input_scale_tril = torch.linalg.cholesky(input_covariance)

        matrix_root      = np.sqrt(self.dim + self.lambd) * input_scale_tril

        n = input_mean.shape[0]
        b = 2 * self.dim + 1
        zeros = torch.zeros(n,self.dim,1).to(input_mean.device)
        try:
            sigma_point_offsets = torch.cat([zeros, matrix_root, -matrix_root],dim=2).to(input_mean.device)
        except:
            import ipdb; ipdb.set_trace() # noqa
        #sigma_point_offsets[:,:, 1 : 1 + self.dim] = matrix_root
        #sigma_point_offsets[:,:, 1 + self.dim :]   = -matrix_root

        sigma_points = input_mean.reshape(n,self.dim,1) + sigma_point_offsets

        return sigma_points

    def transform(self,input_mean, input_covariance,return_cross_cov=False):
        #Reference: https://towardsdatascience.com/the-unscented-kalman-filter-anything-ekf-can-do-i-can-do-it-better-ce7c773cf88d

        #Get sigma points and weights
        sigma_points = self.compute_sigma_points(input_mean, input_covariance)
        sigma_points = sigma_points.permute([2,0,1])
        B,n,D = sigma_points.shape

        # n,D = input_mean.shape
        # B = 200

        # samples = torch.randn((B,n,D)).to(input_mean.device) 
        # input_cov_diag = torch.diagonal(input_covariance,dim1=1,dim2=2)
        # sigma_points = input_mean.unsqueeze(0) + samples*torch.sqrt(input_cov_diag).unsqueeze(0)

        
        assert not torch.isnan(sigma_points).any().item(), "Sigma point is nan"
        assert not torch.isinf(sigma_points).any().item(), "Sigma point is inf"

        #Compute weights mean and covariance of transformed points

        transformed_points = self.f(sigma_points.to(self.device))
        assert not torch.isnan(transformed_points).any().item(), "Transformed point is nan"

        #transformed_points = torch.zeros(B,3)
        #for i,point in enumerate(sigma_points):
        #    transformed_points[i,:]     = self.f(point)

        # self.weights_m = torch.ones(1).to(self.device) / n
        # self.weights_c = torch.ones(1).to(self.device) / n

        #transformed_mean       = torch.sum(transformed_points*self.weights_m.to(self.device),dim=0,keepdim=True)
        #diffs                  = transformed_points - transformed_mean

        transformed_mean       = self.wmean_func(self.weights_m.to(self.device), transformed_points,0)
        diffs                  = self.diff_func(transformed_points,transformed_mean)

        transformed_covariance = (self.weights_c*diffs).permute(1,2,0) @ diffs.permute(1,0,2)  #B x 3 x 3

        d = transformed_covariance.shape[-1]
        I = torch.eye(d).reshape(1,d,d).to(input_mean.device)
        transformed_covariance = transformed_covariance

        diag = torch.diagonal(transformed_covariance,dim1=1,dim2=2)
        # if torch.any(diag<=0).item():
            # print("Negative covariance diag")
            # import ipdb; ipdb.set_trace() # noqa
        #assert torch.all(diag>0).item(), print("Negative covariance")

        if(return_cross_cov):
            cross_covariance  =  (self.weights_c*(sigma_points-input_mean)).permute(1,2,0) @ diffs.permute(1,0,2)
            return transformed_mean.squeeze(dim=-1),transformed_covariance, cross_covariance
        else:
            return transformed_mean.squeeze(dim=-1),transformed_covariance

        #assert torch.abs(torch.max(transformed_mean))<10, "Large Z"

    def pd_project(X,lam=0):
        X      = torch.tril(X) + torch.tril(X.T,-1).T #Force symmetry
        u,s,v  = torch.linalg.svd(X+lam*torch.eye(X.shape[0])) #Decompose
        X      = (u@torch.diag(s)@u.T + v.T@torch.diag(s)@v)/2.0 #Reconstruct
        return(X)


def get_angle_corrections(data_file):

    df_recal   = pd.read_csv(data_file,sep="\t")
    recal_mean =  df_recal.median()

    corrections={}

    for n in range(4):
        corrections[f"node{n+1}"]={}
        corrections[f"node{n+1}"]["pitch"] = np.deg2rad(recal_mean[f"NOD_{n+1} Pitch"])
        corrections[f"node{n+1}"]["roll"] = np.deg2rad(recal_mean[f"NOD_{n+1} Roll"])
        corrections[f"node{n+1}"]["yaw"] = np.deg2rad(recal_mean[f"NOD_{n+1} Yaw"])

    #corrections["node_1"]["yaw"] -= np.deg2rad(1.5)
    corrections["node2"]["pitch"] -= np.deg2rad(5)
    corrections["node2"]["yaw"] += np.deg2rad(1)
    corrections["node3"]["yaw"] = 1.5*corrections["node3"]["yaw"]
    corrections["node3"]["pitch"] = -np.deg2rad(2)
    corrections["node4"]["pitch"] -= np.deg2rad(2)
    return corrections


def update_node_info(node_info,mocap_info,angle_corrections=None):

    for n in range(4):
        node = f"node_{n+1}"

        node_info[node]["zed"]["location_offset_X"]= 0.0381
        node_info[node]["zed"]["location_offset_Y"]= 0.2181
        node_info[node]["zed"]["location_offset_Z"]= -0.156

        node_info[node]['location']['X'] = mocap_info[n]['location']['X']/1000
        node_info[node]['location']['Y'] = mocap_info[n]['location']['Y']/1000
        node_info[node]['location']['Z'] = mocap_info[n]['location']['Z']/1000

        node_info[node]['zed']['roll']  = np.deg2rad(mocap_info[n]['zed']['roll']) 
        node_info[node]['zed']['pitch'] = np.deg2rad(mocap_info[n]['zed']['pitch']) 
        node_info[node]['zed']['yaw']   = np.deg2rad(mocap_info[n]['zed']['yaw']) 

        if angle_corrections is not None:
            node_info[node]['zed']['roll']  -= angle_corrections[node]["roll"]   
            node_info[node]['zed']['pitch'] -= angle_corrections[node]["pitch"] 
            node_info[node]['zed']['yaw']   -= angle_corrections[node]["yaw"] 

    return node_info

class point_projector(nn.Module):
    def __init__(self, node_info, node=None, image_size=None, mode='mocap', 
            apply_corrections=False,corrections_file="environments/umass/recalibration.tsv",
            scenario=None, node_str=None):
        super().__init__()
        self.info = copy.copy(node_info)
        self.scenario = scenario
        self.node_str = node_str
        
        if mode == 'mocap':
            X = node_info['location']['X']
            Y = node_info['location']['Y']
            Z = node_info['location']['Z']
            roll = node_info['zed']['roll']
            pitch = node_info['zed']['pitch']
            yaw = node_info['zed']['yaw']

            X = X / 1000
            Y = Y / 1000
            Z = Z / 1000
            dX = 0.0381
            dY = 0.2181
            dZ = -0.156
            roll = np.deg2rad(roll)
            pitch = np.deg2rad(pitch)
            yaw = np.deg2rad(yaw)

            if apply_corrections:
                angle_corrections = get_angle_corrections(corrections_file)
                roll -= angle_corrections[node]["roll"]
                pitch -= angle_corrections[node]["pitch"]
                yaw -= angle_corrections[node]["yaw"]

        elif mode == 'gq':
            X = node_info['location']['X']
            Y = node_info['location']['Y']
            Z = node_info['location']['Z']
            dX = node_info['zed']['location_offset_X']
            dY = node_info['zed']['location_offset_Y']
            dZ = node_info['zed']['location_offset_Z']
            roll = node_info['zed']['roll']
            # roll = np.deg2rad(roll)
            pitch = node_info['zed']['pitch']
            # pitch = np.deg2rad(pitch)
            yaw = node_info['zed']['yaw']
            # yaw = np.deg2rad(yaw)

        
        self.X = nn.Parameter(torch.ones(1)*X)
        self.Y = nn.Parameter(torch.ones(1)*Y)
        self.Z = nn.Parameter(torch.ones(1)*Z)

        # self.register_buffer("X",torch.ones(1)*X)
        # self.register_buffer("Y",torch.ones(1)*Y)
        # self.register_buffer("Z",torch.ones(1)*Z)

        self.register_buffer("dX", torch.ones(1)*dX)
        self.register_buffer("dY", torch.ones(1)*dY)
        self.register_buffer("dZ", torch.ones(1)*dZ)

        # self.dX = nn.Parameter(torch.ones(1)*dX)
        # self.dY = nn.Parameter(torch.ones(1)*dY)
        # self.dZ = nn.Parameter(torch.ones(1)*dZ)

        
        # self.dX = nn.Parameter(torch.ones(1)*0.0381)
        # self.dY = nn.Parameter(torch.ones(1)*0.2181)
        # self.dZ = nn.Parameter(torch.ones(1)*-0.156)

        
        # roll = np.deg2rad(node_info["zed"]["roll"])
        # roll -= angle_corrections[node]["roll"]
        # pitch = np.deg2rad(node_info["zed"]["pitch"])
        # pitch -= angle_corrections[node]["pitch"]
        # yaw = np.deg2rad(node_info["zed"]["yaw"])
        # yaw -= angle_corrections[node]["yaw"]

        
        #self.register_buffer("roll",torch.tensor(roll,dtype=torch.float32))
        #self.register_buffer("pitch",torch.tensor(pitch,dtype=torch.float32))
        #self.register_buffer("yaw",torch.tensor(yaw,dtype=torch.float32))
        
        self.roll = nn.Parameter(torch.tensor(roll,dtype=torch.float32))
        self.pitch = nn.Parameter(torch.tensor(pitch,dtype=torch.float32))
        self.yaw = nn.Parameter(torch.tensor(yaw,dtype=torch.float32))

        self.std_X = nn.Parameter(inverse_softplus(torch.tensor(0.02)))
        self.std_Y = nn.Parameter(inverse_softplus(torch.tensor(0.02)))
        self.std_Z = nn.Parameter(inverse_softplus(torch.tensor(0.02)))

        self.std_dX = nn.Parameter(inverse_softplus(torch.tensor(0.02)))
        self.std_dY = nn.Parameter(inverse_softplus(torch.tensor(0.02)))
        self.std_dZ = nn.Parameter(inverse_softplus(torch.tensor(0.02)))

        self.std_pitch = nn.Parameter(inverse_softplus(torch.tensor(0.001*2)))
        self.std_roll  = nn.Parameter(inverse_softplus(torch.tensor(0.01*2)))
        self.std_yaw   = nn.Parameter(inverse_softplus(torch.tensor(0.01*2)))

        if(image_size is None):
            self.image_size = [self.info["zed"]["width"],self.info["zed"]["height"]]
        else:
            self.image_size = image_size

        #Node->Camera rotation matrix
        #Static property of coordinate frames
        rot=Rotation.from_euler("xyz",[-1*np.pi/2,-np.pi/2,0])
        self.register_buffer("R_n2c",torch.tensor(rot.as_matrix()).float())
        # self.R_n2c = torch.tensor(rot.as_matrix()).float()

        #Camera->Image+Depth rotation matrix
        #Static property of coordinate frames
        rot=Rotation.from_euler("xyz",[0,0,np.pi])
        self.register_buffer("R_c2id",torch.tensor(rot.as_matrix()).float())
        # self.R_c2id = torch.tensor(rot.as_matrix()).float()

        #Define the intrinsics matrix
        #Static property of camera intrinsics
        zinfo = node_info["zed"]
        K = torch.zeros(3,3)
        K[0,0]= zinfo["fx"]
        K[1,1]= zinfo["fy"]
        K[2,2]= 1
        K[0,2]= zinfo["cx"]
        K[1,2]= zinfo["cy"]
        self.register_buffer("K",K)

        self.starting_params = self.get_parameter_vector().detach()

    # @property
    # def Z(self):
        # normZ = torch.sigmoid(self._Z)
        # return 1.1 + 0.5*normZ

    @property
    #Camera center in node coordinates
    def p_nc(self):
       #Camera location in node coordinates
        return torch.stack([self.dX,self.dY,self.dZ]).reshape(3,1)
       #return torch.tensor([[self.dX, self.dY, self.dZ]]).T

    @property
    #Node center in world coordinates
    def p_wn(self):
        #Node location in world coordinates
        return torch.stack([self.X,self.Y,self.Z]).reshape(3,1)
        # return torch.tensor([[self.X,self.Y,self.Z]]).T

    @property
    #Camera center in world coordinates
    def p_wc(self):
        #Camera location in world coordinates (plotting only)
        return self.R_w2n.T@self.p_nc + self.p_wn

    @property
    #World to node rotation matrix
    def R_w2n(self):
        return euler_to_rot_torch(
                self.roll.unsqueeze(0),
                self.pitch.unsqueeze(0),
                self.yaw.unsqueeze(0)
        ).squeeze(0).T

    #Transform world to node
    def world_to_node(self,x_w):
        return (self.R_w2n@(x_w.T-self.p_wn)).T
    
    #Transform node to camera
    def node_to_camera(self,x_n):
        return (self.R_n2c@(x_n.T-self.p_nc)).T

    #Transform camera to image
    def camera_to_image(self,x_c):
        x_hid = self.K@self.R_c2id@x_c.T
        x_id = torch.zeros_like(x_hid)
        x_id[:2,...] = x_hid[:2,...]/x_hid[[2],...]
        x_id[2,...] = x_hid[2,...]      
        return x_id.T

    #Transform World to image    
    def world_to_image(self,x_w, normalize=False):
        proj_pixels = self.camera_to_image(self.node_to_camera(self.world_to_node(x_w)))
        if normalize:
            proj_pixels /= torch.tensor([1920,1080,1]).unsqueeze(0).to(proj_pixels.device)
        return proj_pixels

    #Legacy API -- Define local to be camera space
    def world_to_local(self,x_w):
        return self.node_to_camera(self.world_to_node(x_w))

    #Legacy API -- Define local to be camera space    
    def local_to_image(self,x_c):
        return self.camera_to_image(x_c)

    #Transform node to world
    def node_to_world(self,x_n):
        return (self.R_w2n.T@x_n.T+self.p_wn).T
    
    #Transform camera to node
    def camera_to_node(self,x_c):
        return (self.R_n2c.T@x_c.T+self.p_nc).T

    #Transform image+depth to camera
    def image_to_camera(self,x_id):
        x_hid = x_id.clone()
        x_hid[...,:2] = x_hid[...,:2]*x_hid[...,[2]]
        return   (self.R_c2id.T@torch.inverse(self.K)@x_hid.T).T

    #Transform image to world    
    def image_to_world(self,x_id):
        return self.node_to_world(self.camera_to_node(self.image_to_camera(x_id)))


    
    #Image to local    
    def image_to_local(self,x_id):
        return self.image_to_camera(x_id)

    #Legacy API
    def local_to_world(self,x_c):
        return self.node_to_world(self.camera_to_node(x_c))
    
    def get_parameter_vector(self):
        params = torch.cat([self.X.reshape(1),self.Y.reshape(1),self.Z.reshape(1),
                            self.dX.reshape(1),self.dY.reshape(1),self.dZ.reshape(1),
                            self.roll.reshape(1),self.pitch.reshape(1),self.yaw.reshape(1)])
        params = params.reshape(-1,1)
        return(params)
    
    def get_parameter_std(self):
        stds = torch.cat([self.std_X.reshape(1),self.std_Y.reshape(1),self.std_Z.reshape(1),
                          self.std_dX.reshape(1),self.std_dY.reshape(1),self.std_dZ.reshape(1),
                          self.std_roll.reshape(1),self.std_pitch.reshape(1),self.std_yaw.reshape(1)])
        stds = F.softplus(stds)
        return(stds)

    def get_parameter_cov(self):
        stds = self.get_parameter_std()
        cov = torch.diag(stds**2)
        return cov

    def get_info(self):
        self.info["zed"]["yaw"] = self.yaw.detach().item()
        self.info["zed"]["pitch"] = self.pitch.detach().item()
        self.info["zed"]["roll"] = self.roll.detach().item()
        self.info["location"]["X"] = self.X.detach().item()
        self.info["location"]["Y"] = self.Y.detach().item()
        self.info["location"]["Z"] = self.Z.detach().item() 
        return self.info
    

    def world_to_imgage_uq(self,world_mean,world_cov):


        def world_to_image2(x_w, p_wn, angles):
            B,n,_ = x_w.shape
            R_w2n = euler_to_rot_torch(angles[...,0].flatten(),angles[...,1].flatten(),angles[...,2].flatten()).transpose(dim0=2,dim1=1)
            R_w2n = R_w2n.reshape(B,n,3,3)

            x_w = x_w.reshape(B,n,3,1)
            p_wn = p_wn.reshape(B,n,3,1)

            #Transform world to node
            x_n =  R_w2n@(x_w-p_wn)
    
            #Transform node to camera
            x_c = self.R_n2c.reshape(1,1,3,3)@(x_n-self.p_nc.reshape(1,1,3,1))

            #Transform camera to image
            x_hid = self.K.reshape(1,1,3,3)@self.R_c2id.reshape(1,1,3,3)@x_c
            x_id= x_hid[:,:,:2,:]/x_hid[:,:,[2],:]
              
            return x_id.squeeze(-1)

        f = lambda x: world_to_image2(x[...,:3], x[...,3:6], x[...,6:9])

        ut        = unscented_transform(f,9,diag_cov=False)   

        n         = world_mean.shape[0]
        proj_mu   = torch.cat([self.X.reshape(1),self.Y.reshape(1),self.Z.reshape(1), self.roll.reshape(1), self.pitch.reshape(1), self.yaw.reshape(1)]).reshape(1,6)
        proj_std  = torch.cat([self.std_X.reshape(1),self.std_Y.reshape(1),self.std_Z.reshape(1), self.std_pitch.reshape(1), self.std_roll.reshape(1), self.std_yaw.reshape(1)])
        # proj_std = F.softplus(proj_std)
        proj_cov  = torch.diag(proj_std**2).reshape(1,6,6)
        
        mu = torch.cat([world_mean, proj_mu.repeat(n,1)],dim=1).reshape(n,9,1) 
        cov = torch.zeros(n,9,9)
        cov[:,:3,:3] = world_cov
        cov[:,3:,3:] = proj_cov.repeat(n,1,1)

        pred_img_mu, pred_img_cov = ut.transform(mu, cov)

        return pred_img_mu.squeeze(0), pred_img_cov

    
    def image_to_world_uq(self,x_id):
        mean = self.image_to_world(x_id)
        cov = torch.eye(3).to(mean.device).unsqueeze(0)
        cov = cov.repeat(mean.shape[0],1,1)
        return mean, cov


    def image_to_world_ground_uq(self,points_img,z,points_img_std=None,z_std=None,d=None):
            f         = lambda x: self.image_to_world_ground(x[...,-3:-1],z=x[...,[-1]], pose_params=x[...,:-3], d=d)
            ut        = unscented_transform(f,12,diag_cov=True)
            proj_mu   = self.get_parameter_vector()
            proj_std  = self.get_parameter_std()

            if(points_img_std is None): points_img_std = torch.zeros_like(points_img)
            if(z_std is None): z_std = torch.zeros_like(z)

            points_mu  = torch.cat([points_img,z],axis=1)
            points_std = torch.cat([points_img_std,z_std],axis=1)

            n   = points_img.shape[0]
            mu  = torch.cat([proj_mu.T.repeat(n,1),points_mu],axis=1)
            std = torch.cat([proj_std.repeat(n,1),points_std],axis=1)
            cov = torch.diag_embed(std**2)

            pred_world_mu, pred_world_cov = ut.transform(mu, cov)

            pred_world_cov = (pred_world_cov + pred_world_cov.permute(0,2,1))/2

            return pred_world_mu.squeeze(0), pred_world_cov

    def image_to_world_ground(self,points_img,z=0,pose_params=None, d=None):


        if(pose_params is None): 
            m    = points_img.shape[0]
            B    = 1

            R_w2nT  = self.R_w2n.T.reshape(1,3,3).repeat(m,1,1)
            p_wn  = self.p_wn.reshape(1,3,1).repeat(m,1,1)
            p_nc  = self.p_nc.reshape(1,3,1).repeat(m,1,1)
            z     = z.reshape(-1,1,1)

        else:

            B,n,_       = points_img.shape
            m           = B*n 

            points_img  = points_img.reshape(-1,2)
            z           = z.reshape(-1,1,1)
            pose_params = pose_params.reshape(-1,9)

            R_w2nT = euler_to_rot_torch(
                    pose_params[:,6],
                    pose_params[:,7],
                    pose_params[:,8]
                )
            p_wn  = pose_params[:,:3].reshape(-1,3,1).float()
            p_nc  = pose_params[:,3:6].reshape(-1,3,1).float()
            
        R_n2cT    = self.R_n2c.T.reshape(1,3,3).repeat(m,1,1)
        R_c2idT   = self.R_c2id.T.reshape(1,3,3).repeat(m,1,1)
        Kinv      = torch.inverse(self.K).reshape(1,3,3).repeat(m,1,1)
        R_w2i_inv =  R_w2nT@R_n2cT@R_c2idT@Kinv


        ones = torch.ones(m,1).to(points_img.device)
        x_id     = torch.concat([points_img, ones],axis=1).unsqueeze(2)

        num      = z - R_w2nT[:,[2],:]@p_nc - p_wn[:,[2],:]
        denom    = R_w2i_inv[:,[2],:]@x_id
        ground_plane_d        = (num/(1e-10+denom))
    
        ground_plane_x_hid         = ground_plane_d*x_id

        if len(points_img) == 0:
            x_hid = ground_plane_x_hid

        else:
            if d is not None:
                d = d.unsqueeze(0).expand(B,-1,-1)
                d = d.reshape(-1,1)
                try:
                    true_x_hid = torch.cat([points_img*d,d],axis=1).unsqueeze(2)
                except:
                    import ipdb; ipdb.set_trace()
                

                if len(ground_plane_d) == 0:
                    x_hid = true_x_hid
                else:
                    valid_d_mask = d > 0
                    invalid_d_mask = d <= 0
                    valid_d_mask = valid_d_mask.squeeze(1)
                    invalid_d_mask = invalid_d_mask.squeeze(1)
                    x_hid = torch.zeros_like(true_x_hid)
                    x_hid[valid_d_mask] = true_x_hid[valid_d_mask]
                    x_hid[invalid_d_mask] = ground_plane_x_hid[invalid_d_mask]
            else:
                x_hid = ground_plane_x_hid
        
            # x_hid = x_hid.reshape(B,n,3)


        # if d is None:
            # ones = torch.ones(m,1).to(points_img.device)
            # x_id     = torch.concat([points_img, ones],axis=1).unsqueeze(2)

            # num      = z - R_w2nT[:,[2],:]@p_nc - p_wn[:,[2],:]
            # denom    = R_w2i_inv[:,[2],:]@x_id
            # d        = (num/(1e-10+denom))
        
            # x_hid         = d*x_id     
        # else:
            # d = d.unsqueeze(0).expand(B,-1,-1)
            # d = d.reshape(-1,1)
            # x_hid = torch.cat([points_img*d,d],axis=1).unsqueeze(2)

        x_c = (R_c2idT@Kinv@x_hid)
        x_n = (R_n2cT@x_c+p_nc)
        x_w = (R_w2nT@x_n+p_wn)

        if(pose_params is not None):
            x_w    = x_w.reshape(B,n,3)
        else:
            x_w = x_w.squeeze(2)

        return x_w

    def local_to_world_cov(self,cov):
        return self.R_w2n.T @ cov @ self.R_w2n
        # return self.R @ cov @ self.R.T   

    def camera_to_world_cov(self,cov):
        return self.R_w2n.T @ self.R_n2c.T @ cov @ self.R_n2c @ self.R_w2n
     
    def is_visible(self, env, points_world,points_img=None):
        #Takes as input an environement and a tensor of shape (m,3) of points in (x,y,z) world coordinates and
        #optionally a tensor of shape (m,3) in image plane coordinate.
        #Outputs a tensor of shape (m,) indicating whether point m is visible from this camera location.
        #To be visible, a point needs to be in the image frame and not occluded.

        occluded, _ = self.is_occluded(env, points_world)
        if(points_img is None):
            points_img = self.world_to_image(points_world)

        in_frame = (points_img[:,0]>0) * (points_img[:,0]<self.image_size[0]) * (points_img[:,1]>0) * (points_img[:,1]<self.image_size[1]) * (points_img[:,2]>0)

        return torch.logical_and(torch.logical_not(occluded),in_frame)

    def dist_to_occluders(self,env):
        #Return distance of node to each occluder segment
        nodep  = torch.tensor([[self.X, self.Y]])
        p      = env.occluders[:,1,:] - env.occluders[:,0,:]
        norm   = torch.sum(p**2,axis=1,keepdims=True)
        u      = torch.sum((nodep -  env.occluders[:,0,:]) * p / norm, axis=1, keepdims=True)
        u[u<0] = 0
        u[u>1] = 1

        p2 =  env.occluders[:,0,:] + u*p

        D  = torch.norm(p2 - nodep,dim=1)

        return D

    def is_occluded(self, env, points_world):
        #Takes as input an environement and a tensor of shape (m,3) of points in (x,y,z) world coordinates
        #If the environment has occluders defined, checks to see which points in points_world are occluded from
        #the perspective of this node. The computation is performed for the 2D XY plane only.
        #The output is a tensor of shape (m,) indicating whether each point is occluded, and a tensor
        #of shape (n,m) indicating for each point, which line segments in the 2D scene map are responsible 
        #for occluding the point

        def ccw(A,B,C):
            return (C[:,1,:]-A[:,1,:]) * (B[:,0,:]-A[:,0,:]) > (B[:,1,:]-A[:,1,:]) * (C[:,0,:]-A[:,0,:])

        def intersect(A,B,C,D):
            #ref: https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
            #Assumes A=(1,2),B=(m,2),C=(n,2),D=(n,2)
            #Checks for intersections between the m lines joining point A[0,:] and all points B[i,:] compared to the
            #n lines joining C[j,:] to D[j,:]. Returns an array of shape(m,n) with all intersections indicated
            m = B.shape[0]
            n = C.shape[0]

            A = A.reshape([1,2,1])
            B = B.reshape([m,2,1])
            C = C.reshape([n,2,1]).transpose(dim0=0,dim1=2) #C will be (1,2,n)
            D = D.reshape([n,2,1]).transpose(dim0=0,dim1=2) #D will be (1,2,n)

            return torch.logical_and(ccw(A,C,D) != ccw(B,C,D), ccw(A,B,C) != ccw(A,B,D))

        nodep = torch.tensor([[self.X, self.Y]])
        objp  = points_world[:,:2] 

        #Filter occluders to avoid node inside walls due to 
        #impercise node placements resulting in all objects occluded
        dists = self.dist_to_occluders(env)
        inds = dists<2

        occluded_by         = intersect(nodep,objp, env.occluders[:,0,:], env.occluders[:,1,:])
        occluded_by[:,inds] = False

        occluded    = torch.any(occluded_by,axis=1) 
        return occluded, occluded_by 

