from abc import ABC, abstractmethod 
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.dates as mpl_dates
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import pandas as pd
import cv2
from ipywidgets import Layout
import ipywidgets as widgets
import numpy as np
import datetime
import time
#from pypdf import PdfMerger
import sys, os, glob
import copy
import tqdm
import threading
import glob
import soundfile as sf
from scipy.fftpack import fft

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(base_path)

import spatial_transform_utils as st  
import data_utils as du          
import vis

class environment():

    def __init__(self,name,env_file="env_info.json", base_path=None):

        config          = du.get_config()
        self.name       = name
        if base_path is not None:
            self.base_path = base_path
        else:
            self.base_path = config["base_path"]
        self.env_file   = env_file
    
        self.env_dir       = os.path.join(self.base_path,"environments",self.name)
        self.env_file_path = os.path.join(self.env_dir,self.env_file)
        self.map_cache_dir = os.path.join(self.env_dir,"map_cache")

        with open(self.env_file_path) as f:
            self.info = json.load(f)

        self.get_occluders()

    def get_occluders(self):

        self.occluders=None

        if(self.info["map"]['type']=="satellite"):
            filename = os.path.join(self.map_cache_dir, "map_vec_lat%.6f_lon%.6f_zoom%d.svg"%(self.info["map"]["center_lat"],self.info["map"]["center_lon"],self.info["map"]["zoom"]))
            if(os.path.exists(filename)):
                self.occluders=vis.segments_from_svg(filename)

    def log(self,msg):
        print(f"{self.name}: {msg}")

    def get_nodes(self):
        return list(self.info["nodes"].keys())

    def get_node_ids(self):
        return [x["id"] for _,x in self.info["nodes"].items()]

    def plot_map(self,ax):
        vis.plot_map_meters(ax,**self.info["map"],cache_dir=self.map_cache_dir)
        
        ax.set_xlim(self.info["map"]["xlim"][0],self.info["map"]["xlim"][1])
        ax.set_ylim(self.info["map"]["ylim"][0],self.info["map"]["ylim"][1])

    def plot_node(self,ax,node,scale=None):
        if(scale is None):
            scale = 0.1*(self.info["map"]["xlim"][1]-self.info["map"]["xlim"][0])
        vis.plot_zed_fov_2D(ax,self.info["nodes"][node],scale=scale)
        vis.plot_node(ax,self.info["nodes"][node])

    def plot_nodes(self,ax):
        for node in self.info["nodes"]:
            self.plot_node(ax,node)

class scenario_loader():
     
    def __init__(self,env):
        self.env           = env
        self.base_path     = env.base_path
        self.scenario_dir = os.path.join(self.env.env_dir,"scenarios")

        
        self.sceanrio_files = glob.glob(os.path.join(self.scenario_dir,"*.json"))
        self.sceanrio_files.sort()   

        self.info={}
        for file in self.sceanrio_files:
            with open(file) as f:
                scenario_data  = json.load(f)
                start_time     = pd.to_datetime(scenario_data["start_time"])
                end_time       = pd.to_datetime(scenario_data["end_time"])
                start_time_str = start_time.strftime("%Y/%m/%d %H:%M:%S")
                name           = f"{start_time_str}: {scenario_data['name']}"

                scenario_data["start_time"] = int(start_time.timestamp()*1000) 
                scenario_data["end_time"]   = int(end_time.timestamp()*1000) 

                self.info[name] = scenario_data

    def get_scenarios(self):
        scenarios = {}
        for scenario_name, info in self.info.items():
            scenarios[scenario_name] = scenario(scenario_name, self.env,info)
        return scenarios

class scenario():
    def __init__(self,name,env,info):
        self.env           = env
        self.base_path     = env.base_path
        self.info          = info
        self.name          = name
        self.scenario_dir = os.path.join(self.env.env_dir,"scenarios")
    
    def get_name(self):
        return self.name
    
    def get_times(self):
        return [self.info["start_time"],self.info["end_time"]] 
    
    def save(self):
        start_time     = pd.to_datetime(self.info["start_time"])
        filename       = start_time.strftime("%Y%m%d-%H%M%S") + ".json"
        filepath       = os.path.join(self.scenario_dir,filename)

        with open(filepath, 'w') as f:
            json.dump(self.info, f,indent=4)

    def local_to_world(self,node,data):
        local_meters = torch.tensor(data[node][["x","y","z"]].to_numpy(),dtype=torch.float)
        pp           = st.point_projector(self.info["nodes"][node])
        world_meters = pp.local_to_world(local_meters).detach().numpy()

        return pd.DataFrame(data=world_meters,columns=["x","y","z"],index=data["world"].index)

    def plot_node(self,ax,node,scale=8,local=False):

        if(local):
            scenario_local = copy.deepcopy(self)
            scenario_local.info["nodes"][node]["location"]["X"]=0.0
            scenario_local.info["nodes"][node]["location"]["Y"]=0.0
            scenario_local.info["nodes"][node]["zed"]["yaw"]=0.0
            vis.plot_zed_fov_2D(ax,scenario_local.info["nodes"][node],scale=scale)
            vis.plot_node(ax,scenario_local.info["nodes"][node])           
        else:
            vis.plot_zed_fov_2D(ax,self.info["nodes"][node],scale=scale)
            vis.plot_node(ax,self.info["nodes"][node])

    def plot_nodes(self,ax,scale=8):
        for node in self.info["nodes"]:
            self.plot_node(ax,node,scale=scale)

    def plot_tracks(self, objects, min_count=0, title=None):

        if (title is None):
            title = self.name

        localized_objects = {}
        for id , object in objects.items():
            localized_objects[id] = object.localize(self,inplace=False,min_count=min_count)

        f=plt.figure(figsize=(9,4))

        gs  = GridSpec(2, 2, figure=f)
        ax1 = f.add_subplot(gs[:, 0])
        ax2 = f.add_subplot(gs[0, 1])
        ax3 = f.add_subplot(gs[1, 1])

        self.env.plot_map(ax1)

        valid_ojects=False
        for id, obj in localized_objects.items():
            if(len(obj.data["world"].index)>min_count):
                obj.plot_track(ax1)
                obj.plot_altitude(ax2)
                obj.plot_accuracy(ax3)
                valid_ojects=True
        
        self.plot_nodes(ax1)
        ax1.set_title(title)
        if valid_ojects: ax1.legend(bbox_to_anchor=(-0.15, 1.0))
        
        ax2.set_title("Altitude (m)")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True)
        
        ax3.set_title("GPS precision (m)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylim(0.001, 10)
        ax3.grid(True)

        plt.tight_layout()
        plt.show()


class track():
    def __init__(self,mean,cov=None,color="b",name=None):
        self.mean = mean
        self.cov  = cov
        self.num  = mean.shape[0] 
        self.color = color
        self.name = name

    def plot_dist(self,ax,mean,cov,chi_sq=5.991,alpha=1):
        eigvals, eigvecs = np.linalg.eig(cov[:2,:2])
        ind = np.argsort(-1*eigvals)
        eigvals = eigvals[ind]
        eigvecs = eigvecs[:,ind]

        angle = np.arctan2(np.real(eigvecs[1, 0]), np.real(eigvecs[0, 0])) 
        angle *= (180 / np.pi)

        axes_lengths = np.sqrt(eigvals * chi_sq)
        ellipse=Ellipse(mean.squeeze(),width=axes_lengths[0],height=axes_lengths[1],angle=1*angle,alpha=alpha,color=self.color)
        ax.add_patch(ellipse)
        return ellipse
    
    def plot_track(self, ax, show_mean=True, show_cov=True, line_style='.-', alpha=1,zorder=3):

        if(self.num==0): return 

        if show_mean:
            ax.plot(self.mean[:,0], self.mean[:,1], line_style ,alpha=alpha, linewidth=3,color=self.color,label=self.name,zorder=zorder)[0]
        
        if(show_cov and self.cov is not None):
            for i in range(self.num):
                self.plot_dist(ax,self.mean[i,:].numpy(),self.cov[i,:,:].numpy(),alpha=alpha)

class tracking_result():
    def __init__(self,name,env,node,local_data,color,scenario=None):

        self.node  = node
        self.name  = name
        self.env   = env
        self.color = color
        self.scenario=scenario
        self.pp = st.point_projector(self.scenario.info["nodes"][node])
        self.h_mean = {"world":None, "local":None}
        self.h_cov  = {"world":None, "local":None}
        self.from_torch_tuple_list(local_data)

    def plot_dist(self,ax,mean,cov,chi_sq=5.991):
        eigvals, eigvecs = np.linalg.eig(cov[:2,:2])
        ind = np.argsort(-1*eigvals)
        eigvals = eigvals[ind]
        eigvecs = eigvecs[:,ind]

        angle = np.arctan2(np.real(eigvecs[1, 0]), np.real(eigvecs[0, 0])) #TODO: y, x order?
        angle *= (180 / np.pi)

        axes_lengths = np.sqrt(eigvals * chi_sq)
        ellipse=Ellipse(mean,width=axes_lengths[0],height=axes_lengths[1],angle=1*angle,alpha=0.5,color=self.color)
        ax.add_patch(ellipse)
        return ellipse

    def from_torch_tuple_list(self, local_data):

        self.mean = {}
        self.cov  = {}
        self.ts   = np.array([item[2].numpy() for item in local_data])
        loc       = [item[0].reshape(1,3) for item in local_data]
        
        self.mean["local"] = torch.vstack(loc)
        self.mean["world"] = self.pp.local_to_world(self.mean["local"]).detach().numpy()
        self.cov["local"]  = torch.concatenate([item[1].reshape(1,3,3) for item in local_data],axis=0)
        self.cov["world"]  = self.pp.local_to_world_cov(self.cov["local"]).detach().numpy()
        self.num           = self.mean["local"].shape[0]


    def plot_track(self, ax, view="world", show_mean=True, show_cov=True):

        if(self.num==0): return 

        if show_mean:
            ax.plot(self.mean[view][:,0], self.mean[view][:,1], ".-",color=self.color,label=self.name)[0]
        
        if(show_cov):
            for i in range(self.num):
                self.plot_dist(ax,self.mean[view][i,:],self.cov[view][i,:,:])

    def plot_update_loc(self,ax, ind, view="world"):
        if ind<0 or ind>self.num-1: return

        if(self.h_mean[view] is None):
            self.h_mean[view] = ax.plot(self.mean[view][ind,0],self.mean[view][ind,1],"o", color=self.color)[0]
        else:
            self.h_mean[view].set_data([self.mean[view][ind,0]],[self.mean[view][ind,1]])

        if(self.h_cov[view] is not None):
            self.h_cov[view].remove()
        self.h_cov[view] =self.plot_dist(ax,self.mean[view][ind,:],self.cov[view][ind,:,:])
        
class tracking_object():
    def __init__(self,name,env,data,color,scenario=None):
        self.name=name
        self.data=data
        self.env=env
        self.color = color
        self.scenario=scenario

    def log(self,msg):
        print(f"{self.name}: {msg}")

    def set_altitude(self, a):
        alt_fields = ["z","al"]
        for _,view in self.data.items():
            for f in alt_fields: 
                if(f in view):
                    view[f]=a

    def slice(self,start_time,end_time,inplace=True):

        start_ind, end_ind = du.get_slice_inds(self.data["world"]["t"],start_time=start_time,end_time=end_time)

        if(inplace):
            new_data = self.data
        else:
            new_data={}

        for view in self.data:
            new_data[view] = self.data[view][start_ind:end_ind].copy(deep=True)
        
        if(not inplace):
            new_object = tracking_object(self.name,self.env,new_data,self.color)
            return new_object


    def localize(self,scenario,inplace=True,min_count=5):

        start_time = scenario.info["start_time"]
        end_time   = scenario.info["end_time"]

        start_ind, end_ind = du.get_slice_inds(self.data["world"]["t"],start_time=start_time,end_time=end_time)

        if(inplace):
            new_data={}
            new_data["world"] = self.data["world"].iloc[start_ind:end_ind]
        else:
            new_data={}
            new_data["world"] = self.data["world"].iloc[start_ind:end_ind].copy(deep=True)

        #Object restricted to scenario has less than min_count observations 
        if(len(new_data["world"])<min_count):
            new_data["world"] = new_data["world"].loc[[]]
            for node in scenario.info["nodes"]:
                new_data[node] = pd.DataFrame(data=[],columns=["x","y","z","h","v","d","visible"],index=new_data["world"].index)

            if(not inplace):
                new_object = tracking_object(self.name,self.env,new_data,self.color,scenario=scenario)
                return new_object
        
        world_meters = torch.tensor(new_data["world"][["x","y","z"]].to_numpy(),dtype=torch.float)

        for node in scenario.info["nodes"]:

            pp = st.point_projector(scenario.info["nodes"][node])
            local_meters = pp.world_to_local(world_meters).detach()
            image_frame  = pp.world_to_image(world_meters).detach()
            visible      = pp.is_visible(self.env, world_meters,points_img=image_frame).reshape([-1,1])

            all = torch.hstack([local_meters,image_frame,visible])

            new_data[node] = pd.DataFrame(data=all.numpy(),columns=["x","y","z","h","v","d","visible"],index=new_data["world"].index)
        
        if(inplace):
            self.data = new_data
        else:
            new_object = tracking_object(self.name,self.env,new_data,self.color)
            return new_object

    def geofence_outside(self,left,right,bottom,top,thresh=5,inplace=True):

        ind1    = np.logical_and(self.data["world"]["x"] >= left, self.data["world"]["x"] <= right)
        ind2    = np.logical_and(self.data["world"]["y"] <= top, self.data["world"]["y"] >= bottom)
        ind_geo = np.logical_not(np.logical_and(ind1, ind2))

        if(inplace):
            new_data = self.data
        else:
            new_data = self.data.copy(deep=True)

        for view in new_data:
            new_data[view]=new_data[view].loc[ind_geo,:]

        if(not inplace):
            new_object = tracking_object(self.name,self.env,new_data,self.color)
            return new_object

    def geofence_inside(self,left,right,bottom,top,thresh=5,inplace=True):

        ind1    = np.logical_and(self.data["world"]["x"] >= left, self.data["world"]["x"] <= right)
        ind2    = np.logical_and(self.data["world"]["y"] <= top, self.data["world"]["y"] >= bottom)
        ind_geo = np.logical_and(ind1, ind2)

        if(inplace):
            new_data = self.data
        else:
            new_data= self.data.copy(deep=True)

        for view in new_data:
            new_data[view]=new_data[view].loc[ind_geo,:]

        if(not inplace):
            new_object = tracking_object(self.name,self.env,new_data,self.color)
            return new_object


    def threshold_accuracy(self,thresh,inplace=True):
         
        ind = self.data["world"]["ac"]<thresh

        if(inplace):
            new_data = self.data
        else:
            new_data = self.data.copy(deep=True)

        for view in self.data:
            new_data[view]=new_data[view].loc[ind,:]

        if(not inplace):
            new_object = tracking_object(self.name,self.env,new_data,self.color)
            return new_object

             
    def gps_to_metric(self,inplace=True):        
        gps_data    = torch.tensor(self.data["world"][["lt","ln","al"]].to_numpy(),dtype=torch.double)
        # data_meters = st.gps_to_meters(gps_data, **self.env.info["map"]).detach().numpy()
        data_meters = st.gps_to_meters(
            gps_data,
            center_lat=self.env.info["map"]["center_lat"],
            center_lon=self.env.info["map"]["center_lon"],
            zoom=self.env.info["map"]["zoom"],
            map_width=self.env.info["map"]["map_width"],
            map_height=self.env.info["map"]["map_height"],
            type=self.env.info["map"]["type"],
        )

        if(inplace):
            new_data = self.data
        else:
            new_data = self.data.copy(deep=True)

        new_data["world"]["x"] =  data_meters[:,0]
        new_data["world"]["y"] =  data_meters[:,1]
        new_data["world"]["z"] =  data_meters[:,2]

        if(not inplace):
            new_object = tracking_object(self.name,self.env,new_data,self.color)
            return new_object

    def get_data(self,t,thresh=100):
        i         = np.searchsorted(self.data["world"]["t"],t,side="right")-1
        if(i<0 or i>len(self.data["world"]["t"])):
            return None
        else:
            if(np.abs(self.data["world"]["t"][self.data["world"].index[i]]-t)>thresh):
                return None
            else:
               return {view: self.data[view].loc[self.data[view].index[i]] for view in self.data}

    def is_empty(self):
        return(len(self.data["world"])==0)

    def plot_track(self, ax, min_count=0, title="", range=None):

        if(len(self.data["world"].index)==0):
            return None

        if(range is not None):
            start_ind, end_ind = du.get_slice_inds(self.data["world"]["t"],start_time=range[0],end_time=range[1])
            this_data = self.data["world"][start_ind:end_ind]
        else:
            this_data = self.data["world"]
        
        count = len(this_data)

        if(count>min_count):
            h = ax.plot(this_data["x"], this_data["y"], "-o",color=self.color,markersize=2,label=self.name)[0]
        else:
            h=None

        return h

    def plot_altitude(self, ax, min_count=0, title="",range=None):

        if(len(self.data["world"].index)==0):
            return None

        if(range is not None):
            start_ind, end_ind = du.get_slice_inds(self.data["world"]["t"],start_time=range[0],end_time=range[1])
            this_data = self.data["world"][start_ind:end_ind]
        else:
            this_data = self.data["world"]

        count = len(this_data)

        if(count>min_count):
            #local_time = (self.data["world"].loc[start_ind:end_ind,"t"] - self.data["world"]["t"].iloc[0])/(60*1000)
            local_time = mpl_dates.date2num(np.array(pd.to_datetime(this_data["t"],unit="ms").dt.to_pydatetime()))
            
            h=ax.plot(local_time,this_data["z"],  "-o",color=self.color,markersize=2,label=self.name)[0]     
            myFmt = mpl_dates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(myFmt)
            ax.set_xlim(local_time[0],local_time[-1])
            ax.xaxis.set_major_locator(mpl_dates.MinuteLocator(interval=2))
        else:
            h=None 

        return h

    def plot_accuracy(self, ax, min_count=0, title="",range=None):

        if(len(self.data["world"].index)==0):
            return None

        if(range is not None):
            start_ind, end_ind = du.get_slice_inds(self.data["world"]["t"],start_time=range[0],end_time=range[1])
            this_data = self.data["world"][start_ind:end_ind]
        else:
            this_data = self.data["world"]

        count = len(this_data)

        if(count>min_count):
            local_time = mpl_dates.date2num(np.array(pd.to_datetime(this_data["t"],unit="ms").dt.to_pydatetime()))
            h=ax.semilogy(local_time,this_data["ac"],  "-o",color=self.color,markersize=2,label=self.name)[0]     
            myFmt = mpl_dates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(myFmt)
            ax.set_xlim(local_time[0],local_time[-1])   
            ax.xaxis.set_major_locator(mpl_dates.MinuteLocator(interval=2))         
        else:
            h=None   

        return h       

class gps_loader():

    def __init__(self,env,from_scratch=False):
        self.name="GPS Loader"
        self.data={}
        self.env=env

        config=du.get_config()
        gps_log_file=config["gps_data_file"]
        self.load_gps_log(gps_log_file,from_scratch=from_scratch)

    def log(self,msg):
        print(f"{self.name}: {msg}")

    def load_gps_log(self,gps_log_file,from_scratch=False):
        
        self.gps_log_file = gps_log_file
        self.base_dir     = os.path.dirname(self.gps_log_file)
        self.gps_cache_file = os.path.join(self.base_dir,"processed_gps.pkl")

        if(os.path.exists(self.gps_cache_file) and not from_scratch):
            self.log(f"Loading cached GPS object data")
            self.data["world"] = pd.read_pickle(self.gps_cache_file)
            return
                    
        self.log("Reading GPS csv data")
        self.data["world"] = pd.read_csv(self.gps_log_file)
        self.log("Saving GPS pkl df cache file")
        self.data["world"].to_pickle(self.gps_cache_file)

    def get_objects(self):

        objects={}
        obj_names   = list(self.data["world"]["id"].unique())
        obj_names.sort()
        num_objects = len(obj_names)

        colors      = cm.prism(np.linspace(0, 1, num_objects))
        color_map   = {id:colors[i] for i,id in enumerate(obj_names)}

        objects={}
        for obj in obj_names:
            ind =  self.data["world"]["id"]==obj
            obj_df = self.data["world"][ind]
            obj_df = obj_df.reset_index()
            new_data = {"world": obj_df}
            objects[obj]=tracking_object(obj,self.env,new_data,color_map[obj])
            
        return(objects)

class video_reader():

    def __init__(self,node,id,env,modality="rgb",scale=1,verbose=True,from_scratch=False):
        self.name=f"{node}:{modality}"
        self.node=node
        self.data={}
        self.env=env
        self.id=id
        self.scale=scale
        self.verbose = verbose
        self.modality=modality

        self.last_frame_ind = None
        self.last_video_file = None
        self.video_source  = None

        self.file_map={"rgb":"left.mp4", "depth":"depth.mp4","rd":"range_doppler.mp4"}
        self.dir_map={"rgb":"zed", "depth":"zed","rd":"mmwave"}
        self.aspect_map={"rgb":"1", "depth":"1","rd":"10"}

        self.aspect = self.aspect_map[modality]

        self.load_timestamps_json(from_scratch=from_scratch)

    def log(self,msg):
        if self.verbose: print(f"Zed {self.name}: {msg}")

    def get_video_file(self,date,time):
        file          = self.file_map[self.modality]
        orin_name     = f"orin{self.id}"
        wildcard_path = os.path.join(self.base_dir,str(date),orin_name,f"{date}_{time}*_dvpg_gq_orin_{self.id}_{self.dir_map[self.modality]}",file)
        file_paths    = list(glob.glob(wildcard_path))

        if len(file_paths)==1:
            return(file_paths[0])
        else:
            self.log(f"Can not find file {wildcard_path}")
            return None

    def load_timestamps_json(self,from_scratch=False):
        
        config = du.get_config()
        data_path = config["data_path"]

        self.base_dir     = data_path 
        cachefile         = os.path.join(self.base_dir,f"{self.modality}_timestamps{self.id}.pkl")

        if(os.path.exists(cachefile) and not from_scratch):
            self.log("Loading cached zed timestamp data")
            self.timestamps = pd.read_pickle(cachefile)

        else:
            pattern = f"{data_path}**/orin{self.id}/**/{self.file_map[self.modality]}"
            files = glob.glob(pattern)
            files.sort()

            dfs = []
            for file in files:
                parts       = file.split("/")
                identifiers = parts[-2].split("_")

                date = int(identifiers[0])
                time = int(identifiers[1])
                
                time_stamps_file = os.path.join(os.path.dirname(file),"timestamps.json")

                with open(time_stamps_file, 'r') as f:
                    timestamps = json.load(f)

                df = pd.DataFrame(data={"t":timestamps})
                df["t"]=df["t"].astype(np.int64)
                df["node"]= self.id
                df["date"]= date
                df["time"]= time
                df["frame"] = list(range(len(timestamps)))
                dfs.append(df)
                
            self.timestamps = pd.concat(dfs).reset_index()
            self.log(f"Saving timestamps to {cachefile}")
            self.timestamps.to_pickle(cachefile)

        self.log(f"Frames available: {len(self.timestamps)}")

    def load_timestamps(self,zed_bags_file,from_scratch=False):

        self.base_dir     = os.path.dirname(zed_bags_file)
        cachefile         = os.path.join(self.base_dir,"zed_timestamps.pkl")
        if(os.path.exists(cachefile) and not from_scratch):
            self.log("Loading cached zed timestamp data")
            df = pd.read_pickle(cachefile)
        else:
                
            self.log("Reading raw zed bag json")
            with open(zed_bags_file) as file:
                zed_data = json.load(file)

            def map_zedbag_data(x):
                parts = x['filename'].split("_")
                d = parts[0]
                t = parts[1]
                n = parts[5]
                return [np.int64(x['timestamp']),int(n), int(d), int(t), int(x["frame"])]

            bag_files = list(zed_data['bag_files'].keys())
            bag_files.sort()

            self.log("Extracting timestamps")
            timestamps_list = np.array([map_zedbag_data(x) for bag_file in bag_files for x in zed_data['bag_files'][bag_file]['frame_timestamps']])
            df = pd.DataFrame(data = timestamps_list, columns=["t","node","date","time","frame"])            

            self.log(f"Saving results to {cachefile}")
            df.to_pickle(cachefile)

        self.timestamps = df[df["node"]==self.id].reset_index()

    def get_frame(self,ind=None,t=None):

        if(ind is None):
            ind = np.searchsorted(self.timestamps["t"],t,side="right")-1
            if(ind<0 or ind>=len(self.timestamps["t"]) or self.timestamps.loc[ind,"t"]>t):
               return None,None

        frame_record = self.timestamps.loc[ind]
        frame_date   = str(int(frame_record["date"]))
        frame_time   = f"{int(frame_record['time']):06d}"
        frame_ind    = frame_record["frame"]
        video_file   = self.get_video_file(frame_date,frame_time)
        
        if video_file is None: return None,None

        if(video_file!=self.last_video_file):
            if(self.last_video_file is not None):
                self.video_source.release()
            self.log(f"Opening new file {video_file}")
            self.video_source = cv2.VideoCapture(video_file)
            self.last_video_file   = video_file
            self.last_frame_ind = None

        if(self.last_frame_ind is None or frame_ind!=self.last_frame_ind+1):
            self.video_source.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)

        status, frame = self.video_source.read()
        if status==True:
            height, width = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame, (int(width*self.scale), int(height*self.scale)))
        else:
            return None,None

        return img, self.timestamps.loc[ind,"t"]

    def show_frame(self,ax,ind):
        img,t = self.get_frame(ind)
        if(img is None):
            img = np.zeros((int(1920*self.scale), int(1080*self.scale),3))
        h = ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(self.name)
        return h

    def get_frames(self,start=None,end=None,scenario=None,mode="generator"):

        if(scenario is not None):
            start = scenario.info["start_time"]
            end   = scenario.info["end_time"]
        else:
            if(start is None and end is not None):
                start = self.timestamps.loc[0,"t"]
            elif(start is not None and end is None):
                L = len(self.timestamps)
                end = self.timestamps.loc[L-1,"t"]

        slice  = self.timestamps[self.timestamps["t"].between(start,end)]

        if(mode=="batch"):
            return self.get_frames_batch(slice.index)
        elif(mode=="generator"):
            return self.get_frames_generator(slice.index)
        elif(mode=="count"):
            return (len(slice))
        elif(mode=="inds"):
            return list(slice.index)
        elif(mode=="files"):
            file_list = []
            df = slice[["date", "time"]].drop_duplicates()
            for i, row in df.iterrows():
                video_file = self.get_video_file(str(row["date"]),f"{row['time']:06d}")
                file_list.append(video_file)
            return file_list
        else:
            self.log(f"Unknwon get_frames mode {mode}")

    def get_frames_batch(self, inds):

        if len(inds)==0:
            self.log("No frames in time slice")
            return [],[]
            
        frames = []
        ts=[]
        for _, ind in enumerate(tqdm.tqdm(inds)):

            img, t = self.get_frame(ind)
            frames.append(img)
            ts.append(t)

        return frames,ts
    
    def get_frames_generator(self, inds):

        if len(inds)==0:
            self.log("No frames in time slice")
            yield [],[]
            
        frames = []
        ts=[]
        for _, ind in enumerate(inds):

            img, t = self.get_frame(ind)
            yield img, t

def get_reader(node,id,env,modality="rgb",verbose=True,from_scratch=False,scale=1):
        if(modality=="audio"):
            return audio_reader(node,id,env,modality=modality,verbose=verbose,from_scratch=from_scratch)
        if(modality in ["rgb","depth","rd"]):
            return video_reader(node,id,env,modality=modality,verbose=verbose,from_scratch=from_scratch,scale=scale)

class audio_reader():

    def __init__(self,node,id,env,modality="audio",verbose=True,from_scratch=False,freq=16000):
        self.name=f"{node}:{modality}"
        self.node=node
        self.data={}
        self.env=env
        self.id=id
        self.verbose = verbose
        self.modality=modality

        self.last_frame_ind = None
        self.last_file = None
        self.source  = None
        self.file_map={"audio":"respeaker.flac"}
        self.dir_map={"audio":"respeaker"}
        self.freq=freq
        self.aspect=1/800
        self.scale=0.25
        
        self.load_timestamps_json(from_scratch=from_scratch)

        self.max = None
        self.min = None

    def log(self,msg):
        if self.verbose: print(f"Zed {self.name}: {msg}")

    def get_data_file(self,date,time):
        file          = self.file_map[self.modality]
        orin_name     = f"orin{self.id}"
        wildcard_path = os.path.join(self.base_dir,str(date),orin_name,f"{date}_{time}*_dvpg_gq_orin_{self.id}_{self.dir_map[self.modality]}",file)
        file_paths    = list(glob.glob(wildcard_path))

        if len(file_paths)==1:
            return(file_paths[0])
        else:
            self.log(f"Can not find file {wildcard_path}")
            return None

    def load_timestamps_json(self,from_scratch=False):
        
        #config = du.get_config()
        #data_path = config["data_path"]
        data_path = "/Users/marlin/Downloads/gq_data_processed/"
        
        self.base_dir     = data_path 
        cachefile         = os.path.join(self.base_dir,f"{self.modality}_timestamps{self.id}.pkl")

        if(os.path.exists(cachefile) and not from_scratch):
            self.log("Loading cached timestamp data")
            self.timestamps = pd.read_pickle(cachefile)

        else:
            pattern = f"{data_path}**/orin{self.id}/**/{self.file_map[self.modality]}"
            files = glob.glob(pattern)
            files.sort()

            timestamp_array = np.zeros((len(files),7),dtype=np.int64)
            for i,file in enumerate(files):
                parts       = file.split("/")
                identifiers = parts[-2].split("_")

                date = int(identifiers[0])
                time = int(identifiers[1])
                
                time_stamps_file = os.path.join(os.path.dirname(file),"timestamps.json")

                with open(time_stamps_file, 'r') as f:
                    timestamps = json.load(f)

                timestamp_array[i,0]=timestamps[0]
                timestamp_array[i,1]=timestamps[1]
                timestamp_array[i,2]=date
                timestamp_array[i,3]=time

                data,freq = sf.read(file)
                timestamp_array[i,4] = data.shape[0]
                timestamp_array[i,5] = freq
                timestamp_array[i,6] = data.shape[1]

            df = pd.DataFrame(data=timestamp_array,columns=["t_start","t_end","date","time","samples","freq","channels"])
            df["node"]= self.id
                
            self.timestamps = df
            self.log(f"Saving timestamps to {cachefile}")
            self.timestamps.to_pickle(cachefile)

        self.log(f"Frames available: {len(self.timestamps)}")

    def get_waveform(self,ind=None,t=None,width=0.25):

        if(ind is None):

            t_start = t-1000*width
            t_end   = t
            ind     = np.searchsorted(self.timestamps["t_start"],t_start,side="right")-1
            if(ind<0 or ind>=len(self.timestamps["t_start"])):
               return None,None

        frame_record = self.timestamps.loc[ind]
        frame_date   = str(int(frame_record["date"]))
        frame_time   = f"{int(frame_record['time']):06d}"
        data_file    = self.get_data_file(frame_date,frame_time)
        
        if data_file is None: return None,None

        if(data_file!=self.last_file):
            self.log(f"Opening new file {data_file}")
            self.source={}
            self.source["data"],_  = sf.read(data_file)
            self.source["t_start"] = frame_record["t_start"]
            self.source["t_end"]   = frame_record["t_end"]
            self.source["samples"] = frame_record["samples"]
            self.source["duration"]= frame_record["t_end"] - frame_record["t_start"]
            self.last_file         = data_file

        frame_start_t = max(0,t_start)
        frame_end_t   = min(self.source["t_end"],t_end)

        ind_start = int((frame_start_t-self.source["t_start"])*(self.freq/1000))
        ind_end   = int((frame_end_t-self.source["t_start"])*(self.freq/1000))

        data = self.source["data"][ind_start:ind_end,1:5]
        if(data.shape[0]==0): return None, None

        return data, frame_end_t 

    def get_frame(self,ind=None,t=None,width=0.25):
        data,frame_end_t = self.get_waveform(ind=ind,t=t,width=width)

        if(data is None): return None, None

        f= fft(data,axis=0)
        f= np.flipud(np.log(1+np.abs(f[:len(f)//2,:])))

        if(self.max==None): self.max = np.max(f)
        if(self.min==None): self.min = np.min(f)

        self.max= max(self.max, np.max(f))
        self.min= min(self.min, np.min(f))    
            
        f = (f-self.min)/self.max

        return f, frame_end_t 
    
    def get_frames(self,start=None,end=None,scenario=None,mode="generator"):

        if(scenario is not None):
            start = scenario.info["start_time"]
            end   = scenario.info["end_time"]
        else:
            if(start is None and end is not None):
                start = self.timestamps.loc[0,"t"]
            elif(start is not None and end is None):
                L = len(self.timestamps)
                end = self.timestamps.loc[L-1,"t"]

        if(mode=="batch"):
            return self.get_frames_batch(start,end)
        elif(mode=="generator"):
            return self.get_frames_generator(start,end)
        elif(mode=="count"):
            return (end-start)//250
        else:
            self.log(f"Unknwon get_frames mode {mode}")

    def get_frames_batch(self, start,end):
            
        frames = []
        ts=[]

        duration = (end-start)
        frames   = duration//250

        for _, ind in enumerate(tqdm.tqdm(range(frames))):

            t_in = start + ind*250
            img, t_out = self.get_frame(t=t_in)

            if(img is None): break

            frames.append(img)
            ts.append(t_out)

        return frames,ts
    
    def get_frames_generator(self, start,end):
            
        frames = []
        ts=[]
        duration = (end-start)
        frames   = duration//250

        for _, ind in enumerate(range(frames)):

            t_in = start + ind*250
            img, t_out = self.get_frame(t=t_in)

            if(img is None): break

            yield img, t_out

