from abc import ABC, abstractmethod 
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

base_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(base_path)

import spatial_transform_utils as st            

def get_config():
    config_file=os.path.join(os.path.dirname(__file__),"config.json")
    with open(config_file) as f:
        config = json.load(f)
        config["gps_data_file"]=os.path.join(config["data_path"],config["gps_data_file"])
        config["zed_bags_file"]=os.path.join(config["data_path"],config["zed_bags_file"])
        config["base_path"] = os.path.dirname(__file__)
        return config

def to_timestamp(self,year,month,day,hour,minute,second=0,milliseconds=0):
    #Return a miliseconds timestamp as int64
    date_time = datetime.datetime(year, month, day, hour, minute,second,milliseconds*1000)
    return(np.int64(np.round(1000*date_time.timestamp())))

def from_timestamp(self,t):
    #Returns date from miliseconds timestamp
    return datetime.datetime.fromtimestamp(t/1000.0)

def time_delta(self,**kwargs):
    #Returns a millisecond timestamp timedelta
    if "milliseconds" in kwargs:
        kwargs["microseconds"]=kwargs["milliseconds"]*1000
        del kwargs["microseconds"]
    return np.int64(np.round(datetime.timedelta(**kwargs).total_seconds()*1000))

def get_slice_inds(index,start_time=0,end_time=None, time_delta=None):
    #Slice a tensor based on timestamp start/end/delta
    #Assumes dim 0 is timestamps in milisecond format

    if(type(index)==torch.Tensor):
        searchsorted = torch.searchsorted
    else:
        searchsorted = np.searchsorted

    if(start_time is None):
        start_time = index[0]
    if(end_time is None and time_delta  is not None):
        end_time = start_time+time_delta
    if(end_time is None and time_delta is None):
        end_time = index[-1]

    start_ind = searchsorted(index,start_time,side="left")
    end_ind   = searchsorted(index,end_time,side="right")

    if(start_ind==end_ind and end_ind < len(index)):
        end_ind = end_ind + 1

    return start_ind,end_ind

def string_encoder(x):
    labels = list(np.unique(x))
    new    = torch.zeros(x.shape,dtype=torch.int)
    for i,id in enumerate(labels):
            new[x==id] = i
    return(new,labels)



def load_track_gps_data(gps_log, env_info, date, hour, minute, duration=5,alt_correction=None):

    date_time  = datetime.datetime(date[0], date[1], date[2], hour, minute)
    start      = np.int64(time.mktime(date_time.timetuple())*1000)
    end        = start+duration*60*1000
    object_ids = list(gps_log["id"].unique())

    this_gps_log = gps_log.loc[start:end].copy()
    object_ids   = list(this_gps_log["id"].unique())
    object_ids.sort()

    tracks={}

    print(date_time)
    for id in object_ids:

        this_gps_log_for_id = this_gps_log[this_gps_log["id"]==id]
        this_gps_log_for_id = this_gps_log_for_id[this_gps_log_for_id["ac"]<5]
        
        if(len(this_gps_log)>1):

            gps_data        = torch.tensor(this_gps_log_for_id[["lt","ln","al"]].to_numpy(),dtype=torch.double)
            gps_data_meters = st.gps_to_meters(gps_data, **env_info["map"])
            time_data       = torch.tensor(this_gps_log_for_id.index.to_numpy())
            ac_data         = torch.tensor(this_gps_log_for_id["ac"].to_numpy())

            ind1 = torch.logical_not(torch.logical_and(gps_data_meters[:,0]>30,gps_data_meters[:,1]<-25))
            ind2 = torch.logical_and(ind1, gps_data_meters[:,0]>-40)
            ind2 = torch.logical_and(ind2, gps_data_meters[:,0]<40)
            ind2 = torch.logical_and(ind2, gps_data_meters[:,1]>-40)
            ind2 = torch.logical_and(ind2, gps_data_meters[:,1]<40)
            
            if(torch.sum(ind2)<50): continue

            gps_data_meters = gps_data_meters[ind1,:]
            time_data       = time_data[ind1]
            ac_data         = ac_data[ind1]
            L               = len(gps_data_meters)

            if(alt_correction is not None):
                gps_data_meters[:,2]=alt_correction

            tracks[id]={}
            tracks[id]["loc"] = gps_data_meters
            tracks[id]["t"]   = (time_data - time_data[0])/1000
            tracks[id]["ac"]  = ac_data 
            print("   ", id, "gps observations:", len(tracks[id]["t"]))

    return tracks

def load_object_gps_data(gps_log_file, from_scratch=False):
    datafile = os.path.join(data_path,"object_gps.npz")
    if(os.path.exists(datafile) and not from_scratch):
        print("Loading preprocessed data")
        data = np.load(datafile,allow_pickle=True)["gps"]
        return data
    
    print("Reading GPS csv log")
    data = pd.read_csv(gps_log_file).to_numpy()
    print("Saving preprocessed GPS data")
    np.savez_compressed(datafile,gps=data)

    return data

def to_timestamp(year,month,day,hour,minute,second=0,milliseconds=0):
    #Return a miliseconds timestamp as int64
    date_time = datetime.datetime(year, month, day, hour, minute,second,milliseconds*1000)
    return(np.int64(np.round(1000*date_time.timestamp())))

def from_timestamp(t):
    #Returns date from miliseconds timestamp
    return datetime.datetime.fromtimestamp(t/1000.0)

def time_delta(**kwargs):
    if "milliseconds" in kwargs:
         kwargs["microseconds"]=kwargs["milliseconds"]*1000
         del kwargs["microseconds"]
    return np.int64(np.round(datetime.timedelta(**kwargs).total_seconds()*1000))

def time_slice(data,start=0,dim=0,end=None, delta=None):

    if(type(data)==np.ndarray):
        searchsorted = np.searchsorted
    else:
        searchsorted = torch.searchsorted

    if(start is None):
        start = data[0,dim]
    if(end is None and delta  is not None):
        end = start+delta
    if(end is None and delta is None):
        end = data[-1,dim]

    start_ind = searchsorted(data[:,dim],start,side="left")
    end_ind   = searchsorted(data[:,dim],end,side="right")

    if(start_ind==end_ind and end_ind < data.shape[0]):
        end_ind = end_ind + 1

    return data[start_ind:end_ind,...]

def get_zed_frames(node, start, delta, zed_time_stamps):
    id=env_info["nodes"][node]["id"]

    frames = []
    vs     = {}
    zed_time_stamps_slice = time_slice(zed_time_stamps[node],start=start,delta=delta)

    if(len(zed_time_stamps_slice)<0):
        return [],[]

    last_frame_ind = None
    last_vide_file = None
    vs             = None
    for i in range(zed_time_stamps_slice.shape[0]):
        frame_date       = str(zed_time_stamps_slice[i,1].item())
        frame_time       = f"{zed_time_stamps_slice[i,2].item():06d}"
        frame_ind        = zed_time_stamps_slice[i,3].item()
        video_file = du.get_file(data_path,id,frame_date,frame_time,"left.mp4")

        if(video_file!=last_vide_file):
            vs = cv2.VideoCapture(video_file)
            last_frame_ind = None

        if(last_frame_ind is None or frame_ind!=last_frame_ind+1):
            vs.set(cv2.CAP_PROP_POS_FRAMES, int(frame_ind))

        _, frame = vs.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(img)

    return frames, list(zed_time_stamps_slice[:,0])