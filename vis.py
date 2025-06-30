import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from ipywidgets import Layout
import ipywidgets as widgets
import numpy as np
import threading
from IPython.display import display
import datetime
import data_classes as dc
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mpl_dates
import pickle
import os

from PIL import Image
import spatial_transform_utils as st
import torch
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from svgpathtools import svg2paths

def segments_from_svg(filename):
    paths, attributes,svga = svg2paths(filename,return_svg_attributes=True)
    segments=[]
    for i, path in enumerate(paths):
        for line in path:
            p1 = (line.start - 100)/2
            p2 = (line.end-100)/2
            segment = [[p1.real, 50-p1.imag], [p2.real, 50-p2.imag]]
            segments.append(segment)
    return torch.tensor(segments)

def plot_segments(ax, segments, cond=None):
    for i in range(segments.shape[0]):
        p1 = segments[i,0,:]
        p2 = segments[i,1,:]
        if(cond is not None and cond[i]):
            ax.plot([p1[0], p2[0]],[p1[1], p2[1]],"r-",alpha=0.8)
        else:
            ax.plot([p1[0], p2[0]],[p1[1], p2[1]],"b-",alpha=0.8)

def plot_map_meters(ax,center_lat=0,center_lon=0,zoom=18,map_width=800,map_height=800,xlim=[],ylim=[],type="",map_file="",cache_dir = "map_cache"):
    if(type=="satellite"):
        mppx      = st.meters_per_pixel(center_lat, zoom)
        file_name = os.path.join(cache_dir  , "map_%s_lat%.6f_lon%.6f_zoom%d.png"%(type, center_lat,center_lon,zoom))
        xlim = [-map_width*mppx/2,map_width*mppx/2]
        ylim = [-map_height*mppx/2, map_height*mppx/2]
        img       = Image.open(file_name)

        if ax is None:
            min_vals = torch.tensor([xlim[0],ylim[0]])
            max_vals = torch.tensor([xlim[1],ylim[1]])
            return img, min_vals, max_vals

        ax.imshow(img,extent=[xlim[0],xlim[1],ylim[0], ylim[1]])

        file_name = os.path.join(cache_dir  , "map_vec_lat%.6f_lon%.6f_zoom%d.svg"%(center_lat,center_lon,zoom))
        if(os.path.exists(file_name)):
            segments = segments_from_svg(file_name)
            plot_segments(ax, segments, cond=None)

    elif(type=="indoor"):
        file_name = os.path.join(cache_dir, map_file)
        img       = Image.open(file_name)

        if ax is None:
            min_vals = torch.tensor([xlim[0],ylim[0]])
            max_vals = torch.tensor([xlim[1],ylim[1]])
            return img, min_vals, max_vals

        ax.imshow(img,extent=[xlim[0],xlim[1],ylim[0], ylim[1]])

    else:
        raise(ValueError,"Unknown map type specified")
    
    #extent floats (left, right, bottom, top)
    ax.imshow(img,extent=[xlim[0],xlim[1],ylim[0], ylim[1]])
    ax.set_aspect('equal', 'box')

def plot_node(ax,node):
    #plot node location
    nodex = node["location"]["X"]
    nodey = node["location"]["Y"]

    ax.plot(nodex,nodey,'wo',markersize=12,markeredgecolor='k')
    ax.text(nodex,nodey,node["id"],size="x-small",ha="center",va="center")

def plot_zed_fov_2D(ax,node,scale=8):

    nodex = node["location"]["X"]
    nodey = node["location"]["Y"]
    fov_w = node["zed"]["fov_h_rad"]
    yaw   = node["zed"]["yaw"]

    point  = torch.tensor([[scale,0.0,0.0]],dtype=torch.float).T
    point1 = st.euler_to_rot(torch.tensor(0.0),torch.tensor(0.0), torch.tensor(fov_w/2).float()).float()@point
    point2 = st.euler_to_rot(torch.tensor(0.0),torch.tensor(0.0), torch.tensor(-1*fov_w/2).float()).float()@point
    
    cam_points = torch.hstack([point1, torch.zeros(3,1,dtype=torch.float),point2,point1])
    cam_points_rot = st.euler_to_rot(torch.tensor(0.0),torch.tensor(0.0), torch.tensor(yaw).float()).float()@cam_points

    ax.plot(nodex+cam_points_rot[0,:].numpy(), nodey+cam_points_rot[1,:].numpy(),"k-",linewidth=1)


def plot_zed_fov_3D(ax,node,scale=8):

    def add_line(v1,v2,c="k"):
        ax.plot([v1[0],v2[0]],[v1[1],v2[1]],[v1[2],v2[2]],'-',color=c)

    nodex = node["location"]["X"]
    nodey = node["location"]["Y"]
    nodez = node["location"]["Z"]

    node_pos = torch.tensor([[nodex,nodey,nodez]],dtype=torch.float).T

    fov_h = node["zed"]["fov_h_rad"]
    fov_v = node["zed"]["fov_v_rad"]

    yaw   = node["zed"]["yaw"]
    pitch = node["zed"]["pitch"]
    roll  = node["zed"]["roll"]

    dx = node['zed']['location_offset_X']
    dy = node['zed']['location_offset_Y']
    dz = node['zed']['location_offset_Z']

    R = st.euler_to_rot(torch.tensor(roll).float(), torch.tensor(pitch).float(), torch.tensor(yaw).float())
    R2=R
    #R2 = st.euler_to_rot_pytorch3D(torch.tensor(0).float(),torch.tensor(0).float(), torch.tensor(yaw).float())
    #R2 = torch.eye(3)
    #node_pos= node_pos*0

    camera_offset = torch.tensor([[dx,dy,dz]],dtype=torch.float).T

    point  = torch.tensor([[0.0,scale,0.0]],dtype=torch.float).T
    tr  = node_pos+ R2@(camera_offset+st.euler_to_rot(torch.tensor(0.0), torch.tensor(fov_v/2).float(),   -1*torch.tensor(fov_h/2).float()).float()@point)
    tl  = node_pos+ R2@(camera_offset+st.euler_to_rot(torch.tensor(0.0), torch.tensor(fov_v/2).float(),   1*torch.tensor(fov_h/2).float()).float()@point)
    br  = node_pos+ R2@(camera_offset+st.euler_to_rot(torch.tensor(0.0),-1*torch.tensor(fov_v/2).float(), -1*torch.tensor(fov_h/2).float()).float()@point)
    bl  = node_pos+ R2@(camera_offset+st.euler_to_rot(torch.tensor(0.0),-1*torch.tensor(fov_v/2).float(), 1*torch.tensor(fov_h/2).float()).float()@point)
    camera_center = node_pos+ R2@camera_offset
    floor = camera_center.clone().detach()
    floor[2,0]=0

    add_line(camera_center,tr,'k')
    add_line(camera_center,tl,'k')
    add_line(camera_center,br,'k')
    add_line(camera_center,bl,'k')
    add_line(tr,tl,'k')
    add_line(tl,bl,'k')
    add_line(bl,br,'k')
    add_line(br,tr,'k')
    add_line(camera_center,floor,'b')
    add_line(camera_center,node_pos,'r')

def zoom_on_node(ax,node,d=4):
    x = node["location"]["X"]
    y = node["location"]["Y"]

    ax.set_xlim([x-d,x+d])
    ax.set_ylim([y-d,y+d])

def plot_tracks(tracks, env_info, color_map, title="", cache_dir = "map_cache"):

    object_ids = list(tracks.keys())
    num_objects = len(object_ids)

    f=plt.figure(figsize=(9,4))

    gs  = GridSpec(2, 2, figure=f)
    ax1 = f.add_subplot(gs[:, 0])
    ax2 = f.add_subplot(gs[0, 1])
    ax3 = f.add_subplot(gs[1, 1])

    for i,id in enumerate(object_ids):
        ind = tracks[id]["ac"]<5
        ax1.plot(tracks[id]["loc"][ind,0].numpy(), tracks[id]["loc"][ind,1].numpy(), "-o",color=color_map[id],markersize=2,label=id)

    plot_map_meters(ax1,**env_info["map"],cache_dir=cache_dir)

    for node in env_info["nodes"]:
        plot_node(ax1,env_info["nodes"][node])
        plot_zed_fov_2D(ax1,env_info["nodes"][node])

    ax1.set_xlim(-40,40)
    ax1.set_ylim(-40,40)
    ax1.legend(bbox_to_anchor=(-0.15, 1.0))
    ax1.set_title(title)

    for i,id in enumerate(object_ids):
        ind = tracks[id]["ac"]<5
        ax2.plot(tracks[id]["t"][ind],tracks[id]["loc"][ind,2].numpy(),  "-o",color=color_map[id],markersize=2,label=id)
    ax2.set_title("Altitude (m)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True)

    for i,id in enumerate(object_ids):
        ind = tracks[id]["ac"]<5
        ax3.semilogy(tracks[id]["t"],tracks[id]["ac"].numpy(),  "-o",color=color_map[id],markersize=2,label=id)
    ax3.set_title("GPS precision (m)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylim(0.001, 10)
    ax3.grid(True)

    plt.tight_layout()


def confidence_sector(pos,mean_angle,std_angle,mean_dist, std_dist,ax,color="b",num_points=31):

    if(torch.is_tensor( std_dist)):std_dist  = std_dist.detach().numpy()
    if(torch.is_tensor( mean_dist)):mean_dist = mean_dist.detach().numpy()
    if(torch.is_tensor( std_angle)):std_angle  = std_angle.detach().numpy()
    if(torch.is_tensor( mean_angle)):mean_angle = mean_angle.detach().numpy()
    if(torch.is_tensor( pos)):pos        = pos.detach().numpy()

    pos = pos.reshape((1,2))

    n_std=2 #95% confidence sector
    
    lowd  = mean_dist - n_std*std_dist
    highd = mean_dist + n_std*std_dist

    #Get rotation matrix (for left multiplication)
    theta    = mean_angle
    R        = np.array([[np.cos(theta) , -np.sin(theta)],[np.sin(theta) , np.cos(theta)]]).T 

    #Get base sector
    t        = np.linspace(- n_std*std_angle, n_std*std_angle, num_points).reshape(num_points,1)
    ell      = np.hstack([np.cos(t),np.sin(t)])

    #Scale and rotate base sector
    high_ell = np.sign(highd)*(ell*0.5).dot(R)+pos 
    low_ell  = np.sign(highd)*(ell*0.1).dot(R)+ pos   
    ell      = np.vstack([high_ell,np.flipud(low_ell),high_ell[0,:]])
    h1=ax.fill(ell[:,0],ell[:,1],"-",color=color,alpha=0.25) 

    #Plot velocity interval
    pl=lowd*np.array([[1,0]]).dot(R)+pos
    ph=highd*np.array([[1,0]]).dot(R)+pos
    h2=ax.plot([pl[0,0], ph[0,0]],[pl[0,1], ph[0,1]] ,'-',color=color)
    
    return([h1,h2])


def plot_zed_data_availability(ax,zeds, start_date, start_time, end_time):
    
    start = pd.to_datetime(f"{start_date} {start_time}")
    end   = pd.to_datetime(f"{start_date} {end_time}")
    duration = (end-start).total_seconds()/(60*60)

    index = pd.date_range(start, periods=duration*60, freq="min")
    df_base = pd.DataFrame(index=index)

    for id,zed in zeds.items():
        
        df = zed.timestamps
        df["t"] = pd.to_datetime(df["t"],unit="ms")
        df = df.set_index("t")
        df = df.resample("1min").count()
        df[id] = df["frame"]

        df_base=df_base.merge(df[[id]], how="left",left_index=True, right_index=True)

    num_zeds = len(df_base.columns)
    num_timepoints = len(df_base)

    start = mpl_dates.date2num(index[0].to_pydatetime())
    end = mpl_dates.date2num(index[-1].to_pydatetime())

    img = df_base.fillna(0).to_numpy().T
    ax.imshow(img,extent=[start,end,num_zeds,0],interpolation="none",cmap=plt.gray())
    ax.axis("auto")
    #ax.colorbar()
    ax.grid(True)
    ax.set_yticks(range(num_zeds))
    #ßax.set_xticks(np.linspace(0,(duration)*60,(duration+1)))
    ax.set_ylim(0,num_zeds)
    ax.set_yticklabels(df_base.columns,va='bottom')
    ax.set_title(f"Video Availability: {start_date}")

    myFmt = mpl_dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    #ax.set_xlim(local_time[0],local_time[-1])
    ax.xaxis.set_major_locator(mpl_dates.MinuteLocator(interval=15))
    ax.tick_params(axis='x', labelrotation=90)

def plot_object_activity(ax,objects, start_date, start_time, end_time,motion_threshold=0):
    
    start = pd.to_datetime(f"{start_date} {start_time}")
    end   = pd.to_datetime(f"{start_date} {end_time}")
    duration = (end-start).total_seconds()/(60*60)

    index = pd.date_range(start, periods=duration*60, freq="min")
    df_base = pd.DataFrame(index=index)

    for object_id,object in objects.items():
        
        df = object.data["world"]
        df["t"] = pd.to_datetime(df["t"],unit="ms")
        df = df.set_index("t")
        df = df[["x","y"]]
        df = df.resample("1min").std()
        df[object_id] = 1*((df["x"]+df["y"])/2 > motion_threshold)

        df_base=df_base.merge(df[[object_id]], how="left",left_index=True, right_index=True)

    num_objects = len(df_base.columns)
    num_timepoints = len(df_base)

    start = mpl_dates.date2num(index[0].to_pydatetime())
    end = mpl_dates.date2num(index[-1].to_pydatetime())

    img = df_base.fillna(0).to_numpy().T
    ax.imshow(img,extent=[start,end,num_objects,0],interpolation="none",cmap=plt.gray())
    ax.axis("auto")
    #plt.colorbar()
    ax.grid(True)
    ax.set_yticks(range(num_objects))
    #ax.set_xticks(np.linspace(0,(duration)*60,(duration+1)))
    ax.set_ylim(0,num_objects)
    ax.set_yticklabels(df_base.columns,va='bottom')
    ax.set_title(f"Object Activity: {start_date}")

    myFmt = mpl_dates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    #ax.set_xlim(local_time[0],local_time[-1])
    ax.xaxis.set_major_locator(mpl_dates.MinuteLocator(interval=15))
    ax.tick_params(axis='x', labelrotation=90)


class data_viewer():
    def __init__(self, env, scenarios, objects, sources,cols=3,threaded=False,continuous_update=False,show_obj_label=False,max_staleness=500):
        self.objects       = objects
        self.sources      = sources
        self.num_zeds      = len(sources)
        self.env           = env
        self.scenarios     = scenarios
        self.duration      = 1
        self.available_objects = []
        self.scenario      = None
        self.loading_scenario = False
        self.continuous_update = continuous_update
        self.threaded=threaded
        self.show_obj_label = show_obj_label
        self.max_staleness=max_staleness

        self.object_ids    = list(objects.keys())
        self.num_objects   = len(self.object_ids)

        self.subplot_cols  = cols

        self.updating_plots=False
        self.video_timer=None

        self.updating={}
        self.updating["map"]=False
        for i,(source_key,source) in enumerate(self.sources.items()):
            self.updating[source.name]=False

        self.f = None
        self.init_widgets()

    def load_scenario(self,event):

        self.loading_scenario=True

        name = self.wscenario_picker.value

        self.scenario   = self.scenarios[name]
        self.node_names = list(self.scenario.info["nodes"])
        self.num_nodes  = len(self.node_names)
        self.start_time = self.scenario.info["start_time"]
        self.end_time   = self.scenario.info["end_time"]
        self.duration   = (self.end_time - self.start_time)/1000

        #Localize objects to the scenario
        self.localized_objects = {}
        self.available_objects = []
        for id, object in self.objects.items():
            self.localized_objects[id] = object.localize(self.scenario,inplace=False)
            if(not self.localized_objects[id].is_empty()): self.available_objects.append(id)

        self.num_objects = len(self.available_objects)
        self.have_objects = self.num_objects>0
        if(self.have_objects):
            self.current_object = self.available_objects[0]

        self.current_locations = {}
        self.current_frames  = {}

        #Get frame indices for this scenario
        self.source_frame_inds={}
        for _, source in self.sources.items():
            self.source_frame_inds[source.name] = source.get_frames(scenario=self.scenario,mode="inds")

        #Update widgets
        self.wtime.max   = self.duration
        self.wtime.value = 0 
        self.wobject.options=self.available_objects
        self.wobject.value=self.available_objects[0]

        self.init_plots()
        self.loading_scenario=False
        self.update(selected_t=self.wtime.value)

    def video_step(self,increment):
        value = max(0, min(self.duration, self.wtime.value + increment))
        self.wtime.value=value

    def video_next(self,event):
        self.video_step(self.wstep.value)

    def video_prev(self,event):
        self.video_step(-1*self.wstep.value)

    def save_annotations(self):
        num=0
        name = self.scenario.name.replace(":","").replace("/","").replace(" ","-") + f"-{num}.pkl"

        while(os.path.exists(name)):
            num+=1
            name = self.scenario.name.replace(":","").replace("/","").replace(" ","-") + f"-{num}.pkl"

        with open(name,"wb") as f:
            pickle.dump(self.correction_data,f)

    def wobject_on_change(self,change):
        if change['type'] == 'change' and change['name'] == 'value':
            self.update(selected_t = self.wtime.value)

    def init_widgets(self):
        self.wtime   = widgets.FloatSlider(value=0, min=0, max=self.duration,step=0.25,layout=Layout(width='1000px'),continuous_update=self.continuous_update,readout_format='.3f',description="Time")
        self.wobject = widgets.Dropdown(value=None,options=self.available_objects,description="Label Object")
        self.wstep   = widgets.FloatText(value=1, min=0,description="Step")
        self.wnext   = widgets.Button(description="Next")
        self.wprev   = widgets.Button(description="Prev")

        self.wnext.on_click(self.video_next)
        self.wprev.on_click(self.video_prev)


        if(len(self.scenarios))>0:
            scenario_list = list(self.scenarios.keys())
            self.wscenario_picker   = widgets.Dropdown(value=None,options=scenario_list,description="Scenario")
        else:
            self.wscenario_picker   = widgets.Dropdown(value=None,options=[""],description="Scenario")

        self.wload = widgets.Button(description='Load')
        self.wload.on_click(self.load_scenario)

        self.wsave = widgets.Button(description='Save Labels')
        self.wsave.on_click(self.save_annotations)
        self.wsave.layout.visibility = 'hidden'

        hbox = widgets.HBox([self.wscenario_picker, self.wobject, self.wload, self.wsave])
        display(hbox)

        hbox = widgets.HBox([self.wstep,self.wnext,self.wprev])
        display(hbox)

        self.wobject.observe(self.wobject_on_change)

        widgets.interact(self.update, selected_t=self.wtime, selected_object_id=self.wobject)

    def init_plots(self):

        #Setup plots
        self.subplot_rows = int(np.ceil((1+self.num_zeds)/self.subplot_cols))

        if(self.f is not None):
            for ax in self.axs:
                ax.clear()
        else:
            fig_height = self.subplot_rows*2.5
            fig_width  = self.subplot_cols*4
            self.f, self.axs=plt.subplots(self.subplot_rows,self.subplot_cols,figsize=(fig_width,fig_height))
            self.axs = self.axs.flatten()

        #Plot the map
        self.ax_map=self.axs[0]
        self.env.plot_map(self.ax_map)
        self.env.plot_nodes(self.ax_map)
        
        #Create handles for objects
        self.h_objects_map = {}    
        for j,id in enumerate(self.available_objects):
            h=self.ax_map.plot(0, 0, "o", color=self.localized_objects[id].color,markersize=5,label=id)
            self.h_objects_map[id]=h[0]
            self.current_locations[id]=[]

        self.ax_map.legend(bbox_to_anchor=(0, 1.0))
        self.ax_map.set_xlim(-40,40)
        self.ax_map.set_ylim(-40,40)
        self.ax_map.set_axis_off()
        self.ax_map.set_title("Map")

        #Plot image frame 
        self.source_axs={}
        self.h_objects_video_label={}
        self.h_objects_video_marker={}
        self.h_frames = {}
        self.h_status={}

        #Plot initial frames
        for i,(source_key,source) in enumerate(self.sources.items()):
            source_name=source.name
            self.source_axs[source_name]  = self.axs[i+1]
            self.h_status[source_name] = self.source_axs[source_name].text(0,int(source.scale*1080), "Status:", va="bottom", ha="left",color='r')
            self.h_frames[source_name] = self.source_axs[source_name].imshow(np.random.rand(int(source.scale*1080),int(source.scale*1920),3),interpolation="nearest")
            self.source_axs[source_name].set_axis_off()

            #Show markers on frame
            self.h_objects_video_label[source_name]={}
            self.h_objects_video_marker[source_name]={}
            for id in self.available_objects:

                if(self.show_obj_label):
                    h=self.source_axs[source_name].text(0, 0, id,color=self.objects[id].color,clip_on=True,va="bottom", ha="center")
                    h.set_bbox(dict(facecolor='white', alpha=0.5,linewidth=0,boxstyle="round,pad=0"))
                    self.h_objects_video_label[source_name][id]=h

                h=self.source_axs[source_name].plot(100,100,"o",markerfacecolor="None",markeredgecolor=self.objects[id].color,alpha=0.8,markersize=12,clip_on=True)[0]
                self.h_objects_video_marker[source_name][id]=h
                
        #Turn off unused axes
        for j in np.arange(len(self.sources),self.subplot_cols*self.subplot_rows):
            self.axs[j].set_axis_off()

        self.f.canvas.toolbar_visible = False 
        self.f.canvas.header_visible = False

        #plt.tight_layout()

    def update(self,selected_t=0):

        if self.scenario is None: return
        if(self.loading_scenario): return

        ts = self.start_time + selected_t*1000
        threads={}

        #Map update thread
        if(self.threaded):
            threads["map"]=threading.Thread(target=self.update_map,args=(ts,))
            threads["map"].start()
        else:
            self.update_map(ts)

        #Video update threads
        for i,(source_key,source) in enumerate(self.sources.items()):
            source_name=source.name
            if(self.threaded):
                threads[source_name]=threading.Thread(target=self.update_video,args=(source,ts))
                threads[source_name].start()
            else:
                self.update_video(source,ts)

        #Wait for all threads to finish all threads
        if(self.threaded):
            for name, thread in threads.items():
                thread.join()

    def update_map(self,ts):
        #Update map:
        #if self.updating["map"]: return
        self.updating["map"]=True

        for id in self.available_objects:

            object = self.localized_objects[id]
            object_data = object.get_data(ts,thresh=self.max_staleness)

            if(object_data is not None):
                self.current_locations[id] = object_data
                self.h_objects_map[id].set_data([object_data["world"]["x"]], [object_data["world"]["y"]])
                self.ax_map.set_title(f"{pd.to_datetime(object_data['world']['t'],unit='ms').strftime('%d/%m/%y %H:%M:%S.%f')[:-3]}")

            else:
                self.current_locations[id]=None
                self.ax_map.set_title("No data")
                self.h_objects_map[id].set_data([-100],[-100])

        self.updating["map"]=False

    def update_video(self, source, ts):
        #Fetch and update one video
        if self.updating[source.name]: return
        self.updating[source.name]=True
        img,t = source.get_frame(t=ts)
        
        if(img is None):
            img = np.random.rand(int(1080*source.scale),int(1920*source.scale),3)
            t_str="NA"
            self.current_frames[source.name] = None
        else:
            t_str = pd.to_datetime(t,unit='ms').strftime('%d/%m/%y %H:%M:%S.%f')[:-3]
            self.current_frames[source.name] = t

        if(source.modality=="rd"):
            img=img.swapaxes(0,1)
            img = np.flipud(img)

        self.h_frames[source.name].set_data(img)
        self.source_axs[source.name].set_title(f"{source.name}: {t_str}")
        
        #Show marker of selected object on frame
        for id in self.available_objects:

            loc = self.localized_objects[id].get_data(ts,thresh=self.max_staleness)

            if(id==self.wobject.value and loc is not None and loc[source.node]["visible"]):
                self.h_objects_video_marker[source.name][id].set_data([loc[source.node]["h"]*source.scale],[(loc[source.node]["v"])*source.scale])      
                self.h_objects_video_marker[source.name][id].set_color(self.objects[id].color)  
                if(self.show_obj_label):       
                    self.h_objects_video_label[source.name][id].set_position([loc[source.node]["h"]*source.scale,(-40+loc[source.node]["v"])*source.scale])
            else:
                self.h_objects_video_marker[source.name][id].set_data([-10,-10])
                if(self.show_obj_label):
                    self.h_objects_video_label[source.name][id].set_position([-10,-10])  

            if(source.modality in ["rd", "audio"]):
                self.h_objects_video_marker[source.name][id].set_data([-100,-100])
                if(self.show_obj_label):
                    self.h_objects_video_label[source.name][id].set_position([-100,-100])                               

        self.updating[source.name]=False

class data_labeler(data_viewer):

    def __init__(self,env, scenarios, objects, zed_data,cols=3,continuous_update=False):
        super().__init__(env, scenarios, objects, zed_data,cols,continuous_update=continuous_update)

    def load_scenario(self,event):
        
        super().load_scenario(event)

        self.f.canvas.mpl_connect('button_press_event', self.onclick)
        self.correction_data = {}


    def onclick(self,event):

            for zed_name, ax in self.source_axs.items():

                if event.inaxes is ax:

                    object_id = self.wobject.value

                    if(self.current_frames[zed_name] is None or self.current_locations[object_id] is None): 
                        self.h_status[zed_name].set_text("No data")
                        return

                    if zed_name not in self.correction_data:
                        self.correction_data[zed_name] = []

                    correction={}
                    correction["obj"]        = object_id 
                    correction["frame_time"] = self.current_frames[zed_name]
                    correction["loc_world"]  = self.current_locations[object_id]["world"]
                    correction["loc_img"]    = [event.xdata/self.sources[zed_name].scale ,event.ydata/self.sources[zed_name].scale]
                    self.correction_data[zed_name].append(correction)    

                    text = f"Labeled: {zed_name} {object_id} {len(self.correction_data[zed_name])}"
                    
                    self.h_status[zed_name].set_text(text)

class scenario_segmenter():
    def __init__(self, env, objects,scenarios):
        self.objects       = objects
        self.env           = env

        self.object_ids    = list(objects.keys())
        self.num_objects   = len(self.object_ids)

        self.updating={}
        self.updating["map"]=False

        self.day_start = 9*60
        self.day_end   = 17*60
        self.duration =  (self.day_end - self.day_start)
        self.dates=['2023-09-18','2023-09-19','2023-09-20']

        self.scenarios = scenarios

        self.init_plots()
        self.init_widgets()

    def set_range_slider(self,event):
        start = self.wstart.value.hour*60 + self.wstart.value.minute - self.day_start
        end   = self.wend.value.hour*60   + self.wend.value.minute - self.day_start
        if(end>start):
            self.wtimerange.value = [start,end]
        print(f"set to {start} {end}")

    def set_time_pickers(self):
        start,end = self.wtimerange.value

        start_hour = (start + self.day_start)//60
        start_minute = (start + self.day_start) % 60

        end_hour = (end + self.day_start)//60
        end_minute = (end + self.day_start) % 60

        self.wstart.value = datetime.time(start_hour,start_minute)
        self.wend.value = datetime.time(end_hour,end_minute)

    def load_scenario(self,event):
        scenario_id = self.wscenario_picker.value
        scenario    = self.scenarios[scenario_id]
        start       = pd.to_datetime(scenario.info["start_time"],unit="ms")
        end         = pd.to_datetime(scenario.info["end_time"],unit="ms")

        date_str = start.strftime("%Y-%m-%d")
        self.wdate.value = date_str

        start_hour   = start.hour
        start_minute = start.minute
        end_hour     = end.hour 
        end_minute   = end.minute

        self.wstart.value = datetime.time(start_hour,start_minute)
        self.wend.value = datetime.time(end_hour,end_minute)   

        self.wdescription.value = self.scenarios[scenario_id].info["name"]

        self.set_range_slider(None)    

    def save_scenario(self,event):

        start_time = pd.to_datetime(self.wdate.value) + pd.Timedelta(minutes=self.wstart.value.hour*60 + self.wstart.value.minute)
        end_time   = pd.to_datetime(self.wdate.value) + pd.Timedelta(minutes=self.wend.value.hour*60 + self.wend.value.minute)

        info               = {}
        info["start_time"] = str(start_time) 
        info["end_time"]   = str(end_time) 
        info["name"]       = self.wdescription.value
        info["nodes"]      = self.env.info["nodes"]

        scenario = dc.scenario(info["name"],self.env,info)
        scenario.save()

        display_name = f"{start_time.strftime('%Y/%m/%d %H:%M:%S')}: {info['name']}"       
        self.scenarios[display_name] = scenario 

        scenario_names = list(self.scenarios.keys())
        scenario_names.sort()
        self.wscenario_picker.options = scenario_names
        self.wscenario_picker.value   = display_name 

    def init_widgets(self):
        self.wdate   = widgets.Dropdown(value=self.dates[0],options=self.dates,description="Date")
        self.wtimerange   = widgets.IntRangeSlider(value=[0,5], min=0, max=self.duration,step=1,layout=Layout(width='1000px'),continuous_update=True,readout_format='d',description="Time")
        
        if(len(self.scenarios))>0:
            scenario_list = list(self.scenarios.keys())
            self.wscenario_picker   = widgets.Dropdown(value=None,options=scenario_list,description="Scenario")
        else:
            self.wscenario_picker   = widgets.Dropdown(value=None,options=[""],description="Scenario")

        self.wload     = widgets.Button(description='Load')
        self.wload.on_click(self.load_scenario)

        self.wstart  = widgets.TimePicker(description='Start',value=datetime.time(9,0))
        self.wend    = widgets.TimePicker(description='End',value=datetime.time(9,5))
        self.wgo     = widgets.Button(description='Go')
        self.wgo.on_click(self.set_range_slider)

        self.wdescription = widgets.Text(value="",layout=Layout(width='500px'), description="Name")
        self.wsave   = widgets.Button(description='Save') 
        self.wsave.on_click(self.save_scenario)

        hbox_date     = widgets.HBox([self.wdate,self.wstart,self.wend,self.wgo])
        hbox_scenario = widgets.HBox([self.wscenario_picker,self.wload,self.wdescription,self.wsave])
        out           = widgets.interactive_output(self.update, {"date":self.wdate, "time_range": self.wtimerange})
        vbox          = widgets.VBox([hbox_date,self.wtimerange,hbox_scenario])

        display(vbox , out)

    def init_plots(self):

        self.f=plt.figure(figsize=(9,4))
        gs  = GridSpec(2, 2, figure=self.f)
        self.ax_map = self.f.add_subplot(gs[:, 0])
        self.ax_alt = self.f.add_subplot(gs[0, 1])
        self.ax_acc = self.f.add_subplot(gs[1, 1])

        self.ax_alt.set_title("Altitude (m)")
        self.ax_alt.set_xlabel("Time (s)")
        self.ax_alt.grid(True)
        
        self.ax_acc.set_title("GPS precision (m)")
        self.ax_acc.set_xlabel("Time (s)")
        self.ax_acc.set_ylim(0.001, 10)
        self.ax_acc.grid(True)

        #Plot the map
        self.env.plot_map(self.ax_map)
        self.env.plot_nodes(self.ax_map)
        
        #Create handles for objects
        self.h_objects_map = {}    
        self.h_objects_acc = {} 
        self.h_objects_alt = {} 
        for j,id in enumerate(self.objects):
            h=self.ax_map.plot(0, 0, "o", color=self.objects[id].color,markersize=5,label=id)[0]
            self.h_objects_map[id]=h
            h=self.ax_alt.plot(0, 0, "o", color=self.objects[id].color,markersize=5,label=id)[0]
            self.h_objects_alt[id]=h
            h=self.ax_acc.semilogy(0, 1, "o", color=self.objects[id].color,markersize=5,label=id)[0]
            self.h_objects_acc[id]=h

        self.hlegend = self.ax_map.legend(bbox_to_anchor=(0, 1.0))
        self.ax_map.set_xlim(-40,40)
        self.ax_map.set_ylim(-40,40)
        self.ax_map.set_axis_off()
        self.ax_map.set_title("Map")

        self.f.canvas.toolbar_visible = False 
        self.f.canvas.header_visible = False

        plt.tight_layout()


    def update(self,date,time_range):

        self.set_time_pickers()

        start = int(pd.to_datetime(date).timestamp()*1000) + (self.day_start+time_range[0])*60*1000
        end   = int(pd.to_datetime(date).timestamp()*1000) + (self.day_start+time_range[1])*60*1000

        have_objects = False
        for id in self.objects:
            if(self.h_objects_map[id] is not None):
                self.h_objects_map[id].remove()
            if(self.h_objects_alt[id] is not None):
                self.h_objects_alt[id].remove()
            if(self.h_objects_acc[id] is not None):
                self.h_objects_acc[id].remove()

            hmap = self.objects[id].plot_track(self.ax_map, range=[start,end],min_count=5)
            halt = self.objects[id].plot_altitude(self.ax_alt, range=[start,end],min_count=5)
            hacc = self.objects[id].plot_accuracy(self.ax_acc, range=[start,end],min_count=5)

            if(hmap is None):
                self.h_objects_map[id] = None
                self.h_objects_alt[id] = None
                self.h_objects_acc[id] = None
            else:
                self.h_objects_map[id] = hmap
                self.h_objects_alt[id] = halt
                self.h_objects_acc[id] = hacc
                have_objects = True

        if(self.hlegend is not None):
            self.hlegend.remove()

        if(have_objects):
            self.hlegend = self.ax_map.legend(bbox_to_anchor=(0, 1.0))
        else:
            self.hlegend = None
        
        self.ax_map.set_title(f"{pd.to_datetime(start,unit='ms').strftime('%Y-%m-%d %H:%M:%S')} to {pd.to_datetime(end,unit='ms').strftime('%Y-%m-%d %H:%M:%S')}")

def figure_to_array(fig):
    fig.canvas.draw()
    arr = np.array(fig.canvas.renderer._renderer)
    return arr[:,:,[2,1,0]]

def plot3d_box_points(ax,points,color='b'):
    for i in range(4):
        j=(i+1)%4
        ax.plot([points[i,0],points[j,0]],[points[i,1],points[j,1]],':',color=color)
        ax.plot([points[4+i,0],points[4+j,0]],[points[4+i,1],points[4+j,1]],':',color=color)
        ax.plot([points[i,0],points[4+i,0]],[points[i,1],points[4+i,1]],':',color=color)

def plot_2dbox_chw(ax,boxchw,color='g'):
    points = torch.tensor([[-1,1], [1,1],[1,-1],[-1,-1]])/2
    points = points * boxchw[2:]
    points = points + boxchw[:2]
    for i in range(4):
        j=(i+1)%4
        ax.plot([points[i,0],points[j,0]],[points[i,1],points[j,1]],'-',color=color,alpha=0.5,linewidth=3)