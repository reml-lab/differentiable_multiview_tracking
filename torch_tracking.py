import torch
import torch.nn as nn
import abc
import tqdm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import data_classes as dc
import spatial_transform_utils as st
from scipy.optimize import linear_sum_assignment


def vec2str(x):
    l=["%.3f"%v.item() for v in x]
    return "[" + ", ".join(l) + "]"

def plot_mocap_tracks(ax,env,all_dets=None, objs=None,Zs=None,Ps=None):
    
    env.plot_map(ax)
    env.plot_nodes(ax)

    #Plot detections
    if(all_dets is not None):
        colors=["k","r","g",'b','k']
        nodes = list(all_dets[0].keys())
        for n, node in enumerate(nodes):
            dets = torch.vstack([det[node]["X"]   for det in all_dets if node in det])
            dc.track(dets,color=colors[n]).plot_track(ax,line_style=".",alpha=0.25)

    colors = list(mcolors.TABLEAU_COLORS.values())

    #Plot tracks
    if(Zs is not None):
        T,K,DZ = Zs.shape

        for k in range(K):
            dc.track(Zs[:,k,:],cov=Ps[:,k,:,:],color=colors[k],name=f"Track{k}").plot_track(ax,line_style="-",alpha=0.25)

    #Plot ground truth
    if(objs is not None):
        K = objs.shape[1]
        for i in range(K):
            dc.track(objs[:,i,:],color=colors[i],name=f"Obj{i}").plot_track(ax,line_style="-")

def plot_mocap_orientations(ax,env,objs,all_pos, all_yaw, l, w, alpha=0.5):
        
    env.plot_map(ax)
    env.plot_nodes(ax)

    T,K,DZ = all_pos.shape

    colors = list(mcolors.TABLEAU_COLORS.values())

    for k in range(K):
        pos = all_pos[0:T:5,k,:]
        yaw = all_yaw[0:T:5,k]

        #Plot ground truth
        dc.track(objs[:,k,:],color=colors[k],name=f"Obj{k}").plot_track(ax,line_style="-",alpha=0.5,zorder=1)

        for t in range(pos.shape[0]):

            rot    = torch.tensor([[torch.cos(yaw[t]),-torch.sin(yaw[t])],[torch.sin(yaw[t]),torch.cos(yaw[t])]])
            bcords = torch.tensor([[l,-l,-l,l],[w,w,-w,-w]])/2
            bcords = rot@bcords +  pos[t,:].reshape((2,1))

            #h1=ax.plot(bcords[0,:].numpy(),bcords[1,:].numpy(),"-",color=colors[k],linewidth=2,alpha=alpha)

            # Create a Rectangle patch
            #rect = patches.Rectangle((pos[t,0], pos[t,1]), l, w, linewidth=2, edgecolor=colors[k], facecolor='w', rotation_point="center", angle=torch.rad2deg(yaw[t]),zorder=2)
            rect = patches.Polygon(bcords.T.numpy(), linewidth=2, edgecolor=colors[k], facecolor='w', zorder=2, alpha=alpha)
            ax.add_patch(rect)

            dax = l/2*torch.cos(yaw[t])
            day = l/2*torch.sin(yaw[t])
            h2=ax.arrow(pos[t,0],pos[t,1],dax,day,width=0.05,color=colors[k],head_width=0.1,head_length=0.1,length_includes_head=False,alpha=alpha,zorder=3)

def inverse_softplus(a):
    return torch.log(torch.exp(a)-1)

def inverse_logistic(p):
    eps = 1e-10
    q   = 0.5*eps + (1-eps)*p
    return torch.log(q/(1-q))

class bbox_measurement_model():
    
    def __init__(self, obj_dims):
        self.obj_dims = obj_dims
        self.linear=False

    def project_box(self,Z,proj):

        Zshape = list(Z.shape)
        Z      = Z.reshape(-1,5)

        K,DZ       = Z.shape
        zeroszK    = torch.zeros(K,1)  
        p          = torch.cat([Z[:,:2],torch.zeros(K,1)],dim=1)
        rot        = st.euler_to_rot_torch(zeroszK,zeroszK,Z[:,[2]])
        box3d      = st.get_3d_box_points(p,rot,self.obj_dims)
        box3d_proj = proj.world_to_image(box3d.reshape(-1,3)).reshape(-1,8,3)
        box2d      = st.box3d_points_to_box2d_chw(box3d_proj)

        Xshape     = Zshape
        Xshape[-1] = 4
        box2d      = box2d.reshape(Xshape)

        return box2d

    def project_Z2X(self,Z,P,det):

        K,DZ = Z.shape

        proj = det["Proj"]
        f  = lambda zz: self.project_box(zz,proj)
        us = st.unscented_transform(f,5,diag_cov=False,lmbda=1e-6,mode="basic")

        #Check which Z's are visible from this perspective
        #Counts if object center is in the frame
        p           = torch.cat([Z[:,:2],torch.zeros(K,1)],dim=1)
        center_proj = proj.world_to_image(p).detach()
        ind         = center_proj[:,2]>0 
        ind         = torch.logical_and(ind, center_proj[:,0]>0)
        ind         = torch.logical_and(ind, center_proj[:,0]<1920)
        ind         = torch.logical_and(ind, center_proj[:,1]>0)
        ind         = torch.logical_and(ind, center_proj[:,1]<1080)
        ind         = ind.float()

        if(ind.sum()>0):
            Xnew, Snew, Cross_new =  us.transform(Z,P,return_cross_cov=True)
            Xnew = ind.reshape(K,1)*Xnew + torch.nan**((1-ind).reshape(K,1))
            Snew = ind.reshape(K,1,1)*Snew + torch.nan**((1-ind).reshape(K,1,1))
            Cross_new = ind.reshape(K,1,1)*Cross_new + torch.nan**((1-ind).reshape(K,1,1))
        else:
            Xnew = torch.nan  * torch.ones(K,4)
            Snew = torch.nan  * torch.ones(K,4,4)
            Cross_new =  torch.nan  * torch.ones(K,5,4)

        return Xnew.reshape(K,4), Snew, Cross_new   #was Xnew.squeeze(0)

class linear_measurement_model():
    
    def __init__(self,H):
        self.H = H
        self.DX,self.DZ = H.shape
        self.linear=True

    def project_state_Z2X(self,Z):
        Zdims = len(Z.shape)

        Hshape = [1]*(Zdims -1) + [self.DX,self.DZ]
        X = (self.H.reshape(Hshape)@(Z.unsqueeze(-1))).squeeze(-1)
        return(X)
    
    def project_cov_Z2X(self,P):
        Pdims   = len(P.shape)
        Hshape  = [1]*(Pdims -2) + [self.DX,self.DZ]
        HshapeT = [1]*(Pdims -2) + [self.DZ,self.DX]

        X = (self.H.reshape(Hshape))@P@((self.H.T).reshape(HshapeT))
        return(X)

    def project_Z2X(self,Z,P,det):

        return self.project_state_Z2X(Z), self.project_cov_Z2X(P)



        #if(dets is None):
        #    return self.project_state_Z2X(Z), self.project_cov_Z2X(P)
        #else:
        #    N,DX = dets["X"].shape
        #    K,DZ = Z.shape
        #   return self.project_state_Z2X(Z).unsqueeze(1).expand(K,N,self.DX), self.project_cov_Z2X(P).unsqueeze(1).expand(K,N,self.DX,self.DX)
    def project_Z2X_dist(self,Z,P):
        normals = []
        for z, p in zip(Z,P):
            z = self.project_state_Z2X(z)
            p = self.project_cov_Z2X(p)
            p = (p + p.T)/2
            normal = torch.distributions.MultivariateNormal(z.unsqueeze(0),p.unsqueeze(0))
            normals.append(normal)
        return normals


    def project_state_X2Z(self,X):
        Xdims = len(X.shape)

        HshapeT = [1]*(Xdims -1) + [self.DZ,self.DX]
        X = ((self.H.T).reshape(HshapeT)@(X.unsqueeze(-1))).squeeze(-1)
        return(X)

    def project_cov_X2Z(self,R):
        Rdims   = len(R.shape)
        Hshape  = [1]*(Rdims -2) + [self.DX,self.DZ]
        HshapeT = [1]*(Rdims -2) + [self.DZ,self.DX]

        X = ((self.H.T).reshape(HshapeT))@R@(self.H.reshape(Hshape))
        return(X)

class composite_motion_model(nn.Module):
    def __init__(self,models,name):
        super().__init__()
        self.models=models
        self.name = name

    def advance_cov(self,dt,P):
        Pnew = []
        ind_start = 0
        for m in self.models:
            ind = ind_start + torch.IntTensor(range(m.DZ))
            Pnew.append(m.advance_cov(dt,P[:,ind,:][:,:,ind]))
            ind_start += m.DZ 

        K = P.shape[0]
        Pcombos=[]
        for k in range(K):
            Pcombos.append(torch.block_diag(*[x[k,:,:] for x in Pnew]))
        Pnew = torch.stack(Pcombos,dim=0)

        return Pnew
    
    def advance_state(self,dt,Z):
        Znew = []
        ind_start = 0
        for m in self.models:
            ind = ind_start + torch.IntTensor(range(m.DZ))
            Znew.append(m.advance_state(dt,Z[:,ind]))
            ind_start += m.DZ
        return torch.hstack(Znew)

class constant_value_model(nn.Module):
    def __init__(self,D):
        super().__init__()
        self.D = D
        self.name="Constant Value"

    @property
    def DZ(self):
        return self.D

    def advance_cov(self,dt,P):
        return P

    def advance_state(self,dt,Z):
        return Z

class constant_velocity_model(nn.Module):
    # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
    def __init__(self,std_acc,D):
        super().__init__()
        self.std_acc_prime = nn.Parameter(inverse_softplus(torch.tensor(std_acc)))
        self.D = D
        self.name = "CV"

    def __repr__(self):
        return f"{self.name}: acc: {self.std_acc:0.3f}"

    @property
    def DZ(self):
        return 2*self.D

    @property
    def std_acc(self):
        return torch.nn.functional.softplus(self.std_acc_prime)

    def getF(self,dt):
        baseF = torch.FloatTensor([[1, dt],[0, 1]])
        return torch.block_diag(*([baseF]*self.D))

    def getQ(self,dt):

        baseQ = torch.FloatTensor([[(dt**4)/4, (dt**3)/2],[(dt**3)/2, dt**2,]]).to(self.std_acc.device)
        baseQ = (self.std_acc**2)*baseQ
        return torch.block_diag(*([baseQ]*self.D))

    def advance_state(self,dt,Z,P):
        DZ = P.shape[-1]
        F = self.getF(dt).to(Z.device)
        Q = self.getQ(dt).to(Z.device)
        Zdims = len(Z.shape)
        Fshape = [1]*(Zdims-1) + [2*self.D,2*self.D]

        Znew = (F.reshape(Fshape)@Z.unsqueeze(-1)).squeeze(-1)
        Pnew = F.reshape(1,DZ,DZ) @ P @ F.T.reshape(1,DZ,DZ) + Q.reshape(1,DZ,DZ)

        return Znew, Pnew

class steerable_model(nn.Module):

    #Reference: https://users.ece.cmu.edu/~hyunggic/papers/2011_icra_bicycle_tracking.pdf
    #Reference: https://math.stackexchange.com/questions/881182/how-to-derive-the-process-noise-co-variance-matrix-q-in-this-kalman-filter-examp

    def __init__(self,std_accl=1,std_accy=1,use_circular=True,num_grid_points=8,use_us=False,verbose=False):
        super().__init__()

        self.dim=5
        self.linear_dynamics=False
        self.linear_measurement=True
        self.name = "CTRV"
        self.use_circular=use_circular
        self.verbose=verbose

        self.std_accl_prime = nn.Parameter(inverse_softplus(torch.tensor(std_accl)))
        self.std_accy_prime = nn.Parameter(inverse_softplus(torch.tensor(std_accy)))

        self.use_us = use_us

        extent = (num_grid_points-1)/2
        x = torch.linspace(-1*extent,1*extent,num_grid_points)
        grids = torch.meshgrid(*([x]*5))
        self.sigma_points = torch.stack([x.reshape(-1) for x in grids]).T
        self.sigma_points = torch.vstack([torch.zeros(1,5), self.sigma_points])
        self.S = self.sigma_points.shape[0]

        self.sigma_weights = torch.distributions.Normal(0,scale=1).log_prob(self.sigma_points).sum(dim=1,keepdims=True)
        self.sigma_weights = torch.exp(self.sigma_weights - torch.logsumexp(self.sigma_weights,dim=0))

    def __repr__(self):
        return f"{self.name}: accl: {self.std_accl:0.3f} accy: {self.std_accy:0.3f}"

    @property
    def std_accl(self):
        return torch.nn.functional.softplus(self.std_accl_prime)

    @property
    def std_accy(self):
        return torch.nn.functional.softplus(self.std_accy_prime)

    def state_from_components(self,pos=torch.zeros(2,1),psi=torch.tensor(0),v=torch.tensor(4.0),omega=torch.tensor(0)):     
        #L = pos.shape[0]
        if(psi.shape==0): psi = psi.unsqueeze(0)
        if(v.shape==0): v = v.unsqueeze(0)
        if(omega.shape==0): omega = omega.unsqueeze(0)

        state = torch.hstack([pos,psi,v,omega])
        if len(pos.shape)==1:
            state = state.unsqueeze(0)
        return(state)     

    def get_pos(self,state):
        return state[:,:2]
    
    def get_psi(self,state):
        return state[:,[2]]

    def get_v(self,state):
        return state[:,[3]]

    def get_omega(self,state):
        return state[:,[4]]

    def to_str(self,state):
        return f'Pos:{vec2str(state[:2])} Psi:{state[2]} Vel:{state[3]} Omega:{state[4]}'

    def get_init_P(self,pos_cov_init=None, psi_cov_init=None, v_cov_init=None, omega_cov_init=None):
        dim = self.dim
        P = torch.eye(dim)
        if(pos_cov_init is not None):
            P[:2,:2] = pos_cov_init
        if(v_cov_init is not None):
            P[3,3]=v_cov_init
        else:
            P[3,3]=20
        if(psi_cov_init is not None):
            P[2,2]=psi_cov_init
        else:
            P[2,2]=torch.pi/2
        if(omega_cov_init is not None):
            P[4,4]=psi_cov_init
        else:
            P[4,4]=1

        return(P)

    def get_process_noise(self,dt,state):
        std_accl = self.std_accl
        std_accy = self.std_accy

        batch_dims = list(state.shape[:-1])
        L = int(torch.prod(torch.tensor(batch_dims)).item())
        state = state.reshape([L,self.dim])

        std_vec  = torch.stack([std_accl, std_accy])
        Sigma = torch.diag(std_vec**2)

        G = torch.FloatTensor([[0,0,0,1,0],[0,0,0,0,1]]).T.cuda()
        Q2 = torch.tile((G@Sigma@G.T).reshape(1,5,5),(L,1,1))

        vk   = self.get_v(state)
        psik = self.get_psi(state)

        zeros = torch.zeros(L,1).cuda()
        ones  = torch.ones(L,1).cuda()

        #Non-vectorized
        #gradA = torch.stack([[0, 0, -vk*torch.sin(psik), torch.cos(psik), 0], 
        #                     [0, 0, vk*torch.cos(psik), torch.sin(psik), 0], 
        #                     [0, 0, 0, 0, 1], 
        #                     [0, 0, 0, 0, 0], 
        #                     [0, 0, 0, 0, 0]])
        
        #Vectorized
        gradA = torch.concat(
                  [torch.hstack([zeros, zeros, -vk*torch.sin(psik), torch.cos(psik), zeros]).reshape(L,1,5),
                   torch.hstack([zeros, zeros,  vk*torch.cos(psik), torch.sin(psik), zeros]).reshape(L,1,5),
                   torch.hstack([torch.zeros(L,4).cuda(),ones]).reshape(L,1,5),
                   torch.zeros(L,2,5).cuda()],
                   dim=1
                )  
        
        #Non-vectorized      
        #gradA2 = torch.stack([[0, 0, 0, 0, -vk*torch.sin(psik)],
        #                      [0, 0, 0, 0, vk*torch.cos(psik)], 
        #                      [0, 0, 0, 0, 0], 
        #                      [0, 0, 0, 0, 0], 
        #                      [0, 0, 0, 0, 0]])

        gradA2 = torch.concat(
                   [torch.hstack([torch.zeros(L,4).cuda(),-vk*torch.sin(psik)]).reshape(L,1,5),
                    torch.hstack([torch.zeros(L,4).cuda(),vk*torch.cos(psik)]).reshape(L,1,5),
                    torch.zeros(L,3,5).cuda()],
                   dim=1
                ) 

        Q2gA  = (dt**2)/2*Q2@gradA.transpose(1,2)
        Q2gA2 = (dt**3)/6*Q2@gradA2.transpose(1,2)
        gAQ2gA2 = (dt**4)/8*gradA@Q2@gradA2.transpose(1,2)

        Q = dt*Q2 + (dt**3/3)*gradA@Q2@gradA.transpose(1,2) +  (dt**5)/20* gradA2@Q2@gradA2.transpose(1,2) \
           + Q2gA + Q2gA.transpose(1,2) + Q2gA2 + Q2gA2.transpose(1,2) + gAQ2gA2 + gAQ2gA2.transpose(1,2)

        if(torch.any(torch.isnan(Q))):
            raise ValueError("Q has nans")

        return Q

    def get_observation_function(self,L=1):
        H = torch.FloatTensor([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0]])
        H = torch.tile(H,[L,1])
        return lambda Z: (H@Z.T).T

    def normalize(self,state):
        #Normalize the state. Ensures that psi is in [0,2pi]
        state[:,2] = state[:,2]%(2*torch.pi)
        return state

    def get_forward_dynamics_function(self,dt, acc_linear=0, acc_angular=0,normalize=False):
        return lambda state: self.advance_state_value(dt,state,acc_linear=acc_linear, acc_angular=acc_angular,normalize=normalize)



    def advance_state_value(self,dt,state,acc_linear=0, acc_angular=0,normalize=False):

            batch_dims = list(state.shape[:-1])
            L = int(torch.prod(torch.tensor(batch_dims)).item())
            state = state.reshape([L,self.dim])

            Px       = state[:,[0]]
            Py       = state[:,[1]]

            psi      = state[:,[2]]
            v        = state[:,[3]]
            omega    = state[:,[4]]


            Pxnew    = Px + dt*v*torch.cos(psi + dt*omega/2)*torch.sinc(dt*omega/2)
            Pynew    = Py + dt*v*torch.sin(psi + dt*omega/2)*torch.sinc(dt*omega/2)
            psinew   = psi + dt*omega
            vnew     = v + dt*acc_linear
            omeganew = omega + dt*acc_angular

            new_state = torch.hstack([Pxnew, Pynew, psinew, vnew, omeganew])

            return new_state.reshape(batch_dims+[self.dim])
    
    def weighted_mean(self, weights, x, dim):
        wmean    = torch.sum(weights*x,axis=dim,keepdim=True)
        if(self.use_circular):
            mean_x   = torch.sum(weights*torch.cos(x[...,[2]]),axis=dim,keepdim=True)
            mean_y   = torch.sum(weights*torch.sin(x[...,[2]]),axis=dim,keepdim=True)
            mean_psi = torch.atan2(mean_y, mean_x)
            wmean = torch.cat([wmean[...,:2], mean_psi, wmean[...,3:5]],dim=-1)

        return wmean 

    def circular_normlaize(self,x):
        return (x+torch.pi)%(2*torch.pi)-torch.pi

    def diff(self,a,b):
        diff=a-b
        if(self.use_circular):
            diff_psi = self.circular_normlaize(a[...,[2]]-b[...,[2]])
            diff = torch.cat([diff[...,:2], diff_psi, diff[...,3:5]],dim=-1)
        return diff

    def weighted_cov(self, weights, samples, mean,dim):
        K=mean.shape[1]
        diff = self.diff(samples,mean)
        return (weights*diff).permute(1,2,0)@diff.permute(1,0,2)

    def advance_state(self,dt,Z,P,acc_linear=0, acc_angular=0,):

        K,DZ = Z.shape

        Q  = self.get_process_noise(dt,Z)
        f  = self.get_forward_dynamics_function(dt, acc_linear=acc_linear, acc_angular=acc_angular,normalize=False)

        if(self.use_us):
            us = st.unscented_transform(f,DZ,alpha=0.01,beta=2,diag_cov=False,
                                        lmbda=1e-6,diff_func=self.diff, 
                                        wmean_func=self.weighted_mean,mode="basic")
            Znew, Pnew0 =  us.transform(Z,P)
            Pnew        = Pnew0 + Q
            Znew=Znew.reshape(K,self.dim)
        else:
            factor = torch.linalg.cholesky(P)
            samps  = self.sigma_points.permute(1,0).reshape(1,self.dim,-1).cuda()
            samps  = Z.reshape(1,K,5) + (factor @ samps).permute(2,0,1)

            transformed_samps = f(samps)

            Znew = self.weighted_mean(self.sigma_weights.reshape(self.S,1,1).cuda(), transformed_samps, 0)
            Pnew =  self.weighted_cov(self.sigma_weights.reshape(self.S,1,1).cuda(), transformed_samps, Znew,0)+ Q
            Znew = Znew.reshape(K,self.dim)

        Pnews=[]
        for k in range(K):
            Pnews.append(self.pd_project(Pnew[k,:,:],clip_psi_var=False,verbose=True))
        Pnew  = torch.stack(Pnews,dim=0)

        if self.verbose: print("stat:",[f"{x:.5f}" for x in Znew.detach().numpy().squeeze()],"\n")

        return Znew,Pnew

    def pd_project(self,X,lam=0,clip_psi_var=False, verbose=False):
        X      = torch.tril(X) + torch.tril(X,-1).T #Force symmetry 
        u,s,v  = torch.linalg.svd(X+lam*torch.eye(X.shape[0]).cuda()) #Decompose
        X      = (u@torch.diag(s)@u.T + v.T@torch.diag(s)@v)/2.0 #Reconstruct
        X      = torch.tril(X) + torch.tril(X,-1).T #Force symmetry
        assert torch.all(X==X.T).item(), "Error: pd projected matrix is not PSD" 

        if self.verbose:
            print("vars:",[f"{x:.5f}" for x in X.diag().detach().numpy()])

            eigs=torch.linalg.eigvals(X)
            print("eigs:",[f"{x:.5f}" for x in eigs.detach().numpy()])

        return(X)

class constant_discrete_model():
    def __init__(self):
        pass

    def advance(self,dt,P):
        return P
    
class multinomial_updater():
    def __init__(self,omega):
        self.omega = omega
        self.C = omega.shape[0]

    def update(self,DP,t,logP):

        N       = DP.shape[0]
        logPnew = []

        for c in range(self.C):
            dir = torch.distributions.dirichlet.Dirichlet( self.omega[c,:].reshape(1,self.C)/t.reshape(N,1))
            logp = torch.sum(dir.log_prob(DP)) + logP[c]
            logPnew.append(logp)

        logPnew = torch.hstack(logPnew)
        logPnew = logPnew-torch.logsumexp(logPnew,dim=0)

        return logPnew

class full_data_associator(nn.Module):
    def __init__(self,clutter_prob,clutter_log_density,mm,Xfield="X", Rfield="R", Rscale=1):
        super().__init__()
        self.clutter_log_density = nn.Parameter(clutter_log_density,requires_grad=False)
        self.clutter_prob_prime    = nn.Parameter(inverse_logistic(clutter_prob))
        self.mm = mm #measurement model
        self.name="FDA"
        self.Xfield=Xfield
        self.Rfield=Rfield
        self.Rscale_prime = nn.Parameter(inverse_softplus(torch.tensor(Rscale)))

    @property
    def Rscale(self):
        return torch.nn.functional.softplus(self.Rscale_prime)  

    def __repr__(self):
        return f"{self.name}: CP: {self.clutter_det_prob:0.8f} CD: {self.clutter_density:0.8f}"

    @property
    def clutter_density(self):
        return torch.exp(self.clutter_log_density)

    @property
    def clutter_det_prob(self):
        return torch.sigmoid(self.clutter_prob_prime)

    @property
    def true_det_prob(self):    
        return (1-self.clutter_det_prob)

    def gen_all_binary_vectors(self,N):
        #Returns float array of all joint binary vectors of length N
        #Output shape is (N**2,2)
        return ((torch.arange(2**N).unsqueeze(1) >> torch.arange(N-1, -1, -1)) & 1).float()

    def associate(self,dets,Z,P):

        K,DZ = Z.shape

        pAgXs=[]
        Xmu0s=[]
        XSigmas=[]
        ZXcrosss=[]
        inds=[]
        Xs=[]
        Rs=[]

        #Get 
        i = 0 
        for n, node in enumerate(dets):
            N = dets[node][self.Xfield].shape[0]
            if(N>0):
                if(self.mm.linear):
                    Xmu0,XSigma0 = self.mm.project_Z2X(Z, P, dets[node])
                else:
                    Xmu0,XSigma0,ZXcross = self.mm.project_Z2X(Z, P, dets[node])
                    ZXcrosss.append(ZXcross)

                #assert Xmu0.shape[0]==1, "Incorrect dimension"

                X = dets[node][self.Xfield]
                R = self.Rscale*dets[node][self.Rfield]

                N,DX=X.shape

                if(R.shape[0]==N):
                    #One R per data case
                    S = XSigma0.reshape(K,1,DX,DX) + R.reshape(1,N,DX,DX)
                elif(R.shape[0]==K):
                    #One R per object
                    S = XSigma0.reshape(K,1,DX,DX) + R.reshape(1,1,DX,DX)

                Sinv          = torch.linalg.inv(S)
                X_minus_Xmu0  = X.reshape(1,N,DX,1)-Xmu0.reshape(K,1,DX,1)
                X_minus_Xmu0T = X_minus_Xmu0.transpose(2,3)
                true_det_prob = (1-self.clutter_det_prob)/K*torch.ones(K,1).to(X.device)
                logpXAD       = torch.log(true_det_prob) -1/2*torch.logdet(2*torch.pi*S) -1/2*(X_minus_Xmu0T@Sinv@X_minus_Xmu0).squeeze(-1).squeeze(-1)
                logpXAC       = (torch.log(self.clutter_det_prob) + self.clutter_log_density )*torch.ones(1,N).to(X.device)  #

                ##Non-differentiable op!!
                ind_nan       = torch.isnan(logpXAD)
                logpXAD[ind_nan] = -torch.inf

                logpXA        = torch.vstack([logpXAC,logpXAD])
                pAgX          = torch.exp(logpXA - torch.logsumexp(logpXA,dim=0))
                ind           = i*torch.ones(N).int()

                pAgXs.append(pAgX)
                Xmu0s.append(Xmu0)
                XSigmas.append(XSigma0)
                inds.append(ind)
                Xs.append(X)
                Rs.append(R)

                i +=1 

        X       = torch.cat(Xs,dim=0)       #DxDX
        pAgX    = torch.cat(pAgXs,dim=1)    #(K+1)xN    -- will contain a 0 when object not visible in view matching det
        Xmu     = torch.stack(Xmu0s,dim=0)  #VxKxDX     -- will contain nans for invalid object/view combos
        XSigma  = torch.stack(XSigmas, dim=0)  #VxKxDXxDX  -- will contain nans for invalid object/view combos
        ind     = torch.cat(inds)           #N
        R       = torch.cat(Rs,dim=0)       #NxDXxDX

        if(self.mm.linear):
            ZXcross = None
        else:
            ZXcross = torch.stack(ZXcrosss,dim=0) #VxKxDZxDX  -- will contain nans for invalid object/view combos
        
        return pAgX, X, R, ind, Xmu, XSigma, ZXcross,  #Return measurement to track association, row 0 is clutter association

    def get_posterior_mixture_us(self,A,a_prob,X,R,ind,Xmu,Xcov,ZXcross,Z,P):
        #L is number of association configurations
        #N is number of observations
        #DX is dimensionality of observations
        #DZ is dimensionality of state vector
        #A is LxN - binary matrix of observation-to-track associations to consider as mixture components
        #a_prob is Nx1 - the probability that observations are associated with this track
        #X is NxD - matrix of observations
        #ind is (N,) vector of observation to view mappings
        #Xmu is VxDX matrix of predicted measurement means in X space for each view
        #Xcov is VxDXxDX matrix of predicted measurement covariances in X space for each view
        #ZXcross is VxDZxDX matrix of predicted Z-X cross covariances for each view
        #Z is 1xDZ - mean of predicted state in Z space
        #P is DZxDZ - covariance matrix of predicted state in Z space
        #K is number of objects

        _,DZ = Z.shape
        L   = A.shape[0]
        N,DX = X.shape

        ##Need to move to log space. Can underflow.
        pi = ((a_prob**A)*((1-a_prob)**(1-A))).prod(dim=1)

        #Find views and detections that are valid 
        valid_views = torch.logical_not(torch.isnan(Xmu[:,0]))
        valid_dets  = valid_views[ind]
        valid_ind   = ind[valid_dets]
        N           = len(valid_ind)

        if(N==0):
            return pi, Z, P.reshape(1,DZ,DZ)

        #Get weight of each combination of detections
        pi = ((a_prob**A)*((1-a_prob)**(1-A))).prod(dim=1)

        X       = X[valid_dets,:]
        Xcov    = Xcov[valid_ind,:,:] #NxDXxDX
        R       = R[valid_dets,:,:]  #NxDXxDX
        ZXcross = ZXcross[valid_ind,:,:]
        a_prob  = a_prob[valid_dets]
        A       = A[:, valid_dets]

        #Sk     = Xcov[0] + R[0]
        #Skinv  = torch.linalg.inv(Sk)
        #KG     = ZXcross[0] @ Skinv
        #Znew    = Z + KG@(X[0]-Xmu[0])
        #Pnew    = P-KG@Sk@KG.T

        XXCov = torch.block_diag(*(Xcov + 0*R))      #(NxDX, NxDX)
        ZXCov = torch.hstack([*ZXcross])  #(DZ,   D*DX)
        ZZCov = P

        C=torch.vstack([torch.hstack([XXCov, ZXCov.T ]), 
                        torch.hstack([ZXCov, ZZCov ])])
        Cinv        = torch.linalg.inv(C)

        RinvH       = -1*Cinv[:N*DX,N*DX:].reshape(N,DX,DZ)
        
        #HTRinvH     = RinvH.transpose(-1,-2)@R@RinvH
        HTRinvH     = RinvH.transpose(-1,-2)@Xcov@RinvH

        HTRinvsumH  = torch.sum(A.reshape(L,N,1,1) * HTRinvH.reshape(1,N,DZ,DZ), dim =1) 
        Sigma0inv   = torch.linalg.inv(P)
        Sigmas      = torch.linalg.inv(Sigma0inv.reshape(1,DZ,DZ) + HTRinvsumH )

        HTRinvX     = torch.sum(A.reshape(L,N,1) * (RinvH.transpose(-1,-2)@X.reshape(N,DX,1)).reshape(L,N,DZ), dim=1)
        
        SigmaInvmu  = (Sigma0inv@Z.T).T + HTRinvX
        mus         = (Sigmas @ SigmaInvmu.unsqueeze(-1)).squeeze(-1)

        return pi, mus, Sigmas

    def get_posterior_mixture_lin(self,A,a_prob,X,R,ind,Z,P):
        #L is number of association configurations
        #N is number of observations
        #DX is dimensionality of observations
        #DZ is dimensionality of state vector
        #A is LxN - binary matrix of observation-to-track associations to consider as mixture components
        #a_prob is Nx1 - the probability that observations are associated with this track
        #X is NxDX - matrix of observations
        #R is NxDxD - tensor of observation covariances
        #Z is 1xK - mean of predicted state
        #P is KxK - covariance matrix of predicted state 

        N,DX = X.shape
        _,DZ = Z.shape
        L = A.shape[0]

        #Get weight of each combination of detections
        a_prob = a_prob.to(X.device)
        A = A.to(X.device)
        pi = ((a_prob**A)*((1-a_prob)**(1-A))).prod(dim=1)

        Rinv        = torch.linalg.inv(R)
        Sigma0inv   = torch.linalg.inv(P)

        Rinvsum     = torch.sum(A.reshape(L,N,1,1) * Rinv.reshape(1,N,DX,DX),dim=1)
        HTRinvsumH  = self.mm.project_cov_X2Z(Rinvsum)
        Sigmas      = torch.linalg.inv(HTRinvsumH + Sigma0inv)

        RinvXsum    = torch.sum(A.reshape(L,N,1,1) * Rinv.reshape(1,N,DX,DX)@X.reshape(1,N,DX,1) ,dim=1).squeeze(-1)
        HTRinvX     = self.mm.project_state_X2Z(RinvXsum)

        SigmaInvmu  = (Sigma0inv@Z.T).reshape(1,DZ) + HTRinvX
        mus         = (Sigmas @ SigmaInvmu.unsqueeze(-1)).squeeze(-1)

        return pi, mus, Sigmas

    def compress_mixture(self,pi, mus, Sigmas):
        #mus:    KxD
        #Sigmas: KxDxD
        #pi:     Kx1
        K,D = mus.shape[:2]

        mu     = pi.reshape(1,K)@mus
        mudiff = mus - mu
        Sigma  = torch.sum(pi.reshape(K,1,1) * (Sigmas + (mudiff.reshape(K,D,1))@(mudiff.reshape(K,1,D))), dim=0,keepdim=True)

        return mu, Sigma

class updater(abc.ABC):

    def __init__(self,da):
        super().__init__()
        self.da=da

    @abc.abstractmethod
    def update(self,X,R,Z,P):
        pass

class full_updater(updater):
    def __init__(self, da):
        super().__init__(da)
        self.name="FU"

    def update(self,dets,Z,P):
  
        N = torch.sum(torch.tensor([len(dets[v]["X"]) for v in dets]))
        K,DZ = Z.shape

        if(N>0):
            PAgX, X, R, ind, Xmu, Xcov, ZXcross = self.da.associate(dets,Z,P)
            A    = self.da.gen_all_binary_vectors(N)
            Znews = []
            Pnews = []
            for k in range(K):
                if(self.da.mm.linear):
                    pi, mus, Sigmas = self.da.get_posterior_mixture_lin(A,PAgX[k+1,:],X,R,ind,Z[[k],:],P[k,:,:])
                else:
                    pi, mus, Sigmas = self.da.get_posterior_mixture_us(A,PAgX[k+1,:],X,R,ind,Xmu[[k],:],Xcov[k,:,:],ZXcross[k,:,:],Z[[k],:],P[k,:,:])
                
                Znew, Pnew = self.da.compress_mixture(pi,mus,Sigmas)

                Znews.append(Znew)
                Pnews.append(Pnew)
            
            Znew = torch.cat(Znews,dim=0)
            Pnew = torch.cat(Pnews,dim=0)

        else:
            Znew=Z
            Pnew=P

        return Znew, Pnew

class map_updater(updater):
    def __init__(self, da):
        super().__init__(da)
        self.name="MAP Updater"

    def update(self,dets,Z,P):
  
        N = torch.sum(torch.tensor([len(dets[v]["X"]) for v in dets]))
        K,DZ = Z.shape

        if(N>0):
            PAgX, X, R, ind, Xmu, Xcov, ZXcross = self.da.associate(dets,Z,P)
            mapA = torch.argmax(PAgX,dim=0,keepdim=True)-1 #Value 0 indicates clutter
            Znews = []
            Pnews = []
            for k in range(K):
                mapAk = (mapA==k).float()
                #pi, mus, Sigmas        = self.da.get_posterior_mixture(mapAk,PAgX[k+1,:],X,R,ind,Z[[k],:],P[k,:,:])
                #pi, mus, Sigmas        = self.da.get_posterior_mixture(mapAk,PAgX[k+1,:],X,R,ind,Xmu[[k],:],Xcov[k,:,:],ZXcross[k,:,:],Z[[k],:],P[k,:,:])

                if(self.da.mm.linear):
                    _, mus, Sigmas = self.da.get_posterior_mixture_lin(mapAk,PAgX[k+1,:],X,R,ind,Z[[k],:],P[k,:,:])
                else:
                    _, mus, Sigmas = self.da.get_posterior_mixture_us(mapAk,PAgX[k+1,:],X,R,ind,Xmu[[k],:],Xcov[k,:,:],ZXcross[k,:,:],Z[[k],:],P[k,:,:])

                Znews.append(mus)
                Pnews.append(Sigmas)
            
            Znew = torch.cat(Znews,dim=0)
            Pnew = torch.cat(Pnews,dim=0)

        else:
            Znew=Z
            Pnew=P

        return Znew, Pnew
    
class soft_map_updater(updater):
    def __init__(self, da,temp=1):
        super().__init__(da)
        self.name="SMU"
        self.temp=temp
  
    def update(self,dets,Z,P):
  
        K,DZ = Z.shape
        N = torch.sum(torch.tensor([len(dets[v]["X"]) for v in dets]))

        if(N>0):
            PAgX, X, R, ind, Xmu, Xcov, ZXcross = self.da.associate(dets,Z,P)
            softMAPA = torch.sigmoid(self.temp*inverse_logistic(PAgX[1:,:]))
            Znews = []
            Pnews = []
            for k in range(K):
                #pi, mus, Sigmas       = self.da.get_posterior_mixture(softMAPA[[k],:],PAgX[k+1,:],X,R,Z[[k],:],P[k,:,:])
                #pi, mus, Sigmas        = self.da.get_posterior_mixture_us(softMAPA[[k],:],PAgX[k+1,:],X,X0,S,ZXcross,Z[[k],:],P[k,:,:])
                #pi, mus, Sigmas        = self.da.get_posterior_mixture(softMAPA[[k],:],PAgX[k+1,:],X,R,ind,Z[[k],:],P[k,:,:])
                #pi, mus, Sigmas        = self.da.get_posterior_mixture(softMAPA[[k],:],PAgX[k+1,:],X,R,ind,Xmu[[k],:],Xcov[k,:,:],ZXcross[k,:,:],Z[[k],:],P[k,:,:])


                #X      #DxDX
                #pAgX   #(K+1)xN    -- will contain a 0 when object not visible in view matching det
                #Xmu    #VxKxDX     -- will contain nans for invalid object/view combos
                #XSigma #VxKxDXxDX  -- will contain nans for invalid object/view combos
                #ind    #N
                #R      #NxDXxDX
                #ZXcross #VxKxDZxDX


                if(self.da.mm.linear):
                    _, mus, Sigmas = self.da.get_posterior_mixture_lin(softMAPA[[k],:],PAgX[k+1,:],X,R,ind,Z[[k],:],P[k,:,:])
                else:
                    _, mus, Sigmas = self.da.get_posterior_mixture_us(softMAPA[[k],:],PAgX[k+1,:],X,R,ind,Xmu[:,k,:],Xcov[:,k,:,:],ZXcross[:,k,:,:],Z[[k],:],P[k,:,:])

                Znews.append(mus)
                Pnews.append(Sigmas)
            
            Znew = torch.cat(Znews,dim=0)
            Pnew = torch.cat(Pnews,dim=0)

        else:
            Znew=Z
            Pnew=P

        return Znew, Pnew

class multi_object_tracker(nn.Module):
    #A mulit-object tracker without track initialization or deletion
    #Requires initialization of track state distribution mean Z and cov P

    def __init__(self,Z,P,classes,mm, dm ,da ,um: updater, const_dt=None,scheduler=None):
        super().__init__()
 
        self.Z=Z
        self.P=P
        self.classes=classes
        self.K = self.Z.shape[0]

        self.mm=mm #measurement model
        self.dm=dm #dynamics model
        self.da=da #data associator
        self.um=um #updater
        self.scheduler = scheduler #sensor scheduler

        self.dt = const_dt

        self.name = f"UKF-{um.name}-{dm[0].name}"

    def update(self,Znew=None,Pnew=None):

        if(Znew is not None): self.Z = Znew
        if(Pnew is not None): self.P = Pnew

    def get_measurement_dist(self,Z=None,P=None):
        if(Z is None): Z = self.Z
        if(P is None): P = self.P

        mu  = self.mm.project_state_Z2X(Z) 
        cov = self.mm.project_cov_Z2X(P) 

        return mu, cov      

    def advance(self,Z,P,dt=None):
        #Z: NxK - current track state means
        #P: NxKxK - current track state covariances

        if(dt is None): dt = self.dt
        K,DZ = Z.shape

        C = torch.max(self.classes)+1
        newZs = []
        newPs = []

        Znew = torch.zeros(K,DZ).cuda()
        Pnew = torch.zeros(K,DZ,DZ).cuda()

        for c in range(C):

            Ztmp, Ptmp = self.dm[c].advance_state(dt,Z,P)
            Znew = Znew + (self.classes.reshape(K,1)==c)*Ztmp
            Pnew = Pnew + (self.classes.reshape(K,1,1)==c)*Ptmp

        return Znew, Pnew
    
        #return self.dm.advance_state(dt,Z), self.dm.advance_cov(dt,P)
    
    def forward(self,Z,P,all_dets,t=0,use_tqdm=True,use_real_time=True, verbose=True,use_sched_grad=False):

        if(use_tqdm):
            iteration_wrapper = tqdm.tqdm
        else:
            iteration_wrapper = lambda x: x

        T    = len(all_dets) 
        Zs   = []
        Ps   = []
        Ss   = []
        Ts   = []

        for i in iteration_wrapper(range(T)):

            #Get the timestamp of event
            if(use_real_time):
                node = next(iter(all_dets[i].keys()))
                new_t = all_dets[i][node]["t"]

                if(verbose): print(f"Iteration {i} time {t}")

                if(new_t<t): 
                    if(verbose): print(f"  Got stale detection: {node}, skipping")
                    continue
                else:
                    if(verbose): print(f"  Got detections: {list(all_dets[i].keys())}")
            else:
                new_t = t + self.dt
            
            #Compute elapsed time since last detection
            full_dt = new_t - t
            t  = new_t

            #Advance in steps of at most self.dt
            steps = int(torch.ceil(torch.tensor(full_dt / self.dt)).item())
            dt    = full_dt/steps
            for j in range(steps):
                Z,P     = self.advance(Z,P,dt=dt)
                if(verbose): print(f"  advancing by {dt}")

            #Update sensor selections
            with torch.set_grad_enabled(use_sched_grad): 
                active_sensors = self.scheduler.update(self,Z,P,t,self.dt)
                active_dets    = {node: all_dets[i][node] for node in active_sensors if node in all_dets[i] }
                if(verbose): print(f"  active sensors: {active_sensors}")
                if(verbose): print(f"  active dets: {list(active_dets.keys())}")

            #Update the tracker state
            if(len(active_dets)>0):
                Znew,Pnew = self.um.update(active_dets,Z,P)

            if(torch.any(torch.isnan(Znew))): raise ValueError("Znew is nan")
            if(torch.any(torch.isnan(Pnew))): raise ValueError("Pnew is nan")

            Z=Znew
            P=Pnew

            Zs.append(Z.clone())
            Ps.append(P.clone())
            Ss.append(active_sensors)
            Ts.append(new_t)

        return torch.stack(Zs,dim=0), torch.stack(Ps,dim=0), Ss, torch.tensor(Ts)
    

    def log_prob(self,truth,truth_times,Zs,Ps,Ts,verbose=False):

        obj_ids = list(truth.keys())
        N    = len(truth[obj_ids[0]]) 
        inds = torch.searchsorted(Ts,truth_times,right=True)-1
        K    = Zs.shape[1]

        M=0
        nlls=[]
        for i in range(N):

            if(inds[i]<0): continue #skip true time stamps before first detection
            if(i>=len(Zs)): i = len(Zs)-1 
            M+=1

            full_dt =  truth_times[i] - Ts[inds[i]]
            steps   = int(torch.ceil(full_dt / self.dt).item())
            dt      = full_dt/steps
            Z       = Zs[i]
            P       = Ps[i]
            P       = (P+P.transpose(1,2))/2.0

            for j in range(steps):
                Z,P  = self.advance(Z,P,dt=dt)
                P       = (P+P.transpose(1,2))/2.0
                if(verbose): print(f"  advancing by {dt}")

            track_dist = torch.distributions.multivariate_normal.MultivariateNormal(self.mm.project_state_Z2X(Z).reshape(K,1,2), self.mm.project_cov_Z2X(P).reshape(K,1,2,2),validate_args=True)

            tmp=torch.stack([truth[id][i] for id in obj_ids])

            all_nll =  -1*track_dist.log_prob(tmp.reshape(1,K,2))
            rows,cols=linear_sum_assignment(all_nll.detach().numpy())
            nll = all_nll[rows,cols].mean()

            assert not torch.any(torch.isinf(nll)),"nll is inf"
            assert not torch.any(torch.isnan(nll)),"nll is nan"
            assert torch.all(torch.isreal(nll)),"nll is nan"

            assert not torch.any(torch.isinf(Z)),"Z is inf"
            assert not torch.any(torch.isnan(Z)),"Z is nan"
            assert torch.all(torch.isreal(Z)),"Z is nan"

            assert not torch.any(torch.isinf(P)),"P is inf"
            assert not torch.any(torch.isnan(P)),"P is nan"
            assert torch.all(torch.isreal(P)),"P is nan"

            nlls.append(nll)

        nll = torch.stack(nlls)
        assert(nll.shape[0]==M and len(nll.shape)==1)

        return nll.mean()
