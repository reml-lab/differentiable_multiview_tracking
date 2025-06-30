import torch

class proj_sensor_model:
    def __init__(self, dummy_cov_scale=1, proj=None):
        self.dummy_cov_scale=dummy_cov_scale
        self.proj = proj

    def predict(self,Xs):
        N          = Xs.shape[0]

        if(Xs.shape[1]==2):
            Xs = torch.hstack([Xs,torch.zeros(N,1).cuda()])

        means = torch.zeros(N,2).cuda()
        covs  = torch.zeros(N,2,2).cuda()

        cam        = self.proj
        points_img = cam.world_to_image(Xs)
        in_frame   = (points_img[:,0]>0) * (points_img[:,0]<cam.image_size[0]) * (points_img[:,1]>0) * (points_img[:,1]<cam.image_size[1]) * (points_img[:,2]>0)

        if(any(in_frame)):
            Nvalid = in_frame.sum()
            m3, c3 = cam.image_to_world_ground_uq(points_img[in_frame,:2],torch.zeros(Nvalid,1).cuda() )  
            means[in_frame,:] = m3[:,:2]
            covs[in_frame,:] = c3[:,:2,:][:,:,:2]

        if(any(~in_frame)):
            Ninvalid = N - in_frame.sum()
            means[~in_frame,:] = Xs[~in_frame,:2]
            covs[~in_frame,:,:] = torch.diag_embed(self.dummy_cov_scale*torch.ones(Ninvalid,2).cuda())

        return covs
    
class Scheduler():
    def __init__(self,sensor_models, max_sensors=2, max_uncertainty=torch.inf,update_interval=0.1, verbose=True, mode="fixed", obj=None):

        self.max_sensors     = max_sensors
        self.sensor_models   = sensor_models
        self.mode            = mode
        self.max_uncertainty = max_uncertainty
        self.update_interval = update_interval
        self.verbose         = verbose
        self.obj             = obj
        self.valid_configs   = None

        if(mode=="fixed"):
            self.max_sensors = len(sensor_models.keys())
            self.name = f"Fixed Scheduler (num={self.max_sensors})"
        elif(mode=="random"):
            self.name = f"Random Scheduler (num={self.max_sensors})"
        elif("optimize" in mode):

            self.constraints   = [lambda a: torch.sum(a)<=self.max_sensors]

            if(mode=="optimize_exhaustive"):
                self.name = f"Exhaustive Optimizing Scheduler (num={self.max_sensors}, obj={self.obj})"
            elif(mode=="optimize_greedy"):
                self.name = f"Greedy Optimizing Scheduler (num={self.max_sensors}, obj={self.obj}, maxu={self.max_uncertainty})"
            else:
                raise ValueError(f"Scheduler mode '{mode}' is unknown")

            if(obj=="entropy"):
                self.objective  = self.entropy_objective
            elif(obj=="eig"):
                self.objective  = self.eig_objective
            elif(obj is None):
                pass
            else:
                raise ValueError(f"Scheduler objective '{obj}' is unknown")   
        else:
            raise ValueError(f"Scheduler mode '{mode}' is unknown")

        self.sensor_ids     = list(sensor_models.keys())
        self.num_sensors    = len(self.sensor_ids)
        self.a              = torch.ones((self.num_sensors))
        self.active_sensors = [self.sensor_ids[i] for i in range(self.num_sensors) if self.a[i]==1]
        self.last_update    = None

    def entropy_objective(self,P,tracker):
        Pobs    = tracker.mm.project_cov_Z2X(P)
        obj     = torch.logdet(Pobs).max() #maximum predicted entropy of any object
        return(obj)

    def eig_objective(self,P,tracker):
        Pobs    = tracker.mm.project_cov_Z2X(P)
        obj     = 5.991*torch.linalg.eigvals(Pobs).real.max() #maximum 95% uncertainty ellipse axis length of any object
        return(obj)

    def optimize_exhaustive(self,tracker, Zpred, Ppred, Xpred, Rpred):

        #Run single threaded
        min_val = torch.inf
        for a in self.valid_configs:
                    
            dets    = {self.sensor_ids[i]:{"X":Xpred,"R":Rpred[self.sensor_ids[i]]} for i in range(self.num_sensors) if a[i]==1}
            _,Pnew  = tracker.um.update(dets,Zpred,Ppred)

            val = self.objective(Pnew, tracker)

            if(val<min_val):
                min_val=val
                best_a = a

        return(best_a,min_val)

    def optimize_greedy(self,tracker, Zpred, Ppred, Xpred, Rpred):

        #Start with no sensors selected
        selected=[]
        not_selected = list(range(self.num_sensors))

        #Run max_senors rounds of greedy optimization
        for s in range(self.max_sensors):

            min_val = torch.inf
            for s in not_selected:

                #Build selection vector
                a           = torch.zeros(self.num_sensors)
                a[selected] = 1
                a[s]        = 1
                        
                #Compute objective
                dets    = {self.sensor_ids[i]:{"X":Xpred,"R":Rpred[self.sensor_ids[i]]} for i in range(self.num_sensors) if a[i]==1}
                _,Pnew  = tracker.um.update(dets,Zpred,Ppred)
                val     = self.objective(Pnew, tracker)

                #Check optimality
                if(val<min_val):
                    min_val=val
                    best_s = s
            
            #Add best sensor found to selected set
            selected.append(best_s)
            not_selected.remove(best_s)

            #If objective value is lower than maximum uncertainty,
            #stop greedy search early  
            if(min_val<self.max_uncertainty): break

        #Return best sensor subset found
        best_a           = torch.zeros(self.num_sensors)
        best_a[selected] = 1

        return(best_a,min_val)


    def get_valid_configs(self):
        configs=[]
        for i in range(1,2**self.num_sensors):
            mask = 2**torch.arange(self.num_sensors)
            a    = torch.tensor(i).unsqueeze(-1).bitwise_and(mask).ne(0).byte()
            #a = torch.tensor([int(x) for x in np.binary_repr(i,self.S)])
            valid=True
            for c in self.constraints:
                if(not c(a)):
                    valid=False
                    break
            if valid: configs.append(a)
        return configs

    def update(self,tracker,Z,P,t,dt):

        if(self.last_update is None):
            if self.verbose: print(f"  Initial sensor selection (t = t)")
        else:
            if(t-self.last_update<self.update_interval):
                return self.active_sensors
            else: 
                if self.verbose: print(f"  Updating sensor selection (delta_t = {t-self.last_update})")

        self.last_update = t

        if self.mode=="fixed":
            return self.active_sensors
        
        elif self.mode =="random":
            inds = torch.multinomial(torch.FloatTensor(range(self.num_sensors)), self.max_sensors, replacement=False)
            self.a = torch.zeros(self.num_sensors)
            self.a[inds] = 1
            self.active_sensors = [self.sensor_ids[i] for i in range(self.num_sensors) if self.a[i]==1]
            return(self.active_sensors)

        elif("optimize" in self.mode):

            #Advance tracker state for all objects
            Zpred, Ppred = tracker.advance(Z,P,dt)
            obj0 = self.objective(Ppred, tracker)

            #Get predicted observed positions for all objects
            Xpred = tracker.mm.project_state_Z2X(Zpred)

            #Get predicted measurement covariances for all objects
            Rpred = {}
            for sensor in self.sensor_models:
                Rpred[sensor] = self.sensor_models[sensor].predict(Xpred.detach())

            #Choose optimization_mode
            if(self.mode=="optimize_exhaustive"):
                #Exhaustive optimization of sensor selections
                if(self.valid_configs is None):
                    self.valid_configs = self.get_valid_configs()
                self.a, obj_new = self.optimize_exhaustive(tracker, Zpred, Ppred, Xpred, Rpred)

            elif(self.mode=="optimize_greedy"):
                #Greedy optimization of sensor selections
                self.a, obj_new = self.optimize_greedy(tracker, Zpred, Ppred, Xpred, Rpred)

            else:
                raise ValueError("Scehuler mode {self.mode} is unknown")

            if(self.verbose): print(f"  obj0: {obj0} obj_new: {obj_new}")

            self.active_sensors = [self.sensor_ids[i] for i in range(self.num_sensors) if self.a[i]==1]

            return self.active_sensors

        else:
            raise ValueError("Scehuler mode {self.mode} is unknown")

    
