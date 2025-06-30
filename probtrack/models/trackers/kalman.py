import torch
import torch.nn as nn
from datetime import datetime
import probtrack
import argparse
import numpy as np
import cv2
from tqdm import tqdm, trange
import probtrack.viz.utils as vutils 
from mmengine.config import Config 
from mmengine.registry import OPTIMIZERS, DATASETS, MODELS, PARAM_SCHEDULERS
import probtrack.datasets.utils as dutils
from probtrack.geometry.distributions import reduce_dim, threshold_dist, scale_dist, dist_to_np
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.detection import Detection, GaussianDetection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis, Euclidean
from stonesoup.dataassociator.neighbour import GlobalNearestNeighbour
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.types.track import Track
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
import torch.distributions as D

def state2dist(state):
    mean = torch.tensor(state.state_vector[[0, 2]]).squeeze(-1).cuda().float()
    cov = torch.tensor(state.covar[[0, 2]][:, [0, 2]]).cuda().float()
    normals = D.MultivariateNormal(mean.unsqueeze(0), cov.unsqueeze(0))
    mix = D.Categorical(probs=torch.tensor([1.]).cuda())
    return D.MixtureSameFamily(mix, normals)

class KalmanFilter(nn.Module):
    def __init__(self, dt=1):
        super().__init__()
        self.dt = dt
        
        self.F = torch.tensor([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]).float()

        self.H = torch.tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]).float()

        q = torch.tensor([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]).float()
        self.Q = q * 0.01

        self.R = torch.eye(2) * 0.01
        self.P = torch.eye(4) * 1
        self.x = torch.zeros(4, 1)

    def predict(self):
        self.x = torch.matmul(self.F, self.x)
        self.P = torch.matmul(torch.matmul(self.F, self.P), self.F.t()) + self.Q
        return self.x

    def update(self, z, R=None):
        if R is not None:
            self.R = R

        S = torch.mm(self.H.mm(self.P), self.H.t()) + self.R
        K = torch.mm(self.P.mm(self.H.t()), torch.inverse(S))

        y = z - torch.mm(self.H, self.x)
        self.x = self.x + torch.mm(K, y)
        I = torch.eye(4)
        self.P = (I - torch.mm(K, self.H)).mm(self.P)



class KalmanTracker:
    def __init__(self, timestamp):
        self.start_timestamp = timestamp
        #vec_noise_diff_coeff = 1e-2 / 2

        self.kf = KalmanFilter(dt=7/1000)
        vec_noise_diff_coeff = 0.1

        velocties = [ConstantVelocity(noise_diff_coeff=vec_noise_diff_coeff)] * 2
        self.transition_model = CombinedLinearGaussianTransitionModel(velocties)

        # self.measurement_model = LinearGaussian(
            # ndim_state=4, mapping=(0, 2),
            # noise_covar=np.eye(2) * 1e-2 / 2
        # )
        self.init(timestamp)

    def init(self, timestamp):
        self.start_timestamp = timestamp
        initial_state = GaussianState(
            StateVector([0.0, 0.0, 0.0, 0.0]),
            CovarianceMatrix(np.eye(4)),
            timestamp=self.start_timestamp
        )
        self.current_state = initial_state

        measurement_model = LinearGaussian(
            ndim_state=4, mapping=(0, 2),
            noise_covar=torch.eye(2)
        )

        self.predictor = KalmanPredictor(self.transition_model)
        self.updater = KalmanUpdater(measurement_model, force_symmetric_covariance=True)
        self.hypothesiser = DistanceHypothesiser(self.predictor, 
                self.updater, measure=Euclidean(),
                missed_distance=100)
        # self.associator = GlobalNearestNeighbour(self.hypothesiser)
        self.associator = GNNWith2DAssignment(self.hypothesiser)
        
        
        self.tracks = {Track([initial_state])}
        self.deleter = CovarianceBasedDeleter(covar_trace_thresh=4)
        self.initiator = MultiMeasurementInitiator(
            prior_state=GaussianState([[0], [0], [0], [0]], np.diag([0, 1, 0, 1])),
            measurement_model=measurement_model,
            deleter=self.deleter,
            data_associator=self.associator,
            updater=self.updater,
            min_points=2
        )

    def update(self, dets, timestamp):
        measurements = set()
        # self.kf.predict()
        for det in dets:
            # self.kf.update(det.mean.cpu().unsqueeze(-1), R=det.covariance_matrix.cpu())
            mean = det.mean.cpu().numpy()
            cov = det.covariance_matrix.cpu().numpy()
            measurement = Detection(
                StateVector(mean),
                measurement_model=LinearGaussian(
                    ndim_state=4, mapping=(0, 2),
                    noise_covar=cov
                ),
                timestamp=timestamp
            )
            measurements.add(measurement)
       
        hypotheses = self.associator.associate(self.tracks, measurements, timestamp)

        associated_measurements = set()
        for track in self.tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = self.updater.update(hypothesis)
                track.append(post)
                associated_measurements.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)
        
        self.tracks -= self.deleter.delete_tracks(self.tracks)
        self.tracks |= self.initiator.initiate(measurements - associated_measurements, timestamp)
        
        dists = []
        for track in self.tracks:
            dist = state2dist(track.state)
            dists.append(dist)
        # prediction = self.predictor.predict(self.current_state, timestamp=timestamp)
        # hypotheses = [SingleHypothesis(prediction, measurement) for measurement in measurements]
        # for hypothesis in hypotheses:
            # self.current_state = self.updater.update(hypothesis)
        # dist = state2dist(self.current_state)

        # dists = []
        # track_mean = self.kf.x.cuda().squeeze()[0:2].unsqueeze(0)
        # track_cov = self.kf.P.cuda().squeeze()[0:2, 0:2].unsqueeze(0)
        # weights = torch.ones(1).cuda()
        # dist = D.MixtureSameFamily(D.Categorical(probs=weights), D.MultivariateNormal(track_mean, track_cov))
        # dists.append(dist)
        return {'kf': {'dist': dists}}
