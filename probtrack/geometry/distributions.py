import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

def to_torch_dist(means, covs, device='cuda'):
    num_dets = len(means)
    normals = []
    for d in range(num_dets):
        mean = means[d][0:2]
        cov = covs[d][0:2, 0:2]
        eigenvals, eigenvecs = torch.linalg.eigh(cov)
        eigenvals = torch.clamp(eigenvals, min=1e-6)
        diag = torch.diag(torch.sqrt(eigenvals))
        scale_tril = eigenvecs @ diag @ eigenvecs.T
        scale_tril = torch.tril(scale_tril)
        normal = D.MultivariateNormal(
            mean.unsqueeze(0).to(device),
            scale_tril=scale_tril.unsqueeze(0).to(device)
        )
        normals.append(normal)
    return normals

def rotate_dist1d(dist, rot):
    mean = dist.mean @ rot
    cov = rot @ dist.covariance_matrix @ rot.T
    try:
        normal = D.MultivariateNormal(mean, cov)
    except ValueError: #cov is not symmetric?
        cov = (cov + cov.transpose(1, 2)) / 2.0
        normal = D.MultivariateNormal(mean, cov)
    return normal

def scale_dist1d(dist, scale):
    scale_matrix = torch.diag_embed(scale)
    new_dist = rotate_dist1d(dist, scale_matrix)
    return new_dist

def shift_dist1d(dist, shift):
    mean = dist.mean + shift
    normal = D.MultivariateNormal(mean, dist.covariance_matrix)
    return normal

def rotate_dist(dist, rot):
    is_mixture = isinstance(dist, D.MixtureSameFamily)
    if is_mixture:
        comp_dist = dist.component_distribution
    else:
        comp_dist = dist
    mean = comp_dist.mean @ rot
    cov = rot @ comp_dist.covariance_matrix @ rot.T
    try:
        normal = D.MultivariateNormal(mean, cov)
    except ValueError: #cov is not symmetric?
        cov = (cov + cov.transpose(1, 2)) / 2.0
        normal = D.MultivariateNormal(mean, cov)
    if is_mixture:
        new_dist = D.MixtureSameFamily(dist.mixture_distribution, normal)
    else:
        new_dist = normal
    return new_dist

def scale_dist(dist, scale):
    scale_matrix = torch.diag_embed(scale)
    new_dist = rotate_dist(dist, scale_matrix)
    return new_dist

def shift_dist(dist, shift):
    is_mixture = isinstance(dist, D.MixtureSameFamily)
    if is_mixture:
        comp_dist = dist.component_distribution
    else:
        comp_dist = dist
    mean = comp_dist.mean + shift
    normal = D.MultivariateNormal(mean, comp_dist.covariance_matrix)
    if is_mixture:
        new_dist = D.MixtureSameFamily(dist.mixture_distribution, normal)
    else:
        new_dist = normal
    return new_dist

def reduce_dim(dist):
    is_mixture = isinstance(dist, D.MixtureSameFamily)
    if is_mixture:
        comp_dist = dist.component_distribution
    else:
        comp_dist = dist
    normal = D.MultivariateNormal(
        comp_dist.mean[:, 0:2], 
        comp_dist.covariance_matrix[:, 0:2, 0:2]
    )
    if is_mixture:
        new_dist = D.MixtureSameFamily(dist.mixture_distribution, normal)
    else:
        new_dist = normal
    return new_dist

def break_mixture(mix_dist):
    normals = mix_dist.component_distribution
    num_dists = len(normals.mean)
    output = []
    for i in range(num_dists):
        dist = D.MultivariateNormal(normals.mean[i], normals.covariance_matrix[i])
        output.append(dist)
    return output

def threshold_dist(dist, threshold):
    comp_dist = dist.component_distribution
    probs = dist.mixture_distribution.probs
    selected_idx = torch.where(probs >= threshold)

    selected_dists = []
    for idx in selected_idx:
        mean = comp_dist.mean[idx]
        cov = comp_dist.covariance_matrix[idx]
        normal = D.MultivariateNormal(mean, cov)
        probs = cov.new_ones(mean.shape[0])
        mix = D.Categorical(probs=probs)
        dist = D.MixtureSameFamily(mix, normal)
        selected_dists.append(dist)
    return selected_dists

def dist_to_np(dist):
    if isinstance(dist, D.MixtureSameFamily):
        mean = dist.component_distribution.mean.detach().cpu().numpy()
        cov = dist.component_distribution.covariance_matrix.detach().cpu().numpy()
        probs = dist.mixture_distribution.probs.detach().cpu().numpy()
    else:
        mean = dist.mean.detach().cpu().numpy()
        cov = dist.covariance_matrix.detach().cpu().numpy()
        probs = torch.ones(len(mean)).detach().cpu().numpy()
    return mean, cov, probs

def transform_dist_for_viz(dist, min_vals, max_vals, img_scale):
    dist = reduce_dim(dist)
    dist = shift_dist(dist, -min_vals)
    scale = 1 / (max_vals - min_vals)
    dist = scale_dist(dist, scale)
    dist = scale_dist(dist, img_scale)
    return dist
