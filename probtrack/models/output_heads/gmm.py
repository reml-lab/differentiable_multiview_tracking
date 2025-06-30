import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from mmengine.registry import MODELS
from probtrack.geometry.distributions import rotate_dist, scale_dist, shift_dist, break_mixture
import lap


def linear_assignment(cost_matrix):
    cost_matrix = cost_matrix.cpu().detach().numpy()
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    assign_idx = np.array([[y[i], i] for i in x if i >= 0])
    sort_idx = np.argsort(assign_idx[:, 1])
    assign_idx = assign_idx[sort_idx]
    assign_idx = torch.from_numpy(assign_idx)
    return assign_idx.long()

@MODELS.register_module()
class CovDropout(nn.Module):
    def __init__(self, p=0.5, cov_eps=1e-8):
        super().__init__()
        self.register_buffer('drop_cov', torch.eye(3) * cov_eps)
        self.p = p

    def forward(self, cov):
        if self.training:
            B, N, D, _ = cov.shape
            cov = cov.reshape(B*N, D, D)
            flip = torch.rand(B*N).to(cov)
            mask = flip >= self.p
            cov[~mask] = 0*cov[~mask] + self.drop_cov
            cov = cov.reshape(B, N, D, D)
        return cov

@MODELS.register_module()
class GaussianMixtureOutputHead(nn.Module):
    def __init__(self,
                 dim=256,
                 mean_min=[-1, -1, -1],
                 mean_max=[1, 1, 1],
                 cov_predictor_cfg=dict(type='DiagCovPredictor'),
                 loss_weights={
                    'local_nll_loss': 1.0,
                    'global_nll_loss': 0.0,
                    'entropy_loss': 0.0,
                    'mse_loss': 1.0,
                 },
                 cov_dropout_cfg=dict(type='CovDropout', 
                     p=0.0, 
                     cov_eps=1e-4
                 ),
                 nonlinear_proj=False
                 ): 
        super().__init__()
        self.register_buffer('mean_min', torch.tensor(mean_min))
        self.register_buffer('mean_max', torch.tensor(mean_max))
        self.cov_predictor = MODELS.build(cov_predictor_cfg)
        self.loss_weights = loss_weights

        self.cov_dropout = MODELS.build(cov_dropout_cfg)

        self.mean_head = []
        if nonlinear_proj:
            self.mean_head.append(nn.Linear(dim, dim))
            self.mean_head.append(nn.GELU())
        self.mean_head.append(nn.Linear(dim, 3))
        self.mean_head.append(nn.Sigmoid())
        self.mean_head = nn.Sequential(*self.mean_head)

        self.mix_head = nn.Linear(dim, 1)
        
    def forward(self, x):
        B, N, C = x.shape
        means = self.mean_head(x)
        means = means * (self.mean_max - self.mean_min) + self.mean_min
        mix_logits = self.mix_head(x)

        cov = self.cov_predictor(x)
        cov = self.cov_dropout(cov)

        dists = []
        for i in range(B):
            normal = D.MultivariateNormal(means[i], cov[i])
            mix = D.Categorical(logits=mix_logits[i].squeeze(-1))
            dist = D.MixtureSameFamily(mix, normal)
            dists.append(dist)
        return dists

    def forward_test(self, x, gt=None):
        if len(x.shape) == 2:
            import ipdb; ipdb.set_trace() # noqa
        dists = self.forward(x)
        global_dists = []
        if len(gt['node_rot'].shape) == 2:
            gt['node_rot'] = gt['node_rot'].unsqueeze(0)
        if len(gt['node_pos'].shape) == 2:
            gt['node_pos'] = gt['node_pos'].unsqueeze(0)
            # import ipdb; ipdb.set_trace() # noqa
        for i, dist in enumerate(dists):
            rot = gt['node_rot']
            pos = gt['node_pos']
            if len(rot.shape) == 3:
                rot = rot.unsqueeze(0)
            if len(pos.shape) == 2:
                pos = pos.unsqueeze(0)
            dist = rotate_dist(dist, rot[i][0])

            dist = shift_dist(dist, pos[i][0])
            global_dists.append(dist)
        return global_dists

    def forward_train(self, x, gt):
        dists = self.forward(x)
        
        viewable = gt['viewable']
        B = len(viewable)
        try:
            num_viewable = viewable.sum(dim=1)
        except:
            num_viewable = viewable
        loss_dict = {}
        
        #compute nll loss
        log_probs = []
        for i, gt_pos in enumerate(gt['local_obj_pos']):
            log_prob = dists[i].log_prob(gt_pos)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs, dim=0) #B x N
    
        nll_vals = -log_probs
        nll_vals = nll_vals * viewable
        Z = viewable.sum()
        if Z == 0:
            loss_dict['nll_loss'] = nll_vals.sum() * 0
        else:
            loss_dict['nll_loss'] = nll_vals.sum() / Z

        #compute matched mse loss
        # mse_vals = []
        # for i, gt_pos in enumerate(gt['local_obj_pos']):
            # means = dists[i].component_distribution.mean
            # dist_matrix = torch.cdist(means, gt_pos, p=2)
            # assign_idx = linear_assignment(dist_matrix)
            # mse = 0
            # for row in assign_idx:
                # mse += dist_matrix[row[0], row[1]]
            # mse = mse / len(assign_idx)
            # mse_vals.append(mse)
        # mse_vals = torch.stack(mse_vals, dim=0) #B
        # mse_vals = mse_vals * viewable.sum(dim=-1)
        # if Z == 0:
            # loss_dict['mse_loss'] = mse_vals.sum() * 0
        # else:
            # loss_dict['mse_loss'] = mse_vals.sum() / Z
    
        entropies = [dists[i].mixture_distribution.entropy() for i in range(B)]
        entropies = torch.stack(entropies, dim=0) #B
        entropy_targets = torch.log(num_viewable)
        # entropy_targets[torch.isnan(entropy_targets)] = np.log(3)
        entropy_targets[torch.isinf(entropy_targets)] = 0 #log(0) -> -inf
        entropy_loss_vals = F.mse_loss(entropies, entropy_targets, reduction='none')
        try:
            entropy_loss_vals = entropy_loss_vals * viewable.sum(dim=-1)
        except:
            import ipdb; ipdb.set_trace()
        if Z == 0:
            loss_dict['entropy_loss'] = entropy_loss_vals.sum() * 0
        else:
            loss_dict['entropy_loss'] = entropy_loss_vals.sum() / Z
        # loss_dict['entropy_loss'] = entropy_loss.mean()

        # log_probs = [global_dists[i].log_prob(pos) for i, pos in enumerate(gt['global_obj_pos'])]
        # log_probs = torch.stack(log_probs, dim=0) #B x num_objects
        # global_nll_vals = -log_probs
        # global_nll_vals = global_nll_vals * viewable
        # loss_dict['global_nll_loss'] = global_nll_vals.sum() / viewable.sum()
        

        for k, v in loss_dict.items():
            loss_dict[k] = self.loss_weights[k] * v
        
        return loss_dict
