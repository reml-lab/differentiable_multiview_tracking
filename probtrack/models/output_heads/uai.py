import torch
import torch.nn as nn
from mmengine.registry import MODELS
import torch.nn.functional as F
from probtrack.geometry.distributions import to_torch_dist
from probtrack.models.output_heads.proj import compute_gt_bboxes

@MODELS.register_module()
class UAIOutputHead(nn.Module):
    def __init__(self, 
            projs=None,
            adapter_cfg=dict(type='LinearAdapter'),
            predictor_cfg=dict(type='GaussianPredictor', freeze_non_cov=False),
            freeze_adapter=False,
            shared_adapter=False,
            predict_in_local=False,
        ):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.shared_adapter = shared_adapter
        self.predict_in_local = predict_in_local
        self.build_adapter(projs, adapter_cfg)
        for param in self.adapter.parameters():
            param.requires_grad = not freeze_adapter
        self.predictor = MODELS.build(predictor_cfg)

    def build_adapter(self, projs, adapter_cfg):
        if self.shared_adapter:
            self.adapter = MODELS.build(adapter_cfg)
        else:
            self.adapter = nn.ModuleDict()
            for scenario_name in projs.keys():
                self.adapter[scenario_name] = nn.ModuleDict()
                for node_name, proj in projs[scenario_name].items():
                    self.adapter[scenario_name][node_name] = MODELS.build(adapter_cfg)

    def forward_adapter(self, embeds, proj):
        if self.shared_adapter:
            adapter = self.adapter
        else:
            adapter = self.adapter[proj.scenario][proj.node_str]
        embeds = adapter(embeds)
        return embeds[:, 0]

    def forward_train_embeds(self, embeds, proj, gt):
        embeds = self.forward_adapter(embeds, proj)
        det_means, det_covs = self.predictor(embeds)
        if self.predict_in_local:
            det_means = proj.local_to_world(det_means)
            det_covs = proj.local_to_world_cov(det_covs)
            det_means = det_means[..., 0:2]
            det_covs = det_covs[..., 0:2, 0:2]
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        gt_pos = gt['obj_position'].cuda()
        det_nll_loss = torch.zeros(1).cuda()
        det_mse_loss = torch.zeros(1).cuda()
        for j in range(len(det_normals)):
            nll_loss_j = -det_normals[j].log_prob(gt_pos[j]).mean()
            det_nll_loss += nll_loss_j
            mse_loss_j = F.mse_loss(det_means[j], gt_pos[j])
            det_mse_loss += mse_loss_j
        det_nll_loss /= len(gt_pos)
        det_mse_loss /= len(gt_pos)
        return {'det_nll_loss': det_nll_loss.mean(), 'det_mse_loss': det_mse_loss.mean()}

    def forward_train(self, dets, proj, gt):
        num_objs = len(gt['obj_position'])
        embeds = self.forward_adapter(dets.embeds, proj)
        det_means, det_covs = self.predictor(embeds)
        if self.predict_in_local:
            det_means = proj.local_to_world(det_means)
            det_covs = proj.local_to_world_cov(det_covs)
            det_means = det_means[..., 0:2]
            det_covs = det_covs[..., 0:2, 0:2]
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        assert len(det_normals) == num_objs
        
        det_nll_loss = torch.zeros(1).cuda()
        for j in range(num_objs):
            gt_pos = gt['obj_position'][j][0:2].cuda()
            nll_loss_j = -det_normals[j].log_prob(gt_pos).mean()
            det_nll_loss += nll_loss_j
        if num_objs > 0:
            det_nll_loss /= num_objs
        loss_dict = {
            'giou_loss': torch.zeros(1).cuda().mean() * self.dummy_param.mean(),
            'l1_loss': torch.zeros(1).cuda().mean() * self.dummy_param.mean(),
            'ce_loss': torch.zeros(1).cuda().mean() * self.dummy_param.mean(),
            'det_nll_loss': det_nll_loss
        }
        return loss_dict, det_normals, dets

    def forward_test(self, dets, proj):
        embeds = self.forward_adapter(dets.embeds, proj)
        det_means, det_covs = self.predictor(embeds)
        if self.predict_in_local:
            det_means = proj.local_to_world(det_means)
            det_covs = proj.local_to_world_cov(det_covs)
            det_means = det_means[..., 0:2]
            det_covs = det_covs[..., 0:2, 0:2]
        det_normals = to_torch_dist(det_means, det_covs, device='cuda')
        return det_normals, dets


@MODELS.register_module()
class FlexLocOutputHead(UAIOutputHead):
    def __init__(self, projs, 
            adapter_cfg=dict(type='NodePoseAdapter'),
            predictor_cfg=dict(type='GaussianPredictor', freeze_non_cov=False),
            freeze_adapter=False,
        ):
        super().__init__(projs, adapter_cfg, predictor_cfg, freeze_adapter)

    def build_adapter(self, projs, adapter_cfg, *args, **kwargs):
        self.adapter = MODELS.build(adapter_cfg)

    def forward_adapter(self, embeds, proj):
        params = proj.get_parameter_vector().T #1 x 9
        embeds = self.adapter(embeds, params)
        return embeds[:, 0]
