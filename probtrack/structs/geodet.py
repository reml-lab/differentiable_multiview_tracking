import torch
import torch.distributions as D

def prune_pixels(pixels, return_mask=False, normalize=True):
    if normalize:
        pixels = pixels / torch.tensor([1920, 1080, 1], device=pixels.device)
    mask = pixels[..., 0] > 0
    mask = mask & (pixels[..., 0] < 1)
    mask = mask & (pixels[..., 1] > 0)
    mask = mask & (pixels[..., 1] < 1)
    mask = mask & (pixels[..., 2] > 0)
    if return_mask:
        return mask
    return pixels[mask]


class GeospatialDetections(torch.nn.Module):
    def __init__(self, bboxes, proj):
        super().__init__()
        self.bboxes = bboxes
        self.proj = proj

    def __len__(self):
        return len(self.bboxes)

    @property
    def dists(self):
        scaled_bboxes = self.bboxes.as_scaled()
        bbox_bottoms = scaled_bboxes.bottoms
        z = torch.zeros(len(bbox_bottoms), 1, device=bbox_bottoms.device)
        means, covs = self.proj.image_to_world_ground_uq(
            bbox_bottoms,
            z=z
        )
        covs = (covs + covs.transpose(-1, -2)) / 2

        dists = []
        for i in range(len(covs)):
            mixture_weights = torch.ones(1, device=means.device)
            dist = D.MixtureSameFamily(
                D.Categorical(probs=mixture_weights),
                D.MultivariateNormal(means[i].unsqueeze(0), covs[i].unsqueeze(0))
            )
            dists.append(dist)
        return dists

    @property
    def dist_from_node(self, world_points):
        node_pos = self.proj.get_parameter_vector()[0:3].squeeze() 
        dists = self.dists
        means = torch.stack([dist.components_distribution.mean for dist in dists], dim=0)
        rmse = torch.sqrt(((means - node_pos) ** 2).sum(dim=-1))
        import ipdb; ipdb.set_trace() # noqa
        return rmse

    def to(self, device):
        self.bboxes = self.bboxes.to(device)
        self.proj = self.proj.to(device)
        return self

    def is_viewable(self, world_points):
        proj_pixels = self.world_to_image(world_points, prune=False)
        mask = prune_pixels(proj_pixels, return_mask=True)
        return mask

    def world_to_image(self, world_points, prune=False, normalize=False):
        proj_pixels = [self.proj.world_to_image(obj_pos.unsqueeze(0)) for obj_pos in world_points]
        proj_pixels = torch.cat(proj_pixels, dim=0)
        if prune:
            mask = prune_pixels(proj_pixels, return_mask=True)
            proj_pixels = proj_pixels[mask]
        if normalize:
            proj_pixels = proj_pixels / torch.tensor([1920, 1080, 1], device=proj_pixels.device)
        return proj_pixels
