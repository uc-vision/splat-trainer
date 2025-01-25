from typing import Optional
from taichi_splatting import Rendering
import torch
from splat_trainer.gaussians.split import clone_points
from splat_trainer.logger.logger import Logger
from splat_trainer.scene.scene import GaussianScene
from splat_trainer.util.misc import exp_lerp, lerp, pow_lerp

from tensordict import tensorclass

@tensorclass
class PointState:
    prune_cost:torch.Tensor
    split_score:torch.Tensor
    max_scale_px:torch.Tensor        # (N,) - maximum scale seen for each near point (pixels)

    points_in_view:torch.Tensor   # (N,) - number of times each point was in view\
    visibility:torch.Tensor       # (N,) - sum visibility (weights) for each point


    @staticmethod
    def new_zeros(num_points:int, device:torch.device) -> 'PointState':
        return PointState(
            prune_cost=torch.zeros(num_points, device=device),
            split_score=torch.zeros(num_points, device=device),
            max_scale_px=torch.zeros(num_points, device=device),

            points_in_view=torch.zeros(num_points, dtype=torch.int16, device=device),
            visibility=torch.zeros(num_points, device=device),

            batch_size=(num_points,)
        )



    def add_rendering(self, rendering:Rendering, far_distance:float = 0.75, 
                      split_alpha:float = 0.01, prune_alpha:float = 0.1):
        points = rendering.points
        far_points = points.depths.squeeze(1) > points.depths.quantile(far_distance)

        # measure scale of far points in image
        image_scale_px = points.screen_scale.max(1).values
        image_scale_px[far_points] = 0.

        self.max_scale_px[points.idx] = torch.maximum(self.max_scale_px[points.idx], image_scale_px)
        self.points_in_view[points.idx] += 1

        self.visibility[points.idx] += points.visibility

        # self.split_score[points.idx] = pow_lerp(split_alpha, self.split_score[points.idx], points.split_score, k=6)
        # self.prune_cost[points.idx] = lerp(prune_alpha, self.prune_cost[points.idx], points.prune_cost)
        

        # self.split_score[points.idx] = max_decaying(split_alpha, self.split_score[points.idx], points.split_score)
        self.split_score[points.idx] = exp_lerp(split_alpha, self.split_score[points.idx], points.split_score)
        self.prune_cost[points.idx] = exp_lerp(prune_alpha, self.prune_cost[points.idx], points.prune_cost)

    def masked_heuristics(self, min_views:int):
        enough_views = self.points_in_view >= min_views
        return torch.where(enough_views, self.prune_cost, torch.inf), self.split_score #torch.where(enough_views, self.split_score, 0.)


    


def log_histograms(points: PointState, logger: Logger, name: str = "densify"):

    def log_scale_histogram(k: str, t: torch.Tensor, min_val: float = 1e-12):
        logger.log_histogram(f"{name}/{k}",
                             torch.log10(torch.clamp_min(t, min_val)))

    log_scale_histogram("prune_cost", points.prune_cost)
    log_scale_histogram("split_score", points.split_score)
    log_scale_histogram("max_scale_px", points.max_scale_px, min_val=1e-6)

    logger.log_histogram(f"{name}/points_in_view", points.points_in_view)
    logger.log_histogram(f"{name}/visibility", points.visibility)



def densify_and_prune(points:PointState, scene:GaussianScene, split_mask:torch.Tensor, prune_mask:torch.Tensor,
                      logger:Optional[Logger] = None):


    split_idx = split_mask.nonzero().squeeze(1)
    device = split_mask.device

    n_prune = prune_mask.sum().item()
    n_split = split_idx.shape[0]

    n = points.batch_size[0]
    n_unseen = n - torch.count_nonzero(points.prune_cost).item()

    prune_thresh = points.prune_cost[prune_mask].max().item() if n_prune > 0 else 0.
    split_thresh = points.split_score[split_idx].min().item() if n_split > 0 else 0.

    keep_mask = ~(split_mask | prune_mask)

    scene.split_and_prune(keep_mask, split_idx)
    metrics = dict(n=points.batch_size[0],
        prune=n_prune,
        split=n_split,
        max_prune_score=prune_thresh,
        min_split_score=split_thresh,
        unseen = n_unseen)

    if logger is not None:
        logger.log_values("densify", metrics)
        log_histograms(points, logger, "densify")

    new_points = PointState.new_zeros(n_split * 2, device=device)

    return torch.cat([points[keep_mask], new_points])
