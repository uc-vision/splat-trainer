
from dataclasses import  dataclass
import math
from typing import Optional
import numpy as np
from taichi_splatting import Rendering
from tensordict import tensorclass
import torch

from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from .gaussians import GaussianScene



@tensorclass
class PointStatistics:
  split_heuristics : torch.Tensor  # (N, 2) - accumulated split heuristics
  visible : torch.Tensor  # (N, ) - number of times the point was rasterized



@dataclass
class EMAConfig(ControllerConfig):

  # target point cloud size - if None then optimize for the current size
  target_count:Optional[int] = None

  # base rate (relative to count) to split points every 100 iterations
  split_rate:float = 0.1

  # max ratio of points to split/prune
  max_ratio:float = 2.0

  # min number of times a point must be visible recently to be considered for splitting/pruning
  min_visibility:int = 20 
  decay_steps:float = 100.0 # half life of visibility decay in steps

  def make_controller(self, scene:GaussianScene, logger:Logger, 
               densify_steps:int, total_steps:int):
    return PointController(self, scene, logger, densify_steps, total_steps)


class PointController(Controller):
  def __init__(self, config:EMAConfig, 
               scene:GaussianScene,
               logger:Logger, 
               densify_steps:int, 
               total_steps:int):
    
    self.config = config
    self.scene = scene
    self.logger = logger

    self.densify_steps = densify_steps
    self.total_steps = total_steps

    self.target_count = config.target_count or scene.num_points

    n = scene.num_points
    self.points = PointStatistics(
       split_heuristics=torch.zeros(n, 2, dtype=torch.float32, device=scene.device),
        visible=torch.zeros(n, dtype=torch.float32, device=scene.device),
        batch_size = (n,)
        )
       
    # adapt 0.9 toward target in 'decay_steps' for an exponential moving average
    self.ema_alpha  = (0.1 ** (2.0 / config.decay_steps))

    # decay visibility by half in 'decay_steps'
    self.decay =  math.exp(-math.log(2) / config.decay_steps)

    self.splits_per_densify = (1 + config.split_rate) ** (1.0 / densify_steps) - 1.0


  def log_histograms(self, step:int):
    split_score, prune_cost = self.points.split_heuristics.unbind(1) 

    self.logger.log_histogram("points/log_split_score", split_score[split_score > 0].log(), step)
    self.logger.log_histogram("points/log_prune_cost", prune_cost[prune_cost > 0].log(), step)
    self.logger.log_histogram("points/visible", self.points.visible, step)


  def split_prune_mask(self, step:int):
      config = self.config  

      split_ratio = np.clip(self.target_count / self.scene.num_points, 
                              a_min=1/config.max_ratio, a_max=config.max_ratio)
        
      t = step / self.total_steps

      candidates = torch.nonzero(self.points.visible > config.min_visibility).squeeze(1)
      prune_cost, split_score = self.points.split_heuristics[candidates].unbind(dim=1) 

      # linear decay
      factor = self.splits_per_densify * (1 - t)

      split_thresh = torch.quantile(split_score, 1 - (factor * split_ratio))
      prune_thresh = torch.quantile(prune_cost, factor * 1/split_ratio )

      pruneable = (prune_cost <= prune_thresh) 
      splittable = (split_score > split_thresh) & ~pruneable


      split_idx, prune_idx =  candidates[splittable], candidates[pruneable]

      counts = dict(total=self.points.batch_size[0], 
              candidates=candidates.shape[0],
              split=split_idx.shape[0],
              prune=prune_idx.shape[0])
         
      self.logger.log_values("split_prune", counts, step)
      return split_idx, prune_idx        


  def densify_and_prune(self, step:int):
    self.log_histograms(step)

    split_idx, prune_idx = self.split_prune_mask(step)
    # self.scene.densify_and_prune(splittable)




  def add_rendering(self, rendering:Rendering): 
    idx = rendering.points_in_view

    visible_mask = (rendering.split_heuristics[:, 1] > 0)
    vis_idx = idx[visible_mask]

    # update split heuristics for *visible points* 
    h = self.points.split_heuristics[vis_idx]
    h = self.ema_alpha * h + (1 - self.ema_alpha) * rendering.split_heuristics[visible_mask]

    # points which are visble which have not been seen before
    # update directly with the heuristic
    unseen_mask = self.points.visible[vis_idx] == 0
    h[unseen_mask] = rendering.split_heuristics[visible_mask][unseen_mask]

    self.points.split_heuristics[vis_idx] = h

    # counter with exponential decay update for those points in view
    self.points.visible[idx] = self.points.visible[idx] * self.decay + visible_mask

    return (vis_idx.shape[0], idx.shape[0])



