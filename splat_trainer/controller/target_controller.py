
from dataclasses import  dataclass
from beartype.typing import Optional
import numpy as np
from taichi_splatting import Rendering
from tensordict import tensorclass
import torch

from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene



@tensorclass
class PointStatistics:
  split_heuristics : torch.Tensor  # (N, 2) - accumulated split heuristics
  visible : torch.Tensor  # (N, ) - number of times the point was rasterized
  
  age : torch.Tensor  # (N, ) - age of the point


  @staticmethod
  def zeros(batch_size, device:Optional[torch.device] = None):
    return PointStatistics(
      split_heuristics=torch.zeros(batch_size, 2, dtype=torch.float32, device=device),
      visible=torch.zeros(batch_size, dtype=torch.float32, device=device),
      age=torch.zeros(batch_size, dtype=torch.float32, device=device),
      batch_size=(batch_size,)
    )
    



@dataclass
class TargetConfig(ControllerConfig):

  # target point cloud size - if None then optimize for the current size
  target_count:Optional[int] = None

  # base rate (relative to count) to split points every 100 iterations
  split_rate:float = 0.05

  # max ratio of points to split/prune
  max_ratio:float = 4.0

  # min number of times a point must be visible recently to be considered for splitting/pruning
  min_visibility:int = 20 

  def make_controller(self, scene:GaussianScene, 
               densify_interval:int, total_steps:int):
    return TargetController(self, scene,  densify_interval, total_steps)


class TargetController(Controller):
  def __init__(self, config:TargetConfig, 
               scene:GaussianScene,
               densify_interval:int, 
               total_steps:int):
    
    self.config = config
    self.scene = scene

    self.densify_interval = densify_interval
    self.total_steps = total_steps

    self.target_count = config.target_count or scene.num_points
    self.points = PointStatistics.zeros(scene.num_points, device=scene.device)
       
    self.splits_per_densify = (1 + config.split_rate) ** (densify_interval / 100) - 1.0
    self.ema_alpha  = (0.1 ** (2.0 / config.min_visibility))



  def log_histograms(self, logger:Logger, step:int):
    split_score, prune_cost = self.points.split_heuristics.unbind(1) 
    logger.log_histogram("points/log_split_score", 
            split_score[split_score > 0 & split_score.isfinite()].log(), step)
    logger.log_histogram("points/log_prune_cost", 
            prune_cost[prune_cost > 0 & prune_cost.isfinite()].log(), step)
    logger.log_histogram("points/visible", self.points.visible, step)


  def find_split_prune_indexes(self, step:int):
      config = self.config  

      split_ratio = np.clip(self.target_count / self.scene.num_points, 
                              a_min=1/config.max_ratio, a_max=config.max_ratio)
        
      t = step / self.total_steps

      candidates = torch.nonzero(self.points.visible >= config.min_visibility).squeeze(1)
      points = self.points[candidates]

      prune_cost, split_score = points.split_heuristics.unbind(dim=1) 

      if candidates.shape[0] == 0:
        splittable, pruneable = [
          torch.zeros(0, dtype=torch.bool, device=candidates.device) for _ in range(2)]

      else:
        # linear decay
        quantile = self.splits_per_densify * (1 - t)

        split_thresh = torch.quantile(split_score, 1 - (quantile * split_ratio))
        prune_thresh = torch.quantile(prune_cost, quantile * 1/split_ratio )

        pruneable = (prune_cost <= prune_thresh) 
        splittable = (split_score > split_thresh) & ~pruneable
      
      split_idx, prune_idx =  candidates[splittable], candidates[pruneable]

      counts = dict(n=self.points.batch_size[0], 
              visible=candidates.shape[0],
              split=split_idx.shape[0],
              prune=prune_idx.shape[0])
         
      return split_idx, prune_idx, counts     


  def densify_and_prune(self, step:int):
    split_idx, prune_idx, counts = self.find_split_prune_indexes(step)

    keep_mask = torch.ones(self.points.batch_size[0], dtype=torch.bool, device=self.scene.device)
    keep_mask[prune_idx] = False
    keep_mask[split_idx] = False

    # keep_mask[self.points.age > 100 & (self.points.visible < 1)] = False

    self.scene.split_and_prune(keep_mask, split_idx)
    self.points = self.points[keep_mask]

    new_points = PointStatistics.zeros(split_idx.shape[0] * 2, device=self.scene.device)
    self.points = torch.cat([self.points, new_points], dim=0)
    return counts    


  def add_rendering(self, rendering:Rendering): 
    idx = rendering.points_in_view
    visible_mask = (rendering.split_heuristics[:, 1] > 0)

    # accumulate split heuristics for visible points
    h, vis = self.points.split_heuristics, self.points.visible
    weight = torch.where(vis[idx] > 0, self.ema_alpha, 1.0).unsqueeze(1)

    h[idx] = (1 - weight) * h[idx] + weight * rendering.split_heuristics
    vis[idx] += 1

    self.points.age += 1
    return (visible_mask.sum().item(), idx.shape[0])



