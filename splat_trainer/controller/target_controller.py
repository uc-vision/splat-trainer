
from dataclasses import  dataclass
import math
from beartype import beartype
from beartype.typing import Optional
import numpy as np
from taichi_splatting import Rendering
from tensordict import tensorclass
import torch

from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene

from taichi_splatting import render_gaussians, RasterConfig
from taichi_splatting.perspective import CameraParams


@tensorclass
class PointStatistics:
  split_score : torch.Tensor  # (N, ) - averaged split score
  prune_cost : torch.Tensor  # (N, ) - max prune cost


  learning_rate : torch.Tensor # (N, ) - learning rate



  @staticmethod
  def zeros(batch_size, device:Optional[torch.device] = None):
    return PointStatistics(
      split_score=torch.zeros(batch_size, dtype=torch.float32, device=device),
      prune_cost=torch.zeros(batch_size, dtype=torch.float32, device=device),

      batch_size=(batch_size,)
    )

def smoothstep(x, a, b, interval=(0, 1)):
  # interpolate with smoothstep function
  r = interval[1] - interval[0]
  x =  np.clip((x - interval[0]) / r, 0, 1)
  return a + (b - a) * (3 * x ** 2 - 2 * x ** 3)





@dataclass
class TargetConfig(ControllerConfig):

  # target point cloud size - if None then optimize for the current size
  target_count:Optional[int] = None

  # base rate (relative to count) to prune points 
  prune_rate:float = 0.2

  # max ratio of points to split/prune
  max_ratio:float = 4.0

  # ema half life
  ema_half_life:int = 20

  max_radius:float = 0.1 # max screenspace radius (proportion of longest side) before splitting
  min_radius: float = 1.0 / 1000.

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
    self.start_count = scene.num_points 

    self.points = PointStatistics.zeros(scene.num_points, device=scene.device)

    # alpha which gives number of iterations to decay by half
    self.ema_alpha = 0.5 ** (1.0 / config.ema_half_life)


  def log_histograms(self, logger:Logger, step:int):

    split_score, prune_cost = self.points.split_score.log(), self.points.prune_cost.log()

    logger.log_histogram("points/log_split_score", split_score[split_score.isfinite()], step)
    logger.log_histogram("points/log_prune_cost",  prune_cost[prune_cost.isfinite()], step)

    logger.log_histogram("points/learning_rate",  self.points.learning_rate, step)




  def find_split_prune_indexes(self, step:int):
    config = self.config  
    n = self.points.shape[0]
    t = max(0, step / self.total_steps)
  
    # nonlinear point count schedule
    target = math.ceil(smoothstep(t, self.start_count, config.target_count, interval=(0.0, 0.6)))

    # number of pruned points is controlled by the split rated
    n_prune = math.ceil(config.prune_rate * n * (1 - t))

    n = self.points.split_score.shape[0]
    prune_mask = take_n(self.points.prune_cost, n_prune, descending=False) | (~torch.isfinite(self.points.prune_cost))

    # number of split points is directly set to achieve the target count 
    # (and compensate for pruned points)
    target_split = ((target - n) + n_prune) 

    split_score = self.points.split_score #/ (self.points.visible + 1)
    split_mask = take_n(split_score, target_split, descending=True)

    both = (split_mask & prune_mask)
    return split_mask ^ both, prune_mask ^ both

  def densify_and_prune(self, step:int):

    split_mask, prune_mask = self.find_split_prune_indexes(step)
    split_idx = split_mask.nonzero().squeeze(1)

    n_prune = prune_mask.sum().item()
    n_split = split_idx.shape[0]

    prune_thresh = self.points.prune_cost[prune_mask].max().item() if n_prune > 0 else 0.
    split_thresh = self.points.split_score[split_idx].min().item() if n_split > 0 else 0.

    keep_mask = ~(split_mask | prune_mask)



    self.scene.split_and_prune(keep_mask, split_idx)
    self.points = PointStatistics.zeros(self.scene.num_points, device=self.scene.device)


    stats = dict(n=self.points.batch_size[0], 
            prune=n_prune,       
            split=n_split,
            max_prune_score=prune_thresh, 
            min_split_score=split_thresh)
    
    return stats


  def add_rendering(self, rendering:Rendering): 
    idx = rendering.points_in_view
    split_score, prune_cost = rendering.split_heuristics.unbind(1)

    points = self.points

    points.prune_cost[idx] = torch.maximum(points.prune_cost[idx], prune_cost) 
    points.split_score[idx] = torch.maximum(points.split_score[idx], split_score) 
      

  def render(self, camera_params:CameraParams, config:RasterConfig, cam_idx:torch.Tensor, 
             **options) -> Rendering:
    
      return self.scene.render(config = config,
                  camera_params = camera_params,
                  cam_idx=cam_idx,
                  compute_split_heuristics=True,
                  **options)
  


  def step(self, rendering:Rendering, step:int):
    self.add_rendering(rendering)
    self.scene.step(step)

    return dict(in_view = rendering.points_in_view.shape[0], 
                visible = rendering.visible_indices.shape[0])


@beartype
def take_n(t:torch.Tensor, n:int, descending=False):
  """ Return mask of n largest or smallest values in a tensor."""
  idx = torch.argsort(t, descending=descending)[:n]

  # convert to mask
  mask = torch.zeros_like(t, dtype=torch.bool)
  mask[idx] = True

  return mask
  
