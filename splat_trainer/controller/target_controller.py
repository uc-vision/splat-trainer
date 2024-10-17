
from dataclasses import  dataclass
import math
from typing import Dict
from beartype import beartype
from beartype.typing import Optional
import numpy as np
from splat_trainer.util.misc import exp_lerp
from taichi_splatting import Rendering
from tensordict import tensorclass
import torch

from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene


@tensorclass
class PointStatistics:
  split_score : torch.Tensor  # (N, ) - averaged split score
  prune_cost : torch.Tensor  # (N, ) - max prune cost
  max_scale : torch.Tensor # (N, ) - max scale


  @staticmethod
  def zeros(batch_size, device:Optional[torch.device] = None):
    return PointStatistics(
      split_score=torch.zeros(batch_size, dtype=torch.float32, device=device),
      prune_cost=torch.zeros(batch_size, dtype=torch.float32, device=device),
      max_scale=torch.zeros(batch_size, dtype=torch.float32, device=device),

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
  prune_rate:float = 0.05

  # ema half life
  split_alpha:float = 0.1
  prune_alpha:float = 0.01

  # maximum screen-space size for a floater point (otherwise pruned)
  max_scale:float = 0.5



  def make_controller(self, scene:GaussianScene):
    return TargetController(self, scene)

  def from_state_dict(self, state_dict:dict, scene:GaussianScene) -> Controller:
    controller = TargetController(self, scene)
    
    controller.points.load_state_dict(state_dict['points'])
    controller.start_count = state_dict['start_count']

    return controller

class TargetController(Controller):
  def __init__(self, config:TargetConfig, 
               scene:GaussianScene):
    
    self.config = config
    self.scene = scene

    self.target_count = config.target_count or scene.num_points
    self.start_count = scene.num_points 

    self.points = PointStatistics.zeros(scene.num_points, device=scene.device)



  def __repr__(self):
    return f"TargetController(points={self.points.batch_size[0]})"

  def log_histograms(self, logger:Logger, step:int):

    split_score, prune_cost = self.points.split_score.log(), self.points.prune_cost.log()

    logger.log_histogram("points/log_split_score", split_score[split_score.isfinite()], step)
    logger.log_histogram("points/log_prune_cost",  prune_cost[prune_cost.isfinite()], step)
    logger.log_histogram("points/max_scale", self.points.max_scale, step)

  def state_dict(self) -> dict:
    return dict(points=self.points.state_dict(), 
                start_count=self.start_count)


  def find_split_prune_indexes(self, t:float):
    config = self.config  
    n = self.points.shape[0]

    # nonlinear point count schedule
    target = math.ceil(smoothstep(t, self.start_count, config.target_count, interval=(0.0, 0.6)))

    # number of pruned points is controlled by the split rated
    # prune_rate = (config.prune_rate * config.densify_interval/100)
    n_prune = math.ceil(config.prune_rate * n * (1 - t))

    n = self.points.split_score.shape[0]
    prune_mask = (take_n(self.points.prune_cost, n_prune, descending=False) 
                  | (~torch.isfinite(self.points.prune_cost)) 
                  | (self.points.max_scale > self.config.max_scale))

    # number of split points is directly set to achieve the target count 
    # (and compensate for pruned points)
    target_split = ((target - n) + n_prune) 

    split_score = self.points.split_score #/ (self.points.visible + 1)
    split_mask = take_n(split_score, target_split, descending=True)

    both = (split_mask & prune_mask)
    return split_mask ^ both, prune_mask ^ both

  def densify_and_prune(self, t:float) -> Dict[str, float]:

    points = self.points
    split_mask, prune_mask = self.find_split_prune_indexes(t)
    split_idx = split_mask.nonzero().squeeze(1)

    n_prune = prune_mask.sum().item()
    n_split = split_idx.shape[0]

    self.prune_thresh = points.prune_cost[prune_mask].max().item() if n_prune > 0 else 0.
    self.split_thresh = points.split_score[split_idx].min().item() if n_split > 0 else 0.

    keep_mask = ~(split_mask | prune_mask)
  # maximum scale for a point to not be pruned
    self.scene.split_and_prune(keep_mask, split_idx)


    self.points = PointStatistics.zeros(self.scene.num_points, device=self.scene.device)
    # self.points.prune_cost[:keep_mask.sum().item()] = points.prune_cost[keep_mask]

    stats = dict(n=self.points.batch_size[0], 
            prune=n_prune,       
            split=n_split,
            max_prune_score=self.prune_thresh, 
            min_split_score=self.split_thresh)
    
    return stats


  def step(self, rendering:Rendering, step:int)  -> Dict[str, float]: 
    idx = rendering.points_in_view
    split_score, prune_cost = rendering.split_heuristics.unbind(1)

    points = self.points

    # Some alternative update rule

    points.split_score[idx] = exp_lerp(self.config.split_alpha, split_score, points.split_score[idx])
    points.prune_cost[idx] = exp_lerp(self.config.prune_alpha, prune_cost, points.prune_cost[idx])
    
    # points.prune_cost[idx] = torch.maximum(points.prune_cost[idx], prune_cost) 

    image_size = max(rendering.camera.image_size)
    near_points = rendering.point_depth.squeeze(1) < rendering.point_depth.quantile(0.5)

    # measure scale of near points in image
    image_scale = rendering.point_scale.max(1).values / image_size
    image_scale[near_points] = 0.

    points.max_scale[idx] = torch.maximum(points.max_scale[idx], image_scale)

    return dict(in_view = rendering.points_in_view.shape[0], 
                visible = rendering.visible_indices.shape[0],
                )      



@beartype
def take_n(t:torch.Tensor, n:int, descending=False):
  """ Return mask of n largest or smallest values in a tensor."""
  idx = torch.argsort(t, descending=descending)[:n]

  # convert to mask
  mask = torch.zeros_like(t, dtype=torch.bool)
  mask[idx] = True

  return mask
  
