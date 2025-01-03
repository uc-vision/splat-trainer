
from dataclasses import  dataclass
import math
from typing import Dict
from beartype import beartype

from taichi_splatting import Rendering
import torch

from splat_trainer.logger.logger import Logger
from splat_trainer.trainer.scheduler import Progress
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene

from splat_trainer.config import Between, SmoothStep, VaryingFloat, eval_varying
from tensordict import TensorClass

class PointState  (TensorClass):
    prune_cost:torch.Tensor  
    split_score:torch.Tensor
    max_scale:torch.Tensor        # (N,) - maximum scale seen for each near point (fraction of focal length)
    points_in_view:torch.Tensor   # (N,) - number of times each point was in view

    @staticmethod
    def new_zeros(num_points:int, device:torch.device) -> 'PointState':
        return PointState(
            prune_cost=torch.zeros(num_points, device=device),
            split_score=torch.zeros(num_points, device=device),
            max_scale=torch.zeros(num_points, device=device),
            points_in_view=torch.zeros(num_points, dtype=torch.int16, device=device),
            batch_size=(num_points,)
        )


    def state_dict(self) -> dict:
      return dict(prune_cost=self.prune_cost, 
                  split_score=self.split_score, 
                  max_scale=self.max_scale, 
                  points_in_view=self.points_in_view)

    def from_state_dict(self, state_dict:dict):
      self.prune_cost = state_dict['prune_cost']
      self.split_score = state_dict['split_score']
      self.max_scale = state_dict['max_scale']
      self.points_in_view = state_dict['points_in_view']


@beartype
@dataclass
class TargetConfig(ControllerConfig):

  # target point cloud size - if None then optimize for the current size
  target_count:int

  # base rate (relative to count) to prune points 
  prune_rate:float = 0.05

  # ema half life
  split_alpha:float = 0.1
  prune_alpha:float = 0.01

  # maximum screen-space size for a floater point (otherwise pruned)
  max_scale:float = 0.2
  target_schedule:VaryingFloat = Between(0, 0.6, SmoothStep(0.0, 1.0))



  def make_controller(self, scene:GaussianScene, logger:Logger):
    return TargetController(self, scene, logger)

  def from_state_dict(self, state_dict:dict, scene:GaussianScene, logger:Logger) -> Controller:
    controller = TargetController(self, scene, logger)
    
    controller.points.load_state_dict(state_dict['points'])
    controller.start_count = state_dict['start_count']

    return controller

class TargetController(Controller):
  def __init__(self, config:TargetConfig, 
               scene:GaussianScene, logger:Logger):
    
    self.config = config
    self.scene = scene
    self.logger = logger
    self.target_count = config.target_count or scene.num_points
    self.start_count = scene.num_points 

    self.points = PointState.new_zeros(scene.num_points, device=scene.device)

  def __repr__(self):
    return f"TargetController(points={self.points.batch_size[0]})"

  def log_checkpoint(self):

    # split_score, prune_cost = self.points.split_score.log(), self.points.prune_cost.log()

    # self.logger.log_histogram("points/log_split_score", split_score[split_score.isfinite()])
    # self.logger.log_histogram("points/log_prune_cost",  prune_cost[prune_cost.isfinite()])
    # self.logger.log_histogram("points/max_scale", self.points.max_scale)
    pass

  def state_dict(self) -> dict:
    return dict(points=self.points.state_dict(), 
                start_count=self.start_count)


  def find_split_prune_indexes(self, t:float):
    config = self.config  
    n = self.points.batch_size[0]

    # point count schedule
    schedule = eval_varying(self.config.target_schedule, t)

    total_added = self.config.target_count - self.start_count
    target = max(n, self.start_count + math.ceil(schedule * total_added))

    # number of pruned points is controlled by the split rated
    # prune_rate = (config.prune_rate * config.densify_interval/100)
    n_prune = math.ceil(config.prune_rate * n * (1 - t))

    # n_prune = math.ceil(config.prune_rate * n)

    

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

  def densify_and_prune(self, progress:Progress):

    points = self.points
    split_mask, prune_mask = self.find_split_prune_indexes(progress.t)
    split_idx = split_mask.nonzero().squeeze(1)

    n_prune = prune_mask.sum().item()
    n_split = split_idx.shape[0]

    n = self.points.batch_size[0]
    n_unseen = n - torch.count_nonzero(points.prune_cost).item()

    self.prune_thresh = points.prune_cost[prune_mask].max().item() if n_prune > 0 else 0.
    self.split_thresh = points.split_score[split_idx].min().item() if n_split > 0 else 0.

    keep_mask = ~(split_mask | prune_mask)

  # maximum scale for a point to not be pruned
    self.scene.split_and_prune(keep_mask, split_idx)

    self.points = PointState.new_zeros(self.scene.num_points, device=self.scene.device)
    # self.points.prune_cost[:keep_mask.sum().item()] = points.prune_cost[keep_mask]

    self.logger.log_values("densify", dict(n=self.points.batch_size[0], 
            prune=n_prune,       
            split=n_split,
            max_prune_score=self.prune_thresh, 
            min_split_score=self.split_thresh,
            unseen = n_unseen))
    


  def add_rendering(self, image_idx:int, rendering:Rendering):
    points = self.points
    in_view_idx = rendering.points_in_view

    points.prune_cost[in_view_idx] += rendering.prune_cost
    points.split_score[in_view_idx] += rendering.split_score

    image_size = max(rendering.camera.image_size)
    far_points = rendering.point_depth.squeeze(1) > rendering.point_depth.quantile(0.75)

    # measure scale of far points in image
    image_scale = rendering.point_scale.max(1).values / image_size
    image_scale[far_points] = 0.

    points.max_scale[in_view_idx] = torch.maximum(points.max_scale[in_view_idx], image_scale)
    points.points_in_view[in_view_idx] += 1

  def step(self, progress:Progress): 
    pass





@beartype
def take_n(t:torch.Tensor, n:int, descending=False):
  """ Return mask of n largest or smallest values in a tensor."""
  idx = torch.argsort(t, descending=descending)[:n]

  # convert to mask
  mask = torch.zeros_like(t, dtype=torch.bool)
  mask[idx] = True

  return mask
  
