
from dataclasses import  dataclass
import math
from beartype import beartype

from taichi_splatting import Rendering
import torch

from splat_trainer.config import eval_varying
from splat_trainer.controller.point_state import PointState, densify_and_prune
from splat_trainer.debug.optim import log_histograms
from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene

from splat_trainer.config import Progress, VaryingInt


@beartype
@dataclass
class ThresholdConfig(ControllerConfig):

  # base rate (relative to cout) to prune points 
  threshold: float = 1e-5

  max_split_rate: float = 0.05

  # maximum screen-space size for a floater point (otherwise pruned)
  max_scale_px:float = 200

  # minimum max-screen-space size for a point to be split (don't split tiny points)
  min_split_px:float = 0.5

  densify_interval:VaryingInt = 100

  prune_rate:float = 0.05

  min_views:int = 10


  def make_controller(self, scene:GaussianScene, logger:Logger):
    return ThresholdController(self, scene, logger)

  def from_state_dict(self, state_dict:dict, scene:GaussianScene, logger:Logger) -> Controller:
    controller = ThresholdController(self, scene, logger)
    
    controller.points.load_state_dict(state_dict['points'])

    return controller

class ThresholdController(Controller):
  def __init__(self, config:ThresholdConfig, 
               scene:GaussianScene, logger:Logger):
    
    self.config = config
    self.scene = scene
    self.logger = logger

    self.points = PointState.new_zeros(scene.num_points, device=scene.device)

  def __repr__(self):
    return f"TargetController(points={self.points.batch_size[0]})"

  
  def state_dict(self) -> dict:
    return dict(points=self.points.state_dict())
  
  def find_split_prune_indexes(self, target_count:int, t:float):

    prune_cost, split_score = self.points.masked_heuristics(self.config.min_views)
    split_mask = torch.ones_like(split_score, dtype=torch.bool)

    if self.config.min_split_px > 0:
      # if min split is enabled, only split points above the threshold size
      split_mask = self.points.max_scale_px > self.config.min_split_px
    
    if self.config.max_split_rate > 0:
      # if max split rate is enabled, only split top N points (don't split too fast)
      max_split = int(self.config.max_split_rate * self.scene.num_points)
      split_mask = take_n(split_score * split_mask.float(), max_split, descending=True)
    
    
    if self.config.threshold > 0:
       # if threshold is enabled, only split points above threshold
      thresh_mask = split_score > self.config.threshold
      split_mask &= thresh_mask

    prune_schedule = math.ceil(self.config.prune_rate * self.scene.num_points * (1 - t))
    prune_mask = take_n(prune_cost, prune_schedule, descending=False)

    # prune large points to avoid growing too big
    prune_mask |= self.points.max_scale_px > self.config.max_scale_px 

    return split_mask ^ prune_mask, prune_mask 
  

  def step(self, target_count:int, progress:Progress, log_details:bool=False):

    densify_interval = eval_varying(self.config.densify_interval, progress.t)
    if log_details:
      log_histograms(self.points, self.logger, "step")

    if progress.step % densify_interval == 0:

      split_mask, prune_mask = self.find_split_prune_indexes(target_count, progress.t)
      densify_and_prune(self.points, self.scene, split_mask, prune_mask, self.logger)

      # reset points
      self.points = PointState.new_zeros(self.scene.num_points, device=self.scene.device)

  def add_rendering(self, image_idx:int, rendering:Rendering):
    self.points.lerp_heuristics(rendering)
    self.points.add_in_view(rendering)



@beartype
def take_n(t:torch.Tensor, n:int, descending=False):
  """ Return mask of n largest or smallest values in a tensor."""
  idx = torch.argsort(t, descending=descending)[:n]

  # convert to mask
  mask = torch.zeros_like(t, dtype=torch.bool)
  mask[idx] = True

  return mask
  
