
from dataclasses import  dataclass
import gc
import math
from beartype import beartype

from taichi_splatting import Rendering
import torch

from splat_trainer.config import clamp, eval_varying, smoothstep
from splat_trainer.controller.point_state import PointState, densify_and_prune, log_histograms
from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene

from splat_trainer.config import Progress, VaryingInt

@beartype
@dataclass
class TargetConfig(ControllerConfig):
  # base rate (relative to count) to prune points 
  prune_rate:float = 0.04

  # proportion of training when we hit the target count
  target_count_t:float = 0.8

  # minimum number of times point is in view before it is able to be pruned
  min_views:int = 10

  # maximum screen-space size for a floater point (otherwise pruned)
  max_scale_px:float = 200

  # minimum max-screen-space size for a point to be split (don't split tiny points)
  min_split_px:float = 0.0

  densify_prune_interval:VaryingInt = 100



  def make_controller(self, scene:GaussianScene, target_points:int, progress:Progress, logger:Logger):
    return TargetController(self, scene, target_points, progress, logger)

  def from_state_dict(self, state_dict:dict, scene:GaussianScene, target_points:int, progress:Progress, logger:Logger) -> Controller:
    controller = TargetController(self, scene, target_points, progress, logger, start_points=state_dict['start_points'])
    controller.points.load_state_dict(state_dict['points'])
    return controller



class TargetController(Controller):
  def __init__(self, config:TargetConfig, 
               scene:GaussianScene, target_points:int, progress:Progress, logger:Logger, start_points:int=None):
    
    self.config = config
    self.scene = scene
    self.logger = logger

    self.points = PointState.new_zeros(scene.num_points, device=scene.device)
    self.start_points = start_points or scene.num_points

    self.next_densify = self.find_next_densify(progress) + 50
    self.max_points = target_points
    
  def __repr__(self):
    return f"TargetController(points={self.points.batch_size[0]})"


  def state_dict(self) -> dict:
    return dict(points=self.points.state_dict(), start_points=self.start_points)



  def find_split_prune_indexes(self, t:float, target_points:int):
    config = self.config  
    n = self.points.batch_size[0]

    exceeds_scale = self.points.max_scale_px > config.max_scale_px
    prune_schedule = int(math.ceil(config.prune_rate * n * (1 - t)))

    prune_cost, split_score = self.points.masked_heuristics(config.min_views)
    
    prune_mask = take_n(prune_cost, prune_schedule, descending=False) | exceeds_scale 

    target_split = (target_points - n) + prune_mask.sum().item()
    
    split_score = self.points.split_score 
    split_score[prune_mask] = 0.

    if self.config.min_split_px > 0:
      # if min split is enabled, only split points above the threshold size
      too_small = self.points.max_scale_px < self.config.min_split_px
      split_score[too_small] = 0.

    split_mask = take_n(split_score, target_split, descending=True)

    both = (split_mask & prune_mask)
    return split_mask ^ both, prune_mask ^ both


  def find_next_densify(self, progress:Progress):
    densify_interval =  eval_varying(self.config.densify_prune_interval, progress.t)
    next_densify = progress.step + densify_interval

    return next_densify if (next_densify + densify_interval < progress.total_steps) else None


  def target_points(self, progress:Progress):
    target_step = self.config.target_count_t * progress.total_steps
    t = clamp(progress.step / target_step, 0.0, 1.0)
    return int(smoothstep(t, self.start_points, self.max_points))   

  def step(self, progress:Progress, log_details:bool=False):

    if log_details:
      log_histograms(self.points, self.logger, "step")

    next_densify = self.next_densify
    if next_densify is not None and progress.step >= next_densify:
      split_mask, prune_mask = self.find_split_prune_indexes(progress.t, self.target_points(progress))
      densify_and_prune(self.points, self.scene, split_mask, prune_mask, self.logger)
      
      self.points = PointState.new_zeros(self.scene.num_points, device=self.scene.device)
      self.next_densify = self.find_next_densify(progress)

      gc.collect()
      torch.cuda.empty_cache()      
    # else:
    #   enough_views = self.points.points_in_view > self.config.min_views
    #   # opacity = torch.sigmoid(self.scene.points['alpha_logit'].data)  
    #   prune_threshold = torch.quantile(self.points.prune_cost, self.config.prune_rate / 2)
    #   target = soft_lt(self.points.prune_cost, prune_threshold, margin=16.0)

    #   points = self.scene.points.tensors[enough_views]
    #   position = points['position'].data

    #   noise_level = target[enough_views] * 10.

    #   # print(noise_level.shape, position.shape)
    #   noise = torch.randn_like(position) * noise_level.unsqueeze(1)
    #   position += sample_gaussians(points, noise)


  def add_rendering(self, image_idx:int, rendering:Rendering, progress:Progress):
    self.points.add_rendering(rendering)




@beartype
def take_n(t:torch.Tensor, n:int, descending=False):
  """ Return mask of n largest or smallest values in a tensor."""

  assert n >= 0, f"n must be >= 0, got {n}"
  idx = torch.argsort(t, descending=descending)[:n]

  # convert to mask
  mask = torch.zeros_like(t, dtype=torch.bool)
  mask[idx] = True

  return mask
  
