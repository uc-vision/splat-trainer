from dataclasses import  dataclass
from beartype import beartype

from taichi_splatting import Rendering
import torch

from splat_trainer.controller.point_state import PointState, densify_and_prune
from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene

from taichi_splatting.optim import ParameterClass

from splat_trainer.config import Progress


def saturate(x:torch.Tensor):
  return 1 - 1/torch.exp(2 * x)



@beartype
@dataclass
class MCMCConfig(ControllerConfig):
  # base rate (relative to count) to prune points 
  opacity_threshold:float = 0.1
  prune_interval:int = 100

  min_views:int = 10

  max_scale_px:float = 200

  min_split_px:float = 0.0



  def make_controller(self, scene:GaussianScene, logger:Logger):
    return MCMCController(self, scene, logger)

  def from_state_dict(self, state_dict:dict, scene:GaussianScene, logger:Logger) -> Controller:
    controller = MCMCController(self, scene, logger)
    controller.points.load_state_dict(state_dict['points'])
    return controller

class MCMCController(Controller):
  def __init__(self, config:MCMCConfig, 
               scene:GaussianScene, logger:Logger):
    
    self.config = config
    self.scene = scene
    self.logger = logger

    self.points = PointState.new_zeros(scene.num_points, device=scene.device)
    self.num_points = scene.num_points

  def state_dict(self):
    return {'points': self.points.state_dict()}


  def __repr__(self):
    return f"MCMCController(points={self.points.batch_size[0]})"

    

  def step(self, target_count:int, progress:Progress, log_details:bool=False):
    points:ParameterClass = self.scene.points
    enough_views = self.points.points_in_view > self.config.min_views

    opacity = torch.sigmoid(points.alpha_logit).squeeze(1)

    if progress.step % self.config.prune_interval == 0:
      # prune large points to avoid growing too big
      prune_mask = (
        (self.points.max_scale_px > self.config.max_scale_px) 
        | (opacity < self.config.opacity_threshold)
      )
      n = prune_mask.sum().item()


      too_small = self.points.max_scale_px < self.config.min_split_px
      split_score = torch.where(prune_mask | too_small, 0, self.points.split_score)


      split_mask = take_n(split_score, n, descending=True)
      self.points = densify_and_prune(self.points, self.scene, split_mask, prune_mask, logger=self.logger)

    else:
      ratio = self.config.opacity_threshold / (opacity + 1e-20)  - 1
      
      self.logger.log_histogram("opacity_ratio", ratio)
      
      lr_multiplier = torch.where(enough_views, 0.5 + 1.5 * torch.sigmoid(8.0 * ratio), 0.5)
      self.logger.log_histogram("lr_multiplier", lr_multiplier)

      
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
  
