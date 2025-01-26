from dataclasses import  dataclass
from beartype import beartype

from taichi_splatting import Rendering
import torch

from splat_trainer.controller.point_state import PointState, densify_and_prune
from splat_trainer.gaussians.split import sample_gaussians
from splat_trainer.logger.logger import Logger
from splat_trainer.util.misc import soft_lt
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene

from taichi_splatting.optim import ParameterClass
from splat_trainer.config import Progress, eval_varying, VaryingFloat

def saturate(x:torch.Tensor):
  return 1 - 1/torch.exp(2 * x)



@beartype
@dataclass
class MCMCConfig(ControllerConfig):
  # base rate (relative to count) to prune points 
  opacity_threshold:float = 0.1
  prune_interval:int = 50

  min_views:int = 5

  max_scale_px:float = 200

  min_split_px:float = 0.0
  noise_level:VaryingFloat = 100.0

  max_prune_rate:float = 0.05



  def make_controller(self, scene:GaussianScene, target_points:int, progress:Progress, logger:Logger):
    return MCMCController(self, scene, target_points, progress, logger)

  def from_state_dict(self, state_dict:dict, scene:GaussianScene, target_points:int, progress:Progress, logger:Logger) -> Controller:
    controller = MCMCController(self, scene, target_points, progress, logger)
    controller.points.load_state_dict(state_dict['points'])
    return controller



class MCMCController(Controller):
  def __init__(self, config:MCMCConfig, 
               scene:GaussianScene, target_points:int, progress:Progress, logger:Logger):
    
    self.config = config
    self.scene = scene
    self.logger = logger
    self.target_points = target_points
    self.points = PointState.new_zeros(scene.num_points, device=scene.device)
    self.num_points = scene.num_points

  def state_dict(self):
    return {'points': self.points.state_dict()}


  def __repr__(self):
    return f"MCMCController(points={self.points.batch_size[0]})"

    
  @torch.no_grad()
  def step(self, progress:Progress, log_details:bool=False):
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

      target = soft_lt(opacity, self.config.opacity_threshold / 2, margin=16.0)

      points = self.scene.points.tensors[enough_views]
      position = points['position'].data

      noise_level = target[enough_views] * eval_varying(self.config.noise_level, progress.t)
      noise = torch.randn_like(position) * noise_level.unsqueeze(1)
      position += sample_gaussians(points, noise)

      # mult = 0.5 + 1.5 * target 
      # mult = torch.where(enough_views, mult, 1.0)

      # self.scene.points.update_group('position', point_lr=mult)

      
  def add_rendering(self, image_idx:int, rendering:Rendering, progress:Progress):
    self.points.add_rendering(rendering)




@beartype
def take_n(t:torch.Tensor, n:int, descending=False):
  """ Return mask of n largest or smallest values in a tensor."""
  idx = torch.argsort(t, descending=descending)[:n]

  # convert to mask
  mask = torch.zeros_like(t, dtype=torch.bool)
  mask[idx] = True

  return mask
  
