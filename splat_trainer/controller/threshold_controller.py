
from dataclasses import  dataclass
import math
from typing import Dict
from beartype.typing import Optional
from taichi_splatting import Rendering
from tensordict import tensorclass
import torch

from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene

from taichi_splatting import RasterConfig
from taichi_splatting.perspective import CameraParams



@tensorclass
class PointStatistics:
  point_grad : torch.Tensor  # (N, 2) - accumulated split heuristics
  visible : torch.Tensor  # (N, ) - number of times the point was in view

  @staticmethod
  def zeros(batch_size, device:Optional[torch.device] = None):
    return PointStatistics(
      point_grad=torch.zeros(batch_size, dtype=torch.float32, device=device),
      visible=torch.zeros(batch_size, dtype=torch.float32, device=device),
      batch_size=(batch_size,)
    )



@dataclass
class ThresholdConfig(ControllerConfig):

  grad_threshold:float = 0.0002
  min_split_size:float = 0.001

  min_opacity:float = 0.01

  # min number of times a point must be visible recently to be considered for splitting/cloning
  min_visibility:int = 40 



  def make_controller(self, scene:GaussianScene, 
               densify_interval:int, total_steps:int):
    return ThresholdController(self, scene,  densify_interval, total_steps)


class ThresholdController(Controller):
  def __init__(self, config:ThresholdConfig, 
               scene:GaussianScene):
    
    self.config = config
    self.scene = scene

    self.points = PointStatistics.zeros(scene.num_points, device=scene.device)
 

  def log_histograms(self, logger:Logger, step:int):
    point_grad = self.points.point_grad

    logger.log_histogram("points/log_point_grad", 
        point_grad[(point_grad > 0) & point_grad.isfinite()].log(), step)
    logger.log_histogram("points/visible", self.points.visible, step)


  def find_split_prune_indexes(self):
      config = self.config  

      grad_mean = self.points.point_grad / self.points.visible
      vis_mask = (self.points.visible > config.min_visibility)

      split_mask = (grad_mean > config.grad_threshold) & vis_mask
      prune_mask = self.scene.gaussians.alpha_logit.squeeze(1) < math.log(config.min_opacity)
      keep_mask = ~prune_mask & ~split_mask

      split_idx = split_mask.nonzero(as_tuple=False).squeeze(1)

      counts = dict(n=self.points.batch_size[0], 
          visible=vis_mask.sum().item(),
          split=split_idx.shape[0],
          prune=prune_mask.sum().item())
      
      return keep_mask, split_idx, counts     


  def densify_and_prune(self, t:float) -> Dict[str, float]:

    keep_mask, split_idx, counts = self.find_split_prune_indexes()

    self.scene.split_and_prune(keep_mask, split_idx)
    self.points = self.points[keep_mask]

    new_points = PointStatistics.zeros(split_idx.shape[0] * 2, device=self.scene.device)
    self.points = torch.cat([self.points, new_points], dim=0)

    return counts    




  def step(self, rendering:Rendering, t:float) -> Dict[str, float]: 
    idx = rendering.points_in_view

    longest_side = max(*rendering.image_size)
    point_grad = rendering.split_score * 0.5 * longest_side

    visible_mask = (point_grad > 0)

    self.points.point_grad[idx] += point_grad
    self.points.visible[idx] += visible_mask

    return dict(in_view = rendering.points_in_view.shape[0], 
                visible = rendering.visible_indices.shape[0])  


