from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import Dict, Optional, Tuple
import torch

from taichi_splatting import Gaussians3D, Rendering
from taichi_splatting.perspective import CameraParams

from splat_trainer.camera_table.camera_table import CameraTable
from splat_trainer.logger.logger import Logger



from tensordict import TensorClass, TensorDict

class PointHeuristics(TensorClass):
  """ Accumulated point heuristics and visibility information 
      similar to Rendering, but accumulated over multiple renderings
  """
  point_heuristic:torch.Tensor  # (N, 2) - accumulated point heuristics
  point_visibility:torch.Tensor # (N,) - accumulated point visibility
  points_in_view:torch.Tensor   # (N,) - number of times each point was in view

  @staticmethod
  def new_zeros(num_points:int, device:torch.device) -> 'PointHeuristics':
    return PointHeuristics(
      point_heuristic=torch.zeros((num_points, 2), device=device),
      point_visibility=torch.zeros(num_points, device=device),
      points_in_view=torch.zeros(num_points, dtype=torch.int16, device=device),
      batch_size=(num_points,)
    )

  def add_rendering(self, rendering:Rendering):    
    # Only accumulate for points that were in view for this rendering
    points_in_view = rendering.points_in_view
    
    # Add loss term from point_heuristic
    self.point_heuristic[points_in_view] += rendering.point_heuristic
    self.point_visibility[points_in_view] += rendering.point_visibility
    self.points_in_view[points_in_view] += 1

  @property
  def prune_cost(self):
    return self.point_heuristic[:, 0]

  @property
  def split_score(self):
    return self.point_heuristic[:, 1]
  
  @cached_property
  def visible_mask(self) -> torch.Tensor:
    return self.point_visibility > 0

  @cached_property
  def visible(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns visible point indexes, and their features """
    mask = self.visible_mask
    return self.points_in_view[mask], self.point_visibility[mask]
  

class GaussianScene(metaclass=ABCMeta):  

  @abstractmethod
  def zero_grad(self):
    raise NotImplementedError
  
  @property
  @abstractmethod
  def all_parameters(self) -> TensorDict:
    raise NotImplementedError

  @abstractmethod
  def step(self, rendering:Rendering, t:float) -> Dict[str, float]:
    raise NotImplementedError
  
  @abstractmethod
  def render(self, camera_params:CameraParams, cam_idx:Optional[int], **options) -> Rendering:
    raise NotImplementedError

  @abstractmethod
  def split_and_prune(self, keep_mask:torch.Tensor, split_idx:torch.Tensor):
    raise NotImplementedError

  @abstractmethod
  def log(self, logger:Logger, step:int):
    raise NotImplementedError
  
  @property
  @abstractmethod
  def num_points(self) -> int:
    raise NotImplementedError


  @abstractmethod
  def state_dict(self) -> dict:
    """ Return controller state for checkpointing """
    raise NotImplementedError

  
  @abstractmethod
  def to_sh_gaussians(self) -> Gaussians3D:
    raise NotImplementedError
  

  @property
  @abstractmethod
  def device(self) -> torch.device:
    raise NotImplementedError
  

  @property
  @abstractmethod
  def clone(self) -> 'GaussianScene':
    raise NotImplementedError


class GaussianSceneConfig(metaclass=ABCMeta):

  @abstractmethod
  def from_color_gaussians(self, gaussians:Gaussians3D, camera_table:CameraTable, device:torch.device) -> GaussianScene:
    raise NotImplementedError
  
  @abstractmethod
  def from_state_dict(self, state:dict, camera_table:CameraTable) -> GaussianScene:
    raise NotImplementedError

