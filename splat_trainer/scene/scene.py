from abc import ABCMeta, abstractmethod
from typing import Dict, Optional
import torch

from taichi_splatting import Gaussians3D, Rendering
from taichi_splatting.perspective import CameraParams

from splat_trainer.camera_table.camera_table import CameraTable
from splat_trainer.logger.logger import Logger


  

class GaussianScene(metaclass=ABCMeta):  
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
  def log(self, logger:Logger):
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


class GaussianSceneConfig(metaclass=ABCMeta):

  @abstractmethod
  def from_color_gaussians(self, gaussians:Gaussians3D, camera_table:CameraTable, device:torch.device) -> GaussianScene:
    raise NotImplementedError
  
  @abstractmethod
  def from_state_dict(self, state:dict, camera_table:CameraTable) -> GaussianScene:
    raise NotImplementedError

