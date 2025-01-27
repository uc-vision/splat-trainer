from abc import ABCMeta, abstractmethod
from typing import Dict, Optional
import torch

from taichi_splatting import Gaussians3D, Rendering
from taichi_splatting.perspective import CameraParams

from splat_trainer.camera_table.camera_table import CameraTable
from splat_trainer.config import Progress
from splat_trainer.logger.logger import Logger


from tensordict import TensorDict



class GaussianScene(metaclass=ABCMeta):  

  @abstractmethod
  def zero_grad(self):
    raise NotImplementedError
  
  @property
  @abstractmethod
  def all_parameters(self) -> TensorDict:
    raise NotImplementedError

  @abstractmethod
  def add_rendering(self, image_idx:int, rendering:Rendering):
    raise NotImplementedError
  
  @abstractmethod
  def reg_loss(self, rendering:Rendering, progress:Progress) -> torch.Tensor:
    raise NotImplementedError

  @abstractmethod
  def step(self, progress:Progress, log_details:bool=False):
    raise NotImplementedError
  
  @abstractmethod
  def render(self, camera_params:CameraParams, cam_idx:Optional[int], point_mask:Optional[torch.Tensor]=None, **options) -> Rendering:
    raise NotImplementedError

  @abstractmethod
  def split_and_prune(self, keep_mask:torch.Tensor, split_idx:Optional[torch.Tensor]=None):
    raise NotImplementedError

  @abstractmethod
  def log_checkpoint(self, progress:Progress):
    raise NotImplementedError
  
  @property
  @abstractmethod
  def num_points(self) -> int:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def gaussians(self) -> Gaussians3D:
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
  def from_color_gaussians(self, gaussians:Gaussians3D, camera_table:CameraTable, device:torch.device, logger:Logger) -> GaussianScene:
    raise NotImplementedError
  
  @abstractmethod
  def from_state_dict(self, state:dict, camera_table:CameraTable, logger:Logger) -> GaussianScene:
    raise NotImplementedError

