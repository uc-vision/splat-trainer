from abc import ABCMeta, abstractmethod
import torch

from taichi_splatting import Gaussians3D, RasterConfig, Rendering
from taichi_splatting.perspective import CameraParams

from splat_trainer.camera_table.camera_table import ViewTable
from splat_trainer.logger.logger import Logger


  

class GaussianScene(metaclass=ABCMeta):  
  @abstractmethod
  def step(self, visible:torch.Tensor, t:float):
    raise NotImplementedError

  @abstractmethod
  def render(self, camera_params:CameraParams, config:RasterConfig, cam_idx:torch.Tensor, **options) -> Rendering:
    raise NotImplementedError

  @abstractmethod
  def split_and_prune(self, keep_mask:torch.Tensor, split_idx:torch.Tensor):
    raise NotImplementedError

  @abstractmethod
  def update_learning_rate(self, lr_scale:float):
    raise NotImplementedError

  @abstractmethod
  def write_to(self, output_dir:str):
    raise NotImplementedError
  
  @abstractmethod
  def log(self, logger:Logger):
    raise NotImplementedError
  
  @property
  @abstractmethod
  def scale(self) -> torch.Tensor:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def opacity(self) -> torch.Tensor:
    raise NotImplementedError


  @property
  @abstractmethod
  def num_points(self) -> int:
    raise NotImplementedError


  @abstractmethod
  def state_dict(self) -> dict:
    """ Return controller state for checkpointing """
    raise NotImplementedError


class GaussianSceneConfig(metaclass=ABCMeta):

  @abstractmethod
  def from_color_gaussians(self, gaussians:Gaussians3D, camera_table:ViewTable, device:torch.device) -> GaussianScene:
    raise NotImplementedError
  
  @abstractmethod
  def from_state_dict(self, state:dict, camera_table:ViewTable) -> GaussianScene:
    raise NotImplementedError

