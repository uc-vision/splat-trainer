from abc import ABCMeta, abstractmethod
import torch

from taichi_splatting import Gaussians3D, Rendering
from taichi_splatting.perspective import CameraParams

  

class GaussianScene(metaclass=ABCMeta):  
  @abstractmethod
  def step(self):
    raise NotImplementedError

  @abstractmethod
  def render(self, cam_idx:torch.Tensor, camera_params:CameraParams, **options) -> Rendering:
    raise NotImplementedError

  @abstractmethod
  def split_and_prune(self, keep_mask:torch.Tensor, split_idx:torch.Tensor):
    raise NotImplementedError

  @abstractmethod
  def update_learning_rate(self, lr_scale:float):
    raise NotImplementedError


class GaussianSceneConfig(metaclass=ABCMeta):

  @abstractmethod
  def from_color_gaussians(self, gaussians:Gaussians3D, device:torch.device, camera_shape:torch.Size) -> GaussianScene:
    raise NotImplementedError
