import abc
from dataclasses import dataclass
import math
from beartype import beartype
import torch
import torch.nn.functional as F

from taichi_splatting.perspective import CameraParams



class ImageScaler(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def __call__(self, image:torch.Tensor, camera_params:CameraParams, current_factor:float):
    pass

  @abc.abstractmethod
  def update(self, step:int) -> float:
    pass
    
    

def interpolate_channels_last(image, scale, mode='bilinear'):
    image = image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
    image = F.interpolate(image, scale_factor=scale, mode=mode)
    return image.permute(0, 2, 3, 1).squeeze(0).to(memory_format=torch.contiguous_format)

@dataclass(kw_only=True, frozen=True)
class ExponentialScaler(ImageScaler):
  initial_scale: float = 0.25
  final_scale: float = 1.0

  steps: int = 20000
  scale_mode:str = 'bilinear'

  
  def update(self, step:int) -> float:
    t = min(step / self.steps, 1)
    return math.exp(math.log(self.initial_scale) * (1 - t) +  t * math.log(self.final_scale)) 


  @beartype
  def __call__(self, image:torch.Tensor, camera_params:CameraParams, current_factor:float):
    camera_params = camera_params.scale_image(current_factor)
    image = interpolate_channels_last(image, current_factor, mode=self.scale_mode)
    return image, camera_params
  




@dataclass(kw_only=True, frozen=True)
class NullScaler:

  def update(self, step:int) -> float:
    return 1.0

  @beartype
  def __call__(self, image:torch.Tensor, camera_params:CameraParams, current_factor:float):
    return image, camera_params