from abc import ABCMeta, abstractmethod
from taichi_splatting import Rendering
import torch

from splat_trainer.scene import GaussianScene

class ControllerConfig(metaclass=ABCMeta):

  @abstractmethod
  def make_controller(self, scene:GaussianScene, 
                      densify_interval:int, total_steps:int) -> 'Controller':
    raise NotImplementedError
  

class Controller(metaclass=ABCMeta):
  @abstractmethod
  def densify_and_prune(self, step:int, total_steps:int):
    raise NotImplementedError

  @abstractmethod
  def add_rendering(self, rendering:Rendering) -> tuple[torch.Tensor, torch.Tensor]: 
    raise NotImplementedError
  
  @abstractmethod
  def log_histograms(self, step:int):
    raise NotImplementedError


