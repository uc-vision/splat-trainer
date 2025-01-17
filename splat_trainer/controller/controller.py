from abc import ABCMeta, abstractmethod
from typing import Dict
from taichi_splatting import Rendering

from splat_trainer.config import Progress
from splat_trainer.logger.logger import Logger
from splat_trainer.scene import GaussianScene

class ControllerConfig(metaclass=ABCMeta):

  @abstractmethod
  def make_controller(self, scene:GaussianScene, logger:Logger) -> 'Controller':
    raise NotImplementedError
  
  @abstractmethod
  def from_state_dict(self, state_dict:dict, scene:GaussianScene, logger:Logger) -> 'Controller':
    raise NotImplementedError


class Controller(metaclass=ABCMeta):
  @abstractmethod
  def step(self, target_count:int, progress:Progress):
    """ Perform densification and pruning """
    raise NotImplementedError

  @abstractmethod
  def add_rendering(self, image_idx:int, rendering:Rendering):
    """ Add a rendering to the controller """
    raise NotImplementedError

  

  @abstractmethod
  def state_dict(self) -> dict:
    """ Return controller state for checkpointing """
    raise NotImplementedError





  