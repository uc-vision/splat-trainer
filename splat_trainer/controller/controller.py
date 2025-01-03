from abc import ABCMeta, abstractmethod
from typing import Dict
from taichi_splatting import Rendering

from splat_trainer.logger.logger import Logger
from splat_trainer.scene import GaussianScene
from splat_trainer.trainer.scheduler import Progress

class ControllerConfig(metaclass=ABCMeta):

  @abstractmethod
  def make_controller(self, scene:GaussianScene, logger:Logger) -> 'Controller':
    raise NotImplementedError
  
  @abstractmethod
  def from_state_dict(self, state_dict:dict, scene:GaussianScene, logger:Logger) -> 'Controller':
    raise NotImplementedError


class Controller(metaclass=ABCMeta):
  @abstractmethod
  def densify_and_prune(self, t:float):
    """ Perform densification and pruning """
    raise NotImplementedError

  @abstractmethod
  def add_rendering(self, image_idx:int, rendering:Rendering):
    """ Add a rendering to the controller """
    raise NotImplementedError

  @abstractmethod
  def step(self, progress:Progress):
    """ Step the controller for a batch of images processed"""
    raise NotImplementedError
  
  @abstractmethod
  def log_checkpoint(self, step:int):
    """ Log more detailed statistics when checkpointing """
    raise NotImplementedError


  @abstractmethod
  def state_dict(self) -> dict:
    """ Return controller state for checkpointing """
    raise NotImplementedError



class DisabledConfig(ControllerConfig):
  def make_controller(self, scene:GaussianScene, logger:Logger) -> 'Controller':
    return DisabledController()
  
  def from_state_dict(self, state_dict:dict, scene:GaussianScene, logger:Logger) -> 'Controller':
    return DisabledController()
  


class DisabledController(Controller):
  def densify_and_prune(self, t:float):
    return {}
  
  def add_rendering(self, image_idx:int, rendering:Rendering):
    pass

  def step(self, progress:Progress):
    pass
  
  def log_checkpoint(self, step:int):
    pass

  def state_dict(self) -> dict:
    return {}

  