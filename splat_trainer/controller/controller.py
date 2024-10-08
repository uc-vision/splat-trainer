from abc import ABCMeta, abstractmethod
from typing import Dict
from taichi_splatting import Rendering

from splat_trainer.logger.logger import Logger
from splat_trainer.scene import GaussianScene

class ControllerConfig(metaclass=ABCMeta):

  @abstractmethod
  def make_controller(self, scene:GaussianScene) -> 'Controller':
    raise NotImplementedError
  
  @abstractmethod
  def from_state_dict(self, state_dict:dict, scene:GaussianScene) -> 'Controller':
    raise NotImplementedError


class Controller(metaclass=ABCMeta):
  @abstractmethod
  def densify_and_prune(self, t:float) -> Dict[str, float]: 
    """ Perform densification and pruning, return dict with metrics for logging"""
    raise NotImplementedError

  @abstractmethod
  def step(self, rendering:Rendering, t:float)  -> Dict[str, float]:  
    """ Step the controller (and gradient step), return dict with metrics for logging"""
    raise NotImplementedError
  
  @abstractmethod
  def log_histograms(self, logger:Logger, step:int):
    """ Log histograms of statistics """
    raise NotImplementedError


  @abstractmethod
  def state_dict(self) -> dict:
    """ Return controller state for checkpointing """
    raise NotImplementedError
