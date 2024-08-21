from abc import ABCMeta, abstractmethod
from typing import Dict
from taichi_splatting import Rendering

from splat_trainer.logger.logger import Logger
from splat_trainer.scene import GaussianScene

class ControllerConfig(metaclass=ABCMeta):

  @abstractmethod
  def make_controller(self, scene:GaussianScene, 
                      densify_interval:int, total_steps:int) -> 'Controller':
    raise NotImplementedError
  

class Controller(metaclass=ABCMeta):
  @abstractmethod
  def densify_and_prune(self, step:int, total_steps:int) -> Dict[str, float]: 
    """ Perform densification and pruning, return dict with metrics for logging"""
    raise NotImplementedError

  @abstractmethod
  def step(self, rendering:Rendering, step:int)  -> Dict[str, float]:  
    """ Step the controller (and gradient step), return dict with metrics for logging"""
    raise NotImplementedError
  
  @abstractmethod
  def log_histograms(self, logger:Logger, step:int):
    """ Log histograms of statistics """
    raise NotImplementedError


  

