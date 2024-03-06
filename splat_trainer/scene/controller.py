from abc import ABCMeta, abstractmethod
from taichi_splatting import Rendering
from splat_trainer.logger.logger import Logger

from splat_trainer.scene.gaussians import GaussianScene

class ControllerConfig(metaclass=ABCMeta):

  @abstractmethod
  def make_controller(self, scene:GaussianScene, logger:Logger, 
                      densify_interval:int, total_steps:int) -> 'Controller':
    raise NotImplementedError
  

class Controller(metaclass=ABCMeta):
  @abstractmethod
  def densify_and_prune(self, step:int, total_steps:int):
    raise NotImplementedError

  @abstractmethod
  def add_rendering(self, rendering:Rendering): 
    raise NotImplementedError


