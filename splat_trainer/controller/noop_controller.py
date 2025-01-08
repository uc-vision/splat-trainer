
from dataclasses import  dataclass
from typing import Dict
from beartype import beartype

from taichi_splatting import Rendering

from splat_trainer.logger.logger import Logger
from .controller import Controller, ControllerConfig
from splat_trainer.scene import GaussianScene




@beartype
@dataclass
class NoopConfig(ControllerConfig):

  def make_controller(self, scene:GaussianScene):
    return NoopController(self, scene)

  def from_state_dict(self, state_dict:dict, scene:GaussianScene) -> Controller:
    controller = NoopController(self, scene)

    return controller


class NoopController(Controller):
  def __init__(self, config:NoopConfig, 
               scene:GaussianScene):
    
    self.config = config
    self.scene = scene

  def densify_and_prune(self, t:float) -> Dict[str, float]: 
    
    return dict()

  def step(self, rendering:Rendering, t:float)  -> Dict[str, float]:  
    return dict()
  
  def log_histograms(self, logger:Logger, step:int):
    pass

  def state_dict(self) -> dict:
    return dict()