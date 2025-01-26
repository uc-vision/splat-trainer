from splat_trainer.config import Progress
from splat_trainer.logger.logger import Logger
from splat_trainer.scene.scene import GaussianScene
from taichi_splatting import Rendering

from splat_trainer.controller.controller import Controller, ControllerConfig
from splat_trainer.controller.point_state import PointState, log_histograms


class DisabledConfig(ControllerConfig):
  def make_controller(self, scene:GaussianScene, target_points:int, progress:Progress, logger:Logger) -> 'DisabledController':
    return DisabledController(scene, logger)
  
  def from_state_dict(self, state_dict:dict, scene:GaussianScene, target_points:int, progress:Progress, logger:Logger) -> 'DisabledController':
    return self.make_controller(scene, target_points, progress, logger)
  


class DisabledController(Controller):
  def __init__(self, scene:GaussianScene, logger:Logger):
    self.scene = scene
    self.logger = logger

    self.points = PointState.new_zeros(scene.num_points, device=scene.device)

  def step(self, progress:Progress, log_details:bool=False):

    if log_details:
      log_histograms(self.points, self.logger, "step")

    self.points = PointState.new_zeros(self.points.batch_size[0], device=self.points.prune_cost.device)

  def add_rendering(self, image_idx:int, rendering:Rendering, progress:Progress):
    self.points.add_rendering(rendering)

  def state_dict(self) -> dict:
    return {}