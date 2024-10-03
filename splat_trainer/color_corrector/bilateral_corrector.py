
from dataclasses import  dataclass
from typing import Tuple, Union

import torch

from taichi_splatting import Rendering
from .bilateral_grid import BilateralGridInstance
from .corrector import CorrectorConfig, Corrector




@dataclass
class BilateralCorrectorConfig(CorrectorConfig):

  device: str

  steps: int

  bilateral_grid_shape: Tuple = (16, 16, 8)


  def make_corrector(self, num_images) -> Corrector:
    return BilateralCorrector(self.device, self.steps, num_images, self.bilateral_grid_shape)



class BilateralCorrector(Corrector):
  def __init__(self, device, steps, num_images, bilateral_grid_shape):
    self.bilat = BilateralGridInstance(device, steps, num_images, bilateral_grid_shape)

  def correct(self, name:str, rendering:Rendering, image:Union[int, torch.Tensor]) -> Rendering:
    if name == 'train':
        return self.bilat.correct_rendered_image(rendering, image)
    if name == 'eval':
        return self.bilat.correct_for_evaluation(rendering, image)

  def loss(self):
    return self.bilat.tvloss()

  def step(self):
    return self.bilat.step()