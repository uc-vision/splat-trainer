
from dataclasses import dataclass
from typing import Tuple

import torch

from taichi_splatting import Rendering
from .bilateral_grid import BilateralGridInstance
from .corrector import CorrectorConfig, Corrector



@dataclass
class BilateralCorrectorConfig(CorrectorConfig):

  device: str
  steps: int
  bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)


  def make_corrector(self, num_images:int) -> Corrector:
    return BilateralCorrector(self.device, self.steps, num_images, self.bilateral_grid_shape)



class BilateralCorrector(Corrector):
  def __init__(self, device:str, steps:int, num_images:int, bilateral_grid_shape:Tuple[int, int, int]):
    self.bilat = BilateralGridInstance(device, steps, num_images, bilateral_grid_shape)
    self.value = True

  def correct(self, name:str, rendering:Rendering, image:torch.Tensor, image_idx:int) -> Rendering:
    if name == 'train':
        return self.bilat.correct_for_train(rendering, image_idx)
    if name == 'eval':
        return self.bilat.correct_for_evaluation(rendering, image)

  def loss(self) -> float:
    return self.bilat.tvloss()

  def step(self):
    return self.bilat.step()

  def __bool__(self) -> bool:
    return self.value