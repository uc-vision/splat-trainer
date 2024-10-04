from dataclasses import dataclass

import torch
from taichi_splatting import Rendering

from .corrector import CorrectorConfig, Corrector



@dataclass
class NilCorrectorConfig(CorrectorConfig):

  def make_corrector(self, num_images:int) -> Corrector:
    return NilCorrector()


class NilCorrector(Corrector):
  def __init__(self):
    self.value = False

  def correct(self, name:str, rendering:Rendering, image:torch.Tensor, image_idx:int) -> Rendering:
    return rendering

  def step(self):
    pass

  def loss(self) -> float:
    return 0.

  def __bool__(self) -> bool:
    return self.value
  