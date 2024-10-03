from dataclasses import  dataclass
from typing import Union

import torch
from taichi_splatting import Rendering

from .corrector import CorrectorConfig, Corrector

@dataclass
class NilCorrectorConfig(CorrectorConfig):

  def make_corrector(self, num_images):
    return NilCorrector()


class NilCorrector(Corrector):
  def __init__(self):
    pass

  def correct(self, name:str, rendering:Rendering, image:Union[int, torch.Tensor]):
    return rendering

  def step(self):
    pass

  def loss(self):
    return 0
  