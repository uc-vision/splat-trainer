from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from taichi_splatting import Rendering

from .corrector import CorrectorConfig, Corrector



@dataclass
class NilCorrectorConfig(CorrectorConfig):

  def make_corrector(self, num_images:int, device:torch.device) -> Corrector:
    return NilCorrector(device)

  def from_state_dict(self, state_dict:dict, device:torch.device) -> Corrector:
    return NilCorrector(device)

class NilCorrector(Corrector):
  def __init__(self, device:torch.device):
    self.device = device

  def correct(self, rendering:Rendering, image_idx:int) -> Rendering:
    return rendering.image

  def step(self, t:float) -> Dict[str, float]:
    return {} 

  def state_dict(self) -> Dict[str, Any]:
    return {}

  def zero_grad(self):
    pass
  