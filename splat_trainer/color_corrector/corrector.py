from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple

import torch
from taichi_splatting import Rendering


class CorrectorConfig(metaclass=ABCMeta):

  @abstractmethod
  def make_corrector(self, num_images:int, device:torch.device) -> 'Corrector':
    raise NotImplementedError


class Corrector(metaclass=ABCMeta):

  @abstractmethod
  def correct(self, rendering:Rendering, image_idx:int) -> torch.Tensor: 
    raise NotImplementedError

  @abstractmethod
  def step(self, t:float):
    raise NotImplementedError
  
  @abstractmethod
  def loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
    """ returns loss, dict of metrics to log"""
    raise NotImplementedError

