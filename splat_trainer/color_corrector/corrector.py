from abc import ABCMeta, abstractmethod
from typing import Union

import torch
from taichi_splatting import Rendering


class CorrectorConfig(metaclass=ABCMeta):

  @abstractmethod
  def make_corrector(self, num_images:int) -> 'Corrector':
    raise NotImplementedError


class Corrector(metaclass=ABCMeta):
  @abstractmethod
  def correct(self, name:str, rendering:Rendering, image:Union[int, torch.Tensor]) -> Rendering: 
    raise NotImplementedError

  @abstractmethod
  def step(self):  
    raise NotImplementedError
  
  @abstractmethod
  def loss(self) -> float:
    raise NotImplementedError
