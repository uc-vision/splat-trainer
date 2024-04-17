import abc
from dataclasses import  dataclass
import math

import torch

class Scheduler(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def __call__(self, step:int, total_steps:int) -> float:
    pass


@dataclass(kw_only=True, frozen=True)
class ExponentialDecay(Scheduler):
  base_lr: float = 1.0
  final_lr:float = 1/1000.0
  warmup_steps:int = 0

  def __call__(self, step:int, total_steps:int):
    t = min(step / total_steps, 1)
    lr = math.exp(math.log(self.base_lr) * (1 - t) + math.log(self.final_lr * self.base_lr) *  t) 
    
    if step < self.warmup_steps:
      warmup = math.sin(0.5 * math.pi * step / self.warmup_steps)
      lr *= warmup
    
    return lr
  
@dataclass(kw_only=True, frozen=True)
class Uniform(Scheduler):
  base_lr: float = 1.0

  def __call__(self, step:int, total_steps:int):
    return self.base_lr
  



