import abc
from dataclasses import  dataclass
import math

class Scheduler(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def __call__(self, base_lr:float, step:int, total_steps:int) -> float:
    pass


@dataclass(kw_only=True, frozen=True)
class ExponentialDecay(Scheduler):
  final_decay:float = 1/1000.0
  warmup_steps:int = 0

  def __call__(self, base_lr:float, step:int, total_steps:int):
    t = min(step / total_steps, 1)
    lr = math.exp(math.log(base_lr) * (1 - t) + math.log(self.final_decay * base_lr) *  t) 
    
    if step < self.warmup_steps:
      warmup = math.sin(0.5 * math.pi * step / self.warmup_steps)
      lr *= warmup
    
    return lr
  
@dataclass(kw_only=True, frozen=True)
class Uniform(Scheduler):
  def __call__(self, base_lr:float, step:int, total_steps:int):
    return base_lr