from abc import abstractmethod, ABCMeta

from dataclasses import asdict, fields, replace
import math
from os import path
from omegaconf import OmegaConf 

from wonderwords import RandomWord
from pathlib import Path
from typing import Generic, Mapping, Protocol, Sequence, Tuple, TypeVar, runtime_checkable
from beartype import beartype

from torch.optim import Optimizer

@runtime_checkable
class IsDataclass(Protocol):
    __dataclass_fields__: dict

T = TypeVar('T')



class Varying(Generic[T], metaclass=ABCMeta):
  @abstractmethod
  def __call__(self, t:float) -> T:
    pass

class Constant(Varying[T]):
  def __init__(self, value:T):
    self.value = value

  def __call__(self, t:float) -> T:
    return self.value
  
class Linear(Varying[T]):
  def __init__(self, start:T, end:T):
    self.start = start
    self.end = end

  def __call__(self, t:float) -> T:
    return self.start * (1 - t) + self.end * t
  

class LogLinear(Varying[T]):
  def __init__(self, start:T, end:T):
    self.start = start
    self.end = end

  def __call__(self, t:float) -> T:
    return math.exp(math.log(self.start) * (1 - t) + math.log(self.end) * t)

class Piecewise(Varying[T]):
  def __init__(self, start:T, steps:list[tuple[float, T]]):
    self.start = start
    self.steps = steps

  def __call__(self, t:float) -> T:
    value = self.start
    for t_min, next_value in self.steps:
        if t < t_min:
            return value
        value = next_value

    return value

def clamp(x:float, min_val:float, max_val:float):
  return max(min_val, min(x, max_val))

VaryingFloat = Varying[float] | float
VaryingInt = Varying[int] | int


def eval_varyings(value, t:float):
  if isinstance(value, IsDataclass):
    return resolve_varying(value, t, deep=True)
  if isinstance(value, Mapping):
    return value.__class__(**{k: eval_varyings(v, t) for k, v in value.items()})
  elif isinstance(value, Sequence):
    return value.__class__(eval_varyings(v, t) for v in value)

  elif isinstance(value, Varying):
    return value(t)
  else:
    return value

def resolve_varying(cfg:IsDataclass, t:float, deep:bool = False):
  if deep:
    varying = {field.name: eval_varyings(field.value, t) for field in fields(cfg) if isinstance(field.value, Varying)}
    return cfg.__class__(**varying)
  else:
    varying = {field.name: field.update(t) for field in fields(cfg) if isinstance(field.value, Varying)}
    return replace(cfg, **varying)
  
@beartype
def eval_varying(value:Varying[T] | T, t:float) -> T:
  if isinstance(value, Varying):
    return value(t)
  else:
    return value  


class Between(Varying[T]):
  def __init__(self, t_start:float, t_end:float, varying:Varying[T]):
    self.t_start = t_start
    self.t_end = t_end
    self.varying = varying

  def __call__(self, t:float) -> T:
    t = (t - self.t_start) / (self.t_end - self.t_start)
    t = clamp(t, 0, 1)
    return self.varying(t)


  
@beartype
def schedule_lr(v:Varying[float] | float, t:float,  optimizer:Optimizer):
  lr = v(t) if isinstance(v, Varying) else v
  
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

@beartype
def schedule_groups(groups:dict[str, VaryingFloat], t:float, optimizer:Optimizer):
    for param_group in optimizer.param_groups:
      if param_group['name'] in groups:
        param_group['lr'] = eval_varying(groups[param_group['name']], t)





def target(name:str, **kwargs):
  return OmegaConf.create({
    "_target_": f"splat_trainer.config.{name}",
    **kwargs
  })

def add_resolvers():
    OmegaConf.register_new_resolver("dirname", path.dirname)
    OmegaConf.register_new_resolver("basename", path.basename)
    OmegaConf.register_new_resolver("without_ext", lambda x: path.splitext(x)[0])
    OmegaConf.register_new_resolver("ext", lambda x: path.splitext(x)[1])

    OmegaConf.register_new_resolver("scan_name", lambda scan_file: path.join(
        path.dirname(scan_file), f"gaussian_{path.splitext(path.basename(scan_file))[0]}"))
    
    OmegaConf.register_new_resolver("constant", 
        lambda x: target('Constant', value=x))
    
    OmegaConf.register_new_resolver("linear", 
        lambda x, y: target('Linear', start=x, end=y))

    OmegaConf.register_new_resolver("log_linear", 
        lambda x, y: target('LogLinear', start=x, end=y))

    OmegaConf.register_new_resolver("log_decay", 
        lambda x, y: target('LogLinear', start=x, end=x * y))

    OmegaConf.register_new_resolver("piecewise", 
        lambda init,values: target('Piecewise', init=init, values=values))
    OmegaConf.register_new_resolver("between", 
        lambda t_start,t_end,varying: target('Between', 
            dict(t_start=t_start, t_end=t_end, varying=varying)))


def pretty(cfg):
    return OmegaConf.to_yaml(cfg, resolve=True)




def random_name(max_length:int=8):
  r = RandomWord()
  
  adj = r.word(word_max_length=max_length, include_parts_of_speech=["adjectives"])
  noun = r.word(word_max_length=max_length, include_parts_of_speech=["nouns"])

  return f"{adj}_{noun}"

@beartype
def random_folder(path:Path):

  name = random_name()
  while (path / name).exists():
    name = random_name()
  
  return name  

def number_folders(path:Path, name:str):
  if not (path / name).exists():
    return path / name

  i = 1
  while (path / f"{name}_{i}").exists():
    i += 1
  
  return path / f"{name}_{i}"

@beartype
def setup_project(project_name:str, run_name:str | None, base_path:str | None) -> Tuple[Path, str]:
  if base_path is None:
    base_path = Path.cwd() / project_name
  else:
    base_path = Path(base_path) / project_name

  base_path.mkdir(parents=True, exist_ok=True)

  run_name = run_name or random_folder(base_path)
  run_path = base_path / run_name
  run_path.mkdir(parents=True, exist_ok=True)


  return run_path, run_name