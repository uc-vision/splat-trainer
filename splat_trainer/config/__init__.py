from abc import abstractmethod, ABCMeta

from dataclasses import dataclass, fields, replace
import math
from os import path
from omegaconf import OmegaConf 

import termcolor
from wonderwords import RandomWord
from pathlib import Path
from typing import Generic, Mapping, Protocol, Sequence, Tuple, TypeVar, runtime_checkable
from beartype import beartype

from torch.optim import Optimizer


@runtime_checkable
class IsDataclass(Protocol):
    __dataclass_fields__: dict

T = TypeVar('T')


@dataclass(kw_only=True, frozen=True)
class Progress:
  step:int
  total_steps:int

  @property
  def t(self) -> float:
    return clamp(self.step / self.total_steps, 0.0, 1.0)
  
  def __float__(self) -> float:
    return float(self.t)



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
  
class LogDecay(Varying[T]):
  def __init__(self, start:T, factor:T):
    self.start = start
    self.factor = factor

  def __call__(self, t:float) -> T:
    return self.start * self.factor ** t


class LogLinear(Varying[T]):
  def __init__(self, start:T, end:T):
    self.start = start
    self.end = end

  def __call__(self, t:float) -> T:
    return math.exp(math.log(self.start) * (1 - t) + math.log(self.end) * t)

class Piecewise(Varying[T]):
  def __init__(self, start:T, steps:list[tuple[float, T]], scale:float = 1.0):
    self.start = start
    self.steps = steps
    self.scale = scale

  def __call__(self, t:float) -> T:
    value = self.start
    for t_min, next_value in self.steps:
        if t < t_min:
            return value * self.scale
        value = next_value

    return value * self.scale
  

def smoothstep(t, a, b, interval=(0, 1)):
  # interpolate with smoothstep function
  r = interval[1] - interval[0]
  t =  clamp((t - interval[0]) / r, 0, 1)
  return a + (b - a) * (3 * t ** 2 - 2 * t ** 3)



class SmoothStep(Varying[float]):
  def __init__(self, start:float, end:float):
    self.start = start
    self.end = end

  def __call__(self, t:float) -> float:
    return smoothstep(t, self.start, self.end)
  


def clamp(x:float, min_val:float, max_val:float):
  return max(min_val, min(x, max_val))

VaryingFloat = Varying[float] | float
VaryingInt = Varying[int] | int


def eval_varyings(value, t:float):
  
  if isinstance(value, IsDataclass):
    return resolve_varying(value, t, deep=True)
  if isinstance(value, Mapping):
    return value.__class__(**{k: eval_varyings(v, t) for k, v in value.items()})
  elif isinstance(value, Sequence) and not isinstance(value, str):
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
def eval_varying(value:Varying[T] | T, t:float | Progress) -> T:

  t = float(t)

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
  group_dict = {param_group['name']: param_group for param_group in optimizer.param_groups}

  for name, lr in groups.items(): 
      if not name in group_dict:
          raise KeyError(f"Group {name} not found in optimizer")
      
      group_dict[name]['lr'] = eval_varying(lr, t)

  # reutrn all the learning rates
  return {param_group['name']: param_group['lr'] for param_group in optimizer.param_groups}


def target(name:str, **kwargs):
  return OmegaConf.create({
    "_target_": f"splat_trainer.config.{name}",
    **kwargs
  })


def make_overrides(**kwargs):
  overrides = []
  for k, v in kwargs.items():
    overrides.append(f"{k}={v if v is not None else 'null'}")
  return overrides



def add_resolvers():
    OmegaConf.register_new_resolver("dirname", path.dirname)
    OmegaConf.register_new_resolver("basename", path.basename)

    OmegaConf.register_new_resolver("int_mul", lambda x, y: int(x * y))
    OmegaConf.register_new_resolver("int_div", lambda x, y: x // y)

    


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
    
    OmegaConf.register_new_resolver("linear_decay", 
        lambda x, y: target('Linear', start=x, end=x * y))

    OmegaConf.register_new_resolver("piecewise", 
        lambda init,values: target('Piecewise', init=init, values=values))
    
    OmegaConf.register_new_resolver("between", 
        lambda t_start,t_end,varying: target('Between', 
            dict(t_start=t_start, t_end=t_end, varying=varying)))

    OmegaConf.register_new_resolver("smoothstep", 
        lambda start,end: target('SmoothStep', start=start, end=end))


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
def setup_project(project_name:str, run_name:str | None, base_path: Path) -> Tuple[Path, Path, str]:

    run_name = run_name or random_folder(base_path)
    run_path = base_path / project_name / run_name
    run_path.mkdir(parents=True, exist_ok=True)

    print(f"Running {termcolor.colored(run_name, 'light_yellow')} in {termcolor.colored(run_path, 'light_green')}")
    return base_path, run_path, run_name
