from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from beartype import beartype
from beartype.typing import  List
import torch

from splat_trainer.config import Progress
from splat_trainer.logger.histogram import Histogram
from splat_trainer.util.pointcloud import PointCloud

# os.environ["WANDB_SILENT"] = "true"

class Logger(metaclass=ABCMeta):

  @abstractmethod
  def step(self, progress:Progress):
    raise NotImplementedError

  @abstractmethod
  def log_evaluations(self, name:str, data:Dict[str, Dict]):
    raise NotImplementedError

  @abstractmethod
  def log_config(self, config:dict):
    raise NotImplementedError
  
  @abstractmethod
  def log_image(self, name:str, image:torch.Tensor, compressed:bool = True, caption:str | None = None):
    raise NotImplementedError
  
  
  @abstractmethod
  def log_cloud(self, name_str, points:PointCloud):
    raise NotImplementedError
  
  @abstractmethod
  def log_values(self, name:str, data:dict):
    raise NotImplementedError
  
  @abstractmethod
  def log_value(self, name:str, value:float):
    raise NotImplementedError
  
  @abstractmethod
  def log_histogram(self, name:str, values:torch.Tensor | Histogram):
    raise NotImplementedError


  @abstractmethod
  def log_json(self, name:str, data:dict):
    raise NotImplementedError

  @abstractmethod
  def close(self):
    raise NotImplementedError
  
  

class CompositeLogger(Logger):
  @beartype
  def __init__(self, *loggers:Logger):
    self.loggers = list(loggers)

  def step(self, progress:Progress):
    for logger in self.loggers:
      logger.step(progress)

  def add_logger(self, logger:Logger):
    self.loggers.append(logger)

  def log_evaluations(self, name:str, data:Dict[str, Dict]):
    for logger in self.loggers:
      logger.log_evaluations(name, data)

  def log_config(self, config:dict):
    for logger in self.loggers:
      logger.log_config(config)

  def log_image(self, name:str, image:torch.Tensor, compressed:bool = True, caption:str | None = None):
    for logger in self.loggers:
      logger.log_image(name, image, compressed=compressed, caption=caption)

  def log_cloud(self, name_str, points:PointCloud):
    for logger in self.loggers:
      logger.log_cloud(name_str, points)

  def log_values(self, name:str, data:dict):
    for logger in self.loggers:
      logger.log_values(name, data)

  def log_value(self, name:str, value:float):
    for logger in self.loggers:
      logger.log_value(name, value)

  def log_histogram(self, name:str, values:torch.Tensor | Histogram):
    for logger in self.loggers: 
      logger.log_histogram(name, values)

  def log_json(self, name:str, data:dict):
    for logger in self.loggers:
      logger.log_json(name, data)

  def close(self):
    for logger in self.loggers:
      logger.close()
      

class NullLogger(Logger):
  def __init__(self):
    pass

  def log_config(self, config:dict):
    pass
  
  def log_evaluations(self, name:str,  data:Dict[str, Dict]):
    pass

  def log_image(self, name:str, image:torch.Tensor, compressed:bool = True, caption:str | None = None):
    pass

  def log_cloud(self, name_str, points:PointCloud):
    pass

  def log_values(self, name:str, data:dict):
    pass

  def log_value(self, name:str, value:float):
    pass

  def log_histogram(self, name:str, values:torch.Tensor | Histogram):
    pass

  def log_json(self, name:str, data:dict):
    pass

  def close(self):
    pass



@dataclass
class StepValue:
  step:int
  value:Any



class StateLogger(NullLogger):
  def __init__(self):
    self.state = {}
    self.current_step = 0


  def _getitem(self, path:str) -> StepValue | dict:
    d = self.state

    paths = path.split("/")
    for i, key in enumerate(paths):
      if key not in d:
        context = "at top level" if i == 0 else f"under path {paths[:i]}"
        raise KeyError(f"{key} not found {context}, options are {d.keys()}")
      
      d = d[key]
    return d
  
  def _getvalue(self, path:str) -> StepValue:
    d = self._getitem(path)
    if isinstance(d, StepValue):
      return d
    else:
      raise KeyError(f"Expected a value at {path}, got category")


  def get_value(self, path:str, default:Any = None) -> Any:
    try:    
      return self._getvalue(path).value 
    except KeyError:
      return default
  
  def __getitem__(self, path:str) -> dict | StepValue:
    return self._getitem(path)

  def __contains__(self, path:str) -> bool:
    try:
      self._getitem(path)
      return True
    except KeyError:
      return False
    
  

  def step(self, progress:Progress):
    self.current_step = progress.step

  def log_config(self, config:dict):
    self.state["config"] = config
  
  def log_values(self, name:str, data:dict):
    for key, value in data.items():
      self.log_value(f"{name}/{key}", value)

  def log_value(self, name:str, value:Any):
    path = name.split("/")

    d = self.state
    for i, key in enumerate(path[:-1]):
      if isinstance(d, StepValue):
        raise ValueError(f"Inconsistent path, previously {path[:i]} was logged as a value")
      
      if key not in d:
        d[key] = {}
      d = d[key]
    
    d[path[-1]] = StepValue(self.current_step, value)
      


