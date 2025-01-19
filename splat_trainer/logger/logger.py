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
  def log_histogram(self, name:str, values:torch.Tensor | Histogram, num_bins:Optional[int] = None):
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

  def add_logger(self, logger:Logger) -> 'CompositeLogger':
    self.loggers.append(logger)
    return self

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

  def log_histogram(self, name:str, values:torch.Tensor | Histogram, num_bins:Optional[int] = None):
    for logger in self.loggers: 
      logger.log_histogram(name, values, num_bins=num_bins)

  def log_json(self, name:str, data:dict):
    for logger in self.loggers:
      logger.log_json(name, data)

  def close(self):
    for logger in self.loggers:
      logger.close()
      

class NullLogger(Logger):
  def __init__(self):
    pass

  def step(self, progress:Progress):
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

  def log_histogram(self, name:str, values:torch.Tensor | Histogram, num_bins:Optional[int] = None):
    pass

  def log_json(self, name:str, data:dict):
    pass

  def close(self):
    pass



class StepValue:
  def __init__(self, step: int, value: float):
    self.step = step
    self.value = value



class StateTree:
  def __init__(self):
    self._data = {}

  def get_leaf(self, parts: list[str]) -> Any:
    node = self._data
    for key in parts:
      node = node[key]
    return node

  def _get_parent_and_key(self, parts: list[str]) -> tuple[dict, str]:
    node = self._data
    for key in parts[:-1]:
      if not isinstance(node.get(key, {}), dict):
        raise ValueError(f"Path conflict: '{'/'.join(parts)}' tries to traverse through '{key}' which is a leaf value")
      if key not in node:
        node[key] = {}
      node = node[key]
    return node, parts[-1]

  def update_leaf(self, parts: list[str], default: Any, update_fn) -> Any:
    parent, key = self._get_parent_and_key(parts)
    if isinstance(parent.get(key, {}), dict) and parent[key]:
      raise ValueError(f"Path conflict: Cannot replace subtree at '{'/'.join(parts)}' with a leaf value")
    parent[key] = update_fn(parent.get(key, default))
    return parent[key]

  def set_leaf(self, parts: list[str], value: Any) -> Any:
    return self.update_leaf(parts, value, lambda x: value)

  def has_path(self, parts: list[str]) -> bool:
    try:
      self.get_leaf(parts)
      return True
    except KeyError:
      return False
    
  def flatten(self) -> dict:
    result = {}
    def flatten_dict(prefix: list[str], node: dict):
        for key, value in node.items():
            path = prefix + [key]
            if isinstance(value, dict):
                flatten_dict(path, value)
            else:
                result["/".join(path)] = value
    
    flatten_dict([], self._data)
    return result


class StateLogger(NullLogger):
  def __init__(self):
    self._tree = StateTree()
    self.current_step = 0

  def __getitem__(self, path: str) -> dict | StepValue:
    return self._tree.get_leaf(path.split("/"))

  def __contains__(self, path: str) -> bool:
    return self._tree.has_path(path.split("/"))

  def step(self, progress: Progress):
    self.current_step = progress.step

  def log_config(self, config: dict):
    self._tree._data["config"] = config
  
  def log_values(self, name: str, data: dict[str, float]):
    for key, value in data.items():
      self.log_value(f"{name}/{key}", value)

  def log_value(self, name: str, value: float):
    self._tree.set_leaf(name.split("/"), StepValue(self.current_step, value))

  def flatten(self) -> dict:
    return self._tree.flatten()

class HistoryLogger(NullLogger):
  def __init__(self):
    self._tree = StateTree()

  def __getitem__(self, path: str) -> dict | list[float]:
    return self._tree.get_leaf(path.split("/"))

  def __contains__(self, path: str) -> bool:
    return self._tree.has_path(path.split("/"))

  def step(self, progress: Progress):
    pass

  def log_config(self, config: dict):
    self._tree._data["config"] = config
  
  def log_values(self, name: str, data: dict[str, float]):
    for key, value in data.items():
      self.log_value(f"{name}/{key}", value)

  def log_value(self, name: str, value: float):
    self._tree.update_leaf(name.split("/"), [], lambda l: l + [value])

  def flatten(self) -> dict:
    return self._tree.flatten()


class LoggerWithState(CompositeLogger):
  def __init__(self, logger:Logger):
    self.state = StateLogger()
    self.logger = logger
    super().__init__(self.state, self.logger)

  def get_value(self, path:str, default:Any = None) -> Any:
    return self.state.get_value(path, default)

  def __getitem__(self, path:str) -> dict | StepValue:
    return self.state[path]
  
  def __contains__(self, path:str) -> bool:
    return path in self.state
  