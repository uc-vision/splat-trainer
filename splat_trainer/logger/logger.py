from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, TypeVar
from beartype import beartype
from beartype.typing import  List
import torch

from splat_trainer.config import Progress
from splat_trainer.logger.histogram import Histogram
from splat_trainer.util.pointcloud import PointCloud


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
  def log_histogram(self, name:str, values:torch.Tensor | Histogram, num_bins:int | None = None):
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

  def log_histogram(self, name:str, values:torch.Tensor | Histogram, num_bins:int | None = None):
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

  def log_histogram(self, name:str, values:torch.Tensor | Histogram, num_bins:int | None = None):
    pass

  def log_json(self, name:str, data:dict):
    pass

  def close(self):
    pass



class StepValue:
  def __init__(self, step: int, value: Any):
    self.step = step
    self.value = value

@dataclass
class Path:
  """Represents a path in a tree structure using parts (e.g. 'a/b/c' -> ['a', 'b', 'c'])."""
  parts: List[str]

  @staticmethod
  def from_str(path: str) -> 'Path':
    return Path(path.split("/"))

  def parent(self) -> 'Path':
    return Path(self.parts[:-1])

  def last_part(self) -> str:
    return self.parts[-1]

  def __str__(self) -> str:
    return "/".join(self.parts)

  def __repr__(self) -> str:
    return '/'.join(self.parts)

T = TypeVar("T")

class StateTree(Generic[T]):
  """Tree structure for storing state, with paths like 'a/b/c' mapping to values of type T."""
  def __init__(self):
    self.data: Dict[str, StateTree[T] | T] = {}

  def __getitem__(self, key: str) -> 'StateTree[T]' | T:
    return self.data[key]

  def __contains__(self, key: str) -> bool:
    return key in self.data

  def __setitem__(self, key: str, value: 'StateTree[T]' | T):
    self.data[key] = value

  def items(self):
    return self.data.items()

  def __repr__(self):
    return f"StateTree({', '.join(self.data.keys())})"

  def get_path(self, path: Path) -> 'StateTree[T]' | T:
    node = self
    for k in path.parts:
      if not isinstance(node, StateTree):
        raise ValueError(f"Path {path}: not a leaf: {k} in {node}")
      
      if k not in node:
        raise ValueError(f"Path {path}: {k} not in {node}")    
      
      node = node[k]
    return node

  def get_leaf(self, path: Path) -> T:
    node = self.get_path(path.parent())
    last = path.last_part()
    if last not in node or isinstance(node[last], StateTree):
      raise ValueError(f"Not a leaf: {path}")
    return node[last]

  def get_or_insert_path(self, path: Path) -> 'StateTree[T]':
    node = self
    for k in path.parts:
      if k not in node:
        node[k] = StateTree[T]()
      node = node[k]
    return node

  def update_leaf(self, path: Path, default: T, update_fn) -> T:
    node = self
    for k in path.parent().parts:
      if k not in node:
        node[k] = StateTree[T]()
      node = node[k]

    last = path.last_part()
    if last not in node:
      node[last] = default
    node[last] = update_fn(node[last])
    return node[last]

  def set_leaf(self, path: Path, value: T) -> T:
    return self.update_leaf(path, value, lambda _: value)

  def has_path(self, path: Path) -> bool:
    try:
      self.get_path(path)
      return True
    except ValueError:
      return False

  def flatten(self) -> Dict[str, T]:
    result = {}
    def flatten_dict(prefix: Path, node: StateTree[T]):
      for key, value in node.data.items():
        path = Path(prefix.parts + [key])
        if isinstance(value, StateTree):
          flatten_dict(path, value)
        else:
          result[str(path)] = value
    flatten_dict(Path([]), self)
    return result


class StateLogger(NullLogger):
  def __init__(self):
    self.tree = StateTree[StepValue]()
    self.current_step = 0

  def __getitem__(self, path: str) -> dict | StepValue:
    return self.tree.get_path(Path.from_str(path))

  def __contains__(self, path: str) -> bool:
    return self.tree.has_path(Path.from_str(path))

  def step(self, progress: Progress):
    self.current_step = progress.step

  def log_config(self, config: dict):
    self.tree.data["config"] = config

  def log_values(self, name: str, data: dict[str, float]):
    node = self.tree.get_or_insert_path(Path.from_str(name))
    steps = {k: StepValue(self.current_step, v) for k, v in data.items()}
    node.data.update(steps)
  
  def log_value(self, name: str, value: float):
    self.tree.set_leaf(Path.from_str(name), StepValue(self.current_step, value))

  def flatten(self) -> dict:
    return self.tree.flatten()


class HistoryLogger(NullLogger):

  def __init__(self):
    self.tree = StateTree[list[float]]()

  def __getitem__(self, path: str) -> dict | list[float]:
    return self.tree.get_path(Path.from_str(path))

  def __contains__(self, path: str) -> bool:
    return self.tree.has_path(Path.from_str(path))

  def log_values(self, name: str, data: dict[str, float]):
    node = self.tree.get_or_insert_path(Path.from_str(name))
    for key, value in data.items():
      if key not in node.data:
        node.data[key] = []

      node.data[key].append(value)

  def log_value(self, name: str, value: float):
    self.tree.update_leaf(Path.from_str(name), [], lambda values: values + [value])

  def flatten(self) -> dict:
    return self.tree.flatten()


class LoggerWithState(CompositeLogger):
  def __init__(self, logger: Logger):
    self.state = StateLogger()
    self.logger = logger
    super().__init__(self.state, self.logger)

  def get_value(self, path: str, default: Any = None) -> Any:
    return self.state.get_value(path, default)

  def __getitem__(self, path: str) -> dict | StepValue:
    return self.state[path]

  def __contains__(self, path: str) -> bool:
    return path in self.state
