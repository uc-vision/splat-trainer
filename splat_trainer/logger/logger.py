from abc import ABCMeta, abstractmethod
from typing import Dict
from beartype.typing import  List
import torch

from splat_trainer.logger.histogram import Histogram
from splat_trainer.util.pointcloud import PointCloud

# os.environ["WANDB_SILENT"] = "true"

class Logger(metaclass=ABCMeta):

  @abstractmethod
  def log_evaluations(self, name:str, data:Dict[str, Dict], step:int):
    raise NotImplementedError

  @abstractmethod
  def log_config(self, config:dict):
    raise NotImplementedError
  
  @abstractmethod
  def log_image(self, name:str, image:torch.Tensor, step:int, compressed:bool = True, caption:str | None = None):
    raise NotImplementedError
  
  
  @abstractmethod
  def log_cloud(self, name_str, points:PointCloud, step:int):
    raise NotImplementedError
  
  @abstractmethod
  def log_values(self, name:str, data:dict, step:int):
    raise NotImplementedError
  
  @abstractmethod
  def log_value(self, name:str, value:float, step:int):
    raise NotImplementedError
  
  @abstractmethod
  def log_histogram(self, name:str, values:torch.Tensor | Histogram, step:int):
    raise NotImplementedError

  @abstractmethod
  def close(self):
    raise NotImplementedError
  
  

class CompositeLogger(Logger):
  def __init__(self, *loggers:Logger):
    self.loggers = list(loggers)

  def add_logger(self, logger:Logger):
    self.loggers.append(logger)

  def log_evaluations(self, name:str, data:Dict[str, Dict], step:int):
    for logger in self.loggers:
      logger.log_evaluations(name, data, step)

  def log_config(self, config:dict):
    for logger in self.loggers:
      logger.log_config(config)

  def log_image(self, name:str, image:torch.Tensor, step:int, compressed:bool = True, caption:str | None = None):
    for logger in self.loggers:
      logger.log_image(name, image, step=step, compressed=compressed, caption=caption)

  def log_cloud(self, name_str, points:PointCloud, step:int):
    for logger in self.loggers:
      logger.log_cloud(name_str, points, step)

  def log_values(self, name:str, data:dict, step:int):
    for logger in self.loggers:
      logger.log_values(name, data, step)

  def log_value(self, name:str, value:float, step:int):
    for logger in self.loggers:
      logger.log_value(name, value, step)

  def log_histogram(self, name:str, values:torch.Tensor | Histogram, step:int):
    for logger in self.loggers:
      logger.log_histogram(name, values, step)

  def close(self):
    for logger in self.loggers:
      logger.close()
      

class NullLogger(Logger):
  def __init__(self):
    pass

  def log_config(self, config:dict):
    pass
  
  def log_evaluations(self, name:str,  data:Dict[str, Dict], step:int):
    pass

  def log_image(self, name:str, image:torch.Tensor, step:int, compressed:bool = True, caption:str | None = None):
    pass

  def log_cloud(self, name_str, points:PointCloud, step:int):
    pass

  def log_values(self, name:str, data:dict, step:int):
    pass

  def log_value(self, name:str, value:float, step:int):
    pass

  def log_histogram(self, name:str, values:torch.Tensor | Histogram, step:int):
    pass

  def close(self):
    pass