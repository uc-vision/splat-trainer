from abc import ABCMeta, abstractmethod
from beartype.typing import  List
import torch

from splat_trainer.logger.histogram import Histogram
from splat_trainer.util.pointcloud import PointCloud

# os.environ["WANDB_SILENT"] = "true"

class Logger(metaclass=ABCMeta):

  @abstractmethod
  def log_evaluations(self, data:List[dict], step:int):
    raise NotImplementedError

  
  @abstractmethod
  def log_image(self, name:str, image:torch.Tensor, step:int, caption:str | None = None):
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
  
  

