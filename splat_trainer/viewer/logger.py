from pprint import pprint
from typing import Any, Dict
import torch
from splat_trainer.logger.histogram import Histogram
from splat_trainer.logger.logger import Logger
from splat_trainer.util.pointcloud import PointCloud

import markdown_strings as ms


def insert_dicts(d:dict, keys:list[str], value:float):
  for key in keys[:-1]:
    if key not in d:
      d[key] = {}
    d = d[key]
    
  d[keys[-1]] = value


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    elif isinstance(value, int):
        return f"{value:d}"
    else:
      return f"{value}"

class ViewLogger(Logger):
  def __init__(self):
    self.sections = {}

  def log_config(self, config:dict):
    self.sections["config"] = ms.code_block(pprint.pformat(config))
  
  def log_evaluations(self, name:str,  data:Dict[str, Dict], step:int):
    pass

  def log_image(self, name:str, image:torch.Tensor, step:int, caption:str | None = None):
    pass

  def log_cloud(self, name_str, points:PointCloud, step:int):
    pass

  def log_values(self, name:str, data:dict, step:int):
    for key, value in data.items():
      self.log_value(f"{name}/{key}", format_value(value))

  def log_value(self, name:str, value:float, step:int):
    path = name.split("/")
    insert_dicts(self.sections, path, format_value(value))
      
      
  def log_histogram(self, name:str, values:torch.Tensor | Histogram, step:int):
    pass

  def close(self):
    pass