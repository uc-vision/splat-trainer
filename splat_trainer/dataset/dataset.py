
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from numbers import Number
from typing import List, Tuple
from beartype.typing import Iterator

import numpy as np
import torch

from splat_trainer.camera_table.camera_table import ViewInfo, ViewTable
from splat_trainer.util.pointcloud import PointCloud


CameraView = namedtuple('CameraView', 
  ('filename', 'image', 'index'))

@dataclass
class CameraProjection:
  projection:np.array # Nx4 [fx, fy, cx, cy]
  image_size:Tuple[int, int]


class Dataset(metaclass=ABCMeta):
  
  @abstractmethod
  def train(self, shuffle=True) -> Iterator[CameraView]:
    raise NotImplementedError
    
  @abstractmethod
  def val(self) -> Iterator[CameraView]:
    raise NotImplementedError

  
  @abstractmethod
  def view_table(self) -> ViewTable:
    raise NotImplementedError
  
  @abstractmethod
  def image_sizes(self) -> torch.Tensor:
    raise NotImplementedError

  @abstractmethod
  def depth_range(self) -> Tuple[Number, Number]:
    raise NotImplementedError

  def view_info(self) -> ViewInfo:
    return ViewInfo(
      camera_table=self.view_table(),
      image_sizes=self.image_sizes(),
      depth_range=self.depth_range(),
    )
    
  @abstractmethod
  def unique_projections(self) -> List[CameraProjection]:
    raise NotImplementedError


  @abstractmethod
  def pointcloud(self) -> PointCloud:
    raise NotImplementedError


  @abstractmethod
  def camera_json(self, camera_table:ViewTable):
    """ Export camera information as a json file. List of dictionaries with the following fields:
    - id: int
    - img_name: str
    - width: int
    - height: int
    - position: List[float]  (3)
    - rotation: List[List[float]] (3x3)
    - fx: float
    - fy: float
    - cx: float
    - cy: float
    """
    
    raise NotImplementedError