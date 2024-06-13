
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from numbers import Number
from typing import Tuple
from beartype.typing import Iterator

import torch

from splat_trainer.camera_table.camera_table import CameraInfo, CameraTable
from splat_trainer.util.pointcloud import PointCloud


CameraView = namedtuple('CameraView', 
  ('filename', 'image', 'index'))

class Dataset(metaclass=ABCMeta):
  
  @abstractmethod
  def train(self, shuffle=True) -> Iterator[CameraView]:
    raise NotImplementedError
    
  @abstractmethod
  def val(self) -> Iterator[CameraView]:
    raise NotImplementedError

  
  @abstractmethod
  def camera_table(self) -> CameraTable:
    raise NotImplementedError
  
  @abstractmethod
  def image_sizes(self) -> torch.Tensor:
    raise NotImplementedError

  @abstractmethod
  def depth_range(self) -> Tuple[Number, Number]:
    raise NotImplementedError

  def camera_info(self) -> CameraInfo:
    return CameraInfo(
      camera_table=self.camera_table(),
      image_sizes=self.image_sizes(),
      depth_range=self.depth_range(),
    )
    

  @abstractmethod
  def pointcloud(self) -> PointCloud:
    raise NotImplementedError


  @abstractmethod
  def camera_json(self, camera_table:CameraTable):
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