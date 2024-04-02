
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from beartype.typing import Iterator

import torch

from splat_trainer.camera_table.camera_table import CameraTable
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
  def pointcloud(self) -> PointCloud:
    raise NotImplementedError
