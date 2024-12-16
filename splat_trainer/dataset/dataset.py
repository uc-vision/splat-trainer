
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from typing import Sequence
from beartype.typing import Iterator
import numpy as np


from splat_trainer.camera_table.camera_table import CameraTable
from splat_trainer.util.pointcloud import PointCloud


CameraView = namedtuple('CameraView', 
  ('filename', 'image', 'index'))


class Dataset(metaclass=ABCMeta):
  @abstractmethod
  def loader(self, idx:np.ndarray, shuffle:bool=False) -> Sequence[CameraView]:
    raise NotImplementedError
    
  @abstractmethod
  def train(self, shuffle=True) -> Sequence[CameraView]:
    raise NotImplementedError
    
  @abstractmethod
  def val(self) -> Sequence[CameraView]:
    raise NotImplementedError

  
  @abstractmethod
  def camera_table(self) -> CameraTable:
    raise NotImplementedError
  
    
  @abstractmethod
  def pointcloud(self) -> PointCloud:
    raise NotImplementedError


  

