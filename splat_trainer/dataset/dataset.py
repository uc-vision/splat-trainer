
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from numbers import Number
from typing import List, Tuple
from beartype.typing import Iterator

import numpy as np
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


  

