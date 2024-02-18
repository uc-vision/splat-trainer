
from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
from beartype.typing import Iterator

import torch

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
  def camera_poses(self) -> torch.nn.Module:
    raise NotImplementedError
    
  @abstractmethod
  def camera_projection(self) -> torch.Tensor:
    raise NotImplementedError

  @abstractmethod
  def pointcloud(self) -> PointCloud:
    raise NotImplementedError

