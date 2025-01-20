
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from dataclasses import dataclass
from typing import Sequence
from beartype import beartype
from beartype.typing import Iterator
import numpy as np
import torch


from splat_trainer.camera_table.camera_table import CameraTable
from splat_trainer.util.pointcloud import PointCloud

@beartype
@dataclass
class ImageView:
  filename:str
  image_idx:int
  image:torch.Tensor

class Dataset(metaclass=ABCMeta):
  @abstractmethod
  def loader(self, idx:torch.Tensor, shuffle:bool=False) -> Sequence[ImageView]:
    raise NotImplementedError
    
  @abstractmethod
  def train(self, shuffle=True) -> Sequence[ImageView]:
    raise NotImplementedError
    
  @abstractmethod
  def val(self) -> Sequence[ImageView]:
    raise NotImplementedError

  
  @abstractmethod
  def camera_table(self) -> CameraTable:
    raise NotImplementedError
  
    
  @abstractmethod
  def pointcloud(self) -> PointCloud:
    raise NotImplementedError

  @abstractmethod
  def load_images(self):
    raise NotImplementedError

  

