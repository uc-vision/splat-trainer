
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence
from beartype import beartype
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
  
  @property
  @abstractmethod
  def scene_transform(self) -> tuple[torch.Tensor, float]:
    """ Return a tuple of the translation back to the original scene center and the scale factor """
    raise NotImplementedError
  
  @abstractmethod
  def train(self, shuffle=True) -> Sequence[ImageView]:
    raise NotImplementedError
    
  @abstractmethod
  def val(self) -> Sequence[ImageView]:
    raise NotImplementedError

  @property
  @abstractmethod
  def camera_table(self) -> CameraTable:
    raise NotImplementedError
  
    
  @abstractmethod
  def pointcloud(self) -> Optional[PointCloud]:
    raise NotImplementedError

  @abstractmethod
  def load_images(self):
    raise NotImplementedError

  

