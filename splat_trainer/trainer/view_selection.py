from abc import ABCMeta
from dataclasses import dataclass
from typing import List, Tuple
from pyparsing import abstractmethod
from splat_trainer.camera_table.camera_table import CameraTable, Label
from splat_trainer.config import Progress, VaryingInt, eval_varying
from splat_trainer.visibility.cluster import ViewClustering


from taichi_splatting.perspective import CameraParams

import torch
from torch.nn import functional as F



class ViewSelectionConfig(metaclass=ABCMeta):
  @abstractmethod
  def create(self, camera_table:CameraTable) -> 'ViewSelection':
    raise NotImplementedError

class ViewSelection(metaclass=ABCMeta):
  @abstractmethod
  def select_images(self, view_clustering:ViewClustering, progress:Progress) -> torch.Tensor:
    raise NotImplementedError

  
@dataclass(kw_only=True, frozen=True)
class OverlapSamplerConfig(ViewSelectionConfig): 
  batch_size:VaryingInt

  overlap_threshold:float
  overlap_temperature:float

  def create(self, camera_table:CameraTable) -> 'OverlapSampler':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return OverlapSampler(self, train_idx)

class OverlapSampler(ViewSelection):
  """  Sampler that selects a batch of images to train on in one gradient step.
        The batch is selected by sampling views from the view clustering according to 
        similarity of feature visibility."""
  def __init__(self, config:OverlapSamplerConfig, train_idx:torch.Tensor):
    self.train_idx = train_idx
    self.config = config
    self.view_counts = torch.zeros(len(train_idx), dtype=torch.int32)


  def select_images(self, view_clustering:ViewClustering, progress:Progress) -> torch.Tensor:
    """ Select a batch of images to train on in one gradient step.    """
    batch_size = eval_varying(self.config.batch_size, progress)

    weighting = F.normalize(1 / (self.view_counts + 1), p=1, dim=0)
    batch_idx = view_clustering.sample_batch(weighting, 
              batch_size, self.config.overlap_temperature)
    
    self.view_counts[batch_idx] += 1
    return batch_idx

@dataclass(kw_only=True, frozen=True)
class RandomSamplerConfig(ViewSelectionConfig):
  batch_size:VaryingInt

  def create(self, camera_table:CameraTable) -> 'RandomSampler':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return RandomSampler(self, train_idx)


class RandomSampler(ViewSelection):

  def __init__(self, config:RandomSamplerConfig, train_idx:torch.Tensor):
    self.train_idx = train_idx

    self.next = self.train_idx[torch.randperm(len(train_idx))]
    self.config = config

  def select_images(self, _: ViewClustering, progress: Progress) -> torch.Tensor:
    batch_size = eval_varying(self.config.batch_size, progress)

    if self.next.shape[0] < batch_size:
      perm = torch.randperm(len(self.train_idx))
      self.next = self.train_idx[perm]
      
    indices = self.next[:batch_size]
    self.next = self.next[batch_size:]
    return indices