from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from splat_trainer.camera_table.camera_table import CameraTable, Label
from splat_trainer.config import Progress, VaryingInt, eval_varying
from splat_trainer.visibility.cluster import ViewClustering

import torch
from torch.nn import functional as F


class ViewSelectionConfig(metaclass=ABCMeta):
  @abstractmethod
  def create(self, camera_table:CameraTable) -> 'ViewSelection':
    raise NotImplementedError
  
  @abstractmethod
  def from_state_dict(self, state_dict:dict, camera_table:CameraTable) -> 'ViewSelection':
    raise NotImplementedError

class ViewSelection(metaclass=ABCMeta):
  @abstractmethod
  def select_images(self, view_clustering:ViewClustering, progress:Progress) -> torch.Tensor:
    raise NotImplementedError
  
  @abstractmethod
  def state_dict(self) -> dict:
    raise NotImplementedError

  
@dataclass(kw_only=True, frozen=True)
class OverlapSamplerConfig(ViewSelectionConfig): 
  batch_size:VaryingInt
  overlap_temperature:float

  def create(self, camera_table:CameraTable) -> 'OverlapSampler':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return OverlapSampler(self, train_idx)

  def from_state_dict(self, state_dict:dict, camera_table:CameraTable) -> 'OverlapSampler':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return OverlapSampler(self, train_idx, state_dict['view_counts'])
  

class OverlapSampler(ViewSelection):
  """  Sampler that selects a batch of images to train on in one gradient step.
        The batch is selected by sampling views from the view clustering according to 
        similarity of feature visibility."""
  def __init__(self, config:OverlapSamplerConfig, train_idx:torch.Tensor, view_counts:Optional[torch.Tensor]=None):
    self.train_idx = train_idx
    self.config = config

    if view_counts is not None:
      self.view_counts = view_counts
    else:
      self.view_counts = torch.zeros(len(train_idx), device=train_idx.device)

  def state_dict(self) -> dict:
    return dict(view_counts=self.view_counts)

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

  def create(self, camera_table:CameraTable, next:Optional[torch.Tensor]=None) -> 'RandomSampler':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return RandomSampler(self, train_idx, next)

  def from_state_dict(self, state_dict:dict, camera_table:CameraTable) -> 'RandomSampler':
    return self.create(camera_table, state_dict['next'])


class RandomSampler(ViewSelection):
  def __init__(self, config:RandomSamplerConfig, train_idx:torch.Tensor, next:Optional[torch.Tensor]=None):
    self.train_idx = train_idx

    if next is None:
      next = self.train_idx[torch.randperm(len(train_idx))]
    
    self.next = next
    self.config = config

  def state_dict(self) -> dict:
    return dict(next=self.next)

  def select_images(self, _: ViewClustering, progress: Progress) -> torch.Tensor:
    batch_size = eval_varying(self.config.batch_size, progress)

    if self.next.shape[0] < batch_size:
      perm = torch.randperm(len(self.train_idx))
      self.next = self.train_idx[perm]
      
    indices = self.next[:batch_size]
    self.next = self.next[batch_size:]
    return indices