from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

from splat_trainer.camera_table.camera_table import CameraTable, Label
from splat_trainer.config import Progress, VaryingInt, eval_varying
from splat_trainer.visibility.cluster import ViewClustering, sample_with_temperature

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
class BatchOverlapSamplerConfig(ViewSelectionConfig): 
  batch_size:VaryingInt
  overlap_temperature:float

  def create(self, camera_table:CameraTable) -> 'BatchOverlapSampler':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return BatchOverlapSampler(self, train_idx)

  def from_state_dict(self, state_dict:dict, camera_table:CameraTable) -> 'BatchOverlapSampler':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return BatchOverlapSampler(self, train_idx, state_dict['view_counts'])
  

class BatchOverlapSampler(ViewSelection):
  """  Sampler that selects a batch of images to train on in one gradient step.
        The batch is selected by sampling views from the view clustering according to 
        similarity of feature visibility."""
  def __init__(self, config:BatchOverlapSamplerConfig, train_idx:torch.Tensor, view_counts:Optional[torch.Tensor]=None):
    self.train_idx = train_idx
    self.config = config
    self.used_mask = torch.zeros_like(train_idx, dtype=torch.bool)

    if view_counts is not None:
      self.view_counts = view_counts
    else:
      self.view_counts = torch.zeros(len(train_idx), device=train_idx.device)

  def state_dict(self) -> dict:
    return dict(view_counts=self.view_counts)

  def select_images(self, view_clustering:ViewClustering, progress:Progress) -> torch.Tensor:
    """ Select a batch of images to train on in one gradient step.    """
    batch_size = eval_varying(self.config.batch_size, progress)

    # Reset mask if all used
    if self.used_mask.all():
      self.used_mask.fill_(False)

    weighting = F.normalize(1 / (self.view_counts + 1), p=1, dim=0)
    weighting[self.used_mask] = 0  # Zero out used views
    
    batch_idx = view_clustering.sample_batch(weighting, batch_size, self.config.overlap_temperature)
    
    # Mark selected views as used
    self.used_mask[batch_idx] = True
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

  
@dataclass(kw_only=True, frozen=True)
class TargetOverlapConfig(ViewSelectionConfig): 
  batch_size:VaryingInt = 1
  overlap_temperature:float = 0.5
  history_size:int = 2
  target_overlap:float = 0.5

  def create(self, camera_table:CameraTable) -> 'TargetOverlap':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return TargetOverlap(self, train_idx, None, None)

  def from_state_dict(self, state_dict:dict, camera_table:CameraTable) -> 'TargetOverlap':
    train_idx = camera_table.cameras.has_label(Label.Training)
    return TargetOverlap(self, train_idx, state_dict['history_idx'], state_dict['available_mask'])
  

class TargetOverlap(ViewSelection):
  """  Sampler which attemps to find views near in view overlap compared to a short history buffer."""
  def __init__(self, config:TargetOverlapConfig, train_idx:torch.Tensor, 
               history_idx:Optional[torch.Tensor]=None, available_mask:Optional[torch.Tensor]=None):
    self.train_idx = train_idx
    self.config = config

    if available_mask is None:
      available_mask = torch.ones(len(train_idx), device=train_idx.device, dtype=torch.bool)
    self.available_mask = available_mask

    if history_idx is None:
      history_idx = torch.randperm(len(train_idx), device=train_idx.device)[:self.config.history_size]

    self.history_idx = history_idx
  
  def state_dict(self) -> dict:
    return dict(history_idx=self.history_idx, available_mask=self.available_mask)

  
  def select_images(self, view_clustering:ViewClustering, progress:Progress) -> torch.Tensor:
    batch_size = eval_varying(self.config.batch_size, progress)

    if self.available_mask.sum() < batch_size:
      self.available_mask.fill_(True)

    vis = F.normalize(view_clustering.normalized_visibility[self.history_idx].sum(0), dim=0, p=2)
    overlaps = view_clustering.overlaps_with(vis)
    
    overlaps = 1 - (self.config.target_overlap - overlaps[self.available_mask]).pow(2)

    available_idx = self.train_idx[self.available_mask]
    idx = sample_with_temperature(overlaps + 1e-6, temperature=self.config.overlap_temperature, n=batch_size)
    batch_idx = available_idx[idx]
    
    # Mark as used
    self.available_mask[batch_idx] = False
    
    # update history buffer
    self.history_idx = torch.cat([batch_idx, self.history_idx[batch_idx:]])
    
    return batch_idx
    