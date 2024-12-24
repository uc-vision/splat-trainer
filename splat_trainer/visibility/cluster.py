
from dataclasses import dataclass
from enum import Enum, auto
from functools import cached_property
from typing import List, Optional
from beartype import beartype
from matplotlib import pyplot as plt
import numpy as np
from taichi_splatting import Rendering
import torch
import torch.nn.functional as F
from pykeops.torch import LazyTensor


class PointClusters:
  def __init__(self, points:torch.Tensor, num_clusters:int):
    self.view_visibility = {}
    self.num_clusters = num_clusters

    self.point_labels, self.centroids = kmeans(points, 
                                k = min(num_clusters, points.shape[0]))


  def assign_clusters(self, points:torch.Tensor) -> torch.Tensor:
    return assign_clusters(points, self.centroids)

  @beartype
  def view_features(self, point_idx:torch.Tensor, 
                    point_vis:torch.Tensor, vis_threshold:float=0.01) -> torch.Tensor:
    vector = torch.zeros(self.num_clusters, device=self.point_labels.device)

    # filter points with low visibility
    mask = point_vis > vis_threshold

    point_idx = point_idx[mask]
    point_vis = point_vis[mask]

    # use clustering to reduce number of points
    vector.scatter_add_(0, self.point_labels[point_idx], point_vis)
    return vector
  
  @beartype
  def rendering_features(self, rendering:Rendering) -> torch.Tensor:
    idx, vis = rendering.visible
    return self.view_features(idx, vis)

class ViewClustering:
  def __init__(self, point_clusters:PointClusters, cluster_visibility:torch.Tensor, metric:str='cosine'):
    assert metric in ['cosine', 'euclidean'], f"Unknown metric: {metric}, expected 'cosine' or 'euclidean'"

    self.point_clusters = point_clusters
    self.cluster_visibility = cluster_visibility
    self.metric = metric

  @cached_property  
  def view_overlaps(self) -> torch.Tensor:
    # normalize features by cluster
    self.cluster_visibility = F.normalize(self.cluster_visibility, dim=0, p=2)

    # compute view overlaps
    if self.metric == 'cosine':
      self.view_overlaps = (self.cluster_visibility @ self.cluster_visibility.T)
    elif self.metric == 'euclidean':
      self.view_overlaps = torch.cdist(self.cluster_visibility, self.cluster_visibility, p=2)


  def select_batch(self, batch_size:int, temperature:float=1.0, weighting:Optional[torch.Tensor]=None) -> torch.Tensor:
    return select_batch(batch_size, self.view_overlaps, temperature, weighting)
  
  def select_batch_grouped(self, batch_size:int, temperature:float=1.0, weighting:Optional[torch.Tensor]=None) -> torch.Tensor:
    return select_batch_grouped(batch_size, self.view_overlaps, temperature, weighting)
  
  def visible_points(self, batch_indices:torch.Tensor) -> torch.Tensor:
    cluster_visibility = self.cluster_visibility[batch_indices].sum(dim=0)
    visible_mask = cluster_visibility[self.point_clusters.point_labels] > 0
    
    return torch.nonzero(visible_mask[self.point_clusters.point_labels]).squeeze(1)

  def state_dict(self):
    return {
      'point_clusters': self.point_clusters.state_dict(),
      'cluster_visibility': self.cluster_visibility,
      'metric': self.metric
    }
  
  @classmethod
  def from_state_dict(cls, state_dict):
    point_clusters = PointClusters.from_state_dict(state_dict['point_clusters'])
    features = state_dict['features']
    metric = state_dict['metric']
    return cls(point_clusters, features, metric)


def assign_clusters(x:torch.Tensor, centroids:torch.Tensor) -> torch.Tensor:
  assert x.shape[1] == centroids.shape[1], f"Expected x and centroids to have the same number of features, got: {x.shape[1]} and {centroids.shape[1]}"

  x_i = LazyTensor(x.view(x.shape[0], 1, x.shape[1]))  # (N, 1, D) samples
  c_j = LazyTensor(centroids.view(1, centroids.shape[0], centroids.shape[1]))  # (1, k, D) centroids

  D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
  return D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster


def kmeans_iter(x:torch.Tensor, centroids:torch.Tensor, iters:int=100) -> tuple[torch.Tensor, torch.Tensor]:
    N, D = x.shape  # Number of samples, dimension
    k = centroids.shape[0]

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(centroids.view(1, k, D))  # (1, k, D) centroids


    for i in range(iters):
        # assign points to the closest cluster 
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # pairwise squared distances
        cluster_labels = D_ij.argmin(dim=1).long().view(-1)  # assign points to clusters by minimum distance

        # update centroids
        centroids.zero_()
        centroids.scatter_add_(0, cluster_labels[:, None].repeat(1, D), x)

        cluster_counts = torch.bincount(cluster_labels, minlength=k).type_as(centroids).view(k, 1)
        centroids /= cluster_counts  

    return cluster_labels, centroids  



class InitMethod(Enum):
  Random = auto()
  Farthest = auto()

@beartype
def kmeans(x:torch.Tensor, k:int=10, iters:int=100, init_method:InitMethod=InitMethod.Random) -> tuple[torch.Tensor, torch.Tensor]:
  if init_method == InitMethod.Random:
    centroid_idx = torch.randperm(x.shape[0])[:k]
    centroids = x[centroid_idx]

  elif init_method == InitMethod.Farthest:
    import torch_fpsample
    centroids, _ = torch_fpsample.sample(x, k)

  return kmeans_iter(x, centroids, iters)




def sample_with_temperature(p:torch.Tensor, temperature:float=1.0, n:int=1, weighting:Optional[torch.Tensor]=None):
  # select other cameras with probability proportional to view overlap with the selected camera
  # temperature > 1 means more uniform sampling
  # temperature < 1 means closer to top_k sampling

  if temperature == 0:
    if weighting is not None:
      p = p * weighting
    return torch.topk(p, k=n, dim=0).indices
  else:

    p = F.softmax(p.log() / temperature, dim=0)
    if weighting is not None:
      p = F.normalize(p * weighting, dim=0, p=1)
    return torch.multinomial(p, n, replacement=False)

def select_batch(batch_size:int,  
                  view_overlaps:torch.Tensor, 
                  temperature:float=1.0,
                  weighting:Optional[torch.Tensor]=None,
                  ) -> torch.Tensor: # (N,) camera indices 
  # select initial camera with probability proportional to weighting
  if weighting is not None:
    index = torch.multinomial(weighting, 1)
  else:
    index = torch.randint(0, view_overlaps.shape[0], (1,))


  probs = view_overlaps[index.squeeze(0)] 
  probs[index.squeeze(0)] = 0 

  other_index = sample_with_temperature(probs, temperature=temperature, n=batch_size - 1, weighting=weighting)
  return torch.cat([index, other_index], dim=0)


def select_batch_grouped(batch_size:int,  
                  view_overlaps:torch.Tensor, 
                  temperature:float=1.0,
                  weighting:Optional[torch.Tensor]=None,
                  ) -> torch.Tensor: # (N,) camera indices 
  # select initial camera with probability proportional to weighting
  if weighting is not None:
    index = torch.multinomial(weighting, 1)
  else:
    index = torch.randint(0, view_overlaps.shape[0], (1,), device=view_overlaps.device)

  overlaps = view_overlaps[index.squeeze(0)].clone()

  selected = index
  # select other cameras incrementally proportional to overlap with already selected cameras
  for i in range(batch_size - 1):
    overlaps[selected] = 0
    other_index = sample_with_temperature(overlaps, temperature=temperature, n=1)

    overlaps += view_overlaps[other_index.squeeze(0)]
    selected = torch.cat([selected, other_index], dim=0)

  return selected



def sinkhorn(matrix: torch.Tensor, num_iter: int, epsilon: float = 1e-8) -> torch.Tensor:
    """Applies Sinkhorn-Knopp algorithm to make matrix doubly stochastic.
    
    Args:
        matrix: Input matrix to normalize
        num_iter: Number of normalization iterations
        epsilon: Small value added for numerical stability
    """
    # matrix = matrix.exp()
    for _ in range(num_iter):
        # Symmetrize
        matrix = (matrix + matrix.T) / 2
        
        # Row normalization
        row_sums = matrix.sum(dim=1, keepdim=True)
        matrix = matrix / (row_sums + epsilon)
        
        # Column normalization  
        col_sums = matrix.sum(dim=0, keepdim=True)
        matrix = matrix / (col_sums + epsilon)
        
    return matrix


def plot_visibility(rendering:Rendering, min_vis:int = 12):
  n = rendering.visible_indices.shape[0]
  print(", ".join([f"vis < {10**-x}: {((rendering.point_visibility < 10**-x).sum() * 100 / n):.1f}%" for x in range(min_vis)]))
  
  # Create and show visibility histogram
  plt.figure(figsize=(10, 7))
  plt.hist(rendering.point_visibility[rendering.point_visibility > 0].cpu().numpy(), 
            bins=np.logspace(np.log10(10**-min_vis), np.log10(100), 200))
  plt.xlabel("Sum transmittance")

  plt.xscale('log')
  plt.ylabel("Count")

  plt.show() 
  plt.close() 