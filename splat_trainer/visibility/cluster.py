
from typing import Optional
from beartype import beartype
from matplotlib import pyplot as plt
import numpy as np
from taichi_splatting import Rendering
import torch
import torch.nn.functional as F


class ClusteredVisibility:
  def __init__(self, points:torch.Tensor, num_clusters:int):
    self.view_visibility = {}
    self.num_clusters = num_clusters

    self.point_cluster_labels, self.centroids = kmeans_keops(points, 
                                k = min(num_clusters, points.shape[0]))

  
  @beartype
  def add_view(self, image_idx:int, rendering:Rendering):
    idx, vis = rendering.visible
    self.add_visible(image_idx, idx, vis)

  @beartype
  def add_visible(self, image_idx:int, idx:torch.Tensor, vis:torch.Tensor):
    vector = torch.zeros(self.num_clusters, device=self.point_cluster_labels.device)

    # use clustering to reduce number of points
    vector.scatter_add_(0, self.point_cluster_labels[idx], vis)
    self.view_visibility[image_idx] = vector


  def feature_matrix(self, normalize:bool=True):
    view_visibility = [self.view_visibility[i] for i in sorted(self.view_visibility.keys())]
    visibility = torch.stack(view_visibility, dim=0)  
    if normalize:
      visibility = F.normalize(visibility, dim=1, p=2)
    return visibility
  
  def view_overlaps(self):
    visibility = self.feature_matrix(normalize=True)
    view_overlaps = (visibility @ visibility.T)
    return view_overlaps

def cluster_points(position:torch.Tensor, num_clusters:int, chunk_size:int=128) -> torch.Tensor:
  cluster_indices = torch.randperm(position.shape[0])[:num_clusters]
  # Process clusters in chunks to avoid memory issues with large point clouds
  num_points = position.shape[0]
  
  min_dist = torch.full((num_points,), float('inf'), device=position.device)
  min_cluster = torch.zeros(num_points, dtype=torch.long, device=position.device)

  for start in range(0, num_clusters, chunk_size):
    end = min(start + chunk_size, num_clusters)
    chunk_indices = cluster_indices[start:end]
    
    # Calculate distances for this chunk of clusters
    chunk_dist = torch.cdist(position[chunk_indices], position)
    
    # Update minimum distances and cluster assignments
    chunk_min_dist, chunk_min_cluster = chunk_dist.min(dim=0)
    update_mask = chunk_min_dist < min_dist
    
    min_dist[update_mask] = chunk_min_dist[update_mask]
    min_cluster[update_mask] = chunk_min_cluster[update_mask] + start

  return min_cluster


def kmeans_keops(x:torch.Tensor, k:int=10, iters:int=100) -> tuple[torch.Tensor, torch.Tensor]:
    from pykeops.torch import LazyTensor
    N, D = x.shape  # Number of samples, dimension of the ambient space

    centroids = x[:k, :].clone()  # Simplistic initialization for the centroids

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(centroids.view(1, k, D))  # (1, k, D) centroids

    for i in range(iters):
        # assign points to the closest cluster 
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cluster_labels = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # Update the centroids to the normalized cluster average: 
        # Compute the sum of points per cluster:
        centroids.zero_()
        centroids.scatter_add_(0, cluster_labels[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        cluster_counts = torch.bincount(cluster_labels, minlength=k).type_as(centroids).view(k, 1)
        centroids /= cluster_counts  # in-place division to compute the average

    return cluster_labels, centroids  



def sample_with_temperature(p:torch.Tensor, temperature:float=1.0, n:int=1, weighting:Optional[torch.Tensor]=None):
  # select other cameras with probability proportional to view overlap with the selected camera
  # temperature > 1 means more uniform sampling
  # temperature < 1 means closer to top_k sampling

  if temperature == 0:
    if weighting is not None:
      p = p * weighting
    return torch.topk(p, k=n, dim=0).indices
  else:

    p = F.softmax(p / temperature, dim=0)
    if weighting is not None:
      p = p * weighting
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
  return torch.cat([index, other_index.squeeze(0)], dim=0)


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
    other_index = sample_with_temperature(overlaps, temperature=temperature, n=1, weighting=weighting)

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