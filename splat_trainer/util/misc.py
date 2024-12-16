import numpy as np
from taichi_splatting import Rendering
import torch
import torch.nn.functional as F

def strided_indexes(subset:int, total:int):
  if subset > 0:
    stride = max(total // subset, 1)
    return torch.arange(0, total, stride).to(torch.long)
  else:
    return torch.tensor([])


def split_stride(images, stride=0):
  assert stride == 0 or stride > 1, f"val_stride {stride}, must be zero, or greater than 1"

  val_cameras = [camera for i, camera in enumerate(images) if i % stride == 0] if stride > 0 else []
  train_cameras = [camera for camera in images if camera not in val_cameras]

  return train_cameras, val_cameras


def next_multiple(x, multiple):
  return x + multiple - x % multiple


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
  return np.log(x) - np.log(1 - x)


sh0 = 0.282094791773878

@torch.compile
def rgb_to_sh(rgb):
    return (rgb - 0.5) / sh0

@torch.compile
def sh_to_rgb(sh):
    return sh * sh0 + 0.5

@torch.compile
def log_lerp(t, a, b):
  return torch.exp(torch.lerp(torch.log(a), torch.log(b), t))

@torch.compile
def inv_lerp(t, a, b):
  return 1 / (torch.lerp(1/a, 1/b, t))

@torch.compile
def exp_lerp(t, a, b):
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(torch.lerp(torch.exp(a - max_ab), torch.exp(b - max_ab), t))

@torch.compile
def lerp(t, a, b):
  return a + (b - a) * t

@torch.compile
def max_decaying(x, t, decay):
  return x * (1 - decay) + torch.maximum(x, t) * decay


class CudaTimer:
  def __init__(self):
    self.start = torch.cuda.Event(enable_timing = True)
    self.end = torch.cuda.Event(enable_timing = True)

  def __enter__(self):
    self.start.record()

  def __exit__(self, *args):
    self.end.record()

  def ellapsed(self):
    return self.start.elapsed_time(self.end)
  
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

def vis_vector(rendering:Rendering, cluster_labels:torch.Tensor, num_clusters:int):
  idx, vis = rendering.visible
  vector = torch.zeros(num_clusters, device=cluster_labels.device)

  # use clustering to reduce number of points
  vector.scatter_add_(0, cluster_labels[idx], vis)
  return vector

def select_batch(batch_size:int,  
                  view_counts:torch.Tensor, 
                  view_overlaps:torch.Tensor, 
                  temperature:float=1.0,
                  eps:float=1e-6,
                  ) -> torch.Tensor: # (N,) camera indices 
  # select initial camera with probability proportional to view count
  inv_counts = 1 / (view_counts + eps)
  index = torch.multinomial(inv_counts, 1)


  if temperature == 0:
    # select other cameras just with topk
    other_index = torch.topk(view_overlaps[index.squeeze(0)] * inv_counts, k=batch_size - 1, dim=0).indices

  else:
    # select other cameras with probability proportional to view overlap with the selected camera
    # temperature > 1 means more uniform sampling
    # temperature < 1 means closer to top_k sampling
    p = F.softmax(view_overlaps[index.squeeze(0)] / temperature, dim=0) * inv_counts

    # select other cameras with probability proportional to view overlap with the selected camera
    other_index = torch.multinomial(p, batch_size - 1, replacement=False)
  
  

  return torch.cat([index, other_index.squeeze(0)], dim=0)



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
