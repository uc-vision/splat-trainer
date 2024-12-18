import numpy as np
import torch

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
  
def cluster_points(position:torch.Tensor, num_clusters:int) -> torch.Tensor:
  cluster_indices = torch.randperm(position.shape[0])[:num_clusters]

  dist = torch.cdist(position[cluster_indices], position)
  return dist.argmin(dim=0)


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
