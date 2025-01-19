from beartype import beartype
import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from splat_trainer.util.misc import inverse_sigmoid
from splat_trainer.util.pointcloud import PointCloud


from taichi_splatting import Gaussians3D
from pykeops.torch import LazyTensor
  

def from_pointcloud(pcd:PointCloud, 
                    initial_scale:float = 0.5,
                    initial_alpha:float = 0.5,
                    num_neighbors:int = 3) -> Gaussians3D:
  scales = estimate_scale(pcd, num_neighbors=num_neighbors) * initial_scale
  return from_scaled_pointcloud(pcd, scales, initial_alpha)

def from_scaled_pointcloud(pcd:PointCloud, scales:torch.Tensor, 
                    initial_alpha:float = 0.5) -> Gaussians3D:

  return Gaussians3D(
    position=pcd.points,
    log_scaling=torch.log(scales).unsqueeze(1).expand(-1, 3),
    rotation=F.normalize(torch.randn(pcd.points.shape[0], 4), dim=1),
    alpha_logit=torch.full( (pcd.points.shape[0], 1), 
                            fill_value=inverse_sigmoid(initial_alpha - 1e-4)),
    feature=pcd.colors,
    batch_size=(pcd.points.shape[0],)
  )

def to_pointcloud(gaussians:Gaussians3D) -> PointCloud:
  return PointCloud(
    points=gaussians.position,
    colors=gaussians.feature
  )
  



def estimate_scale(pointcloud : PointCloud, num_neighbors:int = 3):
  """ Give the mean distance to num_neighbors nearest neighbors """
  points = pointcloud.points
  N, D = points.shape

  x_i = LazyTensor(points.view(N, 1, D))  # (N, 1, D) samples
  x_j = LazyTensor(points.view(1, N, D))  # (1, N, D) samples

  # Compute pairwise squared distances
  D_ij = ((x_i - x_j) ** 2).sum(-1)  # (N, N) symbolic squared distances
  
  # Get k nearest neighbors distances (k = num_neighbors)
  # Sort distances and take mean of k nearest (excluding self)
  knn_dists = D_ij.Kmin(num_neighbors + 1, dim=1)  # +1 to account for self
  scales = knn_dists[:, 1:].sqrt().mean(dim=1)  # Skip first (self) distance
  
  return scales
