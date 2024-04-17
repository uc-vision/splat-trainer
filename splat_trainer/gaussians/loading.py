import numpy as np
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from splat_trainer.util.misc import inverse_sigmoid
from splat_trainer.util.pointcloud import PointCloud


from taichi_splatting import Gaussians3D
  

def from_pointcloud(pcd:PointCloud, 
                    initial_scale:float = 0.5,
                    initial_alpha:float = 0.5,
                    num_neighbors:int = 3) -> Gaussians3D:
  scales = estimate_scale(pcd, num_neighbors=num_neighbors) * initial_scale

  return Gaussians3D(
    position=pcd.points,
    log_scaling=torch.log(scales).unsqueeze(1).expand(-1, 3),
    rotation=F.normalize(torch.randn(pcd.points.shape[0], 4), dim=1),
    alpha_logit=torch.full( (pcd.points.shape[0], 1), 
                            fill_value=inverse_sigmoid(initial_alpha)),
    feature=pcd.colors,
    batch_size=(pcd.points.shape[0],)
  )

  
def estimate_scale(pointcloud : PointCloud, num_neighbors:int = 3):
  points = pointcloud.points.cpu().numpy()
  tree = cKDTree(points)

  dist, idx = tree.query(points, k=num_neighbors + 1, workers=-1)
  
  distance = np.mean(dist[:, 1:], axis=1)
  return torch.from_numpy(distance).to(torch.float32)
