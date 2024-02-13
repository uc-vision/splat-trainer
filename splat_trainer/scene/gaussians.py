
from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import KDTree
from tensordict import TensorDict, tensorclass
import torch
import torch.nn.functional as F

from splat_trainer.util.pointcloud import PointCloud
from splat_viewer.gaussians import Workspace

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D
    

@dataclass(frozen=True)
class LearningRates:
  position: float = 0.001
  log_scaling: float = 0.005
  rotation: float = 0.001
  alpha_logit: float = 0.05

  base_sh: float = 0.0025
  higher_sh: float = 0.0002
  

def estimate_scale(pointcloud : PointCloud, num_neighbors:int = 3):
  valid_point_cloud_np = pointcloud.points.cpu().numpy()
  nearest_neighbor_tree = KDTree(valid_point_cloud_np)
  nearest_three_neighbor_distance, _ = nearest_neighbor_tree.query(
      valid_point_cloud_np, k=num_neighbors + 1)
  
  distance = np.mean(nearest_three_neighbor_distance[:, 1:], axis=1)
  return torch.from_numpy(distance).float()


def scale_gradients(packed, sh_features, lr:LearningRates):
  scales = torch.tensor(3 * [lr['position']] +
                        3 * [lr['log_scaling']] +
                        4 * [lr['rotation']] +
                        [lr['alpha_logit']], device=packed.position.device)

  packed.grad *= scales.unsqueeze(0)

  sh_features.grad[..., 0] *= lr['base_sh']
  sh_features.grad[..., 1:] *= lr['higher_sh']


@tensorclass
class PackedPoints:
  gaussians3d: torch.Tensor  # (N, 11)
  sh_feature: torch.Tensor  # (N, (D+1)**2)


class Scene:
  def __init__(self, points: Gaussians3D, lr:LearningRates, device=torch.device('cuda', 0)):

    self.lr = lr
    
    packed = TensorDict(
      gaussians3d=points.packed(),
      sh_feature=points.feature
    ).to(device)

    self.points = ParameterClass.create(packed, 
      learning_rates=dict(gaussians3d = 1.0, sh_feature = 1.0))


  def step(self):

    scale_gradients(self.points.gaussians3d, self.points.sh_feature, self.lr)
    self.points.step()

  def from_pointcloud(pcd:PointCloud, lr:LearningRates, 
          num_neighbors:int = 3, initial_alpha:float = 0.5,
          sh_degree:int = 2):
    
    scales = estimate_scale(pcd, num_neighbors=num_neighbors)
    sh_features = torch.zeros(pcd.points.shape[0], 3, (sh_degree + 1)**2)
    sh_features[:, :, 0] = rgb_to_sh(pcd.colors)

    gaussians = Gaussians3D(
      position=pcd.points,
      log_scaling=torch.log(scales).unsqueeze(1).expand(-1, 3),
      rotation=F.normalize(torch.randn(pcd.points.shape[0], 4), dim=1),
      alpha_logit=torch.full( (pcd.points.shape[0], 1), 
                             fill_value=inverse_sigmoid(initial_alpha)),

      feature=sh_features
    )

    return Scene(gaussians, lr)


def load_model(workspace_path, model_name = None):
    workspace = Workspace.load(workspace_path)
    return workspace.load_model(model_name=model_name)



def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
  return np.log(x / (1 - x))


sh0 = 0.282094791773878

def rgb_to_sh(rgb):
    return (rgb - 0.5) / sh0

def sh_to_rgb(sh):
    return sh * sh0 + 0.5