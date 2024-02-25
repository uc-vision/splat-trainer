
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from tensordict import TensorDict, tensorclass
import torch
import torch.nn.functional as F

from splat_trainer.util.pointcloud import PointCloud
from splat_viewer.gaussians import Workspace

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D, RasterConfig, render_gaussians
    


@dataclass(frozen=True)
class LearningRates:
  position: float = 0.001
  log_scaling: float = 0.005
  rotation: float = 0.001
  alpha_logit: float = 0.05

  base_sh: float = 0.0025
  higher_sh: float = 0.0002
  

def estimate_scale(pointcloud : PointCloud, num_neighbors:int = 3):
  points = pointcloud.points.cpu().numpy()
  tree = cKDTree(points)

  dist, idx = tree.query(points, k=num_neighbors + 1, workers=-1)
  
  distance = np.mean(dist[:, 1:], axis=1)
  return torch.from_numpy(distance).float()


def scale_gradients(packed, sh_feature, lr:LearningRates):
  scales = torch.tensor(3 * [lr.position] +
                        3 * [lr.log_scaling] +
                        4 * [lr.rotation] +
                        [lr.alpha_logit], device=packed.device)

  packed.grad *= scales.unsqueeze(0)

  sh_feature.grad[..., 0] *= lr.base_sh
  sh_feature.grad[..., 1:] *= lr.higher_sh


@tensorclass
class PackedPoints:
  gaussians3d: torch.Tensor  # (N, 11)
  sh_feature: torch.Tensor  # (N, (D+1)**2)

  split_heuristics : torch.Tensor  # (N, 2) - accumulated split heuristics
  visible : torch.Tensor  # (N, 1) - number of times the point was rasterized
  in_view : torch.Tensor  # (N, 1) - number of times the point was in the view frustum

class GaussianScene:
  def __init__(self, points: Gaussians3D, lr:LearningRates):

    self.lr = lr
    
    packed = TensorDict(dict(
      gaussians3d=points.packed(),
      sh_feature=points.feature,

      split_heuristics=torch.zeros(points.batch_size, 2, dtype=torch.float32),

      visible=torch.zeros(points.batch_size, dtype=torch.int16)),
      in_view=torch.zeros(points.batch_size, dtype=torch.int16),

      batch_size = points.batch_size
    )

    self.raster_config = RasterConfig()

    self.points = ParameterClass.create(packed, 
      learning_rates=dict(gaussians3d = 1.0, sh_feature = 1.0))

  def to(self, device):
    self.points = self.points.to(device)
    
    return self
  

  def __repr__(self):
    return f"GaussianScene({self.points.gaussians3d.shape[0]} points)"

  def step(self):
    
    scale_gradients(self.points.gaussians3d, self.points.sh_feature, self.lr)
    self.points.step()

  def zero_grad(self):
    self.points.zero_grad()

  def render(self, camera_params, compute_radii=False, 
             render_depth=False, compute_split_heuristics=False):
    
    return render_gaussians(self.points.gaussians3d, 
                     features=self.points.sh_feature,
                     use_sh=True,
                     config=self.raster_config,
                     camera_params=camera_params,
                     compute_radii=compute_radii,
                     render_depth=render_depth,
                     compute_split_heuristics=compute_split_heuristics)
  
  def add_training_statistics(self, rendering):
    idx = rendering.point_indexes
    self.points.split_heuristics[idx] += rendering.split_heuristics

    visible = rendering.split_heuristics[:, 1] > 0
    self.points.rendered[idx] += visible
    self.points.in_view[idx] += 1

    return (visible.sum(), idx.shape[0])


  def log_point_statistics(self, logger):
    logger.log_histogram("view_gradient", self.points.split_heuristics[:, 0])
    logger.log_histogram("prune_cost", self.points.split_heuristics[:, 1])


    logger.log_histogram("visible", self.points.visible)
    logger.log_histogram("in_view", self.points.in_view)



  @staticmethod
  def load_model(workspace_path, model_name = None, lr:LearningRates = LearningRates()):
    workspace = Workspace.load(workspace_path)
    gaussians = workspace.load_model(model=model_name).to_gaussians3d()

    return GaussianScene(gaussians, lr)
  

  @staticmethod
  def from_pointcloud(pcd:PointCloud, lr:LearningRates, 
          num_neighbors:int = 3, 
          initial_scale:float = 0.5,
          initial_alpha:float = 0.5,
          sh_degree:int = 2):
    
    scales = estimate_scale(pcd, num_neighbors=num_neighbors) * initial_scale


    sh_feature = torch.zeros(pcd.points.shape[0], 3, (sh_degree + 1)**2)
    sh_feature[:, :, 0] = rgb_to_sh(pcd.colors)

    gaussians = Gaussians3D(
      position=pcd.points,
      log_scaling=torch.log(scales).unsqueeze(1).expand(-1, 3),
      rotation=F.normalize(torch.randn(pcd.points.shape[0], 4), dim=1),
      alpha_logit=torch.full( (pcd.points.shape[0], 1), 
                             fill_value=inverse_sigmoid(initial_alpha)),

      feature=sh_feature,
      batch_size=(pcd.points.shape[0],)
    )
    return GaussianScene(gaussians, lr)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
  return np.log(x / (1 - x))


sh0 = 0.282094791773878

def rgb_to_sh(rgb):
    return (rgb - 0.5) / sh0

def sh_to_rgb(sh):
    return sh * sh0 + 0.5