
from dataclasses import  dataclass
from beartype import beartype
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree
from tensordict import TensorDict, tensorclass
import torch
import torch.nn.functional as F

from splat_trainer.util.pointcloud import PointCloud
from splat_viewer.gaussians import Workspace

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D, RasterConfig, render_gaussians, Rendering
    

@dataclass(kw_only=True, frozen=True)
class SceneConfig:  
  learning_rates : DictConfig
  base_lr:float       = 1.0 
  sh_ratio:float      = 20.0
  
  num_neighbors:int   = 3
  initial_scale:float = 0.5
  initial_alpha:float = 0.5 
  sh_degree:int       = 2



  def load_model(self, workspace_path, model_name = None):
    workspace = Workspace.load(workspace_path)
    gaussians = workspace.load_model(model=model_name).to_gaussians3d()

    return GaussianScene(gaussians, self)
  

  def from_pointcloud(self, pcd:PointCloud):
    
    scales = estimate_scale(pcd, num_neighbors=self.num_neighbors) * self.initial_scale

    sh_feature = torch.zeros(pcd.points.shape[0], 3, (self.sh_degree + 1)**2)
    sh_feature[:, :, 0] = rgb_to_sh(pcd.colors)

    gaussians = Gaussians3D(
      position=pcd.points,
      log_scaling=torch.log(scales).unsqueeze(1).expand(-1, 3),
      rotation=F.normalize(torch.randn(pcd.points.shape[0], 4), dim=1),
      alpha_logit=torch.full( (pcd.points.shape[0], 1), 
                             fill_value=inverse_sigmoid(self.initial_alpha)),

      feature=sh_feature,
      batch_size=(pcd.points.shape[0],)
    )
    return GaussianScene(gaussians, self)

  
def estimate_scale(pointcloud : PointCloud, num_neighbors:int = 3):
  points = pointcloud.points.cpu().numpy()
  tree = cKDTree(points)

  dist, idx = tree.query(points, k=num_neighbors + 1, workers=-1)
  
  distance = np.mean(dist[:, 1:], axis=1)
  return torch.from_numpy(distance).to(torch.float32)


@tensorclass
class Points:
  position    : torch.Tensor  # (N, 3)
  log_scaling : torch.Tensor  # (N, 3)
  rotation    : torch.Tensor  # (N, 4)
  alpha_logit : torch.Tensor  # (N, 1)

  sh_feature  : torch.Tensor  # (N, (D+1)**2)

  split_heuristics : torch.Tensor  # (N, 2) - accumulated split heuristics
  visible : torch.Tensor  # (N, 1) - number of times the point was rasterized
  in_view : torch.Tensor  # (N, 1) - number of times the point was in the view frustum

class GaussianScene:
  def __init__(self, points: Gaussians3D, config: SceneConfig):
    self.config = config

    training_points = TensorDict(dict(
      position=points.position,
      log_scaling=points.log_scaling,
      rotation=points.rotation,
      alpha_logit=points.alpha_logit,

      sh_feature=points.feature,
      split_heuristics=torch.zeros((points.batch_size[0], 2), dtype=torch.float32),

      visible=torch.zeros(points.batch_size, dtype=torch.int16),
      in_view=torch.zeros(points.batch_size, dtype=torch.int16)),

      batch_size = points.batch_size
    )

    self.raster_config = RasterConfig()
    self.learning_rates = OmegaConf.to_container(config.learning_rates)

    self.points = ParameterClass.create(training_points, 
          learning_rates = self.learning_rates)   



  def to(self, device):
    self.points = self.points.to(device)
    return self
  
  @beartype
  def update_learning_rate(self, scene_scale:float, step:int):
    lr = self.learning_rates  
    self.points.set_learning_rate(
      position = lr['position'] * scene_scale)


  def __repr__(self):
    return f"GaussianScene({self.points.position.shape[0]} points)"

  def step(self):
    
    self.points.sh_feature.grad[..., 1:] /= self.config.sh_ratio
    self.points.step()

  def zero_grad(self):
    self.points.zero_grad()

  def gaussians3d(self):
      points = self.points
      return Gaussians3D(
        position    = points.position,
        log_scaling = points.log_scaling,
        rotation    = points.rotation,
        alpha_logit = points.alpha_logit,
        feature     = points.sh_feature,
        batch_size  = points.batch_size
      )

  def render(self, camera_params, compute_radii=False, 
             render_depth=False, compute_split_heuristics=False) -> Rendering:
    
    return render_gaussians(self.gaussians3d(), 
                     use_sh        = True,
                     config        = self.raster_config,
                     camera_params = camera_params,
                     compute_radii = compute_radii,
                     render_depth  = render_depth,
                     compute_split_heuristics=compute_split_heuristics)
  
  def add_training_statistics(self, rendering:Rendering):
    idx = rendering.points_in_view
    self.points.split_heuristics[idx] += rendering.split_heuristics

    visible = rendering.split_heuristics[:, 1] > 0

    self.points.visible[idx] += visible
    self.points.in_view[idx] += 1

    return (visible.sum().item(), idx.shape[0])


  def log_point_statistics(self, logger, step:int):
    h = self.points.split_heuristics / self.points.visible.unsqueeze(1)

    logger.log_histogram("points/log_view_gradient", h[:, 0], step)
    logger.log_histogram("points/log_prune_cost", h[:, 1], step)

    logger.log_histogram("points/visible", self.points.visible / self.points.in_view, step)




def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
  return np.log(x / (1 - x))


sh0 = 0.282094791773878

def rgb_to_sh(rgb):
    return (rgb - 0.5) / sh0

def sh_to_rgb(sh):
    return sh * sh0 + 0.5


def replace_dict(d, **kwargs):
  return {**d, **kwargs}