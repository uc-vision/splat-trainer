
from dataclasses import  dataclass
from beartype import beartype
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial import cKDTree
from tensordict import TensorDict
import torch
import torch.nn.functional as F
from splat_trainer.scene.split import split_gaussians
from splat_trainer.scheduler import Scheduler, Uniform
from splat_trainer.util.misc import inverse_sigmoid, rgb_to_sh

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

  scheduler:Scheduler = Uniform()


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




class GaussianScene:
  def __init__(self, points: Gaussians3D, config: SceneConfig):
    self.config = config

    training_points = TensorDict(dict(
      position=points.position,
      log_scaling=points.log_scaling,
      rotation=points.rotation,
      alpha_logit=points.alpha_logit,
      feature=points.feature),

      batch_size = points.batch_size
    )

    self.raster_config = RasterConfig()
    self.learning_rates = OmegaConf.to_container(config.learning_rates)

    self.points = ParameterClass.create(training_points, 
          learning_rates = self.learning_rates)   



  def to(self, device):
    self.points = self.points.to(device)
    return self
  
  @property 
  def device(self):
    return self.points.position.device
  
  @property
  def num_points(self):
    return self.points.position.shape[0]

  @beartype
  def update_learning_rate(self, scene_scale:float, step:int, total_steps:int):
    scheduler = self.config.scheduler
    base_lr = self.learning_rates ['position'] * scene_scale
    pos_lr = scheduler.schedule(base_lr, step, total_steps)

    self.points.set_learning_rate(position = pos_lr)


  def __repr__(self):
    return f"GaussianScene({self.points.position.shape[0]} points)"

  def step(self):
    
    self.points.feature.grad[..., 1:] /= self.config.sh_ratio
    self.points.step()

  def zero_grad(self):
    self.points.zero_grad()



  def split_and_prune(self, keep_mask, split_idx):

    splits = split_gaussians(self.gaussians[split_idx], n=2, scaling=1 / (0.8 * 2))
    self.points = self.points[keep_mask].append_tensors(splits.to_tensordict())

    return self


  @property
  def gaussians(self):
      return Gaussians3D.from_tensordict(self.points.tensors)
      

  def render(self, camera_params, compute_radii=False, 
             render_depth=False, compute_split_heuristics=False) -> Rendering:
    
    return render_gaussians(self.gaussians, 
                     use_sh        = True,
                     config        = self.raster_config,
                     camera_params = camera_params,
                     compute_radii = compute_radii,
                     render_depth  = render_depth,
                     compute_split_heuristics=compute_split_heuristics)
  




