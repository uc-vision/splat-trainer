
from dataclasses import  dataclass, replace
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F

from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import  split_gaussians_uniform
from splat_trainer.util.misc import  rgb_to_sh

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D, RasterConfig, render_gaussians, Rendering
from taichi_splatting.perspective import CameraParams


@dataclass(kw_only=True, frozen=True)
class SHConfig(GaussianSceneConfig):  
  learning_rates : DictConfig
  sh_ratio:float      = 20.0
  sh_degree:int       = 2

  def from_color_gaussians(self, gaussians:Gaussians3D, device:torch.device, camera_shape:torch.Size):
    sh_feature = torch.zeros(gaussians.batch_size[0], 3, (self.sh_degree + 1)**2)
    sh_feature[:, :, 0] = rgb_to_sh(gaussians.feature)

    return SHScene(gaussians.replace(feature=sh_feature), device, self)


class SHScene(GaussianScene):
  def __init__(self, points: Gaussians3D, device:torch.device, config: SHConfig):
    self.config = config

    self.raster_config = RasterConfig()
    self.learning_rates = OmegaConf.to_container(config.learning_rates)

    self.points = ParameterClass.create(points.to_tensordict().to(device), 
          learning_rates = self.learning_rates)   
        

  
  @property 
  def device(self):
    return self.points.position.device
  
  @property
  def num_points(self):
    return self.points.position.shape[0]

  @beartype
  def update_learning_rate(self, lr_scale:float):
    self.points.set_learning_rate(position = self.learning_rates ['position'] * lr_scale)



  def __repr__(self):
    return f"GaussianScene({self.points.position.shape[0]} points)"

  def step(self):
    self.points.feature.grad[..., 1:] /= self.config.sh_ratio
    self.points.step()

    with torch.no_grad():
      self.points.rotation = torch.nn.Parameter(
        F.normalize(self.points.rotation.detach(), dim=1), requires_grad=True)

  def zero_grad(self):
    self.points.zero_grad()



  def split_and_prune(self, keep_mask:torch.Tensor, split_idx:torch.Tensor):

    splits = split_gaussians_uniform(self.gaussians[split_idx], n=2)
    self.points = self.points[keep_mask].append_tensors(splits.to_tensordict())

    return self


  @property
  def gaussians(self):
      return Gaussians3D.from_tensordict(self.points.tensors)
      

  def render(self, cam_idx:torch.Tensor, camera_params:CameraParams, 
             **options) -> Rendering:
    return render_gaussians(self.gaussians, 
                     use_sh        = True,
                     config        = self.raster_config,
                     camera_params = camera_params,
                     **options)
  




