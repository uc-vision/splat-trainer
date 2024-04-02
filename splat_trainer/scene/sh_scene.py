
import copy
from dataclasses import  dataclass, replace
from pathlib import Path
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F

from splat_trainer.camera_table.camera_table import CameraTable, camera_extents
from splat_trainer.logger.logger import Logger
from splat_trainer.scene.io import write_gaussians
from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import  split_gaussians_uniform
from splat_trainer.util.misc import  rgb_to_sh, sh_to_rgb

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D, RasterConfig, render_gaussians, Rendering
from taichi_splatting.perspective import CameraParams

from splat_viewer.gaussians import Gaussians

from splat_trainer.util.pointcloud import PointCloud


@dataclass(kw_only=True, frozen=True)
class SHConfig(GaussianSceneConfig):  
  learning_rates : DictConfig
  sh_ratio:float      = 20.0
  sh_degree:int       = 2


  def with_scene_extent(self, scene_extent:float) -> 'SHConfig':
    learning_rates = copy.copy(self.learning_rates) 
    learning_rates.position *= scene_extent
    return replace(self, learning_rates=learning_rates)
    

  def from_color_gaussians(self, gaussians:Gaussians3D, camera_table:CameraTable, device:torch.device):
    sh_feature = torch.zeros(gaussians.batch_size[0], 3, (self.sh_degree + 1)**2)
    sh_feature[:, :, 0] = rgb_to_sh(gaussians.feature)

    centre, extent = camera_extents(camera_table)
    config = self.with_scene_extent(extent)
    
    return SHScene(gaussians.replace(feature=sh_feature), camera_table, device, config)


class SHScene(GaussianScene):
  def __init__(self, points: Gaussians3D, camera_table:CameraTable, device:torch.device, config: SHConfig):
    self.config = config
    self.camera_table = camera_table

    self.raster_config = RasterConfig()
    self.learning_rates = OmegaConf.to_container(config.learning_rates)

    self.points = ParameterClass.create(points.to_tensordict().to(device), 
          learning_rates = self.learning_rates)   
        
  def with_scene_extent(self, scene_extent:float) -> 'SHScene':
    learning_rates = copy.copy(self.learning_rates) 
    learning_rates.position *= scene_extent
    return replace(self, learning_rates=learning_rates)
  
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


  def write_to(self, output_dir:Path, base_filename:str):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    write_gaussians(output_dir / f'{base_filename}.ply', self.gaussians, with_sh=True)

    
  def log(self, logger:Logger, step:int):
    point_cloud = PointCloud(self.gaussians.position, sh_to_rgb(self.gaussians.feature[:, :, 0]))
    logger.log_cloud("point_cloud", point_cloud, step=step)
    
    

  @property
  def gaussians(self):
      return Gaussians3D.from_tensordict(self.points.tensors)
      

  def render(self, cam_idx:torch.Tensor, 
             **options) -> Rendering:
    
    camera_params = self.camera_table(cam_idx)
    return render_gaussians(self.gaussians, 
                     use_sh        = True,
                     config        = self.raster_config,
                     camera_params = camera_params,
                     **options)
  




