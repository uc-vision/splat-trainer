
from dataclasses import  dataclass
import math
from pathlib import Path
from typing import Dict
from beartype import beartype
from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from splat_trainer.camera_table.camera_table import CameraTable
from splat_trainer.config import eval_varyings
from splat_trainer.logger.logger import Logger
from splat_trainer.scene.io import write_gaussians
from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import  point_basis, split_gaussians_uniform
from splat_trainer.scene.util import parameters_from_gaussians, pop_raster_config
from splat_trainer.util.misc import   rgb_to_sh

from taichi_splatting.optim.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D, render_gaussians, Rendering
from taichi_splatting.perspective import CameraParams
from taichi_splatting.optim.sparse_adam import SparseAdam


@beartype
@dataclass(kw_only=True, frozen=True)
class SHConfig(GaussianSceneConfig):  
  parameters : DictConfig | Dict

  sh_ratio:float      = 20.0
  sh_degree:int       = 3

  # proportion of training progress per sh degree
  degree_progress: float = 0.1 


  beta1:float = 0.9
  beta2:float = 0.999



  def from_color_gaussians(self, gaussians:Gaussians3D, camera_table:CameraTable, device:torch.device):

    sh_feature = torch.zeros(gaussians.batch_size[0], 3, (self.sh_degree + 1)**2)
    sh_feature[:, :, 0] = rgb_to_sh(gaussians.feature)

    gaussians = gaussians.replace(feature=sh_feature).to(device)

    points = parameters_from_gaussians(gaussians, self.parameters, betas=(self.beta1, self.beta2))
    return SHScene(points, self, camera_table)

  
  def from_state_dict(self, state:dict, camera_table:CameraTable):
    points = ParameterClass.from_state_dict(state['points'], 
          optimizer=SparseAdam, betas=(self.beta1, self.beta2))
    
    return SHScene(points, self, camera_table)


class SHScene(GaussianScene):
  def __init__(self, 
        points: ParameterClass, 
        config: SHConfig,       
        camera_table:CameraTable,     
  ):
    
    self.config = config
    self.points = points

    self.camera_table = camera_table

        
    
  @property
  def device(self):
    return self.points.position.device

  @property
  def num_points(self):
    return self.points.position.shape[0]

  def __repr__(self):
    return f"SHScene({self.num_points} points)"




  def sh_mask(self, t:float):
    """ Learning rate mask for spherical harmonics """
    d = 1 + min(math.floor(t / self.config.degree_progress), self.config.sh_degree)
    
    mask = torch.zeros(self.points.feature.shape[1:], dtype=torch.float32, device=self.device)
    mask[:, 0:1] = 1 # Base color has full learning rate
    mask[:, 1:d**2] = 1/self.config.sh_ratio # higher order SH coefficients have reduced learning rate

    return mask

  def update_learning_rate(self, t:float):
    groups = eval_varyings(self.config.parameters, t)

    self.points.update_groups(**groups)
    self.points.update_group('feature', mask_lr=self.sh_mask(t))

    return self.points.learning_rates


  @beartype
  def step(self, rendering:Rendering, t:float):
    lr = self.update_learning_rate(t)

    vis = rendering.visible_indices
    basis = point_basis(self.points.log_scaling[vis], self.points.rotation[vis]).contiguous()
    self.points.step(visible_indexes=vis, basis=basis)


    self.points.rotation = torch.nn.Parameter(
      F.normalize(self.points.rotation.detach(), dim=1), requires_grad=True)

    self.points.zero_grad()

    return {**lr}


  @beartype
  def split_and_prune(self, keep_mask:torch.Tensor, split_idx:torch.Tensor):

    splits = split_gaussians_uniform(self.points[split_idx].detach(), n=2)
    self.points = self.points[keep_mask].append_tensors(splits)

    return self


  def write_to(self, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    write_gaussians(output_dir / 'point_cloud.ply', self.gaussians, with_sh=True)

    
  def log(self, logger:Logger, step:int):
    for k, v in dict(log_scaling=self.points.log_scaling, alpha_logit=self.points.alpha_logit).items():
      logger.log_histogram(f"points/{k}", v.detach(), step=step)



  def state_dict(self):
    return dict(points=self.points.state_dict())

  @property
  def gaussians(self):
      points = self.points.tensors.select('position', 'rotation', 'log_scaling', 'alpha_logit', 'feature')
      return Gaussians3D.from_tensordict(points)

  def render(self, camera_params:CameraParams, cam_idx:int, **options) -> Rendering:

    raster_config = pop_raster_config(options)  
    return render_gaussians(self.gaussians, 
                     use_sh        = True,
                     config        = raster_config,
                     camera_params = camera_params,
                     **options)
  




