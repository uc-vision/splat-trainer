
from dataclasses import  dataclass, replace
from pathlib import Path
from typing import Dict, Tuple
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F

from splat_trainer.camera_table.camera_table import ViewTable, camera_scene_extents
from splat_trainer.config import VaryingFloat,  eval_varyings
from splat_trainer.logger.logger import Logger
from splat_trainer.scene.color_model import ColorModel
from splat_trainer.scene.io import write_gaussians
from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import split_gaussians_uniform


from taichi_splatting.optim import ParameterClass, SparseAdam
from taichi_splatting import Gaussians3D, RasterConfig, Rendering

from taichi_splatting.renderer import render_projected, project_to_image
from taichi_splatting.perspective import CameraParams


from splat_trainer.scene.util import parameters_from_gaussians, update_depth
from splat_trainer.util.pointcloud import PointCloud

@beartype
@dataclass(kw_only=True, frozen=True)
class TCNNConfig(GaussianSceneConfig):  
  learning_rates : DictConfig | Dict
  
  lr_image_feature: VaryingFloat = 0.001
  lr_nn:VaryingFloat = 0.0001

  image_features:int       = 8
  point_features:int       = 8

  affine_color_model:bool  = False
  hidden_features:int      = 32

  layers:int             = 2

  per_image:bool = True

  depth_ema:float = 0.95
  use_depth_lr:bool = True

  beta1:float = 0.9
  beta2:float = 0.999



  def from_color_gaussians(self, gaussians:Gaussians3D, 
                           camera_table:ViewTable, 
                           device:torch.device):
    

    feature = torch.zeros(gaussians.batch_size[0], self.point_features)
    torch.nn.init.normal_(feature, std=1.0)

    gaussians = gaussians.replace(feature=feature).to(device)
    points = parameters_from_gaussians(gaussians, 
          eval_varyings(self.learning_rates, 0.), betas=(self.beta1, self.beta2))
    
    return TCNNScene(points, self, camera_table)

  
  def from_state_dict(self, state:dict, camera_table:ViewTable):

    points = ParameterClass.from_state_dict(state['points'], 
          optimizer=SparseAdam, betas=(self.beta1, self.beta2))
    
    scene = TCNNScene(points, self, camera_table)

    scene.color_model.load_state_dict(state['color_model'])
    scene.color_opt.load_state_dict(state['color_opt'])

    return scene


@beartype
class TCNNScene(GaussianScene):
  def __init__(self, 
          points: ParameterClass, 
          config: TCNNConfig,       
          camera_table:ViewTable,     
    ):
    self.config = config
    self.points = points

    self.camera_table = camera_table

    num_glo_embeddings = camera_table.num_images if config.per_image else camera_table.num_cameras

    self.color_model = ColorModel(
      num_glo_embeddings, 
      glo_features=config.image_features, 
      point_features=config.point_features, 
      hidden_features=config.hidden_features, 
      layers=config.layers,
      affine_model=config.affine_color_model).to(self.device)
    
    self.color_opt = self.color_model.optimizer(
      config.lr_nn, config.lr_image_feature)
    
    self.scene_extents = camera_scene_extents(camera_table)

  @property
  def device(self):
    return self.points.position.device

  @property
  def num_points(self):
    return self.points.position.shape[0]

  def __repr__(self):
    return f"TCNNScene({self.num_points} points)"

  def update_learning_rate(self, t:float):
    lr = eval_varyings(self.config.learning_rates, t)
    if not self.config.use_depth_lr:
      lr['position'] *= self.scene_extents
    
    self.points.set_learning_rate(**lr)
    self.color_model.schedule(self.color_opt, 
            self.config.lr_nn, self.config.lr_image_feature, t)
    
    return lr


  @beartype
  def step(self, rendering:Rendering, t:float) -> Dict[str, float]:
    lr = self.update_learning_rate(t)
  
    if self.config.use_depth_lr:
      update_depth(self.points, rendering, self.config.depth_ema)

    self.points.step(visible_indexes=rendering.visible_indices)
    # check_finite(self.points)
    self.color_opt.step()
    self.points.rotation = torch.nn.Parameter(
      F.normalize(self.points.rotation.detach(), dim=1), requires_grad=True)
    
    
    self.points.zero_grad()
    self.color_opt.zero_grad()
    
    return {**lr}


  def split_and_prune(self, keep_mask, split_idx):
    splits = split_gaussians_uniform(
      self.points[split_idx].detach(), n=2)

    self.points = self.points[keep_mask].append_tensors(splits)
 
  @property
  def gaussians(self):
      points = self.points.tensors.select('position', 'rotation', 'log_scaling', 'alpha_logit', 'feature')
      return Gaussians3D.from_tensordict(points)
      


  def write_to(self, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    write_gaussians(output_dir / 'point_cloud.ply', self.gaussians.apply(torch.detach), with_sh=False)

    d = self.color_model.state_dict()
    torch.save(d, output_dir / 'color_model.pth')


  def state_dict(self):
    return dict(points=self.points.state_dict(), 
                color_model=self.color_model.state_dict(),
                color_opt = self.color_opt.state_dict())
  

  def get_point_cloud(self, image_idx:int = 0):
    colors = self.evaluate_colors(torch.arange(self.num_points, device=self.device), image_idx)
    return PointCloud(self.gaussians.position.detach(), colors)
    

  def log(self, logger:Logger, step:int):
    for k, v in dict(log_scaling=self.points.log_scaling, 
                     alpha_logit=self.points.alpha_logit,
                     feature=self.points.feature).items():
      logger.log_histogram(f"points/{k}", v.detach(), step=step)


  def evaluate_colors(self, indexes, cam_idx, camera_position):
    return self.color_model(self.points.feature[indexes], self.points.position[indexes], camera_position, cam_idx)

  @beartype
  def render(self, camera_params:CameraParams, config:RasterConfig, 
             image_idx:int, **options) -> Rendering:
    
    idx = image_idx if self.config.per_image else self.camera_table.camera_id(image_idx)

    gaussians2d, depthvars, indexes = project_to_image(self.gaussians, camera_params, config)
    features = self.evaluate_colors(indexes, idx, camera_params.camera_position)

    return render_projected(indexes, gaussians2d, features, depthvars, 
            camera_params, config, **options)


    

