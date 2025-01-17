
from dataclasses import  dataclass
from pathlib import Path
from typing import Dict, Optional
from beartype import beartype
from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from splat_trainer.camera_table.camera_table import CameraTable, camera_scene_extents, camera_similarity
from splat_trainer.config import VaryingFloat,  eval_varyings
from splat_trainer.logger.logger import Logger
from splat_trainer.scene.color_model import ColorModel, GLOTable
from splat_trainer.scene.io import write_gaussians
from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import point_basis, split_gaussians_uniform

import taichi_splatting.optim as optim
from taichi_splatting.optim import ParameterClass
from taichi_splatting.optim.visibility_aware import VisibilityOptimizer
from taichi_splatting import Gaussians3D, Rendering, TaichiQueue

from taichi_splatting.renderer import render_projected, project_to_image
from taichi_splatting.perspective import CameraParams


from splat_trainer.scene.util import parameters_from_gaussians, pop_raster_config
from splat_trainer.util.pointcloud import PointCloud

@beartype
@dataclass(kw_only=True, frozen=True)
class TCNNConfig(GaussianSceneConfig):  
  parameters : DictConfig | Dict
  
  lr_image_feature: VaryingFloat = 0.001
  lr_nn:VaryingFloat = 0.0001

  image_features:int       = 8
  point_features:int       = 8

  affine_color_model:bool  = False
  hidden_features:int      = 32

  layers:int             = 2

  beta1:float = 0.9
  beta2:float = 0.999
  
  optimizer:str = 'SparseAdam'

  per_image:bool = True


  def from_color_gaussians(self, gaussians:Gaussians3D, 
                           camera_table:CameraTable, 
                           device:torch.device):
    

    feature = torch.zeros(gaussians.batch_size[0], self.point_features)
    torch.nn.init.normal_(feature, std=0.5)

    gaussians = gaussians.replace(feature=feature).to(device)
    points = parameters_from_gaussians(gaussians, 
          eval_varyings(self.parameters, 0.), getattr(optim, self.optimizer), betas=(self.beta1, self.beta2))
    
    return TCNNScene(points, self, camera_table)

  
  def from_state_dict(self, state:dict, camera_table:CameraTable):

    points = ParameterClass.from_state_dict(state['points'], 
          optimizer=getattr(optim, self.optimizer), betas=(self.beta1, self.beta2))
    
    scene = TCNNScene(points, self, camera_table)

    scene.color_model.load_state_dict(state['color_model'])
    scene.color_opt.load_state_dict(state['color_opt'])

    return scene


@beartype
class TCNNScene(GaussianScene):
  def __init__(self, 
          points: ParameterClass, 
          config: TCNNConfig,       
          camera_table:CameraTable,     
    ):
    self.config = config
    self.points = points

    self.camera_table = camera_table
    num_glo_embeddings = camera_table.num_images if config.per_image else camera_table.num_cameras


    self.color_model = ColorModel(
      glo_features=config.image_features, 
      point_features=config.point_features, 
      hidden_features=config.hidden_features, 
      layers=config.layers,
      sh_degree=5).to(self.device)
    
    self.color_model = torch.compile(self.color_model, options=dict(max_autotune=True), dynamic=True)
    self.color_opt = self.color_model.optimizer(config.lr_nn)

    self.color_table = GLOTable(num_glo_embeddings, config.image_features).to(self.device)
    self.glo_opt = self.color_table.optimizer(config.lr_image_feature)
    
    
    self.scene_extents = camera_scene_extents(camera_table.cameras)

  @property
  def device(self):
    return self.points.position.device

  @property
  def num_points(self):
    return self.points.position.shape[0]

  def __repr__(self):
    return f"TCNNScene({self.num_points} points)"

  def update_learning_rate(self, t:float):
    groups = eval_varyings(self.config.parameters, t)

    self.points.update_groups(**groups)
    self.color_model.schedule(self.color_opt, self.config.lr_nn, t)
    self.color_table.schedule(self.glo_opt, self.config.lr_image_feature, t)
    

  @beartype
  def step(self, rendering:Rendering, t:float) -> Dict[str, float]:
    vis = rendering.visible_indices
    basis = point_basis(self.points.log_scaling[vis], self.points.rotation[vis]).contiguous()

    if isinstance(self.points.optimizer , VisibilityOptimizer):
      self.points.step(indexes=vis, 
                        visibility=rendering.point_visibility[rendering.visible_mask], 
                        basis=basis)
    else:
      self.points.step(indexes=vis, basis=basis)

    self.color_opt.step()
    self.glo_opt.step()

    self.points.rotation = torch.nn.Parameter(
      F.normalize(self.points.rotation.detach(), dim=1), requires_grad=True)
        
    self.points.zero_grad()
    self.color_opt.zero_grad()
    self.glo_opt.zero_grad()
    
    return self.points.learning_rates


  def split_and_prune(self, keep_mask, split_idx):
    splits = split_gaussians_uniform(
      self.points[split_idx].detach(), n=2, random_axis=True)

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
  def lookup_glo_feature(self, image_idx:int) -> torch.Tensor:
    if not self.config.per_image:
      image_idx = self.camera_table.camera_id(image_idx)

    return self.color_table(image_idx)
    
  @beartype
  def interpolated_glo_feature(self, camera_t_world:torch.Tensor) -> torch.Tensor:
    cameras = self.camera_table.cameras
    similarity = F.softmax(camera_similarity(cameras, camera_t_world), dim=0)
    
    image_idx = torch.arange(self.camera_table.num_images, device=self.device)
    if not self.config.per_image:
      image_idx = self.camera_table.camera_id(image_idx)

    glo_feature = self.color_table(image_idx)
    return torch.sum(similarity.unsqueeze(1) * glo_feature, dim=0)


  def eval_colors(self, indexes, camera_params, image_idx):
    if image_idx is not None:
      glo_feature = self.lookup_glo_feature(image_idx)
    else:
      glo_feature = self.interpolated_glo_feature(torch.inverse(camera_params.T_camera_world))

    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
      return self.color_model(self.points.feature[indexes], self.points.position[indexes], 
                                                        camera_params.camera_position, glo_feature)

  @beartype
  def render(self, camera_params:CameraParams,  
             image_idx:Optional[int] = None, **options) -> Rendering:

    raster_config = pop_raster_config(options)
    gaussians2d, depthvars, indexes = project_to_image(self.gaussians, camera_params, raster_config)

    colour = TaichiQueue.run_sync(self.eval_colors, indexes, camera_params, image_idx)
    # colour = self.eval_colors(indexes, camera_params, image_idx)
    return render_projected(indexes, gaussians2d, colour, depthvars, 
                            camera_params, raster_config, **options)


    

