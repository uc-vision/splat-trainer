
from dataclasses import  dataclass, replace
from functools import partial
from pathlib import Path
from typing import Optional
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

from tensordict import TensorDict
import torch
import torch.nn.functional as F
from tqdm import tqdm

from splat_trainer.camera_table.camera_table import CameraTable, camera_extents
from splat_trainer.logger.logger import Logger
from splat_trainer.scene.io import write_gaussians
from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import split_gaussians_uniform


from taichi_splatting.optim.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D, RasterConfig, Rendering
from taichi_splatting.optim.sparse_adam import SparseAdam

from taichi_splatting.renderer import render_projected, project_to_image
from taichi_splatting.perspective import CameraParams


from splat_trainer.util.misc import lerp
from splat_trainer.util.pointcloud import PointCloud


@dataclass(kw_only=True, frozen=True)
class TCNNConfig(GaussianSceneConfig):  
  learning_rates : DictConfig
  
  lr_image_feature:float= 0.001
  lr_nn:float = 0.0001

  image_features:int       = 8
  point_features:int       = 8

  hidden:int             = 32
  layers:int             = 2

  depth_ema:float = 0.95
  use_depth_lr:bool = True

  scene_extents:Optional[float] = None



  def color_model(self):
    num_features = self.image_features + self.point_features
    import tinycudann as tcnn

    return tcnn.NetworkWithInputEncoding(
      num_features + 3, 3,
      encoding_config=dict(
        otype = "composite",
        nested = [
          dict(otype = "SphericalHarmonics", 
              degree = 4, 
              n_dims_to_encode = 3
          ),
          
          dict(otype = "Identity",
              n_dims_to_encode = num_features)
        ]
      ), 
      
      network_config = dict(
        otype = "FullyFusedMLP",
        activation = "ReLU",
        output_activation = "None",
        n_neurons = self.hidden,
        n_hidden_layers = self.layers,
      )
    )



  def from_color_gaussians(self, gaussians:Gaussians3D, camera_table:CameraTable, device:torch.device):
    color_model = self.color_model().to(device)
    color_targets = gaussians.feature.to(device)

    features = torch.randn(gaussians.batch_size[0], self.point_features)
    gaussians = gaussians.replace(feature=features).to(device)
    
    centre, extents = camera_extents(camera_table)
    if self.scene_extents is None:
      config = replace(self, scene_extents=extents)

    return TCNNScene(gaussians, color_model, camera_table, device, config)





@beartype
class TCNNScene(GaussianScene):
  def __init__(self, points: Gaussians3D, 
               color_model:torch.nn.Module,
               camera_table:CameraTable,

               device:torch.device,            
               config: TCNNConfig):
    self.config = config
    self.device = device

    self.color_model = color_model
    self.learning_rates = OmegaConf.to_container(config.learning_rates)

    parameter_groups = {k:dict(lr=lr) 
            for k, lr in self.learning_rates.items()}

    make_optimizer = partial(SparseAdam, betas=(0.9, 0.999))
    parameter_groups = {k:dict(lr=lr) for k, lr in self.learning_rates.items()}

    d:TensorDict = points.to_tensordict().replace(
      running_depth = torch.zeros(points.batch_size[0], device=device))
    
    self.points = ParameterClass(d, 
          parameter_groups=parameter_groups, 
          optimizer=make_optimizer)   


    self.camera_table = camera_table
    
    image_features = torch.zeros(camera_table.num_cameras,  config.image_features, dtype=torch.float32, device=device)
    self.image_features = torch.nn.Parameter(image_features, requires_grad=True)

    self.color_opt = self.make_color_optimizer()

  def make_color_optimizer(self):
    config = self.config

    param_groups = [
      dict(params=self.color_model.parameters(), lr=config.lr_nn, name="color_model"),
      dict(params=self.image_features, lr=config.lr_image_feature, name="image_features")
    ]

    return torch.optim.Adam(param_groups, betas=(0.7, 0.999))

  
  @property
  def num_points(self):
    return self.points.position.shape[0]

  def __repr__(self):
    return f"GaussianScene({self.points.position.shape[0]} points)"

  @beartype
  def update_learning_rate(self, lr_scale:float):
    # scaled_lr = {k: v * lr_scale for k, v in self.learning_rates.items()}
    # self.points.set_learning_rate(**scaled_lr)
    if not self.config.use_depth_lr:
      lr_scale *= self.config.scene_extents
    
    self.points.set_learning_rate(position = 
              self.learning_rates['position'] * lr_scale)



  def update_depth(self, rendering:Rendering):
    """ Method for scaling learning rates by point depth. 
        Take running average of running_depth = depth/fx.

        Scale gradients by 1/running depth and learning rates by running depth.
    """
    fx = rendering.camera.focal_length[0]
    depth_scales = rendering.point_depth[rendering.visible_mask].squeeze(1) / fx  
    
    running_depth = self.points.running_depth[rendering.visible_indices]
    running_depth[:] = lerp(self.config.depth_ema, depth_scales, running_depth)

    self.points.update_group('position', point_lr=self.points.running_depth)
    self.points.position.grad[rendering.visible_indices] /= depth_scales.unsqueeze(1)

  @beartype
  def step(self, rendering:Rendering, step:int):

    if self.config.use_depth_lr:
      self.update_depth(rendering)


    self.points.step(visible_indexes=rendering.visible_indices)
    # check_finite(self.points)
    self.color_opt.step()
    self.points.rotation = torch.nn.Parameter(
      F.normalize(self.points.rotation.detach(), dim=1), requires_grad=True)
    

    self.points.zero_grad()
    self.color_opt.zero_grad()



  @property
  def scale(self):
    return torch.exp(self.points.log_scaling)
  
  @property
  def opacity(self):
    return torch.sigmoid(self.points.alpha_logit)


  def split_and_prune(self, keep_mask, split_idx):
    splits = split_gaussians_uniform(
      self.points[split_idx].detach(), n=2)

    self.points = self.points[keep_mask].append_tensors(splits)

  @property
  def gaussians(self):
      points = self.points.tensors.select('position', 'rotation', 'log_scaling', 'alpha_logit', 'feature')
      return Gaussians3D.from_tensordict(points)
      
  def evaluate_colors(self, indexes, image_idx, camera_position):
    cam_idx = self.camera_table.camera_id(image_idx)

    cam_feature = self.image_features[cam_idx]  
    cam_feature = cam_feature.unsqueeze(0).expand(indexes.shape[0], -1)

    dir = F.normalize(self.points.position[indexes].detach() - camera_position)

    feature = torch.cat([dir, self.points.feature[indexes], cam_feature], dim=1)
    return self.color_model(feature).to(torch.float32).sigmoid()
  

  def random_camera(self):
      image_idx = [torch.randint(0, j, device=self.device) for j in self.image_features.shape]
      return torch.stack(image_idx)


  def write_to(self, output_dir:Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    write_gaussians(output_dir / 'point_cloud.ply', self.gaussians.apply(torch.detach), with_sh=False)

  def get_point_cloud(self, image_idx:int = 0):
    camera_position = self.camera_table.camera_centers[image_idx]

    colors = self.evaluate_colors(torch.arange(self.num_points, device=self.device), image_idx, camera_position)
    return PointCloud(self.gaussians.position.detach(), colors)
    

  def log(self, logger:Logger, step:int):
    for k, v in dict(log_scaling=self.points.log_scaling, 
                     alpha_logit=self.points.alpha_logit,
                     feature=self.points.feature).items():
      logger.log_histogram(f"points/{k}", v.detach(), step=step)

  @beartype
  def render(self, camera_params:CameraParams, config:RasterConfig, image_idx:int, **options) -> Rendering:


    gaussians2d, depthvars, indexes = project_to_image(self.gaussians, camera_params, config)
    features = self.evaluate_colors(indexes, image_idx, camera_params.camera_position)

    return render_projected(indexes, gaussians2d, features, depthvars, 
            camera_params, config, **options)
      
    



