
from dataclasses import  dataclass, replace
from functools import partial
from pathlib import Path
from typing import Optional
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

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

  scene_extent:Optional[float]     = None


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

    self.pretrain_colors(gaussians.feature, color_model, color_targets, iters=1000)
    
    if self.scene_extent is None:
      centre, extent = camera_extents(camera_table)
      config = replace(self, scene_extent=extent)

    return TCNNScene(gaussians, color_model, camera_table, device, config)


  def pretrain_colors(self, features, color_model, target_colors, iters):
    with torch.enable_grad():
      n = target_colors.shape[0]
      device = features.device

      features = torch.nn.Parameter(features, requires_grad=True)
      param_groups = [
        dict(params=color_model.parameters(), lr=self.lr_nn, name="color_model"),
        dict(params=features, lr=self.learning_rates["feature"], name="point_features")
      ]

      opt = torch.optim.Adam(param_groups)

      pbar = tqdm(total=iters, desc="initializing colors")
      for i in range(iters):
        opt.zero_grad()
        dirs = F.normalize(torch.randn(n, 3, device=device), dim=1)
        cam_feature = torch.zeros(n, self.image_features, device=device)

        features = torch.nn.Parameter(
          F.normalize(features.detach(), dim=1), requires_grad=True)

        feature = torch.cat([dirs, features, cam_feature], dim=1)
        colors = color_model(feature).to(torch.float32).sigmoid()

        loss = F.mse_loss(colors, target_colors)

        loss.backward()
        opt.step()
        
        pbar.update(1)
        pbar.set_postfix(loss=loss.item())


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
    self.learning_rates ['position'] *= self.config.scene_extent

    make_optimizer = partial(SparseAdam, betas=(0.9, 0.999))
    parameter_groups = {k:dict(lr=lr) for k, lr in self.learning_rates.items()}

    self.points = ParameterClass.create(points.to_tensordict(), 
          parameter_groups=parameter_groups, 
          optimizer=make_optimizer)   

    # self.sort_points()

    self.camera_table = camera_table
    
    image_features = torch.zeros(*camera_table.shape,  config.image_features, dtype=torch.float32, device=device)
    self.image_features = torch.nn.Parameter(image_features, requires_grad=True)

    self.color_opt = self.make_color_optimizer()

  def make_color_optimizer(self):
    config = self.config

    param_groups = [
      dict(params=self.color_model.parameters(), lr=config.lr_nn, name="color_model"),
      dict(params=self.image_features, lr=config.lr_image_feature, name="image_features")
    ]

    return torch.optim.Adam(param_groups)

  
  
  @property
  def num_points(self):
    return self.points.position.shape[0]

  @beartype
  def update_learning_rate(self, lr_scale:float):
    # scaled_lr = {k: v * lr_scale for k, v in self.learning_rates.items()}
    # self.points.set_learning_rate(**scaled_lr)
    self.points.set_learning_rate(position = 
              self.learning_rates['position'] * lr_scale)


  def __repr__(self):
    return f"GaussianScene({self.points.position.shape[0]} points)"

  def step(self, visible:torch.Tensor, step:int):
    
    self.points.step(visible_indexes=visible)
    # check_finite(self.points)
    self.color_opt.step()
    self.normalize()

  
  def normalize(self):
    self.points.rotation = torch.nn.Parameter(
      F.normalize(self.points.rotation.detach(), dim=1), requires_grad=True)


  def zero_grad(self):
    self.points.zero_grad()


  @property
  def scale(self):
    return torch.exp(self.points.log_scaling)
  
  @property
  def opacity(self):
    return torch.sigmoid(self.points.alpha_logit)




  def split_and_prune(self, keep_mask, split_idx):
    splits = split_gaussians_uniform(self.gaussians[split_idx], n=2)
    self.points = self.points[keep_mask].append_tensors(splits.to_tensordict())


  @property
  def gaussians(self):
      return Gaussians3D.from_tensordict(self.points.tensors)
      
  def evaluate_colors(self, indexes, image_idx, camera_position):
    cam_feature = self.image_features[image_idx.unbind(0)]  
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

  def get_point_cloud(self):
    image_idx = torch.zeros(len(self.camera_table.shape), device=self.device, dtype=torch.long)
    camera_position = self.camera_table.camera_centers[image_idx.unbind(0)]

    colors = self.evaluate_colors(torch.arange(self.num_points, device=self.device), image_idx, camera_position)
    return PointCloud(self.gaussians.position.detach(), colors)
    

  def log(self, logger:Logger, step:int):
    for k, v in dict(log_scaling=self.points.log_scaling, 
                     alpha_logit=self.points.alpha_logit,
                     feature=self.points.feature).items():
      logger.log_histogram(f"points/{k}", v.detach(), step=step)


  def render(self, camera_params:CameraParams, config:RasterConfig, image_idx:torch.Tensor, **options) -> Rendering:

    gaussians2d, depthvars, indexes = project_to_image(self.gaussians, camera_params, config)
    features = self.evaluate_colors(indexes, image_idx, camera_params.camera_position)

    return render_projected(indexes, gaussians2d, features, depthvars, 
            camera_params, config, **options)




