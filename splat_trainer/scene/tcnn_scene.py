
from dataclasses import  dataclass
from beartype import beartype
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
import tinycudann as tcnn
from tqdm import tqdm

from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import split_gaussians_uniform


from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D, RasterConfig, Rendering

from taichi_splatting.renderer import gaussians_in_view, project_to_image, render_projected
from taichi_splatting.perspective import CameraParams


@dataclass(kw_only=True, frozen=True)
class TCNNConfig(GaussianSceneConfig):  
  learning_rates : DictConfig
  
  lr_image_feature:float= 0.001
  lr_nn:float = 0.0001

  image_features:int       = 8
  image_init:float         = 1e-5
  point_features:int       = 8

  hidden:int             = 32
  layers:int             = 2


  def color_model(self):
    num_features = self.image_features + self.point_features

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

  def from_color_gaussians(self, gaussians:Gaussians3D, device:torch.device, camera_shape:torch.Size):
    color_model = self.color_model().to(device)
    color_targets = gaussians.feature.to(device)

    features = torch.randn(gaussians.batch_size[0], self.point_features)
    gaussians = gaussians.replace(feature=features).to(device)

    self.pretrain_colors(gaussians.feature, color_model, color_targets, iters=1000)
    return TCNNScene(gaussians, color_model, device, camera_shape, self)


  def pretrain_colors(self, features, color_model, target_colors, iters):
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
        cam_feature = self.image_init * torch.randn(n, self.image_features, device=device)
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
               device:torch.device, 

               camera_shape:torch.Size, 
               config: TCNNConfig):
    self.config = config
    self.device = device

    self.color_model = color_model

    self.raster_config = RasterConfig()
    self.learning_rates = OmegaConf.to_container(config.learning_rates)

    self.points = ParameterClass.create(points.to_tensordict(), 
          learning_rates = self.learning_rates)   
    
    image_features = config.image_init * torch.randn(*camera_shape, 
            config.image_features, dtype=torch.float32, device=device)
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
    self.points.set_learning_rate(position = self.learning_rates ['position'] * lr_scale)


  def __repr__(self):
    return f"GaussianScene({self.points.position.shape[0]} points)"

  def step(self):
    self.points.step()
    self.color_opt.step()

    with torch.no_grad():
      self.points.rotation = torch.nn.Parameter(
        F.normalize(self.points.rotation.detach(), dim=1), requires_grad=True)

  def zero_grad(self):
    self.points.zero_grad()



  def split_and_prune(self, keep_mask, split_idx):

    splits = split_gaussians_uniform(self.gaussians[split_idx], n=2)
    self.points = self.points[keep_mask].append_tensors(splits.to_tensordict())

    return self


  @property
  def gaussians(self):
      return Gaussians3D.from_tensordict(self.points.tensors)
      

  def evaluate_colors(self, indexes, cam_idx, camera_position):

    cam_feature = self.image_features[cam_idx.unbind(0)]
    cam_feature = cam_feature.unsqueeze(0).expand(indexes.shape[0], -1)
    dir = F.normalize(self.points.position[indexes].detach() - camera_position)

    feature = torch.cat([dir, self.points.feature[indexes], cam_feature], dim=1)
    return self.color_model(feature).to(torch.float32).sigmoid()
  

  def random_camera(self):
      cam_idx = [torch.randint(0, j, device=self.device) for j in self.image_features.shape]
      return torch.stack(cam_idx)


  def render(self, cam_idx:torch.Tensor, camera_params:CameraParams, 
             **options) -> Rendering:
  
    config = self.raster_config
    indexes = gaussians_in_view(self.points.position, camera_params, config.tile_size, config.margin_tiles)

    features = self.evaluate_colors(indexes, cam_idx, camera_params.camera_position)
    gaussians2d, depthvars = project_to_image(self.gaussians, indexes, camera_params)


    return render_projected(indexes, gaussians2d, features, depthvars, 
                  camera_params, config, **options)




