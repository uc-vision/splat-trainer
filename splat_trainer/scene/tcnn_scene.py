from dataclasses import  dataclass, field, replace
from typing import Dict, Optional
from beartype import beartype
from omegaconf import DictConfig

from tensordict import TensorDict
import torch
import torch.nn.functional as F

from splat_trainer.camera_table.camera_table import CameraTable, camera_scene_extents, camera_similarity
from splat_trainer.config import Progress, VaryingFloat,  eval_varyings
from splat_trainer.logger.logger import Logger
from splat_trainer.scene.transfer_sh import transfer_sh
from splat_trainer.scene.color_model import ColorModel, ColorModelConfig, GLOTable
from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import point_basis, split_gaussians_uniform


from taichi_splatting.optim import ParameterClass, VisibilityAwareLaProp, VisibilityAwareAdam, SparseLaProp
from taichi_splatting import Gaussians3D, RasterConfig, Rendering, TaichiQueue
from taichi_splatting.misc.morton_sort import argsort

from taichi_splatting.renderer import render_projected, project_to_image
from taichi_splatting.perspective import CameraParams


from splat_trainer.scene.util import pop_raster_config

@beartype
@dataclass(kw_only=True, frozen=True)
class TCNNConfig(GaussianSceneConfig):  
  parameters : DictConfig | Dict
  
  lr_image_feature: VaryingFloat = 0.001

  image_features:int       = 8
  point_features:int       = 8

  beta1:float = 0.8
  beta2:float = 0.9

  vis_beta:float = 0.95
  vis_smooth:float = 0.001
  per_image:bool = True

  enable_specular:bool = True
  autotune:bool = False

  color_model:ColorModelConfig = field(default_factory=ColorModelConfig)



  def optim_options(self):
    return dict(optimizer=VisibilityAwareAdam, betas=(self.beta1, self.beta2), vis_beta=self.vis_beta,
                bias_correction=True, vis_smooth=self.vis_smooth)

  # def optim_options(self):
  #   return dict(optimizer=SparseLaProp, betas=(self.beta1, self.beta2), bias_correction=True)

  def from_color_gaussians(self, gaussians:Gaussians3D, 
                           camera_table:CameraTable, 
                           device:torch.device,
                           logger:Logger):
    

    feature = torch.zeros(gaussians.batch_size[0], self.point_features)
    torch.nn.init.normal_(feature, std=5.0)


    point_tensors:TensorDict = (gaussians.to_tensordict().replace(
        visible=torch.zeros(gaussians.batch_size[0], device=device), 
        feature=feature)
    ).to(device)

    points = ParameterClass(point_tensors, parameter_groups=eval_varyings(self.parameters, 0.), **self.optim_options())   

    return TCNNScene(points, self, camera_table, logger)

  
  def from_state_dict(self, state:dict, camera_table:CameraTable, logger:Logger):

    points = ParameterClass.from_state_dict(state['points'], **self.optim_options())
    scene = TCNNScene(points, self, camera_table, logger)

    scene._color_model.load_state_dict(state['color_model'])

    scene.color_table.load_state_dict(state['color_table'])
    scene.color_opt.load_state_dict(state['color_opt'])
    scene.glo_opt.load_state_dict(state['glo_opt'])

    return scene


@beartype
class TCNNScene(GaussianScene):
  def __init__(self, 
          points: ParameterClass, 
          config: TCNNConfig,       
          camera_table:CameraTable,   
          logger:Logger,
    ):

    self.config = config
    self.points = points
    self.logger = logger

    self.camera_table = camera_table
    num_glo_embeddings = camera_table.num_images if config.per_image else camera_table.num_projections

    self._color_model = ColorModel(
      config=config.color_model,
      glo_features=config.image_features, 
      point_features=config.point_features).to(self.device)
    
    self.color_opt = self._color_model.optimizer()

    self.color_model = torch.compile(self._color_model, 
            options=dict(max_autotune=config.autotune), dynamic=True)
    # self.color_model = self._color_model

    self.color_table = GLOTable(num_glo_embeddings, config.image_features).to(self.device)
    self.glo_opt = self.color_table.optimizer(config.lr_image_feature)
    
    
    self.scene_extents = camera_scene_extents(camera_table.cameras)
    self.update_learning_rate(0.)


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


    point_groups = self.points.update_groups(**groups)
    color_groups = self.color_model.schedule(self.color_opt, t)
    glo_groups = self.color_table.schedule(self.glo_opt, self.config.lr_image_feature, t)

    # return a dict of all the learning rates
    return dict(**point_groups, 
                **color_groups, 
                **glo_groups)
    
  @torch.no_grad()
  def zero_grad(self):
    self.points.visible.zero_()

    self.points.zero_grad()
    self.color_opt.zero_grad()
    self.glo_opt.zero_grad()

  def hist_log_nonzero(self, name:str, value:torch.Tensor,  min_value:float=1e-16):
    value = value.detach()[value > min_value]
    self.logger.log_histogram(f"{name}", torch.log10(value))

  @torch.no_grad()
  def log_gradients(self, visibility:torch.Tensor, min_vis:float=0.1):   
    vis_idx = torch.nonzero(visibility > min_vis).squeeze(1)
    visibility = visibility[vis_idx].view(vis_idx.shape[0], 1)

    for key, value in self.points.tensors.items():
      if value.grad is not None:
        grad = value.grad[vis_idx].view(vis_idx.shape[0], -1)  

        self.hist_log_nonzero(f"log10_grad/{key}", grad)
        norm_grad = grad / (self.config.vis_smooth + visibility)
        self.hist_log_nonzero(f"log10_norm_grad/{key}", norm_grad)

  @torch.no_grad()
  def log_optimizer_state(self, name:str="optimizer"):

    state = self.points.tensor_state
    if state.batch_dims > 0:
      for key, value in self.points.tensor_state.items():
         for k, v in value.items():
            self.hist_log_nonzero(f"{name}/{key}/{k}", v)
    

  def log_params(self):
    opacity = torch.sigmoid(self.points.alpha_logit.detach())
    
    self.logger.log_histogram("params/opacity", opacity)
    self.logger.log_histogram("params/log_scale", self.points.log_scaling.detach())
    self.logger.log_histogram("params/feature", self.points.feature.detach())

    self.logger.log_histogram("params/glo_feature", self.color_table.weight.detach())


  def log_checkpoint(self, progress:Progress):
    self.log_params()


  @torch.no_grad()
  @beartype
  def step(self, _:Progress, log_details:bool=False):
    vis_idx = self.points.visible.nonzero().squeeze(1)

    basis = point_basis(self.points.log_scaling[vis_idx], self.points.rotation[vis_idx]).contiguous()

    if log_details:
      self.log_gradients(self.points.visible)
      self.log_optimizer_state()
      self.log_params()

    vis_weight = self.points.visible[vis_idx]
    self.points.step(visibility=vis_weight, indexes=vis_idx, basis=basis)

    self.color_opt.step()
    self.glo_opt.step()
    

    self.points.rotation.data = F.normalize(self.points.rotation.data, dim=1)
    self.points.log_scaling.data.clamp_(min=-10)
      
    self.zero_grad()
  
  @torch.no_grad()
  def add_rendering(self, image_idx:int, rendering:Rendering):
    points = rendering.points
    self.points.visible[points.idx] += points.visibility


  @property
  def all_parameters(self) -> TensorDict:
    def from_model(module:torch.nn.Module):
      return TensorDict(dict(module.named_parameters()))
    
    return TensorDict(points=self.points.tensors.to_dict(), 
                            color_model=from_model(self._color_model),
                            glo_opt = from_model(self.color_table))


  @torch.no_grad()
  def split_and_prune(self, keep_mask, split_idx:Optional[torch.Tensor]=None):  
    splits = None
    if split_idx is not None:
      splits = split_gaussians_uniform(
        self.points[split_idx].detach(), k=2, random_axis=True)
          
    self.points = self.points[keep_mask]
    if splits is not None:
      self.points = self.points.append_tensors(splits)

    # idx = argsort(self.points.position, 0.001)
    # self.points = self.points[idx]
 

  def state_dict(self):
    return dict(points=self.points.state_dict(), 
                color_model=self._color_model.state_dict(),
                color_opt = self.color_opt.state_dict(),
                color_table = self.color_table.state_dict(),
                glo_opt = self.glo_opt.state_dict(),
                )
  

  def clone(self) -> 'TCNNScene':
    return self.config.from_state_dict(self.state_dict(), self.camera_table, self.logger)
    



  @beartype
  def lookup_glo_feature(self, image_idx:int | torch.Tensor) -> torch.Tensor:
    
    if not self.config.per_image:
      image_idx = self.camera_table.cameras[image_idx].camera_idx

    return self.color_table(image_idx)
    
  @beartype
  def interpolated_glo_feature(self, camera_t_world:torch.Tensor) -> torch.Tensor:
    cameras = self.camera_table.cameras
    similarity = F.softmax(camera_similarity(cameras, camera_t_world), dim=0)
    
    image_idx = torch.arange(self.camera_table.num_images, device=self.device)
    if not self.config.per_image:
      image_idx = cameras.camera_idx

    glo_feature = self.color_table(image_idx)
    
    return torch.sum(similarity * glo_feature, dim=0).unsqueeze(0)

  @beartype
  def eval_colors(self, point_indexes:torch.Tensor, camera_params:CameraParams, image_idx:int | None):
    if image_idx is not None:
      glo_feature = self.lookup_glo_feature(image_idx)
    else:
      glo_feature = self.interpolated_glo_feature(torch.inverse(camera_params.T_camera_world))


    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
      feature = self.points.feature[point_indexes]
    
      return self.color_model(feature, self.points.position[point_indexes], 
                                                        camera_params.camera_position, glo_feature.unsqueeze(0),
                                                        enable_specular=self.config.enable_specular)

  def query_visibility(self, camera_params:CameraParams) -> tuple[torch.Tensor, torch.Tensor]:
    config = RasterConfig(compute_visibility=True)
    
    gaussians2d, depth, indexes = project_to_image(self.gaussians, camera_params, config)
    feature = torch.zeros((self.num_points, 1), device=self.device)
    rendering = render_projected(indexes, gaussians2d, feature, depth, 
                            camera_params, config)
    
    visible = rendering.points.visible
    return visible.idx, visible.visibility


  def evaluate_sh_features(self):
    def eval_colors(point_indexes, camera_params, image_idx):
      return self.color_model.post_activation(
        self.eval_colors(point_indexes, camera_params, image_idx))

    glo_features = self.lookup_glo_feature(torch.arange(self.camera_table.num_images, device=self.device))
    return transfer_sh(eval_colors, self.query_visibility, self.camera_table, 
                        self.points.position, glo_features, epochs=2, sh_degree=2)
      

  def to_sh_gaussians(self) -> Gaussians3D:
    gaussians:TensorDict = self.points.tensors.select('position', 'rotation', 'log_scaling', 'alpha_logit')
    gaussians = gaussians.replace(feature=self.evaluate_sh_features())    

    return Gaussians3D.from_dict(gaussians, batch_dims=1)
  

  @property
  def gaussians(self) -> Gaussians3D:
      return Gaussians3D(position=self.points.position, 
                         rotation=self.points.rotation, 
                         log_scaling=self.points.log_scaling, 
                         alpha_logit=self.points.alpha_logit, 
                         feature=self.points.feature)


  @beartype
  def render(self, camera_params:CameraParams,  
             image_idx:Optional[int] = None, **options) -> Rendering:

    config = pop_raster_config(options)
    
    gaussians2d, depth, indexes = project_to_image(self.gaussians, camera_params, config)

    colour = TaichiQueue.run_sync(self.eval_colors, indexes, camera_params, image_idx)
    rendering = render_projected(indexes, gaussians2d, colour, depth, 
                            camera_params, config, **options)
    

    rendering = replace(rendering, image = self.color_model.post_activation(rendering.image))
    return rendering
    


