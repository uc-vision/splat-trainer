from dataclasses import  dataclass, field, replace
from functools import reduce
import operator
from typing import Dict, Optional
from beartype import beartype
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from splat_trainer.camera_table import CameraTable, camera_scene_extents, camera_similarity, Label, Cameras
from splat_trainer.config import Progress, VaryingFloat, eval_varyings
from splat_trainer.logger import Logger
from splat_trainer.scene.transfer_sh import transfer_sh
from splat_trainer.scene.color_model import ColorModel, ColorModelConfig, Colors, GLOTable
from splat_trainer.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import point_basis, split_gaussians_uniform

from splat_trainer.trainer.exception import NaNParameterException
from taichi_splatting.optim import ParameterClass, VisibilityAwareAdam, VisibilityAwareLaProp, VisibilityOptimizer
from taichi_splatting import Gaussians3D, RasterConfig, Rendering, TaichiQueue

from taichi_splatting.renderer import render_projected, project_to_image
from taichi_splatting.perspective import CameraParams

from taichi_splatting.torch_lib.util import check_finite, count_nonfinite

from splat_trainer.scene.util import pop_raster_config
from splat_trainer.util.misc import saturate

@beartype
@dataclass(kw_only=True, frozen=True)
class MLPSceneConfig(GaussianSceneConfig):  
  parameters: DictConfig | Dict
  reg_weight: DictConfig | Dict 
  
  color_model:ColorModelConfig = field(default_factory=ColorModelConfig)

  lr_glo_feature: VaryingFloat = 0.001

  image_features:int       = 8
  point_features:int       = 8

  beta1:float = 0.8
  beta2:float = 0.9

  vis_beta:float = 0.95
  vis_smooth:float = 0.001
  per_image:bool = True

  grad_clip:Optional[float] = 2.0

  autotune:bool = False



  def optim_options(self):
    return dict(optimizer=VisibilityAwareLaProp, betas=(self.beta1, self.beta2), vis_beta=self.vis_beta,
                bias_correction=True, vis_smooth=self.vis_smooth, grad_clip=self.grad_clip) 
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
    return MLPScene(points, self, camera_table, logger)

  
  def from_state_dict(self, state:dict, camera_table:CameraTable, logger:Logger):

    points = ParameterClass.from_state_dict(state['points'], **self.optim_options())
    scene = MLPScene(points, self, camera_table, logger)

    scene._color_model.load_state_dict(state['color_model'])
    scene.color_table.load_state_dict(state['color_table'])
    scene.color_opt.load_state_dict(state['color_opt'])
    scene.glo_opt.load_state_dict(state['glo_opt'])

    return scene


@beartype
class MLPScene(GaussianScene):
  def __init__(self, 
          points: ParameterClass, 
          config: MLPSceneConfig,       
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
    self.glo_opt = self.color_table.optimizer(config.lr_glo_feature)
    
    
    self.scene_extents = camera_scene_extents(camera_table.cameras)
    self.update_learning_rate(0.)


  @property
  def device(self):
    return self.points.position.device

  @property
  def num_points(self):
    return self.points.position.shape[0]

  def __repr__(self):
    return f"MLPScene({self.num_points} points)"

  def update_learning_rate(self, t:float):
    groups = eval_varyings(self.config.parameters, t)


    point_groups = self.points.update_groups(**groups)
    color_groups = self.color_model.schedule(self.color_opt, t)
    glo_groups = self.color_table.schedule(self.glo_opt, self.config.lr_glo_feature, t)

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

    points = self.gaussians.apply(torch.Tensor.detach)
    opacity = torch.sigmoid(points.alpha_logit)
    
    self.logger.log_histogram("params/opacity", opacity)
    self.logger.log_histogram("params/log_scale", self.points.log_scaling)
    self.logger.log_histogram("params/feature", self.points.feature)
    self.logger.log_histogram("params/glo_feature", self.color_table.weight)

    scale = torch.exp(self.points.log_scaling)
    stable_rank =  scale.sum(1) / (scale.max(1).values)

    aspect = scale.max(1).values / (scale.min(1).values + 1e-4)

    self.logger.log_histogram("params/stable_rank", stable_rank)
    self.logger.log_histogram("params/aspect", aspect)
    


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

    if isinstance(self.points.optimizer, VisibilityOptimizer):
      vis_weight = self.points.visible[vis_idx]
      self.points.step(visibility=vis_weight, indexes=vis_idx, basis=basis)
    else:
      self.points.step(indexes=vis_idx, basis=basis)

    self.color_opt.step()
    self.glo_opt.step()
    

    self.points.rotation.data = F.normalize(self.points.rotation.data, dim=1)
    self.points.log_scaling.data.clamp_(min=-8, max=8)
      
    self.zero_grad()
  
  @torch.no_grad()
  def add_rendering(self, image_idx:int, rendering:Rendering):
    points = rendering.points
    self.points.visible[points.idx] += points.visibility

  @torch.compile
  def compute_reg(self, opacity:torch.Tensor, log_scale:torch.Tensor, depths:torch.Tensor, 
                  specular:torch.Tensor, weight:torch.Tensor) -> dict[str, torch.Tensor]:

    scale = torch.exp(log_scale)
    norm_scale =  (scale.pow(2).sum(1) / depths.pow(2).squeeze(-1))

    # stable_rank = scale.sum(1) / scale.max(1).values
    # aspect_term = (stable_rank - 2.0).pow(2) 
    
    aspect_term = (scale.max(1).values / scale.min(1).values)    
    opacity_term = saturate(opacity, gain=4.0, k=2.0) * norm_scale
    spec_term = specular.abs().sum(1)

    return dict(
        scale=(norm_scale * weight).mean(), 
        opacity=(opacity_term * weight).mean(), 
        aspect=(aspect_term * weight).mean(), 
        specular=(spec_term * weight).mean()
      )


  def reg_loss(self, rendering:Rendering, progress:Progress) -> torch.Tensor:
    rendered_points = rendering.points.visible
    log_scale = self.points.log_scaling[rendered_points.idx]


    # if the optimizer is visibility aware, use the visibility as a weight
    # weight = rendered_points.visibility 
    weight = (rendered_points.visibility if isinstance(self.points.optimizer, VisibilityOptimizer) 
              else torch.tensor(1.0, device=self.device))

    regs = self.compute_reg(rendered_points.opacity, log_scale, rendered_points.depths, 
                            rendered_points.attributes.specular, weight)
    
    weights = eval_varyings(self.config.reg_weight, progress.t)
    weighted = {k: v * weights[k] for k, v in regs.items() if k in weights}

    if progress.logging_step:
      with torch.no_grad():
        self.logger.log_values("train/reg", {k:v.item() for k, v in weighted.items()} )

    return reduce(operator.add, weighted.values())


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
  

  def clone(self) -> 'MLPScene':
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
    
    return torch.sum(similarity * glo_feature, dim=0)

  @beartype
  def eval_colors(self, point_indexes:torch.Tensor, camera_params:CameraParams, 
                       image_idx:int | None) -> Colors:
    
    camera:Cameras = self.camera_table.cameras[image_idx]
    if image_idx is not None and camera.count_label(Label.Training) > 0:
      glo_feature = self.lookup_glo_feature(image_idx).unsqueeze(0)
    else:
      # glo_feature = self.interpolated_glo_feature(torch.inverse(camera_params.T_camera_world)).unsqueeze(0)
      glo_feature = torch.zeros((1, self.config.image_features), device=self.device)
      
    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
      feature = self.points.feature[point_indexes]
    
      return self.color_model(feature, 
            self.points.position[point_indexes], 
            camera_params.camera_position, 
            glo_feature)
    
      

  def query_visibility(self, camera_params:CameraParams) -> tuple[torch.Tensor, torch.Tensor]:
    config = RasterConfig(compute_visibility=True)
    
    gaussians2d, depth, indexes = project_to_image(self.gaussians, camera_params, config)
    feature = torch.zeros((indexes.shape[0], 1), device=self.device)
    rendering = render_projected(indexes, gaussians2d, feature, depth, 
                            camera_params, config)
    
    visible = rendering.points.visible
    return visible.idx, visible.visibility


  def evaluate_sh_features(self):
    def f(point_indexes, camera_params, image_idx):
      return self.color_model.post_activation(
        self.eval_colors(point_indexes, camera_params, image_idx).total())

    glo_features = self.lookup_glo_feature(torch.arange(self.camera_table.num_images, device=self.device))
    return transfer_sh(f, self.query_visibility, self.camera_table, 
                        self.points.position, glo_features, epochs=1, sh_degree=2)
      

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
             image_idx:Optional[int] = None,   specular_weight:float = 1.0, **options) -> Rendering:

    config = pop_raster_config(options)
    gaussians2d, depth, indexes = project_to_image(self.gaussians, camera_params, config)

    colors = TaichiQueue.run_sync(self.eval_colors, indexes, camera_params, image_idx)
    rendering = render_projected(indexes, gaussians2d, colors.total(specular_weight), depth, 
                            camera_params, config, **options)
    

    rendering = replace(rendering, 
                        points = rendering.points.replace(attributes=colors),
                        image = self.color_model.post_activation(rendering.image))
    

    return rendering
    


