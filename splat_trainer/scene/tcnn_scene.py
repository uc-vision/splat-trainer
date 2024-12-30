from dataclasses import  dataclass
from typing import Dict, Optional
from beartype import beartype
from omegaconf import DictConfig

from tensordict import TensorDict
import torch
import torch.nn.functional as F

from splat_trainer.camera_table.camera_table import CameraTable, camera_scene_extents, camera_similarity
from splat_trainer.config import VaryingFloat,  eval_varyings
from splat_trainer.logger.logger import Logger
from splat_trainer.debug.optim import compare_tensors
from splat_trainer.scene.transfer_sh import transfer_sh
from splat_trainer.scene.color_model import ColorModel, GLOTable
from splat_trainer.scene.scene import GaussianSceneConfig, GaussianScene
from splat_trainer.gaussians.split import point_basis, split_gaussians_uniform


from taichi_splatting.optim import ParameterClass, VisibilityAwareLaProp
from taichi_splatting import Gaussians3D, RasterConfig, Rendering, TaichiQueue, query_visibility

from taichi_splatting.renderer import render_projected, project_to_image
from taichi_splatting.perspective import CameraParams


from splat_trainer.scene.util import pop_raster_config

@beartype
@dataclass(kw_only=True, frozen=True)
class TCNNConfig(GaussianSceneConfig):  
  parameters : DictConfig | Dict
  
  lr_image_feature: VaryingFloat = 0.001
  lr_nn:VaryingFloat = 0.0001

  image_features:int       = 8
  point_features:int       = 8

  hidden_features:int      = 32

  hidden_layers:int        = 2
  sh_degree:int = 5

  beta1:float = 0.8
  beta2:float = 0.9

  vis_beta:float = 0.95
  per_image:bool = True



  def optim_options(self):
    return dict(optimizer=VisibilityAwareLaProp, betas=(self.beta1, self.beta2), vis_beta=self.vis_beta,
                bias_correction=True, vis_smooth=0.01)

  def from_color_gaussians(self, gaussians:Gaussians3D, 
                           camera_table:CameraTable, 
                           device:torch.device):
    

    feature = torch.zeros(gaussians.batch_size[0], self.point_features)
    torch.nn.init.normal_(feature, std=0.5)

    gaussians = gaussians.replace(feature=feature).to(device)
    points = ParameterClass(gaussians.to_tensordict(), 
              parameter_groups=eval_varyings(self.parameters, 0.), **self.optim_options())   

    return TCNNScene(points, self, camera_table)

  
  def from_state_dict(self, state:dict, camera_table:CameraTable):

    points = ParameterClass.from_state_dict(state['points'], **self.optim_options())
    scene = TCNNScene(points, self, camera_table)

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
    ):
    self.config = config
    self.points = points

    self.camera_table = camera_table
    num_glo_embeddings = camera_table.num_images if config.per_image else camera_table.num_cameras


    self._color_model = ColorModel(
      glo_features=config.image_features, 
      point_features=config.point_features, 
      hidden_features=config.hidden_features, 
      hidden_layers=config.hidden_layers,
      sh_degree=config.sh_degree).to(self.device)
    
    self.color_opt = self._color_model.optimizer(config.lr_nn)
    self.color_model = torch.compile(self._color_model, options=dict(max_autotune=True), dynamic=True)

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
    color_groups = self.color_model.schedule(self.color_opt, self.config.lr_nn, t)
    glo_groups = self.color_table.schedule(self.glo_opt, self.config.lr_image_feature, t)

    # return a dict of all the learning rates
    return dict(**point_groups, 
                **color_groups, 
                **glo_groups)
    

  def zero_grad(self):
    self.points.zero_grad()
    self.color_opt.zero_grad()
    self.glo_opt.zero_grad()

  @beartype
  def step(self, rendering:Rendering, t:float) -> Dict[str, float]:

    learning_rates = self.update_learning_rate(t)
    vis_idx, vis_weight = rendering.visible
    basis = point_basis(self.points.log_scaling[vis_idx], self.points.rotation[vis_idx]).contiguous()

    self.points.step(visibility=vis_weight, indexes=vis_idx, basis=basis)

    self.color_opt.step()
    self.glo_opt.step()

    self.points.rotation = torch.nn.Parameter(
      F.normalize(self.points.rotation.detach(), dim=1), requires_grad=True)
      
    self.zero_grad()
    return learning_rates
  

  @property
  def all_parameters(self) -> TensorDict:
    def from_model(module:torch.nn.Module):
      return TensorDict(dict(module.named_parameters()))
    
    return TensorDict(points=self.points.tensors.to_dict(), 
                            color_model=from_model(self._color_model),
                            glo_opt = from_model(self.color_table))


  def split_and_prune(self, keep_mask, split_idx):
    splits = split_gaussians_uniform(
      self.points[split_idx].detach(), n=2, random_axis=True)

    self.points = self.points[keep_mask].append_tensors(splits)
 



  def state_dict(self):
    return dict(points=self.points.state_dict(), 
                color_model=self._color_model.state_dict(),
                color_opt = self.color_opt.state_dict(),
                color_table = self.color_table.state_dict(),
                glo_opt = self.glo_opt.state_dict(),
                )
  

  def clone(self) -> 'TCNNScene':
    return self.config.from_state_dict(self.state_dict(), self.camera_table)
    

  def log(self, logger:Logger, step:int):


    test = self.clone()


    compare_tensors(self.points.tensor_state, test.points.tensor_state)



  @beartype
  def lookup_glo_feature(self, image_idx:int | torch.Tensor) -> torch.Tensor:
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
    
    return torch.sum(similarity * glo_feature, dim=0).unsqueeze(0)


  def eval_colors(self, point_indexes:torch.Tensor, camera_params:CameraParams, image_idx:Optional[int] = None):
    if image_idx is not None:
      glo_feature = self.lookup_glo_feature(image_idx)
    else:
      glo_feature = self.interpolated_glo_feature(torch.inverse(camera_params.T_camera_world))


    with torch.autocast(device_type=self.device.type, dtype=torch.float16):
      return self.color_model(self.points.feature[point_indexes], self.points.position[point_indexes], 
                                                        camera_params.camera_position, glo_feature)

  def query_visibility(self, camera_params:CameraParams, threshold:float = 0.001):
    config = RasterConfig()
    
    gaussians2d, depth, indexes = project_to_image(self._gaussians, camera_params, config)
    visibility = query_visibility(gaussians2d, depth, camera_params.image_size, config)

    visible = visibility > threshold
    return indexes[visible], visibility[visible]


  def evaluate_sh_features(self):
      glo_features = self.lookup_glo_feature(torch.arange(self.camera_table.num_images, device=self.device))
      return transfer_sh(self.eval_colors, self.query_visibility, self.camera_table, 
                         self.points.position, glo_features, epochs=2, sh_degree=2)
        

  def to_sh_gaussians(self) -> Gaussians3D:
    gaussians:TensorDict = self.points.tensors.select('position', 'rotation', 'log_scaling', 'alpha_logit')
    gaussians = gaussians.replace(feature=self.evaluate_sh_features())    

    return Gaussians3D.from_dict(gaussians, batch_dims=1)
  

  @property
  def _gaussians(self) -> Gaussians3D:
      points = self.points.tensors.select('position', 'rotation', 'log_scaling', 'alpha_logit', 'feature')
      return Gaussians3D.from_tensordict(points)

  @beartype
  def render(self, camera_params:CameraParams,  
             image_idx:Optional[int] = None, **options) -> Rendering:

    config = pop_raster_config(options)
    gaussians2d, depth, indexes = project_to_image(self._gaussians, camera_params, config)


    colour = TaichiQueue.run_sync(self.eval_colors, indexes, camera_params, image_idx)
    raster = render_projected(indexes, gaussians2d, colour, depth, 
                            camera_params, config, **options)


    return raster
    

