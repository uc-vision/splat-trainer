
# Python standard library
from dataclasses import dataclass
from typing import Optional

# Third party packages
from beartype import beartype
from beartype.typing import Optional


from taichi_splatting import RasterConfig

# Local imports


from splat_trainer.config import VaryingFloat, VaryingInt
from splat_trainer.controller import ControllerConfig
from splat_trainer.scene.scene import GaussianSceneConfig


@beartype
@dataclass(kw_only=True, frozen=True)
class TrainConfig:
  device: str

  steps: int 
  scene: GaussianSceneConfig
  controller: ControllerConfig

  load_model: Optional[str] = None

  num_neighbors:int  
  initial_point_scale:float 
  initial_alpha:float 

  limit_points: Optional[int] = None

  initial_points : int 
  add_initial_points: bool = False
  load_dataset_cloud: bool = True

  eval_steps: int    
  log_interval: int  = 20
  batch_size: int = 1


  num_logged_images: int = 8
  log_worst_images: int  = 2

  densify_interval: VaryingInt = 100

  ssim_weight: float
  l1_weight: float
  ssim_levels: int = 4

  raster_config: RasterConfig = RasterConfig()
  
  scale_reg: VaryingFloat = 0.1
  opacity_reg: VaryingFloat = 0.01
  aspect_reg: VaryingFloat = 0.01

  # view similarity
  vis_clusters: int = 1024 # number of point clusters to use

  # amount of randomness in batch view selection, 0 is deterministic, 1 is quite random
  overlap_temperature: float = 0.2

  # minimum cluster size
  min_group_size: int = 25

  # minimum overlap between master view and other views 
  overlap_threshold: float = 0.5


  # renderer settings
  antialias:bool = False
  blur_cov:float = 0.3

  save_checkpoints: bool = False
  save_output: bool = True
  evaluate_first: bool = False