
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
from splat_trainer.trainer.view_selection import ViewSelectionConfig


@dataclass(kw_only=True, frozen=True)
class CloudInitConfig:
  num_neighbors:int  
  initial_point_scale:float 
  initial_alpha:float 

  limit_points: Optional[int] = None

  initial_points : int 
  add_initial_points: bool = False
  load_dataset_cloud: bool = True

@beartype
@dataclass(kw_only=True, frozen=True)
class TrainConfig:

  scene: GaussianSceneConfig
  controller: ControllerConfig
  view_selection: ViewSelectionConfig

  # Point cloud initialization settings
  cloud_init: CloudInitConfig

  # Scheduling settings
  total_steps: int 
  eval_steps: int    
  log_interval: int  = 10

  densify_interval: VaryingInt = 100

  # Evaluation settings
  num_logged_images: int = 8
  log_worst_images: int  = 2

  # Loss function settings
  ssim_weight: float
  l1_weight: float
  ssim_levels: int = 4
  
  scale_reg: VaryingFloat = 0.0
  opacity_reg: VaryingFloat = 0.0
  aspect_reg: VaryingFloat = 0.0

  # view similarity
  vis_clusters: int = 1024 # number of point clusters to use

  # renderer settings
  antialias:bool = False
  blur_cov:float = 0.3

  # General settings
  device: str

  save_checkpoints: bool = False
  save_output: bool = True