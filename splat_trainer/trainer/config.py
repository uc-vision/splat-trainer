
# Python standard library
from dataclasses import dataclass
from typing import Optional

# Third party packages
from beartype import beartype
from beartype.typing import Optional


from taichi_splatting import RasterConfig

# Local imports


from splat_trainer.config import Between, SmoothStep, VaryingFloat, VaryingInt
from splat_trainer.controller import ControllerConfig
from splat_trainer.scene.scene import GaussianSceneConfig
from splat_trainer.trainer.view_selection import ViewSelectionConfig





@dataclass(kw_only=True, frozen=True)
class CloudInitConfig:
  num_neighbors:int  
  initial_point_scale:float 
  initial_alpha:float 

  limit_points: Optional[int] = None
  initial_points : Optional[int] = None
  
  min_view_overlap: int = 4
  clamp_near: float = 0.0

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

  target_points: int
  densify_interval: VaryingInt = 100

  # Minimum steps per second for a training step, if exceeded training will be aborted
  # this will be checked over 10 log_intervals (running mean)
  min_step_rate : Optional[float] = None
  max_ssim_regression: float = 0.04

  # Evaluation settings - note no images are logged if log_images is False
  num_logged_images: int = 8
  log_worst_images: int  = 2
  log_details: bool = False

  # Loss function settings
  ssim_weight: float = 0.0
  mse_weight: float = 0.0
  l1_weight: float = 0.0
  ssim_levels: int = 4

  # view similarity
  vis_clusters: int = 1024 # number of point clusters to use

  # renderer settings
  antialias:bool = False
  blur_cov:float = 0.3

  # General settings
  device: str

  save_checkpoints: bool = False

  save_output: bool = True  # Save outputs (initial point cloud, cameras, output cloud etc.)
  log_images: bool = True  # If false, disable logging any images to logger