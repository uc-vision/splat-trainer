from typing import Dict, Tuple, Any
from taichi_splatting import Gaussians3D, RasterConfig, Rendering

from taichi_splatting.optim import SparseAdam, ParameterClass, VisibilityAwareLaProp

from tensordict import TensorDict
import torch

from splat_trainer.util.misc import lerp

def pop_raster_config(options:Dict[str, Any]) -> RasterConfig:

  keys = set(RasterConfig.__dataclass_fields__.keys())
  raster_options = {}
  for k, v in options.items():
    if k in keys:
      raster_options[k] = v
    
  for k in raster_options:
    del options[k]

  return RasterConfig(**raster_options)





def update_depth(points:ParameterClass, rendering:Rendering, depth_ema:float):
  """ Method for scaling learning rates by point depth. 
      Take running average of running_depth = depth/fx.

      Scale gradients by 1/running depth and learning rates by running depth.
  """
  assert 'running_depth' in points.keys() and 'position' in points.keys(), \
    f"Points must have running_depth and position, got {points.keys()}"

  fx = rendering.camera.focal_length[0]
  depth_scales = rendering.point_depth[rendering.visible_mask].squeeze(1) / fx  
  
  running_depth = points.running_depth
  running_depth[rendering.visible_indices] = lerp(depth_ema, 
                    depth_scales, running_depth[rendering.visible_indices])

  # Use learning rate proportional to running depth
  points.update_group('position', point_lr=points.running_depth)

  # Scale gradients by 1/rendering depth
  points.position.grad[rendering.visible_indices] /= depth_scales.unsqueeze(1)

  return running_depth

