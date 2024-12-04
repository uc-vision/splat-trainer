from typing import Dict, Tuple
from taichi_splatting import Gaussians3D, Rendering

from taichi_splatting.optim.parameter_class import ParameterClass
from taichi_splatting.optim import SparseAdam
from tensordict import TensorDict
import torch

from splat_trainer.util.misc import lerp


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



def parameters_from_gaussians(gaussians:Gaussians3D, parameters:Dict[str, float], betas:Tuple[float, float]) -> ParameterClass:

    points_dict:TensorDict = gaussians.to_tensordict().update(dict(
      running_depth = torch.zeros(gaussians.batch_size[0], device=gaussians.position.device)))
  
    # parameter_groups = {k:dict(lr=lr, type='vector' if k == 'position' else 'scalar') 
    #                     for k, lr in learning_rates.items()}

      
    return ParameterClass(points_dict, 
          parameter_groups=parameters, 
          optimizer=SparseAdam,
          betas=betas)   