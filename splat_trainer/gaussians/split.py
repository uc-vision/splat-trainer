
from dataclasses import replace
from functools import partial
import math
from beartype.typing import Optional
from beartype import beartype

import torch
import torch.nn.functional as F
from taichi_splatting import Gaussians3D
import roma
    

def point_basis(points:Gaussians3D):
  scale = torch.exp(points.log_scaling)

  r = F.normalize(points.rotation, dim=1)
  m = roma.unitquat_to_rotmat(r)


  return m.transpose(1, 2) * scale.unsqueeze(-1)


def split_by_samples(points: Gaussians3D, samples: torch.Tensor) -> Gaussians3D:
  num_points, n, _ = samples.shape

  basis = point_basis(points)
  point_samples = (samples.view(-1, 3).unsqueeze(1) @ basis.repeat_interleave(repeats=n, dim=0)).squeeze(1)

  gaussians = points.apply(
    partial(torch.repeat_interleave, repeats=n, dim=0), 
    batch_size=[num_points * n])
  
  return replace(gaussians,
    position = gaussians.position + point_samples,
    batch_size=(num_points * n, ))
   

@beartype
def split_gaussians(points: Gaussians3D, n:int=2, scaling:Optional[float]=None) -> Gaussians3D:
  """
  Toy implementation of the splitting operation used in gaussian-splatting,
  returns a scaled, randomly sampled splitting of the gaussians.

  Args:
      points: The Gaussians3D parameters to be split
      n: number of gaussians to split into
      scale: scale of the new gaussians relative to the original

  Returns:
      Gaussians2D: the split gaussians 
  """

  samples = torch.randn((points.batch_size[0], n, 3), device=points.position.device) 

  if scaling is None:
    scaling = 1 / math.sqrt(n)

  points = replace(points, 
      log_scaling = points.log_scaling + math.log(scaling),
      batch_size = points.batch_size)

  return split_by_samples(points, samples)


def split_gaussians_uniform(points: Gaussians3D, n:int=2, scaling:Optional[float]=None, noise=0.1) -> Gaussians3D:
  """ Split along most significant axis """
  axis = F.one_hot(torch.argmax(points.log_scaling, dim=1), num_classes=3)
  values = torch.linspace(-1, 1, n, device=points.position.device)

  samples = values.view(1, -1, 1) * axis.view(-1, 1, 3)
  samples += torch.randn_like(samples) * noise

  if scaling is None:
    scaling = 1 / math.sqrt(n)

  points = replace(points, 
      log_scaling = points.log_scaling + math.log(scaling) * axis,
      batch_size = points.batch_size)

  return split_by_samples(points, samples)