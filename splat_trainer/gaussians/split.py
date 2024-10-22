
from dataclasses import replace
from functools import partial
import math
from beartype.typing import Optional
from beartype import beartype

from tensordict import TensorDict
import torch
import torch.nn.functional as F
import roma
    

    

def point_basis(log_scaling:torch.Tensor, rotation_quat:torch.Tensor):
  scale = torch.exp(log_scaling)
  r = F.normalize(rotation_quat, dim=1)
  m = roma.unitquat_to_rotmat(r)

  return m.transpose(1, 2) * scale.unsqueeze(-1)


def split_by_samples(points: TensorDict, samples: torch.Tensor) -> TensorDict:
  num_points, n, _ = samples.shape

  basis = point_basis(points['log_scaling'], points['rotation'])
  point_samples = (samples.view(-1, 3).unsqueeze(1) @ basis.repeat_interleave(repeats=n, dim=0)).squeeze(1)

  gaussians = points.apply(
    partial(torch.repeat_interleave, repeats=n, dim=0), 
    batch_size=[num_points * n])
  
  return gaussians.update(dict(
      position = gaussians['position'] + point_samples,
    ))
   

@beartype
def split_gaussians(points: TensorDict, n:int=2, scaling:Optional[float]=None) -> TensorDict:
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

  samples = torch.randn((points.batch_size[0], n, 3), device=points['position'].device) 

  if scaling is None:
    scaling = 1 / math.sqrt(n)

  scaled = points.update(dict(
      log_scaling = points['log_scaling'] + math.log(scaling)))

  return split_by_samples(scaled, samples)


def split_gaussians_uniform(points: TensorDict, n:int=2, scaling:Optional[float]=None, noise=0.0) -> TensorDict:
  """ Split along most significant axis """
  axis = F.one_hot(torch.argmax(points['log_scaling'], dim=1), num_classes=3)
  values = torch.linspace(-1, 1, n, device=points['position'].device)

  samples = values.view(1, -1, 1) * axis.view(-1, 1, 3)
  if noise > 0:
    samples += torch.randn_like(samples) * noise

  if scaling is None:
    scaling = 1 / math.sqrt(n)

  scaled = points.update(
      dict(log_scaling = points['log_scaling'] + math.log(scaling) * axis))

  return split_by_samples(scaled, samples)