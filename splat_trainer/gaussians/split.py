
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

  return roma.unitquat_to_rotmat(r) * scale.unsqueeze(-2)



def sample_gaussians(points:TensorDict, local_samples:torch.Tensor, n:int=2):
  basis = point_basis(points['log_scaling'], points['rotation'])
  return (basis.repeat_interleave(repeats=n, dim=0) @ local_samples.view(-1, 3).unsqueeze(-1)).view(-1, n, 3)


def split_with_offsets(points: TensorDict, offsets: torch.Tensor) -> TensorDict:
  num_points, n, _ = offsets.shape

  gaussians = points.apply(
    partial(torch.repeat_interleave, repeats=n, dim=0), 
    batch_size=[num_points * n])
  
  return gaussians.update(dict(
      position = gaussians['position'] + offsets.view(-1, 3),
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
  
  offsets = sample_gaussians(points, samples, n)

  return split_with_offsets(scaled, offsets)


def split_gaussians_uniform(points: TensorDict, n:int=2, scaling:Optional[float]=None, sep:float=0.7, random_axis:bool=True) -> TensorDict:
  if random_axis:
    normalized_scaling = F.normalize(points['log_scaling'].exp(), dim=1)

    # Randomly choose axis proportional to the scaling
    axis_probs = normalized_scaling / normalized_scaling.sum(dim=1, keepdim=True)
    axis = torch.multinomial(axis_probs, num_samples=1).squeeze(1)
    axis = F.one_hot(axis, num_classes=3)
  else:
    # Split along most significant axis
    axis = F.one_hot(torch.argmax(points['log_scaling'], dim=1), num_classes=3)

  values = torch.linspace(-sep, sep, n, device=points['position'].device)

  samples = values.view(1, -1, 1) * axis.view(-1, 1, 3)

  if scaling is None:
    scaling = 1 / math.sqrt(n)

  scaled = points.update(
      dict(log_scaling = points['log_scaling'] + math.log(scaling) * axis))

  offsets = sample_gaussians(points, samples, n)
  return split_with_offsets(scaled, offsets)