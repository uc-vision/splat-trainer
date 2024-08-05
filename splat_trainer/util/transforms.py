from beartype.typing import Tuple
import torch
import torch.nn.functional as F




def split_rt(
    transform: torch.Tensor,  # (batch_size, 4, 4)
) -> Tuple[torch.Tensor, torch.Tensor]:
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    return R.contiguous(), t.contiguous()

def join_rt(r, t):
  assert r.shape[-2:] == (3, 3), f"Expected (..., 3, 3) tensor, got: {r.shape}"
  assert t.shape[-1] == 3, f"Expected (..., 3) tensor, got: {t.shape}"

  prefix = t.shape[:-1]
  assert prefix == t.shape[:-1], f"Expected same prefix shape, got: {r.shape} {t.shape}"

  T = torch.eye(4, device=r.device, dtype=r.dtype).view((1, ) * (len(prefix)) + (4, 4)).expand(prefix + (4, 4)).contiguous()

  T[..., 0:3, 0:3] = r
  T[..., 0:3, 3] = t
  return T
  

def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return torch.concatenate([points, torch.ones(shape, dtype=points.dtype, device=points.device)], axis=-1)

def transform44(transform, points):

  points = points.reshape([-1, 4, 1])
  transformed = transform.reshape([-1, 4, 4]) @ points
  return transformed[..., 0].reshape(-1, 4)



def transform33(transform, points):

  points = points.reshape([-1, 3, 1])
  transformed = transform.reshape([1, 3, 3]) @ points
  return transformed[..., 0].reshape(-1, 3)


def expand_proj(transform:torch.Tensor, batch_dims=1):
  # expand 3x3 to 4x4 by padding 
  dims = transform.shape[:batch_dims]

  if transform.shape[batch_dims:] == (3, 3):
    expanded = torch.zeros((*batch_dims, 4, 4), dtype=transform.dtype, device=transform.device)

    expanded[..., :3, :3] = transform
    expanded[..., 3, 3] = 1.0
    return expanded
  elif transform.shape[batch_dims:] == (4,):
    expanded = torch.zeros((*dims, 4, 4), dtype=transform.dtype, device=transform.device)

    fx, fy, cx, cy = transform.unbind(-1)
    expanded[..., 0, 0] = fx
    expanded[..., 1, 1] = fy
    expanded[..., 0, 2] = cx
    expanded[..., 1, 2] = cy

    expanded[..., 2, 2] = 1.0
    expanded[..., 3, 3] = 1.0
    return expanded
  else:
    raise ValueError(f"Expected (..., 3,3) matrix or (..., 4) intrinsic (fx, fy, cx, cy) tensor, got: {transform.shape}")

        
