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
  transformed = transform.reshape([1, 4, 4]) @ points
  return transformed[..., 0].reshape(-1, 4)



def transform33(transform, points):

  points = points.reshape([-1, 3, 1])
  transformed = transform.reshape([1, 3, 3]) @ points
  return transformed[..., 0].reshape(-1, 3)


