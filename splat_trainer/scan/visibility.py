from beartype.typing import List
import torch
import numpy as np

from camera_geometry import Camera

def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return torch.concatenate([points, torch.ones(shape, dtype=torch.float32, device=points.device)], axis=-1)

def _transform_points(transform, points):
  assert points.shape[
      -1] == 3, 'transform_points: expected 3d points of ...x3, got:' + str(
          points.shape)

  homog = make_homog(points).reshape([-1, 4, 1])
  transformed = transform.reshape([1, 4, 4]) @ homog
  return transformed[..., 0].reshape(-1, 4)

def project_points(transform, xyz):
  homog = _transform_points(transform, xyz)
  depth = homog[..., 2:3]
  xy = homog[..., 0:2] 
  return (xy / depth), depth


def visibility(cameras:List[Camera], points:torch.Tensor, device='cuda:0', near=0.1):
  torch_points = points.to(dtype=torch.float32, device=device)
  counts = torch.zeros(len(points), dtype=torch.int32, device=device)
  
  projections = np.array([camera.projection for camera in cameras])
  torch_projections = torch.from_numpy(projections).to(dtype=torch.float32, device=device)

  for camera, proj in zip(cameras, torch_projections):
  
    proj, depth = project_points(proj, torch_points)
    width, height = camera.image_size

    valid = (
      (proj[:, 0] >= 0) & (proj[:, 0] < width) & 
      (proj[:, 1] >= 0) & (proj[:, 1] < height) & 
      (depth[:, 0] > near))
    
    counts[valid] += 1  
  return counts.cpu()