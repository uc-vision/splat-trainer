from dataclasses import replace
from numbers import Number
from typing import Tuple
from beartype import beartype
import torch

from splat_trainer.camera_table.camera_table import ViewInfo, ViewTable
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.util.transforms import expand_proj, transform44



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


def projection(camera_table:ViewTable):
  cam_t_world, image_t_cam = camera_table(torch.arange(len(camera_table), device=camera_table.device))
  return expand_proj(image_t_cam) @ cam_t_world


@beartype
def crop_cloud(info:ViewInfo, pcd:PointCloud) -> PointCloud:
    counts = point_visibility(info, pcd.points)
    return pcd[counts > 0]



def inverse_ndc_depth(ndc_depth: torch.Tensor, near: float, far: float) -> torch.Tensor:
  # ndc from 0 to 1 (instead of -1 to 1)
  return (near * far - ndc_depth * near) / (far - ndc_depth * (far - near))


@beartype 
def random_ndc(n, depth_range:Tuple[Number, Number], device=None) -> torch.Tensor:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    depths = torch.rand((n, 1), device=device) 
    return inverse_ndc_depth(depths, *depth_range)


@beartype
def random_points(info:ViewInfo, count:int) -> torch.Tensor:
    
    depth_range = info.depth_range

    world_t_image  = torch.inverse(projection(info.camera_table))
    device  = world_t_image.device

    camera_idx = torch.randint(0, world_t_image.shape[0], (count,), device=device)

    norm_points = torch.rand(count, 2, device=device)
    image_points = norm_points * info.image_sizes[camera_idx]

    # depths = torch.rand((count, 1), device=device) * (depth_range[1] - depth_range[0]) + depth_range[0]
    depths = random_ndc(count, depth_range, device=device)
    ones = torch.ones((count, 1), device=device)

    homog = torch.cat([image_points * depths, depths, ones], dim=1)

    points_unproj = torch.bmm(world_t_image[camera_idx], homog.unsqueeze(2)).squeeze(-1)

    return points_unproj[..., :3] / points_unproj[..., 3:4]


@beartype
def random_cloud(info:ViewInfo, count:int) -> PointCloud:

  points = random_points(info, count)
  colors = torch.rand(count, 3, device=points.device)
  
  return PointCloud(points, colors, batch_size=(count,))



@beartype
def point_visibility(info:ViewInfo, 
                     points:torch.Tensor) -> torch.Tensor:
  
  counts = torch.zeros(points.shape[0], dtype=torch.int32, device=info.camera_table.device)

  cam_t_world, proj = info.camera_table(torch.arange(len(info.camera_table), device=info.camera_table.device))
  image_t_world = expand_proj(proj, 1) @ cam_t_world

  homog_points = make_homog(points)

  for i in range(image_t_world.shape[0]):
    image_size = info.image_sizes[i] if info.image_sizes.dim() > 1 else info.image_sizes

    proj_points = transform44(image_t_world[i], homog_points)
    proj_points = proj_points / proj_points[..., 2:3]

    counts += (
      (proj_points[..., 0] >= 0) & (proj_points[..., 0] < image_size[0]) 
      & (proj_points[..., 1] >= 0) & (proj_points[..., 1] < image_size[1]) 
      & (proj_points[..., 2] > info.depth_range[0]) & (proj_points[..., 2] < info.depth_range[1])
    )

  return counts