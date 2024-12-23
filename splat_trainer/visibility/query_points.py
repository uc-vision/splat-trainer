from dataclasses import replace
from numbers import Number
from typing import Optional, Tuple
from beartype import beartype
import torch

from splat_trainer.camera_table.camera_table import  Camera, Cameras, CameraTable
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


def projection(cameras:Cameras):
  cam_t_world, image_t_cam = cameras.camera_t_world, cameras.projection.matrix
  return expand_proj(image_t_cam) @ cam_t_world


@beartype
def crop_cloud(cameras:Cameras, pcd:PointCloud) -> PointCloud:
    counts = point_visibility(cameras, pcd.points)
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
def random_points(cameras:Cameras, count:int) -> torch.Tensor:
    
    depth_range = cameras[0].depth_range

    world_t_image  = torch.inverse(projection(cameras))
    device  = world_t_image.device

    camera_idx = torch.randint(0, world_t_image.shape[0], (count,), device=device)

    norm_points = torch.rand(count, 2, device=device)
    image_points = norm_points * cameras.image_sizes[camera_idx]

    depths = random_ndc(count, depth_range, device=device)
    ones = torch.ones((count, 1), device=device)

    homog = torch.cat([image_points * depths, depths, ones], dim=1)
    points_unproj = torch.bmm(world_t_image[camera_idx], homog.unsqueeze(2)).squeeze(-1)
    return points_unproj[..., :3] / points_unproj[..., 3:4]


@beartype
def random_cloud(cameras:Cameras, count:int, seed:int=0) -> PointCloud:

  if seed is not None:
    torch.manual_seed(seed)

  points = random_points(cameras, count)
  colors = torch.rand(count, 3, device=points.device)
  
  return PointCloud(points, colors, batch_size=(count,))


@beartype
def point_visibility(cameras:Cameras, 
                     points:torch.Tensor, 
                     far_threshold:Optional[float]=None, 
                     quantile:float=1.0) -> torch.Tensor:
  
  vis_counts = torch.zeros(points.shape[0], dtype=torch.int32, device=cameras.device)
  image_t_world = expand_proj(cameras.projection.matrix, 1) @ cameras.camera_t_world

  homog_points = make_homog(points)

  for i in range(cameras.batch_size[0]):
    camera:Camera = cameras[i].item()
    
    w, h = camera.image_size
    near, far = camera.depth_range

    proj_points = transform44(image_t_world[i], homog_points)
    depth = proj_points[..., 2]
    xy = proj_points[..., :2] / depth.unsqueeze(-1)

    view_mask = (
      (xy[..., 0] >= 0) & (xy[..., 0] < w) 
      & (xy[..., 1] >= 0) & (xy[..., 1] < h) 
      & (depth > near) & (depth < far)
    )

    if far_threshold is None:
      far_threshold = torch.quantile(depth[view_mask], quantile)

    near_mask = view_mask & (depth < far_threshold)
    vis_counts[near_mask] += 1

  return vis_counts



@beartype
def foreground_points(cameras:Cameras, points:torch.Tensor, 
                      far_threshold:Optional[float]=None, quantile:float=0.25, min_overlap:float=0.01) -> torch.Tensor:
  
  near_counts = point_visibility(cameras, points, far_threshold, quantile=quantile)
  num_views = cameras.batch_size[0]

  return near_counts > (min_overlap * num_views)
