from dataclasses import dataclass
from numbers import Number
from typing import Iterator, Optional, Tuple
from beartype import beartype
import torch

from splat_trainer.camera_table.camera_table import  Camera, Cameras
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.util.transforms import expand_proj, transform44
import torch.nn.functional as F


def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return torch.concatenate([points, torch.ones(shape, dtype=torch.float32, device=points.device)], axis=-1)

def transform_points(transform, points):
  assert points.shape[
      -1] == 3, 'transform_points: expected 3d points of ...x3, got:' + str(
          points.shape)

  homog = make_homog(points).reshape([-1, 4, 1])
  transformed = transform.reshape([1, 4, 4]) @ homog
  return transformed[..., 0].reshape(-1, 4)

def project_points(transform, xyz):
  homog = transform_points(transform, xyz)
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


@dataclass
class Projected:
  camera:Camera

  xy:torch.Tensor
  depths:torch.Tensor

  @property
  def scale(self) -> torch.Tensor:
    return self.camera.focal_length[0] / self.depths[self.visible_mask]
  
  @property
  def visible_mask(self) -> torch.Tensor:
    w, h = self.camera.image_size
    near, far = self.camera.depth_range

    return (
      (self.xy[..., 0] >= 0) & (self.xy[..., 0] < w) 
      & (self.xy[..., 1] >= 0) & (self.xy[..., 1] < h) 
      & (self.depths > near) & (self.depths < far)
    )

  
def projections(cameras:Cameras, points:torch.Tensor) -> Iterator[Projected]:
  image_t_world = expand_proj(cameras.projection.matrix, 1) @ cameras.camera_t_world
  homog_points = make_homog(points)

  for i in range(cameras.batch_size[0]):
    camera:Camera = cameras[i].item()

    proj_points = transform44(image_t_world[i], homog_points)
    depth = proj_points[..., 2]
    xy = proj_points[..., :2] / depth.unsqueeze(-1)

    yield Projected(camera, xy, depth)



@beartype
def point_visibility(cameras:Cameras, points:torch.Tensor) -> torch.Tensor:
  vis_counts = torch.zeros(points.shape[0], dtype=torch.int32, device=cameras.device)
  for proj in projections(cameras, points):
    vis_counts[proj.visible_mask] += 1
  return vis_counts


@beartype
def camera_counts(cameras:Cameras, points:torch.Tensor) -> torch.Tensor:
  cam_counts = torch.zeros(cameras.batch_size[0], dtype=torch.int32, device=cameras.device)
  for i, proj in enumerate(projections(cameras, points)):
    cam_counts[i] = proj.visible_mask.sum()

  return cam_counts

@beartype 
def random_ndc(n, depth_range:Tuple[Number, Number], device=None) -> torch.Tensor:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    depths = torch.rand((n, 1), device=device) 
    return inverse_ndc_depth(depths, *depth_range)


@beartype
def random_points(cameras:Cameras, count:int, weighting:Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Generate random points in cameras.
    """
    camera:Camera = cameras[0].item()
    near, far = camera.depth_range

    world_t_image  = torch.inverse(projection(cameras))
    device  = world_t_image.device

    if weighting is None:
      camera_idx = torch.randint(0, world_t_image.shape[0], (count,), device=device)
    else:
      camera_idx = torch.multinomial(F.normalize(weighting, p=1, dim=0), count, replacement=True)

    norm_points = torch.rand(count, 2, device=device)
    image_points = norm_points * cameras.image_sizes[camera_idx]

    depths = random_ndc(count, (near, far), device=device)
    ones = torch.ones((count, 1), device=device)

    homog = torch.cat([image_points * depths, depths, ones], dim=1)
    points_unproj = torch.bmm(world_t_image[camera_idx], homog.unsqueeze(2)).squeeze(-1)
    return points_unproj[..., :3] / points_unproj[..., 3:4]



def balanced_points(cameras:Cameras, count:int, min_overlap:int=4, existing_points:Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random points in cameras, each point is visible in at least min_views cameras.
    Attempt to balance the number of points visible in each camera.
    """

    if existing_points is not None:
      valid_points = existing_points
      cam_counts = camera_counts(cameras, valid_points)
    else:
      valid_points = torch.empty((0, 3), device=cameras.device)
      cam_counts = torch.zeros(cameras.batch_size[0], dtype=torch.int32, device=cameras.device)


    while valid_points.shape[0] < count:
      points = random_points(cameras, count // 8, weighting= 1 / (cam_counts + 1))
      points = points[point_visibility(cameras, points) >= min_overlap]

      cam_counts += camera_counts(cameras, points)
      valid_points = torch.cat([valid_points, points])

    return valid_points[:count], cam_counts

@beartype
def random_cloud(cameras:Cameras, count:int) -> PointCloud:

  points = random_points(cameras, count)
  colors = torch.rand(count, 3, device=points.device)
  
  return PointCloud(points, colors, batch_size=(count,))


@beartype
def balanced_cloud(cameras:Cameras, count:int, min_overlap:int=4, existing_points:Optional[PointCloud]=None) -> PointCloud:

  if existing_points is not None:
    points, _ = balanced_points(cameras, count, min_overlap, existing_points.points)
    colors = existing_points.colors[:count]
    if colors.shape[0] < count:
      colors = torch.cat([colors, torch.rand(count - colors.shape[0], 3, device=colors.device)])

  else:
    points, _ = balanced_points(cameras, count, min_overlap)
    colors = torch.rand(count, 3, device=points.device)

  return PointCloud(points, colors, batch_size=(count,))


@beartype
def foreground_visibility(cameras:Cameras, 
                     points:torch.Tensor, 
                     far_threshold:Optional[float]=None, 
                     quantile:float=1.0) -> torch.Tensor:
  
  vis_counts = torch.zeros(points.shape[0], dtype=torch.int32, device=cameras.device)

  for proj in projections(cameras, points):
    if far_threshold is None:
      far_threshold = torch.quantile(proj.depths[proj.visible_mask], quantile)

    near_mask = proj.visible_mask & (proj.depths < far_threshold)
    vis_counts[near_mask] += 1
  return vis_counts



@beartype
def foreground_points(cameras:Cameras, points:torch.Tensor, 
                      far_threshold:Optional[float]=None, quantile:float=0.25, min_overlap:float=0.01) -> torch.Tensor:
  
  near_counts = foreground_visibility(cameras, points, far_threshold, quantile=quantile)
  num_views = cameras.batch_size[0]

  return near_counts > (min_overlap * num_views)


