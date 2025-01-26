import math
import numpy as np
from taichi_splatting.perspective.params import CameraParams
import torch
import torch.nn.functional as F

from splat_trainer.camera_table.camera_table import Cameras, Camera, Label
from splat_trainer.util.transforms import join_rt
from splat_trainer.visibility.query_points import transform_points


def fit_plane(points:torch.Tensor, normal_hint:torch.Tensor):
  """ Fit a plane to the points."""
  # Compute mean-centered points
  centroid = points.mean(dim=0)
  centered = points - centroid

  # Get principal components using SVD
  U, S, V = torch.svd(centered)
  
  # Normal is the last principal component (smallest singular value)
  normal = V[:, -1]
  
  # Ensure normal points opposite to the mean camera direction
  if torch.dot(normal, normal_hint) > 0:
    normal = -normal
  
  # Create offset using point-normal form of plane equation
  offset = -torch.dot(normal, centroid)
  return torch.cat([normal, offset.unsqueeze(0)])


def plane_distance(point: torch.Tensor,                   # N, 3
                   plane: torch.Tensor,                   # 4
                   ) -> torch.Tensor:                     # N        
    """Returns the distance from the point to the plane."""
    homog = torch.cat([point, torch.ones_like(point[..., :1])], dim=-1)

    distance = torch.einsum('d,nd->n', plane, homog)
    return distance

def closest_point(point: torch.Tensor,  # N, 3
                   plane: torch.Tensor  # 4
                   ) -> torch.Tensor:   # N, 3
    """Returns the closest point on the plane to the given point."""
    distance = plane_distance(point, plane)
    return point - distance.unsqueeze(-1) * plane[:3].unsqueeze(0)


def look_at(pos:torch.Tensor,     # N, 3
            target:torch.Tensor,  # N, 3
            up:torch.Tensor,      # N, 3
            ) -> torch.Tensor:    # N, 4, 4
  """ Returns a camera looking at the target from the position."""
  forward = F.normalize(target - pos, dim=-1)
  left = F.normalize(torch.cross(up, forward, dim=-1), dim=-1)
  true_up = F.normalize(torch.cross(forward, left, dim=-1), dim=-1)
  
  rotation = torch.stack([left, true_up, forward], dim=1)
  return join_rt(rotation, pos)



def fov_to_focal(fov, image_size):
  return image_size / (2 * torch.tan(fov / 2))


def frame_scene(cameras:Cameras, distance:float, longest_side:int, margin:float=0.0) -> CameraParams:
  """ For a scene where all the cameras exist on one side of the scene,
      this function will frame a view which frames the entire scene.

      Args:
        cameras: The cameras to frame.
        distance: The distance from the scene to the camera.

      Returns:
        A camera which frames the scene.
  """
  
  mean_up = cameras.up.mean(dim=0)
  mean_forward = cameras.forward.mean(dim=0)

  # fit a plane to the cameras
  plane = fit_plane(cameras.centers, mean_forward)
  closest_on_plane = closest_point(cameras.centers, plane)
  
  
  center = (closest_on_plane.min(dim=0).values + closest_on_plane.max(dim=0).values) / 2
  viewpoint = center - distance * plane[:3]

  camera_t_world = look_at(viewpoint, center, up=mean_up).inverse()

  on_plane_in_camera = transform_points(camera_t_world, closest_on_plane)[:, :3]
  bounds_cam = 2*margin + (on_plane_in_camera.max(dim=0).values - on_plane_in_camera.min(dim=0).values)[:2]

  if bounds_cam[0] > bounds_cam[1]:
     image_size = (longest_side, int(longest_side * bounds_cam[1] / bounds_cam[0]))
  else:
     image_size = (int(longest_side * bounds_cam[0] / bounds_cam[1]), longest_side)
  
  torch_size = torch.tensor(image_size, device=cameras.device)

  fov = torch.atan2(bounds_cam, torch.full_like(bounds_cam, distance))
  focal_length = fov_to_focal(fov, torch_size)

  return CameraParams(
    projection=torch.cat([focal_length, torch_size * 0.5], dim=-1),
    T_camera_world=camera_t_world,
    image_size=image_size,
    near_plane=distance,
    far_plane=distance + 100.0,
  )


