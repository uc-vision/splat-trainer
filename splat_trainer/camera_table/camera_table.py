

from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn


from splat_trainer.camera_table.pose_table import PoseTable, RigPoseTable
from splat_trainer.util.transforms import expand_proj, make_homog, split_rt, transform44


class CameraTable(nn.Module):

  @abstractmethod
  def forward(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError
  

  @property
  @abstractmethod
  def shape(self) -> torch.Size:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def all_cameras(self):
    raise NotImplementedError

  @property
  @abstractmethod
  def camera_centers(self) -> torch.Tensor:
    raise NotImplementedError
  




def camera_extents(cameras:CameraTable):
    cam_centers = cameras.camera_centers.reshape(-1, 3)
    avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)

    distances = torch.norm(cam_centers - avg_cam_center, dim=0, keepdim=True)
    diagonal = torch.max(distances)

    return avg_cam_center.reshape(3), (diagonal * 1.1).item()


class CameraRigTable(CameraTable):
  def __init__(self, rig_t_world:torch.Tensor,   # (N, 4, 4) - poses for the whole camera rig
                     camera_t_rig:torch.Tensor,  # (C, 4, 4) - camera poses inside the rig
                     projection:torch.Tensor   # (C, 3, 3) - camera intrinsics for each camera in rig
                    ):
    super().__init__()

    self.camera_projection = torch.nn.Parameter(
        projection.to(torch.float32))
    self.camera_poses = RigPoseTable(
      rig_t_world=rig_t_world, camera_t_rig=camera_t_rig)


  def forward(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:     
    return self.camera_poses(image_idx.unsqueeze(0)), self.camera_projection[image_idx[1]]

  @property
  def shape(self) -> torch.Size:
    return torch.Size([self.camera_poses.num_frames, self.camera_poses.num_cameras])
  
  @property
  def camera_centers(self) -> torch.Tensor:

    world_t_camera = torch.linalg.inv(self.camera_poses(self.all_cameras))
    return world_t_camera[..., :3, 3]
  
  @property
  def device(self):
    return self.camera_projection.device
  
  @property
  def all_cameras(self):
    dims = [torch.arange(0, n, device=self.device) for n in self.shape]
    return torch.stack(torch.meshgrid(*dims, indexing='ij'), dim=-1).view(-1, 2)



class MultiCameraTable(CameraTable):
  """
  A table of camera poses and intrinsics - cameras can have different intrinsics which are stored in the projection table.
  """
  def __init__(self, 
               camera_t_world:torch.Tensor, # (N, 4, 4)
               camera_idx:torch.Tensor,     # (N,) - index into projection table (0, P-1)
               projection:torch.Tensor     # (P, 4, 4),
              ):
    super().__init__()

    self.camera_projection = torch.nn.Parameter(
      projection.to(torch.float32))
    self.register_buffer("camera_idx", camera_idx)
    self.camera_t_world = PoseTable(camera_t_world)


  def forward(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      cam_idx = self.camera_idx[image_idx]
      return self.camera_t_world(image_idx), self.camera_projection[cam_idx]

  @property
  def shape(self) -> torch.Size:
    return self.camera_t_world.shape
  
  @property
  def camera_centers(self) -> torch.Tensor:
    world_t_camera = torch.linalg.inv(self.camera_t_world(self.all_cameras))
    return world_t_camera[..., :3, 3]
  
  @property
  def device(self):
    return self.camera_projection.device

  @property
  def all_cameras(self):
    return torch.arange(0, self.shape[0], device=self.camera_projection.device)




def camera_json(camera_table:CameraTable):
  def export_camera(i, idx:torch.Tensor):
    camera_t_world, proj = camera_table(idx)
    r, t = split_rt(torch.linalg.inv(camera_t_world))

    return {
      "id": i,
      "position": t.cpu().numpy().tolist(),
      "rotation": r.cpu().numpy().tolist(),
      "fy": proj[1, 1].item(),
      "fx": proj[0, 0].item(),
      "cx": proj[0, 2].item(),
      "cy": proj[1, 2].item(),
    }

  return [export_camera(i, idx) for i, idx in enumerate(camera_table.all_cameras.unbind(0))]


def visibility(camera_table:CameraTable, points:torch.Tensor, image_size:Tuple[float, float], near=0.1, far=100.0):
  counts = torch.zeros(camera_table.shape, dtype=torch.bool, device=camera_table.device)

  cam_t_world, image_t_cam = camera_table(camera_table.all_cameras)
  image_t_world = expand_proj(image_t_cam) @ cam_t_world

  homog_points = make_homog(points)

  for i in range(image_t_world.shape[0]):
    proj_points = transform44(image_t_world[i], homog_points)
    proj_points = proj_points / proj_points[..., 2:3]

    counts += (
      (proj_points[..., 0] >= 0) & (proj_points[..., 0] < image_size[0]) 
      & (proj_points[..., 1] >= 0) & (proj_points[..., 1] < image_size[1]) 
      & (proj_points[..., 2] > near) & (proj_points[..., 2] < far)
    )

  return counts