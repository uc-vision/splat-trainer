

from abc import abstractmethod
from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import torch
from torch import nn


from splat_trainer.camera_table.pose_table import PoseTable, RigPoseTable
from splat_trainer.util.transforms import split_rt

from beartype import beartype


class CameraTable(nn.Module):

  @abstractmethod
  def forward(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError


  def lookup(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ Convenience function to get camera pose and projection for a single image index."""
    camera_t_world, proj = self(image_idx.unsqueeze(0))
    return camera_t_world.squeeze(0), proj.squeeze(0)

  @property
  @abstractmethod
  def shape(self) -> torch.Size:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def all_cameras(self) -> torch.Tensor:
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
                     projection:torch.Tensor   # (C, 4) - camera intrinsics (fx, fy, cx, cy) for each camera in rig
                    ):
    super().__init__()

    self.camera_projection = torch.nn.Parameter(
        projection.to(torch.float32))
    self.camera_poses = RigPoseTable(
      rig_t_world=rig_t_world, camera_t_rig=camera_t_rig)


  def forward(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:     
    assert image_idx.dim() == 2 and image_idx.shape[1] == 2, f"Expected 2D Nx2 tensor, got: {image_idx.shape}"

    return self.camera_poses(image_idx), self.camera_projection[image_idx[:, 1]]

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
               projection:torch.Tensor     # (P, 4) fx, fy, cx, cy
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
    camera_t_world, proj = [x.squeeze(0) for x in camera_table(idx.unsqueeze(0))]

    r, t = split_rt(torch.linalg.inv(camera_t_world))
    fx, fy, cx, cy = proj.cpu().tolist()

    return {
      "id": i,
      "position": t.cpu().numpy().tolist(),
      "rotation": r.cpu().numpy().tolist(),
      "fx": fx, "fy": fy, 
      "cx": cx, "cy": cy
      }

  return [export_camera(i, idx) 
          for i, idx in enumerate(camera_table.all_cameras.unbind(0))
        ]




@beartype
@dataclass 
class CameraInfo:
  camera_table:CameraTable
  image_sizes:torch.Tensor
  depth_range:Tuple[Number, Number]

  def to(self, device) -> 'CameraInfo':
    return CameraInfo(
      camera_table=self.camera_table.to(device),
      image_sizes=self.image_sizes.to(device),
      depth_range=self.depth_range,
    )