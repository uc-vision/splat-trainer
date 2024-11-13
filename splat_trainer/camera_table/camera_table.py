

from abc import abstractmethod
import abc
from dataclasses import dataclass
from numbers import Number
from typing import Tuple

import torch
from torch import nn


from splat_trainer.camera_table.pose_table import PoseTable, RigPoseTable
from splat_trainer.util.transforms import split_rt

from beartype import beartype


class ViewTable(nn.Module, metaclass=abc.ABCMeta):

  @abstractmethod
  def forward(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError

  @beartype
  def lookup(self, image_idx:int | torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ Convenience function to get camera pose and projection for a single image index."""

    if isinstance(image_idx, int):
      image_idx = torch.tensor([image_idx], device=self.device)

    assert image_idx.shape == (1,), f"Expected singleton tensor, got: {image_idx.shape}"

    camera_t_world, proj = self(image_idx) # add batch dimension
    return camera_t_world.squeeze(0), proj.squeeze(0)

    


  def __len__(self):
    return self.num_images
  
  @property
  @abstractmethod
  def num_cameras(self) -> int:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def num_frames(self) -> int:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def num_images(self) -> int:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def lookup_projection(self, camera_idx:int) -> torch.Tensor:
    pass
  
  @property
  @abstractmethod
  def camera_centers(self) -> torch.Tensor:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def device(self):
    raise NotImplementedError
  
  
  @property
  @abstractmethod
  def frame_id(self, image_idx):
    """ Returns tuple of frame_id for a given image index/indices."""
    raise NotImplementedError

  @property
  @abstractmethod
  def camera_id(self, image_idx):
      """ Returns tuple of frame_id for a given image index/indices."""
      raise NotImplementedError
  
  @property
  def all_images(self):
    return torch.arange(0, self.num_images, device=self.device)
  
  @property
  def camera_poses(self) -> PoseTable:
    camera_t_world, _ = self(self.all_images)
    return camera_t_world
  

@beartype
def camera_scene_extents(cameras:ViewTable):
    """ 
    Compute centroid and diagonal of camera centers.
    """

    cam_centers = cameras.camera_centers.reshape(-1, 3)
    avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)

    distances = torch.norm(cam_centers - avg_cam_center, dim=0, keepdim=True)
    diagonal = torch.max(distances)

    return avg_cam_center.reshape(3), (diagonal * 1.1).item()


@beartype
def pose_adjacency(poses1:torch.Tensor, poses2:torch.Tensor) -> torch.Tensor:
  """ Compute adjacency matrix between all camera poses in the table."""

  dir_similarity = torch.dot(poses1[..., :3, 2], poses2[..., :3, 2].unsqueeze(0), dim=1)
  distance = torch.norm(poses1[..., :3, 3] - poses2[..., :3, 3].unsqueeze(0), dim=1)
  
  return dir_similarity * distance


@beartype
def camera_similarity(camera_table:ViewTable, pose:torch.Tensor) -> torch.Tensor:
  """ Compute similarity between a camera pose and all camera poses in the table."""
  poses = camera_table.camera_poses
  return pose_adjacency(poses, pose.unsqueeze(0))
  

class CameraRigTable(ViewTable):
  def __init__(self, rig_t_world:torch.Tensor,   # (N, 4, 4) - poses for the whole camera rig
                     camera_t_rig:torch.Tensor,  # (C, 4, 4) - camera poses inside the rig
                     projection:torch.Tensor,   # (C, 4) - camera intrinsics (fx, fy, cx, cy) for each camera in rig
                    ):
    super().__init__()

    assert projection.shape[0] == camera_t_rig.shape[0], f"Expected equal number of cameras and projections, got: {projection.shape[0]} != {camera_t_rig.shape[0]}"

    self.camera_projection = torch.nn.Parameter(
        projection.to(torch.float32), requires_grad=False)
    self.camera_poses = RigPoseTable(
      rig_t_world=rig_t_world, camera_t_rig=camera_t_rig)
    
    
  @beartype
  def forward(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:     
    assert image_idx.dim() == 1, f"Expected 1D tensor, got: {image_idx.shape}"
    assert (image_idx < self.num_images).all(), f"Image index out of range: {image_idx} >= {self.shape[0]}"

    frame_id, camera_id = self.frame_id(image_idx), self.camera_id(image_idx)
    return self.camera_poses(torch.stack([frame_id, camera_id], -1)), self.camera_projection[camera_id]


  @property
  def num_images(self):
    """ Total number of images."""
    return self.camera_poses.num_frames * self.camera_poses.num_cameras
  
  @property
  def num_cameras(self) -> int:
    return self.camera_poses.num_cameras
  
  def lookup_projection(self, camera_idx:int):
    return self.camera_projection[camera_idx]

  @property
  def get_projection(self, camera_idx:int) -> torch.Tensor:
    assert camera_idx < self.num_cameras, f"Camera index out of range: {camera_idx} >= {self.num_cameras}"
    return self.camera_projection[camera_idx]
  
  @property
  def num_frames(self) -> int:
    return self.camera_poses.num_frames
  
  
  def frame_id(self, image_idx):
    """ Returns tuple of frame_id for a given image index/indices."""
    return image_idx // self.num_cameras
    
  def camera_id(self, image_idx):
    """ Returns tuple of frame_id for a given image index/indices."""
    return image_idx % self.num_cameras
  
  @property
  def camera_centers(self) -> torch.Tensor:
    frame_id, camera_id = self.frame_id(self.all_images), self.camera_id(self.all_images)

    world_t_camera = torch.linalg.inv(self.camera_poses(torch.stack([frame_id, camera_id], -1)))
    return world_t_camera[..., :3, 3]
  
  @property
  def device(self):
    return self.camera_projection.device
  




class MultiCameraTable(ViewTable):
  """
  A table of camera poses and intrinsics - cameras can have different intrinsics which are stored in the projection table.
  """
  def __init__(self, 
               camera_t_world:torch.Tensor, # (N, 4, 4)
               camera_idx:torch.Tensor,     # (N,) - index into projection table (0, P-1)
               projection:torch.Tensor     # (P, 4) fx, fy, cx, cy
              ):
    super().__init__()

    assert camera_t_world.shape[0] == camera_idx.shape[0], f"Expected equal number of cameras and indices, got: {camera_t_world.shape[0]} != {camera_idx.shape[0]}"

    self.camera_projection = torch.nn.Parameter(
      projection.to(torch.float32), requires_grad=False)
    self.register_buffer("camera_idx", camera_idx)
    self.camera_t_world = PoseTable(camera_t_world)

  @beartype
  def forward(self, image_idx:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert image_idx.dim() == 1, f"Expected 1D tensor, got: {image_idx.shape}"

    cam_idx = self.camera_idx[image_idx]
    return self.camera_t_world(image_idx), self.camera_projection[cam_idx]


  @property
  def num_images(self) -> int:
    return len(self.camera_t_world)
  
  @property
  def num_cameras(self) -> int:
    return self.camera_projection.shape[0]
  
  def lookup_projection(self, camera_idx:int):
    return self.camera_projection[camera_idx]
  
  @property
  def num_frames(self) -> int:
    return self.num_images
  
  @property
  def camera_id(self, image_idx):
    return self.camera_idx[image_idx]

  @property
  def frame_id(self, image_idx):
    return image_idx

  @property
  def camera_centers(self) -> torch.Tensor:

    all_cameras = torch.arange(0, self.num_images, device=self.device)
    world_t_camera = torch.linalg.inv(self.camera_t_world(all_cameras))
    return world_t_camera[..., :3, 3]
  
  @property
  def device(self):
    return self.camera_projection.device



  


def camera_json(camera_table:ViewTable):
  def export_camera(i):

    camera_t_world, proj = camera_table.lookup(i)

    r, t = split_rt(torch.linalg.inv(camera_t_world))
    fx, fy, cx, cy = proj.cpu().tolist()

    return {
      "id": i,
      "position": t.cpu().numpy().tolist(),
      "rotation": r.cpu().numpy().tolist(),
      "fx": fx, "fy": fy, 
      "cx": cx, "cy": cy
      }

  return [export_camera(i) for i in range(len(camera_table))]




@beartype
@dataclass 
class ViewInfo:
  camera_table:ViewTable
  image_sizes:torch.Tensor
  depth_range:Tuple[Number, Number]

  def to(self, device) -> 'ViewInfo':
    return ViewInfo(
      camera_table=self.camera_table.to(device),
      image_sizes=self.image_sizes.to(device),
      depth_range=self.depth_range,
    )