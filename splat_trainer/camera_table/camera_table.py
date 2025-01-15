from abc import abstractmethod
import abc
from dataclasses import dataclass, replace
from enum import Flag
from functools import cached_property
import math
from typing import List, Tuple

import numpy as np
from tensordict import tensorclass
from tensordict import TensorDictParams
import torch
from torch import nn

from splat_trainer.camera_table.pose_table import PoseTable, RigPoseTable
from splat_trainer.util.transforms import join_rt, split_rt

from taichi_splatting.perspective import CameraParams

from beartype import beartype

class Label(Flag):
  """ Labels for camera tables. Bitwise encoded."""
  Validation = 1 << 0
  Training = 1 << 1
  




def to_matrix(intrinsics:torch.Tensor) -> torch.Tensor:
  fx, fy, cx, cy = intrinsics.unbind(-1)
  m = torch.eye(3, dtype=intrinsics.dtype, device=intrinsics.device)
  m = m.unsqueeze(0).expand(intrinsics.shape[:-1] + (3, 3)).clone()
  m[..., 0, 0] = fx
  m[..., 1, 1] = fy
  m[..., 0, 2] = cx
  m[..., 1, 2] = cy
  return m


@tensorclass
class Projections:
  """ Projection parameters for a camera or a batch of cameras."""
  intrinsics:torch.Tensor      # (..., 4) fx, fy, cx, cy
  image_size:torch.Tensor      # (..., 2) int
  depth_range:torch.Tensor     # (..., 2) float

  @property
  def matrix(self) -> torch.Tensor:
    return to_matrix(self.intrinsics)
  
  
  @property
  def focal_length(self) -> torch.Tensor:
    return self.intrinsics[..., :2]
  
  @property
  def principal_point(self) -> torch.Tensor:
    return self.intrinsics[..., 2:]
  
  @property
  def aspect_ratio(self) -> torch.Tensor:
    return self.image_size[..., 0] / self.image_size[..., 1]

  @property
  def fov(self) -> torch.Tensor:
    f = self.focal_length
    return 2.0 * torch.atan(0.5 * self.image_size / f)



@dataclass(kw_only=True)
class Camera:
  """ Convenience class for a single camera."""
  intrinsics: torch.Tensor      # (4,) float
  camera_t_world:torch.Tensor   # (4, 4) float

  image_size:Tuple[int, int]
  depth_range:Tuple[float, float]

  camera_idx: int
  frame_idx: int
  label: Label

  image_name: str

  @property
  def device(self):
    return self.camera_t_world.device
  
  @property
  def matrix(self) -> torch.Tensor:
    return to_matrix(self.intrinsics)

  @property
  def position(self) -> torch.Tensor:
    return self.world_t_camera[..., :3, 3]

  @property
  def world_t_camera(self) -> torch.Tensor:
    return torch.inverse(self.camera_t_world)

  def translate(self, vector:torch.Tensor) -> 'Camera':
    position = self.position + (self.world_t_camera[..., :3, :3] @ vector.unsqueeze(-1)).squeeze(-1)
    return replace(self, camera_t_world=torch.linalg.inv(join_rt(self.rotation, position)))

  @property
  def rotation(self) -> torch.Tensor:
    return self.world_t_camera[..., :3, :3].transpose(-1, -2)
  
  @property
  def focal_length(self) -> torch.Tensor:
    return self.intrinsics[..., :2]
  
  @property
  def principal_point(self) -> torch.Tensor:
    return self.intrinsics[..., 2:]
  
  @property
  def aspect_ratio(self) -> float:
    return self.image_size[0] / self.image_size[1]
  
  @property
  def fov(self) -> torch.Tensor:
    f = self.focal_length
    return 2.0 * torch.atan(0.5 * torch.tensor(self.image_size, device=self.device) / f)
  
  def has_label(self, label:Label) -> bool:
    return bool(self.label & label)
  
  def resized(self, image_scale:float) -> 'Camera':
    return replace(self, 
                   intrinsics=self.intrinsics * image_scale,
                   image_size=(int(self.image_size[0] * image_scale), int(self.image_size[1] * image_scale)))

  @beartype
  def to_camera_params(self) -> CameraParams:
    return CameraParams(
      projection=self.intrinsics,
      T_camera_world=self.camera_t_world,

      near_plane=self.depth_range[0],
      far_plane=self.depth_range[1],  
      image_size=self.image_size
    )

  

@tensorclass
class Cameras:
  """ Represents either a single camera or a batch of cameras."""

  camera_t_world:torch.Tensor  # (..., 4, 4) float
  projection:Projections       

  camera_idx: torch.Tensor     # (..., 1) int
  frame_idx: torch.Tensor      # (..., 1) int
  labels: torch.Tensor         # (..., 1) int

  image_names: List[str]

  
  @property
  def device(self):
    return self.camera_t_world.device
  
  @property
  def centers(self) -> torch.Tensor:
    return self.world_t_camera[..., :3, 3]
    
  @property
  def rotations(self) -> torch.Tensor:
    return self.camera_t_world[..., :3, :3].transpose(-1, -2)
  
  @cached_property
  def world_t_camera(self) -> torch.Tensor:
    return torch.inverse(self.camera_t_world)
    
  @property
  def right(self) -> torch.Tensor:
    return self.world_t_camera[..., :3, 0]
  
  @property
  def up(self) -> torch.Tensor:
    return self.world_t_camera[..., :3, 1]
  
  @property
  def forward(self) -> torch.Tensor:
    return -self.rotations[..., :3, 2]
  
  @property
  def intrinsics(self) -> torch.Tensor:
    return self.projection.intrinsics
  
  @property
  def image_sizes(self) -> torch.Tensor:
    return self.projection.image_size
  

  def has_label(self, label:Label) -> torch.Tensor:
    """ Get indices of cameras with the given label. 
    Labels are bitwise encoded, so we use bitwise AND to get the indices.
    """
    label_mask = self.labels & label.value
    return torch.nonzero(label_mask, as_tuple=True)[0]
  
  @beartype
  def with_label(self, label:Label) -> 'Cameras':
    """ Get cameras with the given label."""
    return self[self.has_label(label)]
  
  @beartype
  def count_label(self, label:Label) -> int:
    """ Get number of cameras with the given label."""
    return self.has_label(label).shape[0]


  def item(self) -> Camera:
    assert np.prod(self.batch_size) == 1, f"Expected batch size 1, got shape: {self.batch_size}"


    return Camera(
      intrinsics=self.intrinsics.squeeze(0),
      camera_t_world=self.camera_t_world.squeeze(0),

      image_size=tuple(self.image_sizes.squeeze(0).cpu().tolist()),
      depth_range=tuple(self.projection.depth_range.squeeze(0).cpu().tolist()),
      camera_idx=int(self.camera_idx.item()),
      frame_idx=int(self.frame_idx.item()),
      label=Label(self.labels.item()),
      image_name=self.image_names[0]
    )



class CameraTable(nn.Module, metaclass=abc.ABCMeta):

  @abstractmethod
  def forward(self, image_idx: torch.Tensor) -> Cameras:
    raise NotImplementedError
  

  
  @property
  def num_projections(self) -> int:
    return self.projections.batch_size[0]

  @property
  @abstractmethod
  def projections(self) -> Projections:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def num_frames(self) -> int:
    raise NotImplementedError
  
  @property
  @abstractmethod
  def num_images(self) -> int:
    raise NotImplementedError
    
  def __len__(self):
    return self.num_images

  @property
  @abstractmethod
  def device(self):
    raise NotImplementedError
  
  @property
  @abstractmethod
  def image_names(self) -> List[str]:
    raise NotImplementedError
  
  @property
  def cameras(self) -> Cameras:
    return self.forward(torch.arange(self.num_images))
  

  


  @beartype
  def __getitem__(self, image_idx:int | torch.Tensor) -> Cameras:
    if isinstance(image_idx, int):
      image_idx = torch.tensor([image_idx], device=self.device)
      return self.forward(image_idx)
    else:
      return self.forward(image_idx)

@beartype
def camera_scene_extents(cameras:Cameras) -> Tuple[torch.Tensor, float]:
    """ 
    Compute centroid and diagonal of camera centers.
    """

    cam_centers = cameras.centers.reshape(-1, 3)
    avg_cam_center = torch.mean(cam_centers, dim=0, keepdim=True)

    distances = torch.norm(cam_centers - avg_cam_center, dim=0, keepdim=True)
    diagonal = torch.max(distances)

    return avg_cam_center.reshape(3), (diagonal * 1.1).item()


@beartype
def pose_adjacency(poses1: torch.Tensor, poses2: torch.Tensor, k: int = 8) -> torch.Tensor:
    """ Compute adjacency matrix between all camera poses in the table.
    Returns higher values for more similar poses (closer and similarly oriented).
    Uses k-nearest neighbors to adapt to local scale.
    
    Args:
        poses1, poses2: Camera poses as (N, 4, 4) tensors
        k: Number of neighbors to consider for local scale (default 8)
    """
    forward1, forward2 = poses1[..., :3, 2], poses2[..., :3, 2]

    # Outer product of forward vectors (ranges from -1 to 1)
    dir_similarity = torch.einsum('id,jd->ij', forward1, forward2)

    pos1, pos2 = poses1[..., :3, 3], poses2[..., :3, 3]  
    distance = torch.cdist(pos1, pos2)
    
    # For each camera, compute local scale based on kNN distances
    # We use k+1 because the closest point will be the point itself (distance 0)
    knn_distances, _ = torch.topk(distance, k=min(k + 1, distance.shape[1]), dim=1, largest=False)
    local_scale = knn_distances.median()
    
    # Convert distance to similarity using local scale for each source camera
    dist_similarity = 1.0 / (1.0 + distance / local_scale)
    
    return dir_similarity * dist_similarity


@beartype
def camera_similarity(cameras:Cameras, world_t_camera:torch.Tensor) -> torch.Tensor:
  """ Compute similarity between a camera pose and all camera poses in the table."""
  assert world_t_camera.shape == (4, 4), f"Expected shape (4, 4), got: {world_t_camera.shape}"

  return pose_adjacency(cameras.world_t_camera, world_t_camera.unsqueeze(0))

@beartype
def camera_adjacency_matrix(cameras:Cameras) -> torch.Tensor:
  """ Compute adjacency matrix between all camera poses in the table."""
  return pose_adjacency(cameras.world_t_camera, cameras.world_t_camera)

class CameraRigTable(CameraTable):
  def __init__(self, rig_t_world:torch.Tensor,   # (N, 4, 4) - poses for the whole camera rig
                     camera_t_rig:torch.Tensor,  # (C, 4, 4) - camera poses inside the rig
                     projection:Projections,     # (C) Camera projections (image size and intrinsics)
                     image_names:List[str],      # (N,)
                     labels:torch.Tensor        # (N,)
                    ):
    super().__init__()

    assert projection.shape[0] == camera_t_rig.shape[0], \
      f"Expected equal number of cameras and projections, got: {projection.shape[0]} != {camera_t_rig.shape[0]}"

    num_cameras, num_frames = camera_t_rig.shape[0], rig_t_world.shape[0]
    assert len(image_names) == num_cameras * num_frames, \
        f"Incorrect number of image names, got: {len(image_names)} != {num_cameras * num_frames}"

    self._camera_projection = TensorDictParams(projection.to_tensordict())
    self._camera_projection.requires_grad_(False)

    self._camera_poses = RigPoseTable(
      rig_t_world=rig_t_world, camera_t_rig=camera_t_rig)
    self._camera_poses.requires_grad_(False)

    self._image_names = image_names

    self.register_buffer("_labels", labels) 
    self._labels:torch.Tensor = labels

  @beartype
  def forward(self, image_idx:torch.Tensor) -> Cameras:     
    assert image_idx.dim() <= 1, f"Expected 1D tensor, got: {image_idx.shape}"
    assert (image_idx < self.num_images).all(), f"Image index out of range: {image_idx} >= {self.num_images}"

    num_cameras = self._camera_poses.num_cameras
    frame_idx = image_idx // num_cameras
    camera_idx = image_idx % num_cameras

    
    return Cameras(
      camera_t_world=self._camera_poses(frame_idx, camera_idx),
      projection=self.projections[camera_idx],
      camera_idx=camera_idx,
      frame_idx=frame_idx,
      batch_size=image_idx.shape,
      labels=self._labels[image_idx],
      image_names=[self._image_names[i] for i in image_idx]
    )


  @property
  def num_images(self):
    """ Total number of images."""
    return self._camera_poses.num_frames * self._camera_poses.num_cameras
  
  @property
  def image_names(self) -> List[str]:
    return self._image_names


  @property
  def projections(self) -> Projections:  
    return Projections.from_dict(self._camera_projection, batch_dims=1)
  
  @property
  def num_frames(self) -> int:
    return self._camera_poses.num_frames
  
  
  @property
  def device(self):
    return self._camera_projection.device
  



class MultiCameraTable(CameraTable):
  """
  A table of camera poses and intrinsics - cameras can have different intrinsics which are stored in the projection table.
  """
  def __init__(self, 
               camera_t_world:torch.Tensor, # (N, 4, 4)
               camera_idx:torch.Tensor,     # (N,) - index into projection table (0, P-1)
               projection:Projections,     # (P,)
               image_names:List[str],      # (N,)
               labels: torch.Tensor        # (N,) - train=1, val=0
              ):
    super().__init__()

    assert camera_t_world.shape[0] == camera_idx.shape[0], \
        f"Expected equal number of cameras and indices, got: {camera_t_world.shape[0]} != {camera_idx.shape[0]}"
    
    assert len(image_names) == camera_t_world.shape[0], \
        f"Expected equal number of image names and cameras, got: {len(image_names)} != {camera_t_world.shape[0]}"

    self._camera_projection = TensorDictParams(projection.to_tensordict())
    self._camera_projection.requires_grad_(False)

    self._camera_idx = camera_idx
    self.register_buffer("_camera_idx", camera_idx)
    
    self._camera_t_world = PoseTable(camera_t_world)
    self._camera_t_world.requires_grad_(False)

    self._image_names = image_names

    self._labels = labels
    self.register_buffer("_labels", labels)



  @beartype
  def forward(self, image_idx:torch.Tensor) -> Cameras:
    assert image_idx.dim() == 1, f"Expected 1D tensor, got: {image_idx.shape}"

    cam_idx = self._camera_idx[image_idx]
    return Cameras(
      camera_t_world=self._camera_t_world(image_idx),
      projection=self.projections[cam_idx],
      camera_idx=cam_idx,
      frame_idx=image_idx,
      batch_size=image_idx.shape,
      labels=self._labels[image_idx],
      image_names=[self._image_names[i] for i in image_idx]
    )


  @property
  def num_images(self) -> int:
    return len(self._camera_t_world)

  @property
  def projections(self) -> Projections:
    return Projections.from_dict(self._camera_projection)

  @property
  def num_frames(self) -> int:
    return self.num_images

  @property
  def device(self):
    return self._camera_projection.device




def camera_json(camera_table:CameraTable):
  def export_camera(i):
    camera = camera_table[i].item()

    r, t = split_rt(torch.linalg.inv(camera.camera_t_world))
    fx, fy, cx, cy = camera.intrinsics.cpu().tolist()
    near, far = camera.depth_range
    width, height = camera.image_size

    return {
      "id": i,
      "position": t.cpu().numpy().tolist(),
      "rotation": r.cpu().numpy().tolist(),
      "fx": fx, "fy": fy, 
      "cx": cx, "cy": cy,
      "width": width,
      "height": height,
      "near": near,
      "far": far,
      "img_name": camera.image_name
    }

  return [export_camera(i) for i in range(len(camera_table))]





