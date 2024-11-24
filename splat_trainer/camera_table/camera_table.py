from abc import abstractmethod
import abc
from numbers import Number
from typing import List, Tuple

from tensordict import tensorclass
from tensordict import TensorDictParams
import torch
from torch import nn

from splat_trainer.camera_table.pose_table import PoseTable, RigPoseTable
from splat_trainer.util.transforms import split_rt

from beartype import beartype

@tensorclass
class Projections:
  """ Projection parameters for a camera or a batch of cameras."""
  intrinsics:torch.Tensor      # (..., 4) fx, fy, cx, cy
  image_size:torch.Tensor      # (..., 2) int
  depth_range:torch.Tensor     # (..., 2) float

  @property
  def matrix(self) -> torch.Tensor:
    fx, fy, cx, cy = self.intrinsics.unbind(-1)
    m = torch.eye(3, dtype=self.intrinsics.dtype, device=self.intrinsics.device)
    m = m.unsqueeze(0).expand(self.intrinsics.shape[:-1] + (3, 3)).clone()
    m[..., 0, 0] = fx
    m[..., 1, 1] = fy
    m[..., 0, 2] = cx
    m[..., 1, 2] = cy
    return m
  
  @property
  def device(self):
    return self.intrinsics.device
  

@tensorclass
class Cameras:
  """ Represents either a single camera or a batch of cameras."""

  camera_t_world:torch.Tensor  # (..., 4, 4) float
  projection:Projections      # (..., ) Projection-

  camera_idx: torch.Tensor     # (..., 1) int
  frame_idx: torch.Tensor      # (..., 1) int

  @property
  def device(self):
    return self.camera_t_world.device
  
  @property
  def centers(self) -> torch.Tensor:
    return self.camera_t_world[..., :3, 3]
  
  @property
  def rotations(self) -> torch.Tensor:
    return self.camera_t_world[..., :3, :3]
  
  @property
  def intrinsics(self) -> torch.Tensor:
    return self.projection.intrinsics
  
  @property
  def image_sizes(self) -> torch.Tensor:
    return self.projection.image_size
  
  # Convenience properties for retrieving single camera values
  @property
  def depth_range(self) -> Tuple[float, float]:
    assert len(self.batch_size) == 0, "depth_range returns (near, far) for a single camera"
    return tuple(self.projection.depth_range.cpu().tolist())

  @property
  def frame(self) -> int:
    assert len(self.batch_size) == 0, "frame returns a single frame index"
    return self.frame_idx.item()

  @property
  def camera(self) -> int:
    assert len(self.batch_size) == 0, "camera returns a single camera index"
    return self.camera_idx.item()

  @property
  def size_tuple(self) -> Tuple[int, int]:
    assert len(self.batch_size) == 0, "size_tuple returns (width, height) for a single camera"
    return tuple(self.image_sizes.cpu().tolist())


class CameraTable(nn.Module, metaclass=abc.ABCMeta):

  @abstractmethod
  def forward(self, image_idx: torch.Tensor) -> Cameras:
    raise NotImplementedError
  
  @beartype
  def __getitem__(self, image_idx:int | torch.Tensor) -> Cameras:
    if isinstance(image_idx, int):
      image_idx = torch.tensor([image_idx], device=self.device)
      return self.forward(image_idx).squeeze(0)
    else:
      return self.forward(image_idx)
  
  @property
  @abstractmethod
  def image_name(self, image_idx:int) -> str:
    raise NotImplementedError
  
  @property
  def num_projections(self) -> int:
    return self.projections.shape[0]

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
  def cameras(self) -> Cameras:
    return self.forward(torch.arange(self.num_images))
  

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
def pose_adjacency(poses1:torch.Tensor, poses2:torch.Tensor) -> torch.Tensor:
  """ Compute adjacency matrix between all camera poses in the table."""

  forward1, forward2 = poses1[..., :3, 2], poses2[..., :3, 2]
  dir_similarity = torch.sum(forward1 * forward2, dim=1)

  pos1, pos2 = poses1[..., :3, 3], poses2[..., :3, 3]  
  distance = torch.norm(pos1 - pos2, dim=1)

  return dir_similarity * distance


@beartype
def camera_similarity(camera_table:CameraTable, camera_t_world:torch.Tensor) -> torch.Tensor:
  """ Compute similarity between a camera pose and all camera poses in the table."""
  assert camera_t_world.shape == (4, 4), f"Expected shape (4, 4), got: {camera_t_world.shape}"

  poses = camera_table.camera_poses
  return pose_adjacency(poses, camera_t_world.unsqueeze(0))
  

class CameraRigTable(CameraTable):
  def __init__(self, rig_t_world:torch.Tensor,   # (N, 4, 4) - poses for the whole camera rig
                     camera_t_rig:torch.Tensor,  # (C, 4, 4) - camera poses inside the rig
                     projection:Projections,     # (C) Camera projections (image size and intrinsics)
                     image_names:List[str]      # (N,)
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
    self._image_names = image_names
    
  @beartype
  def forward(self, image_idx:torch.Tensor) -> Cameras:     
    assert image_idx.dim() == 1, f"Expected 1D tensor, got: {image_idx.shape}"
    assert (image_idx < self.num_images).all(), f"Image index out of range: {image_idx} >= {self.num_images}"

    num_cameras = self._camera_poses.num_cameras
    frame_idx = image_idx // num_cameras
    camera_idx = image_idx % num_cameras
    
    return Cameras(
      camera_t_world=self._camera_poses(frame_idx, camera_idx),
      projection=self.projections[camera_idx],
      camera_idx=camera_idx,
      frame_idx=frame_idx,
      batch_size=image_idx.shape
    )


  @property
  def num_images(self):
    """ Total number of images."""
    return self._camera_poses.num_frames * self._camera_poses.num_cameras
  
  @property
  def image_name(self, image_idx:int) -> str:
    return self._image_names[image_idx]


  @property
  def projections(self) -> Projections:  
    return Projections.from_tensordict(self._camera_projection)
  
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
               image_names:List[str]        # (N,)
              ):
    super().__init__()

    assert camera_t_world.shape[0] == camera_idx.shape[0], \
        f"Expected equal number of cameras and indices, got: {camera_t_world.shape[0]} != {camera_idx.shape[0]}"
    
    assert len(image_names) == camera_t_world.shape[0], \
        f"Expected equal number of image names and cameras, got: {len(image_names)} != {camera_t_world.shape[0]}"

    self._camera_projection = TensorDictParams(projection.to_tensordict())
    self._camera_projection.requires_grad_(False)

    self.register_buffer("_camera_idx", camera_idx)
    self._camera_t_world = PoseTable(camera_t_world)

  @beartype
  def forward(self, image_idx:torch.Tensor) -> Cameras:
    assert image_idx.dim() == 1, f"Expected 1D tensor, got: {image_idx.shape}"

    cam_idx = self._camera_idx[image_idx]
    return Cameras(
      camera_t_world=self._camera_t_world(image_idx),
      projection=self.projections[cam_idx],
      camera_idx=cam_idx,
      frame_idx=image_idx,
      batch_size=image_idx.shape
    )


  @property
  def num_images(self) -> int:
    return len(self._camera_t_world)

  @property
  def projections(self) -> Projections:
    return Projections.from_tensordict(self._camera_projection)

  @property
  def num_frames(self) -> int:
    return self.num_images

  @property
  def device(self):
    return self._camera_projection.device




def camera_json(camera_table:CameraTable):
  def export_camera(i):
    camera = camera_table[i]

    r, t = split_rt(torch.linalg.inv(camera.camera_t_world))
    fx, fy, cx, cy = camera.projection.intrinsics.cpu().tolist()
    near, far = camera.projection.depth_range.cpu().tolist()
    width, height = camera.image_sizes.cpu().tolist()

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
    }

  return [export_camera(i) for i in range(len(camera_table))]



