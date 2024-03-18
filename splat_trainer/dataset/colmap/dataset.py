from dataclasses import dataclass
from pathlib import Path
from beartype import beartype
from beartype.typing import Iterator, Tuple, List

import torch

import numpy as np
from splat_trainer.dataset.colmap.loading import load_images
from splat_trainer.modules.pose_table import CameraRigTable, PoseTable
from splat_trainer.dataset.dataset import CameraView, Dataset

from splat_trainer.util.misc import split_stride
from splat_trainer.util.pointcloud import PointCloud

import pycolmap 

 


class COLMAPDataset(Dataset):
  def __init__(self, base_path:str,
        model_dir:str = "sparse/0",
        image_dir:str = "images",                
        image_scale:float=1.0,
        val_stride:int=10,
        depth_range:Tuple[float, float] = (0.1, 100.0)):

    self.image_scale = image_scale
    self.depth_range = depth_range

    self.base_path = base_path
    model_path = Path(base_path) / model_dir

    self.reconstruction = pycolmap.Reconstruction(str(model_path))
    self.projections = []
    camera_idx = {}

    for i, (k, camera) in enumerate(self.reconstruction.cameras.items()):
      camera.rescale(image_scale)
      self.projections.append(camera.calibration_matrix())
      camera_idx[k] = i



    def image_info(image:pycolmap.Image) -> Tuple[np.array, str, int]:
      camera_t_world = np.eye(4)
      camera_t_world[:3, :4] =  image.cam_from_world.matrix()        
      return (camera_t_world, image.name, camera_idx[image.camera_id])
    
    self.num_cameras = len(self.reconstruction.images)

    self.camera_t_world, self.image_names, self.camera_idx = zip(
      *[image_info(image) for image in self.reconstruction.images.values()])
    
    images = load_images(list(self.image_names), Path(base_path) / image_dir, image_scale=image_scale)
    cameras = [CameraImage(filename, torch.from_numpy(image).pin_memory(), i) 
               for i, (filename, image) in enumerate(zip(self.image_names, images))]
    
    cam_positions = np.array([image.cam_from_world.translation for image in self.reconstruction.images.values()])
    self.centre, self.camera_extent = camera_extents(cam_positions)    
    
    # Evenly distribute validation images
    self.train_cameras, self.val_cameras = split_stride(cameras, val_stride)

  def __repr__(self) -> str:
    return f"COLMAPDataset({self.base_path}, image_scale={self.image_scale})"

  def train(self, shuffle=True) -> Iterator[CameraView]:
    return Images(self.train_cameras, shuffle=shuffle)
  
  def val(self) -> Iterator[CameraView]:
    return Images(self.val_cameras)


  def camera_poses(self) -> PoseTable:
    return PoseTable(torch.tensor(np.array(self.camera_t_world), dtype=torch.float32))
  
  def camera_shape(self) -> torch.Size:
    return torch.Size([self.num_cameras])

  def camera_projection(self) -> torch.Tensor:
    return CameraProjectionTable(torch.tensor(np.array(self.projections), dtype=torch.float32), 
                                 torch.tensor(self.camera_idx, dtype=torch.long))

  def pointcloud(self) -> PointCloud:
    
    xyz = np.array([p.xyz for p in self.reconstruction.points3D.values()])
    colors = np.array([p.color for p in self.reconstruction.points3D.values()])

    return PointCloud(torch.tensor(xyz, dtype=torch.float32), 
                      torch.tensor(colors, dtype=torch.float32) / 255.0,
                      batch_size=(len(xyz),))
    
  def scene_scale(self) -> float:
    return self.camera_extent


class CameraProjectionTable(torch.nn.Module):
  def __init__(self, projections:torch.Tensor, image_cameras:torch.Tensor):
    super().__init__()

    self.projection = torch.nn.Parameter(projections.to(torch.float32))
    self.register_buffer("image_cameras", image_cameras)

  def forward(self, image_idx):
    return self.projection[self.image_cameras[image_idx]]


def camera_extents(cam_centers):
    avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)

    distances = np.linalg.norm(cam_centers - avg_cam_center, axis=0, keepdims=True)
    diagonal = np.max(distances)

    return avg_cam_center.reshape(3), diagonal * 1.1


@dataclass
class CameraImage:
  filename:str
  image:torch.Tensor
  image_id:int

class Images(torch.utils.data.Dataset):
  @beartype
  def __init__(self, camera_images:List[CameraImage], shuffle:bool=False):
    self.camera_images = camera_images
    self.shuffle = shuffle

  def __len__(self):
      return len(self.camera_images)

  def __getitem__(self, index) -> CameraView:
    camera_image:CameraImage = self.camera_images[index]

    idx = torch.tensor([camera_image.image_id], dtype=torch.long).pin_memory()
    return camera_image.filename, camera_image.image, idx
     
  def __iter__(self) -> Iterator[CameraView]:
    order = torch.randperm(len(self)) if self.shuffle else torch.arange(len(self))
    for idx in order:
      yield self[idx]  

