from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional
from beartype import beartype
from beartype.typing import Iterator, Tuple, List

import torch

import numpy as np
from splat_trainer.camera_table.camera_table import Projections, MultiCameraTable
from splat_trainer.dataset.colmap.loading import load_images
from splat_trainer.dataset.dataset import  CameraView, Dataset

from splat_trainer.util.misc import split_stride
from splat_trainer.util.pointcloud import PointCloud

import pycolmap

@dataclass
class CameraImage:
  filename:str
  image:torch.Tensor
  image_id:int

  @property
  def image_size(self) -> Tuple[int, int]:
    h, w =  self.image.shape[:2]
    return (w, h)
  

def colmap_projection(camera:pycolmap.Camera, image_scale:Optional[float]=None, 
                      resize_longest:Optional[int]=None, depth_range:Tuple[float, float] = (0.1, 100.0)) -> Projections:
  assert camera.model == pycolmap.CameraModelId.PINHOLE, f"Only PINHOLE cameras are supported for now, got {camera.model}"   

  w, h = camera.width, camera.height
  if resize_longest is not None:
    image_scale = resize_longest / max(w, h)
          
  fx = camera.focal_length_x
  fy = camera.focal_length_y
  cx = camera.principal_point_x
  cy = camera.principal_point_y

  proj = np.array([fx, fy, cx, cy])  

  if image_scale is not None:
    proj *= image_scale
    (w, h) = (round(w * image_scale), round(h * image_scale))
  
  w, h = camera.width, camera.height

  return Projections(
    intrinsics=torch.tensor(proj, dtype=torch.float32), 
    image_size=torch.tensor((w, h), dtype=torch.int32),
    depth_range=torch.tensor(depth_range, dtype=torch.float32)
  )

def print_projections(projections:List[Projections]):
  for i, p in enumerate(projections):
    fx, fy, cx, cy = p.intrinsics
    w, h = p.image_size
    print(f"Camera {i}@{w}x{h} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

class COLMAPDataset(Dataset):
  def __init__(self, base_path:str,
        model_dir:str = "sparse/0",
        image_dir:str = "images",          

        image_scale:Optional[float]=None,
        resize_longest:Optional[int]=None,
                
        val_stride:int=10,
        depth_range:Tuple[float, float] = (0.1, 100.0)):

    self.image_scale = image_scale
    self.resize_longest = resize_longest
    self.camera_depth_range = depth_range

    self.image_dir = image_dir
    self.base_path = base_path

    model_path = Path(base_path) / model_dir

    self.reconstruction = pycolmap.Reconstruction(str(model_path))
    id_to_camera_idx = {}

    assert resize_longest is None or image_scale is None, "Specify either resize_longest or image_scale (not both)"
    projections = []

    for i, (k, camera) in enumerate(self.reconstruction.cameras.items()): 
      proj = colmap_projection(camera, image_scale, resize_longest, depth_range)
      projections.append(proj)
      id_to_camera_idx[k] = i

    self.projections = torch.stack(projections)
    print_projections(self.projections)

    def image_info(image:pycolmap.Image) -> Tuple[np.array, str, int]:
      camera_t_world = np.eye(4)
      camera_t_world[:3, :4] =  image.cam_from_world.matrix()    

      return (camera_t_world, image.name, id_to_camera_idx[image.camera_id])
    
    self.num_cameras = len(self.reconstruction.images)
    self.camera_t_world, self.image_names, self.camera_idx = zip(
      *[image_info(image) for image in self.reconstruction.images.values()])
    
    # Evenly distribute validation images
    self.train_idx, self.val_idx = split_stride(np.arange(self.num_cameras), val_stride)

  def __repr__(self) -> str:
    args = []
    if self.image_scale is not None:
      args += [f"image_scale={self.image_scale}"]

    if self.resize_longest is not None:
      args += [f"resize_longest={self.resize_longest}"]
        
    args += [f"near={self.camera_depth_range[0]:.3f}", f"far={self.camera_depth_range[1]:.3f}"]
    return f"COLMAPDataset({self.base_path} {', '.join(args)})"
  
  
  @cached_property
  def camera_images(self) -> List[CameraImage]:
    images = load_images(list(self.image_names), Path(self.base_path) / self.image_dir, 
                         image_scale=self.image_scale, resize_longest=self.resize_longest)
    
    cameras = [CameraImage(filename, torch.from_numpy(image).pin_memory(), i) 
               for i, (filename, image) in enumerate(zip(self.image_names, images))]  
    return cameras
  
  @cached_property
  def train_cameras(self) -> List[CameraImage]:
    return [self.camera_images[i] for i in self.train_idx]

  @cached_property
  def val_cameras(self) -> List[CameraImage]:
    return [self.camera_images[i] for i in self.val_idx]

  def train(self, shuffle=False) -> Iterator[CameraView]:
    return Images(self.train_cameras, shuffle=shuffle)
  
  def val(self) -> Iterator[CameraView]:
    return Images(self.val_cameras)


  def camera_table(self) -> MultiCameraTable:
    return MultiCameraTable(
      camera_t_world = torch.tensor(np.array(self.camera_t_world), dtype=torch.float32),
      projection = self.projections,
      camera_idx = torch.tensor(self.camera_idx, dtype=torch.long))
  


  def pointcloud(self) -> PointCloud:  
    xyz = np.array([p.xyz for p in self.reconstruction.points3D.values()])
    colors = np.array([p.color for p in self.reconstruction.points3D.values()])

    return PointCloud(torch.tensor(xyz, dtype=torch.float32), 
                      torch.tensor(colors, dtype=torch.float32) / 255.0,
                      batch_size=(len(xyz),))
    



class Images(torch.utils.data.Dataset):
  @beartype
  def __init__(self, camera_images:List[CameraImage], shuffle:bool=False):
    self.camera_images = camera_images
    self.shuffle = shuffle

  def __len__(self):
      return len(self.camera_images)

  def __getitem__(self, index) -> CameraView:
    camera_image:CameraImage = self.camera_images[index]
    return camera_image.filename, camera_image.image, camera_image.image_id
     
  def __iter__(self) -> Iterator[CameraView]:
    order = torch.randperm(len(self)) if self.shuffle else torch.arange(len(self))
    for idx in order:
      yield self[idx]  

