from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Sequence
from beartype import beartype
from beartype.typing import Iterator, Tuple, List

from splat_trainer.dataset.normalization import Normalization, NormalizationConfig
import torch

import numpy as np
from splat_trainer.camera_table.camera_table import Label, Projections, MultiCameraTable
from splat_trainer.dataset.colmap.loading import load_images
from splat_trainer.dataset.dataset import  ImageView, Dataset

from splat_trainer.dataset.util import split_every
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
                
        test_every:int=8,
        depth_range:Tuple[float, float] = (0.1, 100.0),

        normalize:NormalizationConfig=NormalizationConfig()
    ):


    self._images = None
    self.image_scale = image_scale
    self.resize_longest = resize_longest
    self.camera_depth_range = [float(f) for f in depth_range]

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

    def image_position(image:pycolmap.Image) -> torch.Tensor:
      return image.cam_from_world.inverse().translation

    camera_positions = np.array([image_position(image) for image in 
                                 self.reconstruction.images.values()])

    self.normalize = normalize.get_transform(torch.from_numpy(camera_positions).to(torch.float32))
    
    def image_info(image:pycolmap.Image) -> Tuple[np.array, str, int]:
      world_t_camera = torch.from_numpy(image.cam_from_world.inverse().matrix()).to(torch.float32)
      camera_t_world = self.normalize.transform_rigid(world_t_camera).inverse()
      
      return (camera_t_world, image.name, id_to_camera_idx[image.camera_id])
    
    self.num_cameras = len(self.reconstruction.images)
    camera_t_world, self.image_names, self.camera_idx = zip(
      *[image_info(image) for image in self.reconstruction.images.values()])
    
    self.camera_t_world = torch.stack(camera_t_world)

    # Evenly distribute validation images
    self.train_idx, self.val_idx = split_every(self.num_cameras, test_every)


  def __repr__(self) -> str:
    args = []
    if self.image_scale is not None:
      args += [f"image_scale={self.image_scale}"]

    if self.resize_longest is not None:
      args += [f"resize_longest={self.resize_longest}"]
        
    args += [f"near={self.camera_depth_range[0]:.3f}", f"far={self.camera_depth_range[1]:.3f}"]
    args += [f"normalization={self.normalize}"]

    args += [f"num_train={len(self.train_idx)}", f"num_val={len(self.val_idx)}"]
    return f"COLMAPDataset({self.base_path} {', '.join(args)})"
  
  @property
  def to_original(self) -> Normalization:
    return self.normalize.inverse


  def _load_camera_images(self) -> List[CameraImage]:
    images = load_images(list(self.image_names), Path(self.base_path) / self.image_dir, 
                         image_scale=self.image_scale, resize_longest=self.resize_longest)
    
    cameras = [CameraImage(filename, torch.from_numpy(image).pin_memory(), i) 
               for i, (filename, image) in enumerate(zip(self.image_names, images))]  
    return cameras
  
  def load_images(self) -> List[CameraImage]:
    """ Load images from disk - don't use this function unless you want to load images (an expensive operation)
    """
    if self._images is None:
      self._images = self._load_camera_images()

    return self._images
  
  @property
  def num_images(self) -> int:
    return len(self.image_names)


  @beartype
  def loader(self, idx:torch.Tensor, shuffle:bool=False) -> Sequence[ImageView]:
    images = self.load_images()
    return Images([images[i] for i in idx.cpu().numpy()], shuffle=shuffle)

  def train(self, shuffle=False) -> Sequence[ImageView]:
    return self.loader(self.train_idx, shuffle=shuffle)
  
  def val(self) -> Sequence[ImageView]:
    return self.loader(self.val_idx)

  @cached_property
  def camera_table(self) -> MultiCameraTable:
    labels = torch.zeros((self.num_images,), dtype=torch.int32)
    labels[self.train_idx] |= Label.Training.value
    labels[self.val_idx] |= Label.Validation.value

    return MultiCameraTable(
      camera_t_world = self.camera_t_world,
      projection = self.projections,
      camera_idx = torch.tensor(self.camera_idx, dtype=torch.long),
      labels=labels,
      image_names=self.image_names
    )
  


  def pointcloud(self) -> PointCloud:  
    xyz = np.array([p.xyz for p in self.reconstruction.points3D.values()])
    colors = np.array([p.color for p in self.reconstruction.points3D.values()])

    cloud = PointCloud(torch.tensor(xyz, dtype=torch.float32), 
                      torch.tensor(colors, dtype=torch.float32) / 255.0,
                      batch_size=(len(xyz),))
    
    return self.normalize.transform_cloud(cloud)


class Images(Sequence[ImageView]):
  @beartype
  def __init__(self, camera_images:List[CameraImage], shuffle:bool=False):
    self.camera_images = camera_images
    self.shuffle = shuffle

  def __len__(self):
      return len(self.camera_images)

  def __getitem__(self, index) -> ImageView:
    camera_image:CameraImage = self.camera_images[index]
    return ImageView(camera_image.filename, camera_image.image_id, camera_image.image)
     
  def __iter__(self) -> Iterator[ImageView]:
    order = torch.randperm(len(self)) if self.shuffle else torch.arange(len(self))
    for idx in order:
      yield self[idx]  

