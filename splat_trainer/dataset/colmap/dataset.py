from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from beartype import beartype
from beartype.typing import Iterator, Tuple, List

import torch

import numpy as np
from splat_trainer.camera_table.camera_table import ViewTable, MultiCameraTable, camera_json
from splat_trainer.dataset.colmap.loading import load_images
from splat_trainer.dataset.dataset import CameraProjection, CameraView, Dataset

from splat_trainer.util.misc import split_stride
from splat_trainer.util.pointcloud import PointCloud

import pycolmap



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

    self.base_path = base_path
    model_path = Path(base_path) / model_dir

    self.reconstruction = pycolmap.Reconstruction(str(model_path))
    self.projections:List[CameraProjection] = []
    camera_idx = {}

    assert resize_longest is None or image_scale is None, "Specify either resize_longest or image_scale"


    for i, (k, camera) in enumerate(self.reconstruction.cameras.items()):

      if image_scale is not None:
        camera.rescale(image_scale)
      elif resize_longest is not None:
        w, h = camera.width, camera.height
        scale = resize_longest / max(w, h)
        camera.rescale(round(w * scale), round(h * scale))
      
      fx = camera.focal_length_x
      fy = camera.focal_length_y
      cx = camera.principal_point_x
      cy = camera.principal_point_y

      proj = np.array([fx, fy, cx, cy])  
      
      self.projections.append(CameraProjection(torch.tensor(proj, dtype=torch.float32), (w, h)))

      camera_idx[k] = i
      print(f"Camera {k}@{camera.width}x{camera.height} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")


    def image_info(image:pycolmap.Image) -> Tuple[np.array, str, int]:
      camera_t_world = np.eye(4)
      camera_t_world[:3, :4] =  image.cam_from_world.matrix()    

      return (camera_t_world, image.name, camera_idx[image.camera_id])
    
    self.num_cameras = len(self.reconstruction.images)
    self.camera_t_world, self.image_names, self.camera_idx = zip(
      *[image_info(image) for image in self.reconstruction.images.values()])
    
    images = load_images(list(self.image_names), Path(base_path) / image_dir, 
                         image_scale=image_scale, resize_longest=resize_longest)
    
    self.all_cameras = [CameraImage(filename, torch.from_numpy(image).pin_memory(), i) 
               for i, (filename, image) in enumerate(zip(self.image_names, images))]  
  
    # Evenly distribute validation images
    self.train_cameras, self.val_cameras = split_stride(self.all_cameras, val_stride)

  def __repr__(self) -> str:
    args = []
    if self.image_scale is not None:
      args += [f"image_scale={self.image_scale}"]

    if self.resize_longest is not None:
      args += [f"resize_longest={self.resize_longest}"]
        
    args += [f"near={self.camera_depth_range[0]:.3f}", f"far={self.camera_depth_range[1]:.3f}"]

    return f"COLMAPDataset({self.base_path} {', '.join(args)})"

  def train(self, shuffle=True) -> Iterator[CameraView]:
    return Images(self.train_cameras, shuffle=shuffle)
  
  def val(self) -> Iterator[CameraView]:
    return Images(self.val_cameras)


  def view_table(self) -> MultiCameraTable:
    projections = [p.projection for p in self.projections]
    return MultiCameraTable(
      camera_t_world = torch.tensor(np.array(self.camera_t_world), dtype=torch.float32),
      projection = torch.tensor(np.array(projections), dtype=torch.float32),
      camera_idx = torch.tensor(self.camera_idx, dtype=torch.long))
  

  def unique_projections(self) -> List[CameraProjection]:
    return list(set(self.projections))
  
  def depth_range(self) -> Tuple[float, float]:
    return tuple(self.camera_depth_range)

  def image_sizes(self) -> torch.Tensor:
    return torch.tensor([image.image_size for image in self.all_cameras], dtype=torch.int32)

  def pointcloud(self) -> PointCloud:  
    xyz = np.array([p.xyz for p in self.reconstruction.points3D.values()])
    colors = np.array([p.color for p in self.reconstruction.points3D.values()])

    return PointCloud(torch.tensor(xyz, dtype=torch.float32), 
                      torch.tensor(colors, dtype=torch.float32) / 255.0,
                      batch_size=(len(xyz),))
    


  def camera_json(self, camera_table:ViewTable):

    def export_camera(i, info):
      image:CameraImage = self.all_cameras[i]
      h, w, _ = image.image.shape

      return {
        "img_name": image.filename,
        "width": w,
        "height" : h,
        **info
      }

    camera_info = camera_json(camera_table)
    return [export_camera(i, info) for i, info in enumerate(camera_info)]


@dataclass
class CameraImage:
  filename:str
  image:torch.Tensor
  image_id:int

  @property
  def image_size(self) -> Tuple[int, int]:
    h, w =  self.image.shape[:2]
    return (w, h)

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

