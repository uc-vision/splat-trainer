import os
from pathlib import Path
from typing import List, Optional
from beartype.typing import Iterator, Tuple
from camera_geometry import FrameSet

from camera_geometry.camera_models import optimal_undistorted
import torch


import numpy as np
from splat_trainer.camera_table.camera_table import CameraRigTable, ViewTable, camera_json
from splat_trainer.dataset.dataset import CameraProjection, CameraView, Dataset
from splat_trainer.util.misc import split_stride


from .loading import  CameraImage, PreloadedImages, camera_rig_table, preload_images
from splat_trainer.util.pointcloud import PointCloud

def load_scan(scan_file:str, image_scale:Optional[float]=None, resize_longest:Optional[int]=None) -> Tuple[FrameSet, List[CameraImage]]:
    scan = FrameSet.load_file(Path(scan_file))

    cameras = {k: optimal_undistorted(camera, alpha=0)
                 for k, camera in scan.cameras.items()}

    assert resize_longest is None or image_scale is None, "Specify either resize_longest or image_scale"

    if resize_longest is not None:
      cameras = {k: camera.resize_longest(longest=resize_longest) for k, camera in cameras.items()}
    elif image_scale is not None:
      cameras = {k: camera.scale_image(image_scale) for k, camera in cameras.items()}


    print("Undistorted cameras:")
    for k, camera in cameras.items():
        print(k, camera)

    print("Loading images...")
    all_cameras = preload_images(scan, cameras)
    return scan.copy(cameras=cameras), all_cameras


class ScanDataset(Dataset):
  def __init__(self, scan_file:str,                
        image_scale:Optional[float]=None,
        resize_longest:Optional[int]=None,
        val_stride:int=0,
        depth_range:Tuple[float, float] = (0.1, 100.0)):

    self.scan_file = scan_file
    self.image_scale = image_scale
    self.resize_longest = resize_longest

    scan = FrameSet.load_file(Path(scan_file))
    self.camera_depth_range = tuple(depth_range)

    cameras = {k: optimal_undistorted(camera, alpha=0)
                 for k, camera in scan.cameras.items()}

    assert resize_longest is None or image_scale is None, "Specify either resize_longest or image_scale"

    if resize_longest is not None:
      cameras = {k: camera.resize_longest(longest=resize_longest) for k, camera in cameras.items()}
    elif image_scale is not None:
      cameras = {k: camera.scale_image(image_scale) for k, camera in cameras.items()}


    print("Undistorted cameras:")
    for k, camera in cameras.items():
        print(k, camera)

    print("Loading images...")
    self.all_cameras = preload_images(scan, cameras)
    self.scan = scan.copy(cameras=cameras)

    # Evenly distribute validation images
    self.train_cameras, self.val_cameras = split_stride(self.all_cameras, val_stride)
    
  def __repr__(self) -> str:
    args = [] 
    if self.image_scale is not None:
      args += [f"image_scale={self.image_scale}"]

    if self.resize_longest is not None:
      args += [f"resize_longest={self.resize_longest}"]

    args += [f"near={self.camera_depth_range[0]:.3f}", f"far={self.camera_depth_range[1]:.3f}"]
    args += [f"pointcloud={self.pointcloud_file()}"]

    return f"ScanDataset({self.scan_file} {', '.join(args)})"

  def train(self, shuffle=True) -> Iterator[CameraView]:
    images = PreloadedImages(self.train_cameras, shuffle=shuffle)
    return images
    # images = PreloadedImages(self.train_cameras)
    # return torch.utils.data.DataLoader(images, batch_size=1, shuffle=shuffle, pin_memory=True, num_workers=2)
    
  def val(self) -> Iterator[CameraView]:
    images = PreloadedImages(self.val_cameras)
    return images
  

  def depth_range(self) -> Tuple[float, float]:
    return self.camera_depth_range


  def image_sizes(self) -> torch.Tensor:
    return torch.Tensor([(cam.camera.image_size) for cam in self.all_cameras]).to(torch.int32)

  def unique_projections(self) -> List[CameraProjection]:
    from camera_geometry import Camera

    def to_projection(camera:Camera):
      projection = np.array([*camera.focal_length, *camera.principal_point])
      return CameraProjection(projection, camera.image_size)

    return [to_projection(camera) for camera in self.scan.cameras.values()]
  

  def view_table(self) -> CameraRigTable:
    return camera_rig_table(self.scan)
  
  def camera_shape(self) -> torch.Size:
    return torch.Size([self.scan.num_frames, len(self.scan.cameras)])
  
  def pointcloud_file(self) -> Optional[str]:
    return find_cloud(self.scan)    


  def pointcloud(self) -> Optional[PointCloud]:
    pcd_filename = self.pointcloud_file()  
    return PointCloud.load(pcd_filename) if pcd_filename is not None else None

  
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
    
  


def find_cloud(scan:FrameSet) -> Tuple[np.ndarray, np.ndarray]:
  if 'sparse' not in scan.models:
    return None
  
  return Path(scan.find_file(scan.models.sparse.filename))
