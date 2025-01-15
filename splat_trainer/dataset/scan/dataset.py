from functools import cached_property

from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from beartype import beartype
from beartype.typing import Iterator, Tuple

from camera_geometry import FrameSet
from camera_geometry.camera_models import optimal_undistorted

import numpy as np
import torch
from tqdm import tqdm
from splat_trainer.camera_table.camera_table import CameraRigTable, Label
from splat_trainer.dataset.dataset import  ImageView, Dataset
from splat_trainer.util.misc import split_stride


from .loading import  CameraImage, PreloadedImages, camera_rig_table, preload_images
from splat_trainer.util.pointcloud import PointCloud


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
    self.loaded_scan = scan

    self.camera_depth_range = [float(f) for f in depth_range]

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
  
    self.cameras = cameras
    self.scan = scan.copy(cameras=cameras)

    # Evenly distribute validation images
    train_idx, val_idx = split_stride(np.arange(scan.num_frames * len(cameras)), val_stride)
    self.train_idx = torch.tensor(train_idx, dtype=torch.long)
    self.val_idx = torch.tensor(val_idx, dtype=torch.long)
    
    
  def __repr__(self) -> str:
    args = [] 
    if self.image_scale is not None:
      args += [f"image_scale={self.image_scale}"]

    if self.resize_longest is not None:
      args += [f"resize_longest={self.resize_longest}"]

    args += [f"near={self.camera_depth_range[0]:.3f}", f"far={self.camera_depth_range[1]:.3f}"]
    args += [f"pointcloud={self.pointcloud_file()}"]

    return f"ScanDataset({self.scan_file} {', '.join(args)})"

  @cached_property
  def _images(self) -> PreloadedImages:
    """ Load all images into memory, DO NOT use this property unless you want to load the images """
    print("Loading images...")
    return preload_images(self.loaded_scan, self.cameras)

  @property
  def num_images(self) -> int:
    return self.scan.num_frames * len(self.cameras)

  @beartype
  def loader(self, idx:torch.Tensor, shuffle:bool=False) -> Sequence[ImageView]:
    images = [self._images[i] for i in idx.cpu().numpy()]
    return PreloadedImages(images, shuffle=shuffle)
    

  def train(self, shuffle=False) -> Sequence[ImageView]:
    return self.loader(self.train_idx, shuffle=shuffle)

    
  def val(self) -> Sequence[ImageView]:
    return self.loader(self.val_idx)
  
  def camera_table(self) -> CameraRigTable:
    labels = torch.zeros((self.num_images, ), dtype=torch.int32)

    labels[self.train_idx] |= Label.Training.value
    labels[self.val_idx] |= Label.Validation.value

    return camera_rig_table(self.scan, self.camera_depth_range, labels)

  
  def pointcloud_file(self) -> Optional[str]:
    return find_cloud(self.scan)    

  def pointcloud(self) -> Optional[PointCloud]:
    pcd_filename = self.pointcloud_file()  
    return PointCloud.load_cloud(pcd_filename) if pcd_filename is not None else None



def find_cloud(scan:FrameSet) -> Tuple[np.ndarray, np.ndarray]:
  if 'sparse' not in scan.models:
    return None
  
  return Path(scan.find_file(scan.models.sparse.filename))
