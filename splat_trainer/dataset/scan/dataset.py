from dataclasses import dataclass
from functools import cached_property, partial
import numpy as np
from splat_trainer.dataset.normalization import Normalization, NormalizationConfig
import torch
from tqdm import tqdm

from pathlib import Path
from typing import Optional, Sequence
from beartype import beartype
from beartype.typing import Tuple

from camera_geometry import FrameSet
from camera_geometry.camera_models import optimal_undistorted

from splat_trainer.camera_table.camera_table import CameraRigTable, Label
from splat_trainer.util.pointcloud import PointCloud

from splat_trainer.dataset.dataset import  ImageView, Dataset
from .loading import  PreloadedImages, camera_rig_table, preload_images

from splat_trainer.dataset.util import split_every, expand_index


def camera_positions(scan:FrameSet) -> torch.Tensor:
  centers = np.array([cam.location for cam in scan.expand_cameras()])
  return torch.from_numpy(centers).to(torch.float32)


class ScanDataset(Dataset):
  @beartype
  def __init__(self, scan_file:str,                
        image_scale:Optional[float]=None,
        resize_longest:Optional[int]=None,
        test_every:int=8,
        depth_range:Sequence[float] = (0.1, 100.0),
        
        normalize:NormalizationConfig=NormalizationConfig()
        ):

    self.scan_file = scan_file
    self.image_scale = image_scale
    self.resize_longest = resize_longest
    self._images = None

    scan = FrameSet.load_file(Path(scan_file))
    self.loaded_scan = scan

    self.camera_depth_range = [float(f) for f in depth_range]
    self.normalize = normalize.get_transform(camera_positions(scan))

    print("Scan cameras:")
    for k, camera in scan.cameras.items():
        print(k, camera)

    scan = scan.translated(self.normalize.translation).scaled(self.normalize.scaling)
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

    # Split frames (each a set of images per camera) into training and test sets 
    # Pad to 2 to avoid first and last frames as quality is lower
    train_frames, val_frames = split_every(scan.num_frames, test_every, padding=2)

    # Get image indexes from train and test frames
    self.train_idx = expand_index(train_frames, len(cameras))
    self.val_idx = expand_index(val_frames, len(cameras))

    
  def load_images(self) -> PreloadedImages:
    if self._images is None:
      self._images = preload_images(self.loaded_scan, self.cameras, progress=partial(tqdm, desc="Loading images"))

    return self._images
    
  def __repr__(self) -> str:
    args = [] 
    if self.image_scale is not None:
      args += [f"image_scale={self.image_scale}"]

    if self.resize_longest is not None:
      args += [f"resize_longest={self.resize_longest}"]

    args += [f"near={self.camera_depth_range[0]:.3f}", f"far={self.camera_depth_range[1]:.3f}"]
    args += [f"pointcloud={self.pointcloud_file()}"]

    args += [f"normalization={self.normalize}"]
    args += [f"num_train={len(self.train_idx)}", f"num_val={len(self.val_idx)}"]
    return f"ScanDataset({self.scan_file} {', '.join(args)})"

  @property
  def to_original(self) -> Normalization:
    return self.normalize.inverse

  @property
  def num_images(self) -> int:
    return self.scan.num_frames * len(self.cameras)

  @beartype
  def loader(self, idx:torch.Tensor, shuffle:bool=False) -> Sequence[ImageView]:
    images = self.load_images()

    images = [images[i] for i in idx.cpu().numpy()]
    return PreloadedImages(images, shuffle=shuffle)
    

  def train(self, shuffle=False) -> Sequence[ImageView]:
    return self.loader(self.train_idx, shuffle=shuffle)

    
  def val(self) -> Sequence[ImageView]:
    return self.loader(self.val_idx)
  
  @cached_property
  def camera_table(self) -> CameraRigTable:
    labels = torch.zeros((self.num_images, ), dtype=torch.int32)

    labels[self.train_idx] |= Label.Training.value
    labels[self.val_idx] |= Label.Validation.value

    return camera_rig_table(self.scan, self.camera_depth_range, labels)

  
  def pointcloud_file(self) -> Optional[str]:
    return find_cloud(self.scan)    

  def pointcloud(self) -> Optional[PointCloud]:
    pcd_filename = self.pointcloud_file()  
    if pcd_filename is None:
      return None

    pcd = PointCloud.load_cloud(pcd_filename)
    return self.normalize.transform_cloud(pcd)


def find_cloud(scan:FrameSet) -> Tuple[np.ndarray, np.ndarray]:
  if 'sparse' not in scan.models:
    return None
  
  return Path(scan.find_file(scan.models.sparse.filename))
