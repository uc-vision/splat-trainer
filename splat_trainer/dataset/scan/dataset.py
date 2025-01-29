from functools import cached_property, partial
import numpy as np
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

from splat_trainer.dataset.util import scene_normalization, split_train_val


class ScanDataset(Dataset):
  @beartype
  def __init__(self, scan_file:str,                
        image_scale:Optional[float]=None,
        resize_longest:Optional[int]=None,
        val_stride:int=0,
        depth_range:Sequence[float] = (0.1, 100.0),
        normalize_scale:bool=False,
        centering:bool=True,
        ):

    self.scan_file = scan_file
    self.image_scale = image_scale
    self.resize_longest = resize_longest
    self._images = None

    scan = FrameSet.load_file(Path(scan_file))
    self.loaded_scan = scan

    self.camera_depth_range = [float(f) for f in depth_range]
    center, scale = scene_normalization([cam.location for cam in scan.expand_cameras()])

    self.centering = np.zeros(3, dtype=np.float32)
    self.scaling = 1.0

    if centering is True:
      scan = scan.translated(-center)
      self.centering = center

    if normalize_scale is True:
      scan = scan.scaled(1 / scale)
      self.scaling = scale


    print("Scan cameras:")
    for k, camera in scan.cameras.items():
        print(k, camera)

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

    self.train_idx, self.val_idx = split_train_val(self.num_images, val_stride)
    
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

    centre, scale = self.scene_transform
    centre = [f"{c:.3f}" for c in centre.tolist()]
    args += [f"centering={centre}", f"scaling={scale:.3f}"]

    return f"ScanDataset({self.scan_file} {', '.join(args)})"

  @property
  def scene_transform(self) -> tuple[torch.Tensor, float]:
    return torch.from_numpy(self.centering).to(torch.float32), self.scaling

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
    centering, scaling = self.scene_transform
    return pcd.translated(-centering).scaled(1.0/scaling)


def find_cloud(scan:FrameSet) -> Tuple[np.ndarray, np.ndarray]:
  if 'sparse' not in scan.models:
    return None
  
  return Path(scan.find_file(scan.models.sparse.filename))
