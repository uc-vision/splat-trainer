from pathlib import Path
from beartype.typing import Iterator, Tuple
from camera_geometry import FrameSet

from camera_geometry.camera_models import optimal_undistorted
import torch


import numpy as np
from splat_trainer.camera_table.camera_table import CameraRigTable, CameraTable
from splat_trainer.dataset.dataset import CameraView, Dataset
from splat_trainer.util.misc import split_stride


from .loading import  PreloadedImages, preload_images
from splat_trainer.util.pointcloud import PointCloud
from .visibility import visibility



class ScanDataset(Dataset):
  def __init__(self, scan_file:str,                
        image_scale:float=1.0,
        val_stride:int=10,
        depth_range:Tuple[float, float] = (0.1, 100.0)):

    self.scan_file = scan_file
    self.image_scale = image_scale

    scan = FrameSet.load_file(Path(scan_file))
    self.depth_range = depth_range

    cameras = {k: optimal_undistorted(camera, alpha=0).scale_image(image_scale)
      for k, camera in scan.cameras.items()}

    print("Undistorted cameras:")
    for k, camera in cameras.items():
        print(k, camera)

    print("Loading images...")
    self.all_cameras = preload_images(scan, cameras)
    self.scan = scan.copy(cameras=cameras)

    # Evenly distribute validation images
    self.train_cameras, self.val_cameras = split_stride(self.all_cameras, val_stride)
    
  def __repr__(self) -> str:
    return f"ScanDataset({self.scan_file}, image_scale={self.image_scale} cloud={find_cloud(self.scan)})"

  def train(self, shuffle=True) -> Iterator[CameraView]:
    images = PreloadedImages(self.train_cameras, shuffle=shuffle)
    return images
    # images = PreloadedImages(self.train_cameras)
    # return torch.utils.data.DataLoader(images, batch_size=1, shuffle=shuffle, pin_memory=True, num_workers=2)
    
  def val(self) -> Iterator[CameraView]:
    images = PreloadedImages(self.val_cameras)
    return images


  def camera_table(self) -> CameraRigTable:
    camera_t_rig = np.array(
       [camera.camera_t_parent for camera in self.scan.cameras.values()])
    
    world_t_rig = torch.from_numpy(np.array(self.scan.rig_poses)).to(torch.float32)
    projections = np.array([camera.intrinsic for camera in self.scan.cameras.values()])

    
    return CameraRigTable(
      rig_t_world=torch.linalg.inv(world_t_rig),
      camera_t_rig=torch.from_numpy(camera_t_rig).to(torch.float32),
      projection=torch.from_numpy(projections).to(torch.float32))
  
  def camera_shape(self) -> torch.Size:
    return torch.Size([self.scan.num_frames, len(self.scan.cameras)])
  

  def pointcloud(self) -> PointCloud:
    pcd_filename = find_cloud(self.scan)    
    pcd = PointCloud.load(pcd_filename)

    counts = visibility(self.scan.expand_cameras(), pcd.points)
    print(f"Visible {(counts > 0).sum()} of {len(counts)} points")
    # pcd = pcd.select_by_index(np.flatnonzero(vis > 0))
    
    return pcd[counts > 0]


def find_cloud(scan:FrameSet) -> Tuple[np.ndarray, np.ndarray]:
  assert 'sparse' in scan.models, "No sparse model found in scene.json"
  return Path(scan.find_file(scan.models.sparse.filename))
