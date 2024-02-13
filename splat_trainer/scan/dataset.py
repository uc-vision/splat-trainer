from pathlib import Path
from typing import Iterator, Tuple
from camera_geometry import FrameSet

from camera_geometry.transforms import translate_44
from camera_geometry.camera_models import optimal_undistorted
import torch


import numpy as np
from splat_trainer.camera_pose import CameraRigTable
from splat_trainer.dataset import CameraView, Dataset

from splat_trainer.scan.loading import  PreloadedImages, preload_images
from splat_trainer.util.pointcloud import PointCloud
from .visibility import visibility

def load_cloud(scan:FrameSet) -> Tuple[np.ndarray, np.ndarray]:
  assert 'sparse' in scan.models, "No sparse model found in scene.json"
  cloud_file = Path(scan.find_file(scan.models.sparse.filename))

  return PointCloud.load_cloud(cloud_file)

def camera_extents(scan:FrameSet):
    cam_centers = np.stack([camera.location for camera in scan.expand_cameras()])
    avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)

    distances = np.linalg.norm(cam_centers - avg_cam_center, axis=0, keepdims=True)
    diagonal = np.max(distances)

    return avg_cam_center.reshape(3), diagonal * 1.1



class ScanDataset(Dataset):
  def __init__(self, scan_file:str,                
        image_scale:float=1.0,
        val_count:int=10,
        depth_range:Tuple[float, float] = (0.1, 100.0)):

    scan = FrameSet.load_file(Path(scan_file))
    self.depth_range = depth_range

    self.centre, self.scene_scale = camera_extents(scan)    
    t = translate_44(*(-self.centre))
    self.scan = scan.transform(t).copy(
        metadata=dict(
          source=scan_file,
          offset=(-self.centre).tolist() )
       )

    cameras = {k: optimal_undistorted(camera, alpha=0).scale_image(image_scale)
      for k, camera in scan.cameras.items()}

    print("Undistorted cameras:")
    for k, camera in cameras.items():
        print(k, camera)

    print("Loading images...")
    self.all_cameras = preload_images(scan, cameras)
    self.scan = scan.copy(cameras=cameras)


    # Evenly distribute validation images
    self.val_cameras = self.all_cameras[::len(self.all_cameras) // val_count]
    self.train_cameras = [c for c in self.all_cameras if c not in self.val_cameras]

  def train(self, shuffle=True) -> Iterator[CameraView]:
    images = PreloadedImages(self.train_cameras, shuffle=shuffle)
    return iter(images)
    # images = PreloadedImages(self.train_cameras)
    # return torch.utils.data.DataLoader(images, batch_size=1, shuffle=shuffle, pin_memory=True, num_workers=2)
    
  def val(self) -> Iterator[CameraView]:
    images = PreloadedImages(self.val_cameras)
    return iter(images)


  def camera_poses(self) -> CameraRigTable:
    camera_t_rig = np.array(
       [camera.camera_t_parent for camera in self.scan.cameras.values()])
    
    world_t_rig = torch.from_numpy(np.array(self.scan.rig_poses)).to(torch.float32)

    return CameraRigTable(
      rig_t_world=torch.linalg.inv(world_t_rig),
      camera_t_rig=torch.from_numpy(camera_t_rig).to(torch.float32))
  
  def camera_projection(self) -> torch.Tensor:
    projections = np.array([camera.intrinsic for camera in self.scan.cameras.values()])
    return torch.from_numpy(projections).to(torch.float32)

  def pointcloud(self) -> PointCloud:
    pcd = load_cloud(self.scan)    

    vis = visibility(self.scan.expand_cameras(), pcd.points)
    print(f"Visible {(vis > 0).sum()} of {len(vis)} points")
    # pcd = pcd.select_by_index(np.flatnonzero(vis > 0))
    
    return pcd[vis]
