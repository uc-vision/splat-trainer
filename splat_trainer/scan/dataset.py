from pathlib import Path
from camera_geometry import FrameSet

from camera_geometry.transforms import translate_44
from camera_geometry.camera_models import optimal_undistorted
import torch


import numpy as np
from splat_trainer.camera_pose import CameraRigTable

from splat_trainer.scan.loading import PreloadedImages, preload_images

from .visibility import visibility
from pyntcloud import PyntCloud


def load_cloud(scan:FrameSet) -> PyntCloud:
  assert 'sparse' in scan.models, "No sparse model found in scene.json"
  pcd_file = scan.find_file(scan.models.sparse.filename)

  return PyntCloud.from_file(pcd_file)


def camera_extents(scan:FrameSet):
    cam_centers = np.stack([camera.location for camera in scan.expand_cameras()])
    avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)

    distances = np.linalg.norm(cam_centers - avg_cam_center, axis=0, keepdims=True)
    diagonal = np.max(distances)

    return avg_cam_center.reshape(3), diagonal * 1.1



class ScanDataset:
  def __init__(self, filename:str,                
        image_scale:float=1.0,
        val_count:int=10):

    scan = FrameSet.load_file(Path(filename))

    self.centre, self.scene_scale = camera_extents(scan)    
    t = translate_44(*(-self.centre))
    self.scan = scan.transform(t).copy(
        metadata=dict(
          source=filename,
          offset=(-self.centre).tolist() )
       )

    cameras = {k: optimal_undistorted(camera, alpha=0)
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

  def train(self):
    return PreloadedImages(self.train_cameras)
  
  def val(self):
     return PreloadedImages(self.val_cameras)
  
  def camera_poses(self):
    camera_t_rig = np.array(
       [camera.camera_t_parent for camera in self.scan.cameras.values()])
    
    world_t_rig = torch.from_numpy(np.array(self.scan.rig_poses)).to(torch.float32)

    return CameraRigTable(
      rig_t_world=torch.linalg.inv(world_t_rig),
      camera_t_rig=torch.from_numpy(camera_t_rig).to(torch.float32))


  def scene(self) -> Scene:
    pcd = load_cloud(self.scan)    
    rgb = self.pyntcloud.points[[
            "red", "green", "blue"]].values.astype(np.float32)
    
    xyz = pcd.xyz - self.centre

    vis = visibility(self.scan.expand_cameras(), pcd.xyz)
    print(f"Visible {(vis > 0).sum()} of {len(vis)} points")
    # pcd = pcd.select_by_index(np.flatnonzero(vis > 0))
    
    xyz, rgb  = xyz[vis], rgb[vis]
    scene =  Scene(xyz, rgb, self.centre)


    return scene