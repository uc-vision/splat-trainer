
from dataclasses import dataclass, replace
from typing import Dict, List

import cv2
from taichi_3d_gaussian_splatting.Camera import CameraInfo, CameraView
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch

from camera_geometry import FrameSet, Camera


from camera_geometry.scan import FrameSet
from camera_geometry.scan.views import load_frames_with, undistort_cameras, Undistortion
from camera_geometry.transforms import translate_44, split_rt
import torch
from tqdm import tqdm

from py_structs import concat_lists

from beartype import beartype
import numpy as np


@dataclass 
class CameraImage:
   camera:Camera
   image : torch.Tensor
   frame_id: int
   camera_id: int
   image_id: int

   filename: str

@beartype
def preload_images(scan:FrameSet, undistorted:Dict[str, Camera]) -> List[CameraImage]:
  undistortions = {k:Undistortion(camera, undistorted[k]) 
    for k, camera in scan.cameras.items() }

  camera_names = list(undistortions.keys())
  num_cameras = len(camera_names)

  @beartype
  def load(undistortion: Undistortion, rig_pose:np.ndarray, camera_name:str, frame_index:int):
    image_file = scan.image_sets.rgb[frame_index][camera_name]
    image = scan.loader.rgb.load_image(image_file)
    image = undistortion.undistort(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    camera_id = camera_names.index(camera_name)
    return CameraImage(
        camera=undistortion.undistorted.transform(rig_pose),
        image=torch.from_numpy(image).permute(2, 0, 1),
        frame_id=frame_index,
        camera_id=camera_id,
        image_id=num_cameras * frame_index + camera_names.index(camera_name),
        filename=image_file
    )

  frames = load_frames_with(scan, undistortions, load)
  return concat_lists(frames)

class PreloadedImages(torch.utils.data.Dataset):
  @beartype
  def __init__(self, camera_images:List[CameraImage]):
    self.camera_images = camera_images

  def __len__(self):
      return len(self.camera_images)

  def __getitem__(self, index):
    camera_image:CameraImage = self.camera_images[index]
    camera:Camera = camera_image.camera

    camera_t_parent = torch.from_numpy(camera.camera_t_parent).to(torch.float32).unsqueeze(0)
    camera_t_parent_qt = SE3_to_quaternion_and_translation_torch(camera_t_parent)

    info = CameraInfo(
        camera_intrinsics=torch.from_numpy(camera.intrinsic).to(torch.float32),
        camera_height = int(camera.image_size[1]),
        camera_width = int(camera.image_size[0]),
        camera_id=camera_image.camera_id
    )

    idx = torch.tensor([camera_image.image_id], dtype=torch.long)
    return camera_image.filename, camera_image.image, camera_t_parent_qt, idx, info
     

