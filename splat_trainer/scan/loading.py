
from dataclasses import dataclass
from typing import Dict, List

import cv2
import torch

from camera_geometry import FrameSet, Camera
from camera_geometry.scan.views import load_frames_with, Undistortion

from beartype import beartype
import numpy as np

def concat_lists(xs):
  return [x for x in xs for x in x]

@dataclass 
class CameraImage:
   image : torch.Tensor
   frame_id: int
   camera_id: int

   filename: str

@beartype
def preload_images(scan:FrameSet, undistorted:Dict[str, Camera]) -> List[CameraImage]:
  undistortions = {k:Undistortion(camera, undistorted[k]) 
    for k, camera in scan.cameras.items() }

  camera_names = list(undistortions.keys())

  @beartype
  def load(undistortion: Undistortion, rig_pose:np.ndarray, camera_name:str, frame_index:int):
    image_file = scan.image_sets.rgb[frame_index][camera_name]
    image = scan.loader.rgb.load_image(image_file)
    image = undistortion.undistort(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    camera_id = camera_names.index(camera_name)
    return CameraImage(
        camera=undistortion.undistorted.transform(rig_pose),
        image=torch.from_numpy(image).permute(2, 0, 1).pin_memory(),
        frame_id=frame_index,
        camera_id=camera_id,
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

    idx = torch.tensor([camera_image.frame_id, camera_image.image_id], dtype=torch.long)
    return camera_image.filename, camera_image.image, idx
     

