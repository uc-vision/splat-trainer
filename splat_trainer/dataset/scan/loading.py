
from dataclasses import dataclass
from beartype.typing import Dict, Iterator, List

import cv2
import torch

from camera_geometry import FrameSet, Camera
from camera_geometry.scan.views import load_frames_with, Undistortion

from beartype import beartype
import numpy as np

from splat_trainer.dataset import CameraView

@dataclass
class CameraImage:
   camera : Camera
   image : torch.Tensor
   image_id: int

   filename: str

   @property
   def image_size(self):
      return self.image.shape[1], self.image.shape[0]


def concat_lists(xs):
  return [x for x in xs for x in x]


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
        image=torch.from_numpy(image).pin_memory(),
        image_id=frame_index * len(camera_names) + camera_id,
        filename=image_file
    )

  frames = load_frames_with(scan, undistortions, load)
  return concat_lists(frames)

class PreloadedImages(torch.utils.data.Dataset):
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

    

