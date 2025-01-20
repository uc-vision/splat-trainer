
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Sequence, Tuple
from beartype.typing import Dict, Iterator, List

import cv2
import torch

from camera_geometry import FrameSet, Camera
from camera_geometry.scan.views import load_frames_with, Undistortion
from camera_geometry.camera_models.camera import optimal_undistorted

from beartype import beartype
import numpy as np
from tqdm import tqdm

from splat_trainer.camera_table.camera_table import CameraRigTable, Projections
from splat_trainer.dataset import ImageView

@dataclass
class CameraImage:
   camera : Camera
   image : torch.Tensor
   image_id: int

   camera_name: str
   frame_index: int

   filename: str

   @property
   def image_size(self):
      return self.image.shape[1], self.image.shape[0]


def load_scan(scan_file:str, image_scale:Optional[float]=None, 
              resize_longest:Optional[int]=None) -> Tuple[FrameSet, List[CameraImage]]:
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

    all_cameras = preload_images(scan, cameras, progress=partial(tqdm, desc="Loading images"))
    return scan.copy(cameras=cameras), all_cameras


def projections(scan:FrameSet, depth_range:Tuple[float, float]) -> Projections:
    def to_projection(camera:Camera):
      return Projections(
        intrinsics = torch.tensor([*camera.focal_length, *camera.principal_point], dtype=torch.float32),
        image_size = torch.tensor(camera.image_size, dtype=torch.long),
        depth_range = torch.tensor(depth_range, dtype=torch.float32))

    return torch.stack([to_projection(camera) for camera in scan.cameras.values()])

def camera_rig_table(scan:FrameSet, depth_range:Tuple[float, float], labels:torch.Tensor) -> CameraRigTable:
    camera_t_rig = np.array(
      [camera.camera_t_parent for camera in scan.cameras.values()])
    
    world_t_rig = torch.from_numpy(np.array(scan.rig_poses)).to(torch.float32)
    image_names = [image_set[k] for k in scan.camera_names for image_set in scan.image_sets.rgb]

    return CameraRigTable(
      rig_t_world=torch.linalg.inv(world_t_rig),
      camera_t_rig=torch.from_numpy(camera_t_rig).to(torch.float32),
      projection=projections(scan, depth_range),
      image_names=image_names,
      labels=labels)


def concat_lists(xs):
  return [x for x in xs for x in x]


@beartype
def preload_images(scan:FrameSet, undistorted:Dict[str, Camera], progress=None) -> List[CameraImage]:
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
        camera_name=camera_name,
        frame_index=frame_index,

        image=torch.from_numpy(image).pin_memory(),
        image_id=frame_index * len(camera_names) + camera_id,
        filename=image_file
    )

  frames = load_frames_with(scan, undistortions, load, progress=progress)
  return concat_lists(frames)

class PreloadedImages(Sequence[ImageView]):
  @beartype
  def __init__(self, camera_images:List[CameraImage], shuffle:bool=False):
    self.camera_images = camera_images
    self.shuffle = shuffle

  def __len__(self):
      return len(self.camera_images)

  def __getitem__(self, index) -> ImageView:
    camera_image:CameraImage = self.camera_images[index]
    return ImageView(camera_image.filename, camera_image.image_id, camera_image.image)
     
  def __iter__(self) -> Iterator[ImageView]:
    order = torch.randperm(len(self)) if self.shuffle else torch.arange(len(self))
    for idx in order:
      yield self[idx]  

    

