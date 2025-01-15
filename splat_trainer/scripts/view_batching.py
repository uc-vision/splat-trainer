from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from taichi_splatting import Gaussians3D

from splat_trainer.dataset.dataset import ImageView
from splat_trainer.scene.io import read_gaussians
from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.trainer import Trainer

from splat_trainer.util.misc import sh_to_rgb
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.util.view_cameras import CameraViewer
from splat_trainer.visibility import cluster
from splat_trainer.visibility.query_points import foreground_points


def display_image(title:str, image:np.ndarray):
  cv2.imshow(title, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
  while cv2.waitKey(1) == -1:
    pass


def transpose_batch(views:Sequence[ImageView]) -> Tuple[List[str], List[torch.Tensor], List[int]]:
  fields = {k:[] for k in ImageView._fields}
  for view in views:
    for k, v in zip(ImageView._fields, view):
      fields[k].append(v)

  return fields["filename"], fields["image"], fields["index"]


def image_grid(images:List[torch.Tensor], rows:int):
  image_row = []
  for i in range(0, len(images), rows):
    row = torch.concatenate(images[i:i+rows], dim=0)
    image_row.append(row)
  
  image = torch.concatenate(image_row, dim=1)
  image = image.cpu().numpy()
  return image



def show_batch(window:str, trainer:Trainer, batch_indexes:torch.Tensor, rows:int=2):
  assert batch_indexes.shape[0] % rows == 0, "Batch size must be divisible by number of rows"


  filenames, images, indexes = transpose_batch(trainer.dataset.loader(batch_indexes.cpu().numpy()))

  print(filenames)
  print(indexes, batch_indexes)

  grid = image_grid(images, rows)
  display_image(window, grid)


def sh_gaussians_to_cloud(gaussians:Gaussians3D) -> PointCloud:
  return PointCloud(gaussians.position, sh_to_rgb(gaussians.feature[:, :, 0]))


def main():
  parser = arguments()
  parser.add_argument("--min_batch", type=int, default=8, help="Minimum batch size to show")
  parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for selecting views")
  
  parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
  args = parser.parse_args()


  def f(trainer:Trainer):
    point_cloud = sh_gaussians_to_cloud(trainer.load_cloud())
    cameras = trainer.camera_table.cameras

    fg_mask = foreground_points(cameras, point_cloud.points)
    point_cloud = point_cloud[fg_mask]

    camera_viewer = CameraViewer(cameras, point_cloud)

    while camera_viewer.is_active():
      batch_indexes = trainer.view_selection.select_images(trainer.view_clustering, trainer.progress)

      camera_viewer.show_selection(batch_indexes)
      camera_viewer.wait_key()

  with_trainer(f, args)
