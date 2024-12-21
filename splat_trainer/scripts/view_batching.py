
from typing import List, Tuple
import cv2
import numpy as np
from taichi_splatting import Gaussians3D
import torch
from splat_trainer.dataset.dataset import CameraView
from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.trainer import Trainer
from splat_trainer.util.misc import sh_to_rgb
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.util.view_cameras import CameraViewer
from splat_trainer.util.visibility import foreground_points


def display_image(title:str, image:np.ndarray):
  cv2.imshow(title, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
  while cv2.waitKey(1) == -1:
    pass


def transpose_batch(views:List[CameraView]) -> Tuple[List[str], List[np.ndarray], List[int]]:
  fields = {k:[] for k in CameraView._fields}
  for view in views:
    for k, v in zip(CameraView._fields, view):
      fields[k].append(v)

  return fields["filename"], fields["image"], fields["index"]


def image_grid(images:List[np.ndarray], rows:int):
  rows = []
  for i in range(0, len(images), rows):
    row = torch.concatenate(images[i:i+rows], dim=0)
    rows.append(row)
  
  image = torch.concatenate(rows, dim=1)
  image = image.cpu().numpy()

def sh_gaussians_to_cloud(gaussians:Gaussians3D) -> PointCloud:
  sh_features = gaussians.feature[:, :, 0] # N, 3
  positions = gaussians.position # N, 3
  colors = sh_to_rgb(sh_features) # N, 3

  return PointCloud(positions, colors, batch_size=(positions.shape[0],))


def show_batch():
  parser = arguments()
  parser.add_argument("--batch_size", type=int, default=8, help="Batch size to show")
  parser.add_argument("--rows", type=int, default=2, help="Number of rows to show")
  
  parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
  parser.add_argument("--show_images", action="store_true", help="Show images")
  parser.add_argument("--show_cameras", action="store_true", help="Show cameras")
  args = parser.parse_args()

  assert args.batch_size % args.rows == 0, "Batch size must be divisible by number of rows"


  def f(trainer:Trainer):
    gaussians = trainer.scene.to_sh_gaussians()
    point_cloud = sh_gaussians_to_cloud(gaussians)
    cameras = trainer.camera_table.cameras

    fg_mask = foreground_points(cameras, point_cloud.points)

    if args.show_cameras:
      camera_viewer = CameraViewer(cameras, point_cloud[fg_mask])

    if args.show_images:
      cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    result = trainer.evaluate()
    print(result)


    while True:
      batch_indexes = trainer.select_batch(args.batch_size, temperature=args.temperature)
      filenames, images, indexes = transpose_batch(trainer.dataset.loader(batch_indexes.cpu().numpy()))

      print(filenames)
      print(indexes, batch_indexes)



      if args.show_images:
        grid = image_grid(images, args.rows)
        display_image("image", grid)

      if camera_viewer is not None:

        # cameras = trainer.camera_table[batch_indexes]
        mask = torch.zeros(len(cameras), dtype=torch.bool)
        mask[batch_indexes] = True
        camera_viewer.highlight_cameras(mask)
        camera_viewer.wait_key()

  with_trainer(f, args)
