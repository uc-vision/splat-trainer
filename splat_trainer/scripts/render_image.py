
from pathlib import Path
import cv2
import numpy as np
from taichi_splatting.perspective.params import CameraParams
import torch
from splat_trainer.scripts.checkpoint import arguments, with_trainer
from splat_trainer.trainer import Trainer
from splat_trainer.util.colorize import colorize_depth
from splat_trainer.util.render_image import frame_scene
from splat_trainer.visibility.query_points import foreground_points


def display_images(images:dict[str, np.ndarray]):
  
  for name, image in images.items():
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)

  try:
    while True:
      cv2.waitKey(1)
  except KeyboardInterrupt:
    pass


def crop_image(camera_params:CameraParams, offset:tuple[int, int], size:tuple[int, int]):
  T_camera_world = camera_params.T_camera_world
  offset = torch.tensor(offset, device=camera_params.device)

  projection = camera_params.projection.clone()
  projection[2:] = projection[2:] - offset 

  
  return CameraParams(
    projection=projection,
    T_camera_world=T_camera_world,
    image_size=size,
    near_plane=camera_params.near_plane,
    far_plane=camera_params.far_plane,
  )


def render_tiled(trainer:Trainer, camera_params:CameraParams, tile_size:int=2048):
  image_size = camera_params.image_size
  image = torch.zeros((image_size[1], image_size[0], 3), device=trainer.device)
  depth = torch.zeros((image_size[1], image_size[0]), device=trainer.device)


  for i in range(0, image_size[0], tile_size):
    for j in range(0, image_size[1], tile_size):
      w, h = min(tile_size, image_size[0] - i), min(tile_size, image_size[1] - j)

      sub_params = crop_image(camera_params, (i, j), (w, h))

      rendering = trainer.render(sub_params, render_median_depth=True)
      h, w = rendering.image.shape[:2]

      image[j:j+h, i:i+w] = rendering.image
      depth[j:j+h, i:i+w] = rendering.median_depth_image

  return image, depth

def main():
  parser = arguments()
  parser.add_argument("--image_size", type=int, default=2048, help="Size of the image to render (longest side)")
  parser.add_argument("--distance", type=float, default=4.0, help="Distance from the camera to the scene")
  parser.add_argument("--margin", type=float, default=0.25, help="Margin to add to camera bounds (m)")

  parser.add_argument("--foreground", action="store_true", help="Crop to foreground points")

  parser.add_argument("--show", action="store_true", help="Show the rendered image/depth image")
  parser.add_argument("--save", default=None, type=Path, help="Save the rendered image and depth image to this path")
  args = parser.parse_args()

  def f(trainer:Trainer):
    cameras = trainer.camera_table.cameras

    if args.foreground:
      mask = foreground_points(cameras, trainer.scene.gaussians.position, quantile=0.25, min_overlap=0.05)
      trainer.scene.split_and_prune(keep_mask=mask)

    camera_params = frame_scene(cameras, distance=args.distance, longest_side=args.image_size, margin=args.margin)

    image, depth = render_tiled(trainer, camera_params, tile_size=2048)
    print(f"Rendered image {image.shape[1]}x{image.shape[0]}")
    
    image = cv2.cvtColor(image.cpu().numpy(), cv2.COLOR_RGB2BGR)
    colorised = cv2.cvtColor(colorize_depth(trainer.color_map, depth - args.distance, near=0.3).cpu().numpy(), cv2.COLOR_RGB2BGR)

    if args.save is not None:
      args.save.mkdir(parents=True, exist_ok=True)
      filenames = ["image.jpg", "depth_colorised.jpg", "depth.tiff"]

      for filename, image in zip(filenames, [image, colorised, depth.cpu().numpy()]):
        cv2.imwrite(str(args.save / filename), image)

      print(f"Wrote {filenames} to {args.save}")

    if args.show:
      display_images(dict(image=image, depth=colorised))






  with_trainer(f, args)

