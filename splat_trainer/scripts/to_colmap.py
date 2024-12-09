import argparse
from pathlib import Path
import traceback
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from termcolor import colored
from tqdm import tqdm


from camera_geometry import FrameSet
from camera_geometry.camera_models import optimal_undistorted

from splat_trainer.dataset.scan.dataset import find_cloud
from splat_trainer.dataset.scan.loading import CameraImage, camera_rig_table, preload_images

import camera_geometry 
from scipy.spatial.transform import Rotation as R

import torch
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.util.visibility import random_cloud

from multiprocessing.pool import ThreadPool


def load_scan(scan_file:str, image_scale:Optional[float]=None, resize_longest:Optional[int]=None) -> Tuple[FrameSet, List[CameraImage]]:
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

    print("Loading images...")
    all_cameras = preload_images(scan, cameras)
    return scan.copy(cameras=cameras), all_cameras


def parse_args():
  parser = argparse.ArgumentParser("Export a dataset to COLMAP format")

  parser.add_argument("--scan", type=str, required=True, help="Scan json scene file to load")
  parser.add_argument("--output", type=str, required=True, help="Project name")

  parser.add_argument("--image_scale", type=float, default=None, help="Image scale")
  parser.add_argument("--resize_longest", type=int, default=None, help="Resize longest side")

  parser.add_argument("--random_points", type=int, default=None, help="Generate random points")

  parser.add_argument("--near", type=float, default=0.1, help="Near distance")
  parser.add_argument("--far", type=float, default=100.0, help="Far distance")


  return parser.parse_args()

def flatten_path(path:str) -> str:
   # remove any / in the path with underscore
   return path.replace("/", "_")

def export_colmap(output_path:Path, cameras:Dict[str, camera_geometry.Camera], cam_images:List[CameraImage], cloud:PointCloud):
    import pycolmap

    # Create a COLMAP reconstruction object
    reconstruction = pycolmap.Reconstruction()

    camera_ids = {}
    for i, (k, camera) in enumerate(cameras.items()):
        fx, fy = camera.focal_length
        cx, cy = camera.principal_point
        # Assuming image sizes are available in the trainer's dataset
        width, height = camera.image_size
        camera_ids[k] = i

        colmap_camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=[fx, fy, cx, cy],
            camera_id=i
        )
        
        reconstruction.add_camera(colmap_camera)

    # Add cameras to the reconstruction
    for i, cam_image in enumerate(cam_images):
        cam = cam_image.camera

        # Add image to the reconstruction
        r = cam.camera_t_parent[0:3, 0:3]
        t = cam.camera_t_parent[0:3, 3]

        q = R.from_matrix(r).as_quat(scalar_first=False)

        cam_from_world = pycolmap.Rigid3d(q, t)
        
        image = pycolmap.Image(
            image_id=i,
            name=flatten_path(cam_image.filename),
            camera_id=camera_ids[cam_image.camera_name],
            cam_from_world=cam_from_world
            )
        reconstruction.add_image(image)
        reconstruction.register_image(image.image_id)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export the reconstruction to a COLMAP format
    model_path = output_path / "sparse" / "0"
    model_path.mkdir(parents=True, exist_ok=True)


    positions = cloud.points.cpu().numpy()
    colors = (cloud.colors.cpu().numpy() * 255).astype(np.uint8)

    # Add points to the reconstruction
    for i, (position, color) in enumerate(zip(positions, colors)):
        reconstruction.add_point3D(
            xyz=position,
            color=color,
            track=pycolmap.Track(),
        )


    reconstruction.check()
    print(reconstruction.summary())

    reconstruction.write_text(model_path)
    print(f"Exported to COLMAP format in {model_path.absolute()}")


    image_path = output_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(cam_images)} images to {image_path.absolute()}")

    def save_image(cam_image):
        try:
            image_rgb = cam_image.image.cpu().numpy()
            cv2.imwrite(str(image_path / flatten_path(cam_image.filename)), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
          traceback.print_exc()
          print(f"Error saving image {cam_image.filename}: {e}")

    with ThreadPool() as pool:
        for _ in tqdm(pool.imap_unordered(save_image, cam_images), total=len(cam_images), desc="Exporting images"):
            pass


def main():
  args = parse_args()

  scan, cam_images = load_scan(args.scan, image_scale=args.image_scale, resize_longest=args.resize_longest)
  print(f"Loaded {len(cam_images)} images from {args.scan}")

  cloud_file = find_cloud(scan)   
  if cloud_file is None and args.random_points is None:
      print(f"scan {args.scan} contains no sparse cloud, generating 50000 random points instead (specify number with --random_points)")
      args.random_points = 50000

  if args.random_points is not None:
     camera_table = camera_rig_table(scan)
     image_sizes = torch.tensor([image.image_size for image in cam_images])


     cloud = random_cloud(camera_table, (args.near, args.far), args.random_points)
     print(f"Generated {cloud.count} random points")

  else:

    cloud = PointCloud.load(cloud_file)
    print(f"Loaded sparse cloud with {cloud.count} points")

     
  print(f"Exporting to {colored(args.output, 'light_green')}")
  export_colmap(args.output, scan.cameras, cam_images, cloud)

if __name__ == "__main__":
  main()