import argparse
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from pyrender import Camera
from termcolor import colored
from tqdm import tqdm

from splat_trainer.camera_table.camera_table import ViewInfo
from splat_trainer.dataset.scan.dataset import find_cloud
from splat_trainer.dataset.scan.loading import CameraImage, camera_rig_table, load_scan

import torch
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.util.transforms import split_rt
from splat_trainer.util.visibility import random_cloud

from multiprocessing.pool import ThreadPool


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



def export_colmap(output_path:Path, cameras:Dict[str, Camera], cam_images:List[CameraImage], cloud:PointCloud):
    import pycolmap

    # Create a COLMAP reconstruction object
    reconstruction = pycolmap.Reconstruction()

    camera_ids = {}
    for i, (k, camera) in enumerate(cameras.items()):
        fx, fy, cx, cy = camera.intrinsics
        
        # Assuming image sizes are available in the trainer's dataset
        width, height = camera.image_size
        camera_ids[k] = i

        colmap_camera = pycolmap.Camera(
            model="PINHOLE",
            width=width,
            height=height,
            params=[fx, fy, cx, cy]
        )
        
        reconstruction.add_camera(colmap_camera)

    # Add cameras to the reconstruction
    for i, cam_image in enumerate(cam_images):
        cam = cam_image.camera

        # Add image to the reconstruction
        r, t = split_rt(cam.camera_t_parent)
        qvec = pycolmap.rotmat_to_qvec(r)
        
        reconstruction.add_image(
            pycolmap.Image(
                name=cam_image.filename,
                camera_id=camera_ids[cam_image.camera_name],
                qvec=qvec,
                tvec=t
            )
        )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export the reconstruction to a COLMAP format
    model_path = output_path / "sparse" / "0"
    model_path.mkdir(parents=True, exist_ok=True)


    positions = cloud.position.cpu().numpy()
    colors = (cloud.color.cpu().numpy() * 255).astype(np.uint8)

    # Add points to the reconstruction
    for i, (position, color) in enumerate(zip(positions, colors)):
        reconstruction.add_point3D(
            xyz=position,
            color=color,
            track=pycolmap.Track()
        )

    print(f"Added {len(cloud.position)} points to the reconstruction")

    reconstruction.write_text(model_path)
    print(f"Exported to COLMAP format in {model_path.absolute()}")


    image_path = output_path / "images"
    image_path.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(cam_images)} images to {image_path.absolute()}")

    def save_image(cam_image):
        cv2.imwrite(str(image_path / cam_image.filename), cam_image.image.cpu().numpy())

    with ThreadPool() as pool:
        list(tqdm(pool.imap(save_image, cam_images), total=len(cam_images), desc="Exporting images"))



def main():
  args = parse_args()

  scan, cam_images = load_scan(args.scan, image_scale=args.image_scale, resize_longest=args.resize_longest)
  print(f"Loaded {len(cam_images)} images from {args.scan}")

  if args.random_points is not None:
     camera_table = camera_rig_table(scan)
     image_sizes = torch.tensor([image.image_size for image in cam_images])

     view_info = ViewInfo(
        camera_table,
        image_sizes,
        depth_range=(args.near, args.far))


     cloud = random_cloud(view_info, args.random_points)
     print(f"Generated {cloud.count} random points")

  else:
    cloud_file = find_cloud(scan)   
    if cloud_file is None:
      raise RuntimeError(f"scan {args.scan} contains no sparse cloud, try random cloud?")
  
    cloud = PointCloud.load(cloud_file)
    print(f"Loaded sparse cloud with {cloud.count} points")

     
  print(f"Exporting to {colored(args.output, 'light_green')}")
  export_colmap(args.output, scan.cameras, cam_images, cloud)

if __name__ == "__main__":
  main()