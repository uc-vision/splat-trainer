from typing import Optional
from taichi_splatting import Gaussians3D
from termcolor import colored
import torch
from splat_trainer.dataset.dataset import Dataset
from splat_trainer.gaussians.loading import estimate_scale, from_pointcloud, from_scaled_pointcloud
from splat_trainer.trainer.config import CloudInitConfig
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.visibility.query_points import balanced_cloud, crop_cloud, projections, random_cloud


def get_initial_gaussians(config:CloudInitConfig, dataset:Dataset, device:torch.device) -> Gaussians3D:
    camera_table = dataset.camera_table().to(device)
    cameras = camera_table.cameras

    dataset_cloud:Optional[PointCloud] = dataset.pointcloud() if config.load_dataset_cloud else None
    points = None

    if dataset_cloud is not None:
      points = dataset_cloud.to(device)
      points = crop_cloud(cameras, points)

      if points.batch_size[0] == 0:
        raise ValueError("No points visible in dataset images, check input data!")

      print(colored(f"Using {points.batch_size[0]} points from original {dataset_cloud.batch_size[0]}", 'yellow'))
    
      if config.limit_points is not None:
        print(f"Limiting {points.batch_size[0]} points to {config.limit_points}")
        random_indices = torch.randperm(points.batch_size[0])[:config.limit_points]
        points = points[random_indices]
      
    if config.add_initial_points or dataset_cloud is None:
      random_points = balanced_cloud(cameras, config.initial_points, config.min_point_overlap)

      if points is not None:
        print(f"Adding {random_points.batch_size[0]} random points")
        points = torch.cat([points, random_points], dim=0)
      else:
        print(f"Using {random_points.batch_size[0]} random points")
        points = random_points

    scales = torch.full(points.batch_size, torch.inf, device=device)
    for proj in projections(cameras, points.points):
      density = proj.camera.image_size[0] * proj.camera.image_size[1] / proj.visible_mask.sum()
      scales[proj.visible_mask] = torch.minimum(scales[proj.visible_mask], density.sqrt() / proj.scale)


    # scales = estimate_scale(points, num_neighbors=config.num_neighbors)
    initial_gaussians:Gaussians3D = from_scaled_pointcloud(points, scales  * config.initial_point_scale, 
                                          initial_alpha=config.initial_alpha)

    return initial_gaussians