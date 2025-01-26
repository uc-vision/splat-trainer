from typing import Optional
from taichi_splatting import Gaussians3D
from termcolor import colored
import torch
from splat_trainer.camera_table.camera_table import Cameras
from splat_trainer.dataset.dataset import Dataset
from splat_trainer.gaussians.loading import estimate_scale, from_pointcloud, from_scaled_pointcloud
from splat_trainer.trainer.config import CloudInitConfig
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.visibility.query_points import balanced_cloud, crop_cloud, projections, random_cloud


def get_initial_gaussians(config:CloudInitConfig, dataset:Dataset, device:torch.device) -> Gaussians3D:
    camera_table = dataset.camera_table().to(device)
    cameras:Cameras = camera_table.cameras

    points:Optional[PointCloud] = dataset.pointcloud()

    if points is not None:
      points = points.to(device)
      num_dataset_points = points.batch_size[0]

      points = crop_cloud(cameras, points)
      if points.batch_size[0] == 0:
        raise ValueError("No points visible in dataset images, check input data!")

      print(colored(f"Found {points.batch_size[0]} visible points from original {num_dataset_points}", 'yellow'))
      limit = points.batch_size[0]
      if config.initial_points is not None:
        limit = min(limit, config.initial_points)

      if config.limit_points is not None:
        limit = min(limit, config.limit_points)

        random_indices = torch.randperm(points.batch_size[0])[:limit]
        points = points[random_indices]

    if config.initial_points is not None or points is None:
      n = config.initial_points or 0
      n_dataset = points.batch_size[0] if points is not None else 0
      n_random = n - n_dataset
      
      print(f"Initializing with {n} total points, from dataset {n_dataset}, random {n_random}")
      cameras = cameras.clamp_near(config.clamp_near)

      points = balanced_cloud(cameras, n, config.min_view_overlap, points)


    scales = estimate_scale(points, num_neighbors=config.num_neighbors)
    initial_gaussians:Gaussians3D = from_scaled_pointcloud(points, scales  * config.initial_point_scale, 
                                          initial_alpha=config.initial_alpha)

    return initial_gaussians
    