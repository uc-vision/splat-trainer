from tensordict import tensorclass
from pathlib import Path

import torch


import numpy as np
import pypcd4
import plyfile

@tensorclass 
class PointCloud:
  points : torch.Tensor # (N, 3)
  colors : torch.Tensor # (N, 3)


  @staticmethod
  def from_numpy(xyz:np.ndarray, rgb:np.ndarray) -> 'PointCloud':
    return PointCloud(
        points = torch.from_numpy(xyz), 
        colors = torch.from_numpy(rgb),
        batch_size = (xyz.shape[0],))

  
  
  @staticmethod
  def load(filename:str | Path):
    filename = Path(filename)

    if filename.suffix == ".pcd": 
      cloud = pypcd4.PointCloud.from_path(filename)
      if 'rgb' in cloud.fields:
        rgb = cloud.numpy(('rgb',))
        rgb = rgb.view(np.uint8).reshape(-1, 4)
        rgb = rgb[:, :3].astype(np.float32) / 255.0

      elif 'r' in cloud.fields:
        rgb = cloud.numpy(('red', 'green', 'blue'))
      else:
        rgb = np.ones((cloud.size, 3), dtype=np.uint8)
        
      return PointCloud.from_numpy(cloud.numpy(("x", "y", "z")), rgb)

    elif filename.suffix == ".ply":
      ply = plyfile.PlyData.read(filename) 
      v = ply['vertex']
      xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
      rgb = np.stack([v['red'], v['green'], v['blue']], axis=1)

      return PointCloud.from_numpy(xyz, rgb)  
    else:
      raise ValueError(f"Unknown file type {filename.suffix}")