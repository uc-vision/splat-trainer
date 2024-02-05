from tensordict import tensorclass
from pathlib import Path

import torch


import numpy as np
import pypcd4
import plyfile

@tensorclass 
class PointCloud:
  points : torch.Tensor
  colors : torch.Tensor
  
  @staticmethod
  def load_cloud(filename:str | Path):
    filename = Path(filename)

    if filename.suffix == ".pcd": 
      cloud = pypcd4.PointCloud.from_path(filename)
      return PointCloud(cloud.numpy(("x", "y", "z")), 
                        cloud.numpy("red", "green", "blue"))
    elif filename.suffix == ".ply":
      with plyfile.PlyData.read(filename) as ply:
        v = ply['vertex']
        return PointCloud(np.stack([v['x'], v['y'], v['z']], axis=1), 
                np.stack([v['red'], v['green'], v['blue']], axis=1))  
    else:
      raise ValueError(f"Unknown file type {filename.suffix}")