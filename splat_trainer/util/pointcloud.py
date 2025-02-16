from tensordict.tensorclass import tensorclass
from pathlib import Path

import torch


import numpy as np
import pypcd4
import plyfile


@tensorclass
class PointCloud: 
  points : torch.Tensor # (N, 3)
  colors : torch.Tensor # (N, 3)

  @property
  def num_points(self) -> int:
    return self.points.shape[0]
  
  @property
  def device(self) -> torch.device:
    return self.points.device


  @staticmethod
  def from_numpy(xyz:np.ndarray, rgb:np.ndarray) -> 'PointCloud':
    if rgb.dtype == np.uint8:
      rgb = rgb.astype(np.float32) / 255.0

    return PointCloud(
        points = torch.from_numpy(xyz.astype(np.float32)), 
        colors = torch.from_numpy(rgb.astype(np.float32)),
        batch_size = (xyz.shape[0],))



  def append(self, other: 'PointCloud') -> 'PointCloud':
    return PointCloud(
        points = torch.cat([self.points, other.points], dim=0),
        colors = torch.cat([self.colors, other.colors], dim=0),
        batch_size = (self.num_points + other.num_points,))
  

  def translated(self, translation:torch.Tensor) -> 'PointCloud':
    return PointCloud(
        points = self.points + translation,
        colors = self.colors,
        batch_size = self.batch_size)
  
  def scaled(self, scale:float) -> 'PointCloud':

    return PointCloud(
        points = self.points * scale,
        colors = self.colors,
        batch_size = self.batch_size)
  
  @staticmethod
  def load_cloud(filename:str | Path) -> 'PointCloud':
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
        rgb = np.ones((len(cloud), 3), dtype=np.uint8)
        
      return PointCloud.from_numpy(cloud.numpy(("x", "y", "z")), rgb)

    elif filename.suffix == ".ply":
      ply = plyfile.PlyData.read(filename) 
      v = ply['vertex']
      xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
      rgb = np.stack([v['red'], v['green'], v['blue']], axis=1)

      return PointCloud.from_numpy(xyz, rgb)  
    else:
      raise ValueError(f"Unknown file type {filename.suffix}")
    

  def as_ply(self) -> plyfile.PlyData:
    vertex = np.zeros(self.points.shape[0], dtype=[
      ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
      ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])

    for i, name in enumerate(['x', 'y', 'z']):
      vertex[name] = self.points[:, i].cpu().numpy()

    for i, name in enumerate(['red', 'green', 'blue']):
      vertex[name] = (self.colors[:, i].cpu().numpy() * 255).astype(np.uint8)

    return plyfile.PlyData([plyfile.PlyElement.describe(vertex, 'vertex')], text=True)


  def save_ply(self, filename:str | Path):
    filename = Path(filename)
    ply = self.as_ply()
    ply.write(filename)


  def as_pcd(self) -> pypcd4.PointCloud:
    fields = ("x", "y", "z", "red", "green", "blue")
    types = (np.float32, np.float32, np.float32, np.uint8, np.uint8, np.uint8)

    return pypcd4.PointCloud.from_points(self.points.cpu().numpy(), fields, types)

  def save_pcd(self, filename:str | Path) -> None:
    filename = Path(filename)
    pc = self.as_pcd()
    pc.save(filename)