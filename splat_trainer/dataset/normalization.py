from dataclasses import dataclass
from beartype import beartype
from splat_trainer.camera_table.camera_table import Cameras
from splat_trainer.util.pointcloud import PointCloud
from splat_trainer.util.transforms import join_rt, split_rt
from taichi_splatting import Gaussians3D
import torch


def median_knn(positions: torch.Tensor, k: int = 10) -> float:
  distances = torch.cdist(positions, positions)
  distances = torch.sort(distances, dim=1).values
  median_distance = torch.median(distances[:, k - 1])
  return median_distance


@dataclass
class Normalization:
  config: 'NormalizationConfig'
  translation: torch.Tensor
  scaling: float

  @beartype
  def transform_cloud(self, cloud: PointCloud) -> PointCloud:
    return cloud.translated(self.translation.to(cloud.device)).scaled(self.scaling)

  @beartype
  def transform_gaussians(self, gaussians: Gaussians3D) -> Gaussians3D:
    return gaussians.translated(
      self.translation.to(gaussians.position.device)
      ).scaled(self.scaling)

  @beartype
  def transform_cameras(self, cameras: Cameras) -> Cameras:
    return cameras.translated(
      self.translation.to(cameras.device)
      ).scaled(self.scaling)

  @beartype
  def transform_points(self, 
                       points: torch.Tensor # (..., 3)
                       ) -> torch.Tensor:   # (..., 3)
    
    t = self.translation.to(points.device)
    while len(t.shape) < len(points.shape):
        t = t.unsqueeze(0)

    return self.scaling * (points + t)
  
  @beartype
  def transform_rigid(self, rigid:torch.Tensor) -> torch.Tensor:
    r, t = split_rt(rigid)
    t = self.transform_points(t)
    return join_rt(r, t)
    

  @property
  def inverse(self) -> 'Normalization':
    return Normalization(
      config=self.config,
      translation=self.scaling * -self.translation,
      scaling=1.0 / self.scaling,
    )

  def __repr__(self):
    translation_str = [f"{val:.3f}" for val in self.translation.tolist()]
    return f"Normalization(translation={translation_str}, scaling={self.scaling:.3f})"


@dataclass
class NormalizationConfig:
  centering: bool = True
  scaling_method: str = "none"  # 'none', 'median_knn'
  normalize_knn: int = 20

  @beartype
  def get_transform(self, positions: torch.Tensor) -> Normalization:
    if self.scaling_method == "none":
      scale = 1.0
    elif self.scaling_method == "median_knn":
      scale = median_knn(positions, k=self.normalize_knn)
    else:
      raise ValueError(f"Unknown scaling method: {self.scaling_method}, options are: none, median_knn")

    mean = (
      torch.mean(positions, dim=0).to(torch.float32)
      if self.centering
      else torch.zeros(3, dtype=torch.float32)
    )

    return Normalization(
      config=self,
      translation=-mean,
      scaling=1.0 / scale,
    )
  
  def __repr__(self) -> str:
    return f"NormalizationConfig(centering={self.centering}, scaling_method={self.scaling_method}, normalize_knn={self.normalize_knn})"