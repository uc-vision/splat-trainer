from dataclasses import dataclass
from tensordict import tensorclass
import torch

from splat_trainer.util.pointcloud import PointCloud



@tensorclass
class PackedPoints:
  gaussians3d: torch.Tensor  # (N, 11)
  sh_features: torch.Tensor  # (N, (D+1)**2)




class Scene:
  def __init__(self):
    pass


  @staticmethod
  def initialize(self, points: PointCloud, centre: torch.Tensor):
    pass