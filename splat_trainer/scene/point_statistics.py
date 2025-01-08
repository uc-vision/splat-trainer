from functools import cached_property
from typing import Tuple
import torch

from taichi_splatting import Rendering


from tensordict import tensorclass

@tensorclass
class PointStatistics:
  """ Accumulated point heuristics and visibility information 
      similar to Rendering, but accumulated over multiple renderings
  """
  point_heuristic:torch.Tensor  # (N, 2) - accumulated point heuristics
  point_visibility:torch.Tensor # (N,) - accumulated point visibility
  points_in_view:torch.Tensor   # (N,) - number of times each point was in view


  @staticmethod
  def new_zeros(num_points:int, device:torch.device) -> 'PointStatistics':
    return PointStatistics(
      point_heuristic=torch.zeros((num_points, 2), device=device),
      point_visibility=torch.zeros(num_points, device=device),
      points_in_view=torch.zeros(num_points, dtype=torch.int16, device=device),
      batch_size=(num_points,)
    )

  def add_rendering(self, rendering:Rendering):    
    # Only accumulate for points that were in view for this rendering
    points_in_view = rendering.points_in_view
    
    # Add loss term from point_heuristic
    self.point_heuristic[points_in_view] += rendering.point_heuristic
    self.point_visibility[points_in_view] += rendering.point_visibility
    self.points_in_view[points_in_view] += 1

  @property
  def prune_cost(self):
    return self.point_heuristic[:, 0]

  @property
  def split_score(self):
    return self.point_heuristic[:, 1]
  
  @cached_property
  def visible_mask(self) -> torch.Tensor:
    return self.point_visibility > 0

  @cached_property
  def visible(self) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Returns visible point indexes, and their features """
    mask = self.visible_mask
    return self.points_in_view[mask], self.point_visibility[mask]