
from dataclasses import dataclass, replace
from typing import Dict, Tuple

import torch

from splat_trainer.config import LogLinear, Varying, schedule_lr
from taichi_splatting import Rendering
from .corrector import CorrectorConfig, Corrector

from splat_trainer.util.lib_bilagrid import (
    BilateralGrid,
    color_affine_transform,
    fit_affine_colors,
    total_variation_loss,
)


@dataclass
class BilateralCorrectorConfig(CorrectorConfig):

  bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
  tv_weight: float = 10.0
  lr: Varying[float] = LogLinear(2e-3, 2e-5)

  
  def make_corrector(self, num_images:int, device:torch.device) -> Corrector:
    return BilateralCorrector(self, num_images, device)

# @torch.compile
def correct_grid(bil_grids:BilateralGrid, image:torch.Tensor, image_idx:int) -> torch.Tensor:
    grid_y, grid_x = torch.meshgrid(
      (torch.arange(image.shape[0], device=image.device) + 0.5) / image.shape[0],
      (torch.arange(image.shape[1], device=image.device) + 0.5) / image.shape[1],
      indexing="ij",
    )
    
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

    idx = torch.full((1,), image_idx, device=image.device)
    affine_mats = bil_grids(grid_xy, image.unsqueeze(0), idx)
    return color_affine_transform(affine_mats, image.unsqueeze(0)).squeeze(0)
    

class BilateralCorrector(Corrector):
  def __init__(self, config:BilateralCorrectorConfig, num_images:int, device:str):
      
      self.config = config
      shape = config.bilateral_grid_shape

      self.bil_grids = BilateralGrid(num_images, grid_X=shape[0], grid_Y=shape[1], grid_W=shape[2])
      self.bil_grids.to(device)

      self.bil_grid_optimizer = torch.optim.Adam(self.bil_grids.parameters(), lr=config.lr(0.0))

  def correct(self, rendering:Rendering, image_idx:int) -> torch.Tensor:
    return correct_grid(self.bil_grids, rendering.image, image_idx)
  

  def fit_image(self, rendering:Rendering, source_image:torch.Tensor) -> torch.Tensor:
    """ Fit an affine color transform between the two images and return corrected image """
    return fit_affine_colors(rendering.image.unsqueeze(0), source_image.unsqueeze(0)).squeeze(0)

  def loss(self) -> Tuple[torch.Tensor, Dict[str, float]]:
    tvloss = self.config.tv_weight * total_variation_loss(self.bil_grids.grids)
    return tvloss, {'tv_loss': tvloss.item()}

  def step(self, t:float):
    # update learning rate 
    schedule_lr(self.config.lr, t, self.bil_grid_optimizer)

    self.bil_grid_optimizer.step()
    self.bil_grid_optimizer.zero_grad()


     


