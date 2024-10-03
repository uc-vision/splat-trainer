import torch
from dataclasses import replace

from splat_trainer.util.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

class BilateralGridInstance():
  def __init__(self, device, steps, num_images, bilateral_grid_shape):
    self.device = device
    self.bil_grids = BilateralGrid(
        num_images,
        grid_X=bilateral_grid_shape[0],
        grid_Y=bilateral_grid_shape[1],
        grid_W=bilateral_grid_shape[2],
    ).to(device)
    self.bil_grid_optimizer = torch.optim.Adam(
        self.bil_grids.parameters(),
        lr=2e-3,
        eps=1e-15,
    )
    self.bil_grid_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                self.bil_grid_optimizer,
                start_factor=0.01,
                total_iters=1000,
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.bil_grid_optimizer, gamma=0.01 ** (1.0 / steps)
            ),
        ]
    )

  def correct_rendered_image(self, rendering, image_idx):
    height, width = rendering.image_size[1], rendering.image_size[0]
    grid_y, grid_x = torch.meshgrid(
        (torch.arange(height, device=self.device) + 0.5) / height,
        (torch.arange(width, device=self.device) + 0.5) / width,
        indexing="ij",
    )
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    cc_image = slice(self.bil_grids, grid_xy, rendering.image.unsqueeze(0), torch.tensor([image_idx]))["rgb"].squeeze(0)
    rendering = replace(rendering, image=cc_image)
    return rendering

  def tvloss(self):
    tvloss = 10 * total_variation_loss(self.bil_grids.grids)
    return tvloss.item()

  def step(self):
    self.bil_grid_optimizer.step()
    self.bil_grid_optimizer.zero_grad(set_to_none=True)
    self.bil_grid_scheduler.step()

  def correct_for_evaluation(self, rendering, image):
    cc_image = color_correct(rendering.image.unsqueeze(0), image.unsqueeze(0)).squeeze(0)
    rendering = replace(rendering, image=cc_image)
    return rendering

