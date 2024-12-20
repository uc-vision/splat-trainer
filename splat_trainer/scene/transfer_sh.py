

from functools import partial
from typing import Callable, Tuple
from beartype import beartype
from taichi_splatting import evaluate_sh_at
from taichi_splatting.perspective import CameraParams
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from splat_trainer.camera_table.camera_table import CameraTable, Label, Camera
from splat_trainer.config import LogDecay, Varying, VaryingFloat, eval_varying
from splat_trainer.scene.mlp.torch_mlp import BasicMLP
from splat_trainer.util.misc import sh_to_rgb



class TransferSH(torch.nn.Module):
  def __init__(self, num_points:int, sh_degree:int=3, glo_features:int=16, hidden:int=32, layers:int=1):
    super().__init__()

    n = (sh_degree + 1) ** 2
    self.base_sh = nn.Parameter(torch.randn(num_points, 3, 1))
    self.higher_sh = nn.Parameter(torch.zeros(num_points, 3, n - 1))

    self.mlp = BasicMLP(glo_features, 4 * 3, hidden, layers, 
                        norm=partial(nn.LayerNorm, elementwise_affine=True), 
                        out_scale=1e-12)
  @property
  def sh_features(self):
    return torch.cat([self.base_sh, self.higher_sh], dim=2)
  
  @property
  def rgb_colors(self):
    return sh_to_rgb(self.base_sh.squeeze(2))

  def optimizer(self, lr_sh: float = 0.05, lr_nn: float = 0.025):
    param_groups = [
      dict(params=self.base_sh, lr=lr_sh, name="base_sh"),
      dict(params=self.higher_sh, lr=lr_sh / 10.0, name="higher_sh", weight_decay=1e-4),
      dict(params=self.mlp.parameters(), lr=lr_nn, name="nn"),
    ]

    return torch.optim.Adam(param_groups, betas=(0.9, 0.999))


  def forward(self, positions:torch.Tensor,          # N, 3
                    indexes:torch.Tensor,            # N
                    cam_pos:torch.Tensor,            # 3
                    glo_feature:torch.Tensor         # 1, glo_features
              ):

    colors = evaluate_sh_at(self.sh_features, positions, indexes, cam_pos) # N, 3
    colors = torch.cat([colors, torch.ones_like(colors[:, :1])], dim=1) # N, 4

    affine_res = self.mlp(glo_feature).reshape(1, 3, 4) 

    affine = affine_res + torch.eye(3, 4, device=glo_feature.device).unsqueeze(0) # 1, 3, 4
    colors = torch.matmul(colors.unsqueeze(1), affine.transpose(-2,-1)).squeeze(1) # N, 3

    return colors.clamp(0, 1), affine_res.abs().mean() 
  

@beartype
def camera_to_camera_params(camera:Camera, image_scale:float = 1.0) -> CameraParams:
  return CameraParams(
    projection=camera.intrinsics * image_scale,
    T_camera_world=camera.camera_t_world,

    near_plane=camera.depth_range[0],
    far_plane=camera.depth_range[1],  
    image_size=(int(camera.image_size[0] * image_scale), int(camera.image_size[1] * image_scale)),
  )

@beartype
def transfer_sh(eval_colors:Callable[[torch.Tensor, CameraParams, int], torch.Tensor], 
                query_visibility:Callable[[CameraParams], Tuple[torch.Tensor, torch.Tensor]], 

                camera_table:CameraTable, 
                positions:torch.Tensor,

                glo_features:torch.Tensor,
                epochs:int = 2,
                sh_degree:int = 3):
  
  train_idx = camera_table.has_label(Label.Training)
  sh_model = TransferSH(num_points=positions.shape[0], sh_degree=sh_degree, glo_features=glo_features.shape[1], hidden=32, layers=1)
  sh_model.to(positions.device)

  optimizer = sh_model.optimizer()

  def iter():
    for _ in range(epochs):
      idx = train_idx[torch.randperm(train_idx.shape[0], device=train_idx.device)]
      for i in idx.cpu().tolist():
        camera_params = camera_to_camera_params(camera_table[i], image_scale=0.5)
        yield i, camera_params

  pbar = tqdm(iter(), total=train_idx.shape[0] * epochs, desc="Transferring SH colors")

  loss_avg = 0.0
  mse_avg = 0.0
  iteration = 0
  for i, camera_params in pbar:

    point_indexes, visibility = query_visibility(camera_params)
    colors = eval_colors(point_indexes, camera_params, i)

    with torch.enable_grad():
      pred_colors, reg = sh_model.forward(positions, point_indexes, camera_params.camera_position, glo_features[i])
      mse = F.l1_loss(pred_colors, colors, reduction='none')
      rgb = F.l1_loss(sh_model.rgb_colors[point_indexes], colors, reduction='none')

      vis = visibility.unsqueeze(1)
      vis_mse = (mse * vis).sum() / vis.sum()
      loss =  rgb.mean() * 0.2 + vis_mse + reg * 0.2
      loss.backward()

      loss_avg += loss.item()
      mse_avg += mse.mean().item()

    if iteration % 50 == 0:
      pbar.set_postfix_str(f"loss: {loss_avg / 50:.4f} mse: {mse_avg / 50:.4f}")
      loss_avg = 0.0
      mse_avg = 0.0
    iteration += 1


    optimizer.step()
    optimizer.zero_grad()

  return sh_model.sh_features.detach()
