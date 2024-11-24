from functools import partial
import math
from typing import Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
from taichi_splatting.perspective import (CameraParams)

from splat_trainer.config import Varying, VaryingFloat, eval_varying, schedule_groups, schedule_lr
from splat_trainer.scene.mlp.tcnn_mlp import directional_mlp
from splat_trainer.scene.mlp.torch_mlp import AffineMLP, DirectionalMLP



class GLOTable(torch.nn.Module):
  def __init__(self, n:int, glo_features:int):
    super().__init__()
    self.embeddings = nn.Embedding(n, glo_features, sparse=True)

  def interpolated(self, weights:torch.Tensor):
    assert weights.shape[0] == self.embeddings.num_embeddings

    idx = torch.arange(weights.shape[0], device=weights.device)
    return (self.embeddings(idx) * F.softmax(weights, dim=0).unsqueeze(1)).sum(dim=0)


  def forward(self, idx:torch.Tensor | int):
    if isinstance(idx, int):
      return self.embeddings(torch.tensor([idx], device=self.embeddings.weight.device))
    else:
      return self.embeddings(idx)

  def optimizer(self, lr_glo:VaryingFloat):
    lr_glo = eval_varying(lr_glo, 0.)
    param_groups = [
      dict(params=self.embeddings.parameters(), lr=lr_glo, name="glo"),
    ]

    return torch.optim.SparseAdam(param_groups, betas=(0.9, 0.999))

  def schedule(self, optimizer, lr_glo: VaryingFloat, t:float):
    schedule_groups(dict(glo=lr_glo), t, optimizer)


class ColorModel(torch.nn.Module):
  def __init__(self, 
               
               glo_features:int = 16, 
               point_features:int       = 16,

               hidden_features:int     = 64,
               layers:int             = 2,

               color_channels:int     = 3,
               sh_degree:int          = 3):
    
    super().__init__()

    self.feature_size = glo_features + point_features
    
    
    self.color_model = directional_mlp( 
        inputs=self.feature_size, outputs=color_channels,
        layers=layers, 
        hidden=hidden_features,
        sh_degree=sh_degree,
        #norm=partial(nn.LayerNorm, elementwise_affine=False),
    )

  def forward(self, point_features:torch.Tensor, # N, point_features
                positions:torch.Tensor,          # N, 3
                cam_pos:torch.Tensor,            # 3
                glo_feature:torch.Tensor         # 1, glo_features
              ):

    glo_feature = glo_feature.expand(positions.shape[0], -1)
    feature = torch.cat([point_features, glo_feature], dim=1)

    dir = F.normalize(positions.detach() - cam_pos.unsqueeze(0), dim=1)
    return self.color_model(torch.cat([dir, feature], dim=1)).to(torch.float32)
  

  def optimizer(self, lr_nn:VaryingFloat):
    lr_nn = eval_varying(lr_nn, 0.)
    param_groups = [
      dict(params=self.color_model.parameters(), lr=lr_nn, name="nn"),
    ]
    return torch.optim.Adam(param_groups, betas=(0.9, 0.999))
  
  def schedule(self, optimizer, lr_nn: VaryingFloat, t:float):
    schedule_groups(dict(nn=lr_nn), t, optimizer)