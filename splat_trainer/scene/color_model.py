from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn

from splat_trainer.config import VaryingFloat, eval_varying, schedule_groups
from splat_trainer.scene.mlp.torch_mlp import AffineMLP



class GLOTable(torch.nn.Module):
  def __init__(self, n:int, glo_features:int):
    super().__init__()
    self.embeddings = nn.Embedding(n, glo_features, sparse=True)
    torch.nn.init.normal_(self.embeddings.weight, mean=0.0, std=1.0)


  @property
  def weight(self):
    return self.embeddings.weight

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

    return torch.optim.SparseAdam(param_groups, betas=(0.8, 0.95))

  def schedule(self, optimizer, lr_glo: VaryingFloat, t:float):
    return schedule_groups(dict(glo=lr_glo), t, optimizer)


class ColorModel(torch.nn.Module):
  def __init__(self, 
               
               glo_features:int = 16, 
               point_features:int       = 16,

               hidden_features:int     = 64,
               hidden_layers:int       = 2,

               color_channels:int     = 3,
               sh_degree:int          = 3):
    
    super().__init__()

    self.feature_size = glo_features + point_features
    
    self.color_model = AffineMLP( 
        inputs=self.feature_size, outputs=color_channels,
        hidden_layers=hidden_layers, 
        hidden=hidden_features,
        sh_degree=sh_degree,
        # output_activation=nn.Sigmoid,
        norm=partial(nn.LayerNorm, elementwise_affine=False),
    )

  def forward(self, point_features:torch.Tensor, # N, point_features
                positions:torch.Tensor,          # N, 3
                cam_pos:torch.Tensor,            # 3
                glo_feature:torch.Tensor         # 1, glo_features
              ):
    

    glo_feature = glo_feature.expand(positions.shape[0], glo_feature.shape[1])
    feature = torch.cat([point_features, glo_feature], dim=1)

    dir = F.normalize(positions.detach() - cam_pos.unsqueeze(0), dim=1)
    return self.color_model(dir, feature).to(torch.float32)
  

  def optimizer(self, lr_nn:VaryingFloat) -> torch.optim.Optimizer:
    lr_nn = eval_varying(lr_nn, 0.)
    param_groups = [
      dict(params=self.color_model.parameters(), lr=lr_nn, name="nn"),
    ]
    return torch.optim.Adam(param_groups, betas=(0.9, 0.99))
  
  def schedule(self, optimizer:torch.optim.Optimizer, lr_nn: VaryingFloat, t:float):
    return schedule_groups(dict(nn=lr_nn), t, optimizer)