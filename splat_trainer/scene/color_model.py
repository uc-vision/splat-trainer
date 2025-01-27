from dataclasses import dataclass
from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn

from splat_trainer.config import VaryingFloat, eval_varying, schedule_groups
from splat_trainer.scene.mlp.torch_mlp import AffineMLP, MLP
from tensordict import tensorclass


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
      return self.embeddings(torch.tensor(idx, device=self.embeddings.weight.device))
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




@dataclass
class ColorModelConfig:
  hidden_features:int      = 32  
  sh_degree:int = 5

  lr_diffuse:VaryingFloat = 1e-3
  lr_specular:VaryingFloat = 1e-3

  color_channels:int = 3
  hdr:bool = False

  def create_model(self, glo_features:int, point_features:int):
    return ColorModel(
      config=self,
      glo_features=glo_features,
      point_features=point_features
    )


def luminance_activation(rgbl:torch.Tensor, intensity_bias:float = 0.):
    colors, intensity = rgbl[:, 1:], rgbl[:, 0:1]
    lum = (intensity + intensity_bias).exp()
    
    return colors.sigmoid() * lum
    # return torch.cat([F.sigmoid(colors) * lum, lum], dim=1)

@tensorclass
class Colors:
  diffuse:torch.Tensor    # N, 4  rgb, luminance
  specular:torch.Tensor   # N, 4  rgb, luminance

  def total(self, specular_weight:float = 1.0):
    return (self.diffuse + self.specular * specular_weight)



class ColorModel(torch.nn.Module):
  def __init__(self, 
               config:ColorModelConfig,
               
               glo_features:int = 16, 
               point_features:int = 16,
            ):
    
    super().__init__()

    self.config = config
    self.feature_size = glo_features + point_features

    self.norm = nn.LayerNorm(self.feature_size, elementwise_affine=False)

    n_out = config.color_channels + 1  # rgb + intensity

    self.directional_model = AffineMLP(
        inputs=self.feature_size,
        outputs=n_out,

        hidden_layers=1, 
        hidden=config.hidden_features,
        proj_hidden_layers=0,
        sh_degree=config.sh_degree
    )

    self.base_model = MLP(
      inputs=self.feature_size, 
      outputs=n_out, 

      hidden=config.hidden_features, 
      hidden_layers=1
    )


  def forward(self, point_features:torch.Tensor, # N, point_features
                positions:torch.Tensor,          # N, 3
                cam_pos:torch.Tensor,            # 3
                glo_feature:torch.Tensor         # 1, glo_features
              ) -> Colors:

    assert glo_feature.dim() == 2 and point_features.dim() == 2, f"got {glo_feature.dim()} and {point_features.dim()}"
    

    glo_feature = glo_feature.expand(positions.shape[0], glo_feature.shape[1])
    feature = torch.cat([point_features, glo_feature], dim=1)
    feature = self.norm(feature)

    diffuse = luminance_activation(
      self.base_model(feature))

    dir = F.normalize(positions.detach() - cam_pos.unsqueeze(0), dim=1)

    # Directional specular/reflected colors
    specular = luminance_activation(
      self.directional_model(dir, feature), intensity_bias=-2.0)
    
    return Colors(
      diffuse=diffuse, 
      specular=specular, 
      batch_size=(positions.shape[0],))

      
    

  def post_activation(self, image:torch.Tensor, eps:float = 1e-6) -> torch.Tensor:      
    if not self.config.hdr:
      rgb = image[..., :3]
      return rgb.clamp(0, 1)

    else:
      return image
  

  def optimizer(self, t:float=0.) -> torch.optim.Optimizer:
    param_groups = [
      dict(params=self.directional_model.parameters(), lr=0.0, name="spec"),
      dict(params=[*self.base_model.parameters(), *self.norm.parameters()], lr=0.0, name="base"),
    ]
    opt = torch.optim.Adam(param_groups, betas=(0.9, 0.999))
    self.schedule(opt, t)

    return opt
  
  def schedule(self, optimizer:torch.optim.Optimizer, t:float):
        return schedule_groups(dict(spec=self.config.lr_specular, base=self.config.lr_diffuse), t, optimizer)