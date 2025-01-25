from dataclasses import dataclass
from functools import partial
import torch
import torch.nn.functional as F
import torch.nn as nn

from splat_trainer.config import VaryingFloat, eval_varying, schedule_groups
from splat_trainer.scene.mlp.torch_mlp import AffineMLP, glu_mlp



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



def luminance_activation(rgbl:torch.Tensor, intensity_bias:float = 0.):
    colors, intensity = rgbl[:, 1:], rgbl[:, 0:1]
    return colors.sigmoid() * (intensity + intensity_bias).exp()


class Luminance(torch.nn.Module):
  def __init__(self, intensity_bias:float = 0.):
    super().__init__()
    self.intensity_bias = intensity_bias

  def forward(self, outputs:torch.Tensor):
    return luminance_activation(outputs, self.intensity_bias)


@dataclass
class ColorModelConfig:
  hidden_features:int      = 32
  hidden_layers:int        = 1
  
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

    self.directional_model = AffineMLP(
        inputs=self.feature_size,
        outputs=(config.color_channels + 1),

        hidden_layers=config.hidden_layers, 
        hidden=config.hidden_features,
        proj_hidden_layers=0,
        sh_degree=config.sh_degree

    )

    self.base_model = glu_mlp(
      inputs=self.feature_size, 
      outputs=(config.color_channels + 1), 

      hidden=config.hidden_features, 
      hidden_layers=config.hidden_layers
    )


  def forward(self, point_features:torch.Tensor, # N, point_features
                positions:torch.Tensor,          # N, 3
                cam_pos:torch.Tensor,            # 3
                glo_feature:torch.Tensor,         # 1, glo_features
                enable_specular:bool = True
              ):

    assert glo_feature.dim() == 2 and point_features.dim() == 2, f"got {glo_feature.dim()} and {point_features.dim()}"
    

    glo_feature = glo_feature.expand(positions.shape[0], glo_feature.shape[1])
    feature = torch.cat([point_features, glo_feature], dim=1)
    feature = self.norm(feature)

    base_feature = self.base_model(feature)
    color = luminance_activation(base_feature)

    if enable_specular:
      dir = F.normalize(positions.detach() - cam_pos.unsqueeze(0), dim=1)

      # Directional specular/reflected colors
      specular = self.directional_model(dir, feature)
      spec_color = luminance_activation(specular, intensity_bias=-2.0)

      color += spec_color
    return color

      
    

  def post_activation(self, image:torch.Tensor) -> torch.Tensor:      
    if not self.config.hdr:
      return image.clamp(0, 1) 
    else:
      return image
  

  def optimizer(self, t:float=0.) -> torch.optim.Optimizer:
    param_groups = [
      dict(params=self.directional_model.parameters(), lr=0.0, name="spec"),
      dict(params=[*self.base_model.parameters(), *self.norm.parameters()], lr=0.0, name="base"),
    ]
    opt = torch.optim.Adam(param_groups, betas=(0.9, 0.99))
    self.schedule(opt, t)

    return opt
  
  def schedule(self, optimizer:torch.optim.Optimizer, t:float):
        return schedule_groups(dict(spec=self.config.lr_specular, base=self.config.lr_diffuse), t, optimizer)