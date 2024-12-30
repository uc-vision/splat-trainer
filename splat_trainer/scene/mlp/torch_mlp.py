
from splat_trainer.scene.mlp import rsh
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F

def layer(in_features: int, out_features: int, norm = nn.Identity, activation = torch.nn.ReLU, out_scale:Optional[float] = None):
    m = torch.nn.Linear(in_features, out_features)
    
    if out_scale is not None:
      nn.init.normal_(m.weight, std=out_scale)
      nn.init.zeros_(m.bias)

    return torch.nn.Sequential(m, activation(), norm(out_features))



sh_coeffs = {
  2: rsh.rsh_cart_2,
  3: rsh.rsh_cart_3,
  4: rsh.rsh_cart_4,
  5: rsh.rsh_cart_5,
  6: rsh.rsh_cart_6,
  7: rsh.rsh_cart_7,
  8: rsh.rsh_cart_8,
}


class ProjectSH(torch.nn.Module):
  def __init__(self, out_features: int, sh_degree: int, hidden: int, hidden_layers: int = 0, norm = nn.Identity):
    super().__init__()

    self.get_coeffs = sh_coeffs[sh_degree]
    self.mlp = BasicMLP((sh_degree + 1)**2, out_features, hidden, hidden_layers, norm=norm)

  def forward(self, dir):
    coeffs = self.get_coeffs(dir).to(dtype=dir.dtype)
    return self.mlp(coeffs)


class DirectionalMLP(torch.nn.Module):
  def __init__(self, inputs: int, outputs: int, hidden: int, hidden_layers: int, 
               norm = nn.Identity, activation = torch.nn.ReLU, output_activation = nn.Identity, sh_degree: int = 3):
    super().__init__()

    self.get_coeffs = sh_coeffs[sh_degree]
    sh_size = (sh_degree + 1) ** 2
    self.mlp = BasicMLP(inputs + sh_size, outputs, hidden, hidden_layers, norm, activation, output_activation)

  def forward(self, dir, x):
    coeffs = self.get_coeffs(dir).to(dtype=x.dtype)
    x = torch.cat([coeffs, x], dim=1)
    
    return self.mlp(x)


class AffineMLP(torch.nn.Module):
  def __init__(self, inputs: int, outputs: int, hidden: int, hidden_layers: int, 
               norm = nn.Identity, activation = torch.nn.ReLU, output_activation = nn.Identity, sh_degree: int = 3):
    super().__init__()

    self.get_coeffs = sh_coeffs[sh_degree]

    self.mlp = BasicMLP(inputs, outputs, hidden, hidden_layers, norm, activation, output_activation)
    self.encode_dir = ProjectSH(inputs * 2, sh_degree, hidden=hidden, norm=norm)

    self.norm_feature = norm(inputs)

  def forward(self, dir, x):
    dir_enc = self.encode_dir(dir)
    a, b = torch.split(dir_enc, dir_enc.shape[1] // 2, dim=1)

    return self.mlp(self.norm_feature(x * a + b))


class BasicMLP(torch.nn.Module):
  def __init__(self, inputs: int, outputs: int, hidden: int, hidden_layers: int, 
               norm = nn.Identity, activation = torch.nn.ReLU, output_activation = nn.Identity, out_scale:Optional[float] = None):
    super().__init__()

    self.layers = nn.ModuleList([
      layer(inputs, hidden, norm=norm, activation=activation),
      *[layer(hidden, hidden, norm=norm, activation=activation) for _ in range(hidden_layers)],
      layer(hidden, outputs, activation=output_activation, out_scale=out_scale)
    ])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
