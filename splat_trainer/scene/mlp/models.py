
from splat_trainer.scene.mlp import rsh
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F

def layer(in_features: int, out_features: int, norm = nn.Identity, activation = torch.nn.ReLU):
    return torch.nn.Sequential(
    torch.nn.Linear(in_features, out_features),
    activation(),
    norm(out_features)
  )



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
  def __init__(self, out_features: int, sh_degree: int):
    super().__init__()

    self.get_coeffs = sh_coeffs[sh_degree]
    self.linear = nn.Linear((sh_degree + 1) ** 2, out_features)

  def forward(self, x):
    coeffs = self.get_coeffs(dir).to(dtype=x.dtype)
    return self.linear(coeffs)


class DirectionalMLP(torch.nn.Module):
  def __init__(self, inputs: int, outputs: int, hidden: int, layers: int, 
               norm = nn.Identity, activation = torch.nn.ReLU, output_activation = nn.Identity, sh_degree: int = 2):
    super().__init__()

    self.get_coeffs = sh_coeffs[sh_degree]
    sh_size = (sh_degree + 1) ** 2

    self.mlp = BasicMLP(inputs + sh_size, outputs, hidden, layers, norm, activation, output_activation)

  def forward(self, dir, x):
    coeffs = self.get_coeffs(dir).to(dtype=x.dtype)
    x = torch.cat([coeffs, x], dim=1)

    return self.mlp(x)



class BasicMLP(torch.nn.Module):
  def __init__(self, inputs: int, outputs: int, hidden: int, layers: int, 
               norm = nn.Identity, activation = torch.nn.ReLU, output_activation = nn.Identity):
    super().__init__()

    self.layers = nn.ModuleList([
      layer(inputs, hidden, norm=norm, activation=activation),
      *[layer(hidden, hidden, norm=norm, activation=activation) for _ in range(layers - 1)],
      layer(hidden, outputs, activation=output_activation)
    ])

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
