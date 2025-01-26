from functools import partial
from beartype import beartype
from splat_trainer.scene.mlp import rsh
import torch
from torch import Tensor, nn
from typing import Optional
import torch.nn.functional as F





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
  def __init__(self, out_features: int, sh_degree: int, hidden: int, hidden_layers: int = 0):
    super().__init__()

    self.get_coeffs = sh_coeffs[sh_degree]
    self.mlp = MLP((sh_degree + 1)**2, out_features, hidden, hidden_layers)

  def forward(self, dir):
    coeffs = self.get_coeffs(dir).to(dtype=dir.dtype)
    return self.mlp(coeffs)


class DirectionalMLP(torch.nn.Module):
  def __init__(self, inputs: int, outputs: int, hidden: int, hidden_layers: int, sh_degree: int = 3):
    super().__init__()

    self.get_coeffs = sh_coeffs[sh_degree]
    sh_size = (sh_degree + 1) ** 2
    self.mlp = MLP(inputs + sh_size, outputs, hidden, hidden_layers)

  def forward(self, dir, x):
    coeffs = self.get_coeffs(dir).to(dtype=x.dtype)
    x = torch.cat([coeffs, x], dim=1)
    
    return self.mlp(x)



class AffineMLP(torch.nn.Module):
  def __init__(self, inputs: int, outputs: int, hidden: int, hidden_layers: int, proj_hidden_layers: int = 0, sh_degree: int = 3):
    super().__init__()

    self.get_coeffs = sh_coeffs[sh_degree]

    self.mlp = MLP(inputs, outputs, hidden, hidden_layers)
    self.encode_dir = ProjectSH(out_features=inputs * 2, sh_degree=sh_degree, hidden=hidden, hidden_layers=proj_hidden_layers)


  def forward(self, dir, x):
    dir_enc = self.encode_dir(dir)
    a, b = torch.split(dir_enc, dir_enc.shape[1] // 2, dim=1)

    return self.mlp(x * a + b)
  

def gated(x: Tensor, f:nn.Module) -> Tensor:
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * f(b)


class Gated(nn.Module):
    """
    References:
        Shazeer et al., "GLU Variants Improve Transformer," 2020.
        https://arxiv.org/abs/2002.05202
    """
    def __init__(self, activation:nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x: Tensor) -> Tensor:
        return gated(x, self.activation)


class GLULayer(nn.Module):
    def __init__(self, in_features: int, out_features:int):
        super().__init__()
        self.m = nn.Linear(in_features, out_features * 2)
        self.act = nn.GLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.m(x))


def layer(activation: nn.Module):
     def _layer(in_features: int, out_features:int):
          return Layer(in_features, out_features, activation())
     return _layer

class Layer(nn.Module):
    @beartype
    def __init__(self, in_features: int, out_features:int, activation: nn.Module):
        super().__init__()
        self.m = nn.Linear(in_features, out_features)
        self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.m(x))


class MLP(torch.nn.Module):
  @beartype
  def __init__(self, inputs: int, outputs: int, hidden: int, hidden_layers: int, 
               layer_type: type[nn.Module]=GLULayer):
    
    super().__init__()
    feature_sizes = [inputs] + [hidden] * hidden_layers

    self.layers = nn.ModuleList(
      [layer_type(feature_sizes[i], feature_sizes[i+1]) 
          for i in range(len(feature_sizes) - 1)] 
    )

    self.layers.append(
      nn.Linear(feature_sizes[-1], outputs))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)

    return x



class EnvMap(torch.nn.Module):
  def __init__(self, features:int = 16, shape:tuple[int, int] = (256, 256)):
    super().__init__()

    texture = torch.zeros((features, shape[0], shape[1]), dtype=torch.float32)
    torch.nn.init.normal_(texture, mean=0.0, std=1.0)
    self.texture = nn.Parameter(texture)


  def forward(self, dir:torch.Tensor):
    """ Sample the environment map at the given direction.
    
    Args:
        dir (torch.Tensor): Direction vectors to sample, shape (N, 3)
        
    Returns:
        torch.Tensor: Sampled environment map values, shape (N, features)
    """

    # Convert direction vectors to spherical coordinates
    theta = torch.atan2(dir[:, 0], dir[:, 2])  # azimuth angle
    phi = torch.asin(torch.clamp(dir[:, 1], -1.0, 1.0))  # elevation angle

    # Convert directly to [-1,1] grid coordinates
    u = theta / torch.pi  # theta/π maps to [-1,1]
    v = phi / (torch.pi/2)  # phi/(π/2) maps to [-1,1]
    
    # Adjust grid coordinates for padded texture
    grid = torch.stack([u, v], dim=-1)
    grid = grid.view(1, 1, -1, 2)  # Add batch and height dims

    # Sample the padded texture
    samples = F.grid_sample(self.texture.unsqueeze(0), grid, align_corners=False)
    return samples[0,:,0].t() # Return [N, features]
