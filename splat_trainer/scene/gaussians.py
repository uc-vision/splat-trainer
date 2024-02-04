from tensordict import tensorclass
import torch



@tensorclass
class PackedPoints:
  gaussians3d: torch.Tensor  # (N, 11)
  sh_features: torch.Tensor  # (N, (D+1)**2)