import numpy as np
import torch

def strided_indexes(subset:int, total:int):
  if subset > 0:
    stride = max(total // subset, 1)
    return torch.arange(0, total, stride).to(torch.long)
  else:
    return torch.tensor([])


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
  return np.log(x / (1 - x))


sh0 = 0.282094791773878

def rgb_to_sh(rgb):
    return (rgb - 0.5) / sh0

def sh_to_rgb(sh):
    return sh * sh0 + 0.5


