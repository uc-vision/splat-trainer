import numpy as np
import torch

def strided_indexes(subset:int, total:int):
  if subset > 0:
    stride = max(total // subset, 1)
    return torch.arange(0, total, stride).to(torch.long)
  else:
    return torch.tensor([])


def split_stride(images, stride=0):
  assert stride == 0 or stride > 1, f"val_stride {stride}, must be zero, or greater than 1"

  val_cameras = [camera for i, camera in enumerate(images) if i % stride == 0] if stride > 0 else []
  train_cameras = [camera for camera in images if camera not in val_cameras]

  return train_cameras, val_cameras



def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
  return np.log(x / (1 - x))


sh0 = 0.282094791773878

@torch.compile
def rgb_to_sh(rgb):
    return (rgb - 0.5) / sh0

@torch.compile
def sh_to_rgb(sh):
    return sh * sh0 + 0.5

@torch.compile
def log_lerp(t, a, b):
  return torch.exp(torch.lerp(torch.log(a), torch.log(b), t))

@torch.compile
def inv_lerp(t, a, b):
  return 1 / (torch.lerp(1/a, 1/b, t))

@torch.compile
def exp_lerp(t, a, b):
    max_ab = torch.max(a, b)
    return max_ab + torch.log(torch.lerp(torch.exp(a - max_ab), torch.exp(b - max_ab), t))

@torch.compile
def lerp(t, a, b):
  return a + (b - a) * t


class CudaTimer:
  def __init__(self):
    self.start = torch.cuda.Event(enable_timing = True)
    self.end = torch.cuda.Event(enable_timing = True)

  def __enter__(self):
    self.start.record()

  def __exit__(self, *args):
    self.end.record()

  def ellapsed(self):
    return self.start.elapsed_time(self.end)