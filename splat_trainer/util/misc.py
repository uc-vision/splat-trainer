import heapq
import numpy as np
import torch


def split_stride(images, stride=0):
  assert stride == 0 or stride > 1, f"val_stride {stride}, must be zero, or greater than 1"

  val_cameras = [camera for i, camera in enumerate(images) if i % stride == 0] if stride > 0 else []
  train_cameras = [camera for camera in images if camera not in val_cameras]

  return train_cameras, val_cameras


def next_multiple(x, multiple):
  return x + multiple - x % multiple

def format_dict(d:dict, precision:int=4, align:str="<"):
  return " ".join([f"{k}: {v:{align}{precision+3}.{precision}g}" for k,v in d.items()])

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def inverse_sigmoid(x):
  return np.log(x) - np.log(1 - x)


def soft_gt(t:torch.Tensor, threshold:float, margin:float=8.0):
  """ Soft threshold (greater than threshold) using sigmoid.
  Args:
    t: tensor to threshold
    threshold: threshold value (sigmoid is centered at this value)
    margin: scales width of the sigmoid response (larger margin -> sharper threshold)
  """  
  return torch.sigmoid((t - threshold)* margin/threshold)

def soft_lt(t:torch.Tensor, threshold:float, margin:float=8.0):
  """ Soft threshold (less than threshold) using sigmoid.
  Args:
    t: tensor to threshold
    threshold: threshold value (sigmoid is centered at this value)
    margin: scales width of the sigmoid response (larger margin -> sharper threshold)
  """  
  return 1 - soft_gt(t, threshold, margin)



sh0 = 0.282094791773878

@torch.compile
def rgb_to_sh(rgb):
    return (rgb - 0.5) / sh0

@torch.compile
def sh_to_rgb(sh):
    return sh * sh0 + 0.5


@torch.compile
def inv_lerp(t:float | torch.Tensor, a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
  return 1 / (lerp(1/a, 1/b, t))

@torch.compile
def exp_lerp(t:float | torch.Tensor, a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(lerp(t, torch.exp(a - max_ab), torch.exp(b - max_ab)))

@torch.compile
def pow_lerp(t:float | torch.Tensor, a:torch.Tensor, b:torch.Tensor, k:float=2) -> torch.Tensor:
    return lerp(t, a**k, b**k)**(1/k)

def lerp(t:float | torch.Tensor, a:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
  return torch.lerp(a, b, t)



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
  
  def wrap(self, f, *args, **kwargs):
    with self:
      return f(*args, **kwargs)
  
  


class Heap:
  def __init__(self, max_size:int):
    self.max_size = max_size
    self.heap = []

  def push(self, value, item):
    heapq.heappush(self.heap, (value, item))
    if len(self.heap) > self.max_size:
      heapq.heappop(self.heap)

  def pop(self):
    return heapq.heappop(self.heap)

  def __len__(self):
    return len(self.heap)

  def __iter__(self):
    return iter(self.heap)