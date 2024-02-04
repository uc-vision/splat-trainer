from typing import Tuple
from torch import nn
import torch

from taichi_splatting.torch_ops.transforms import split_rt



class CameraRigTable(nn.Module):
  def __init__(self, rig_t_world:torch.Tensor, camera_t_rig:torch.Tensor):
    super().__init__()

    self.camera_t_rig = PoseTable(camera_t_rig)
    self.rig_t_world = PoseTable(rig_t_world)

  @property
  def num_cameras(self):
    return self.camera_t_rig.t.shape[0]
  
  @property 
  def num_frames(self):
    return self.rig_t_world.t.shape[0]
  
  @property
  def rig_t_camera(self):
    return torch.linalg.inv(self.camera_t_rig.matrix)
  
  @property
  def world_t_rig(self):
    return torch.linalg.inv(self.rig_t_world.matrix)

  def normalize(self):
    self.camera_t_rig.normalize()
    self.rig_t_world.normalize()

  
  def forward(self, camera_index:torch.Tensor, rig_index:torch.Tensor):
    assert camera_index.shape == rig_index.shape, \
      f"{camera_index.shape} != {rig_index.shape}"

    camera_t_rig = self.camera_t_rig(camera_index)
    rig_t_world = self.rig_t_world(rig_index)

    return camera_t_rig @ rig_t_world


class PoseTable(nn.Module):
  def __init__(self, m:torch.Tensor):
    super().__init__()

    R, t = split_rt(m)

    rot_axis, angle = rot_to_quat(R)

    self.t = nn.Parameter(t.to(torch.float32))
    self.q = nn.Parameter(q.to(torch.float32))


  def forward(self, indices):
    q, t = self.q[indices].contiguous(), self.t[indices].contiguous()
  
  def normalize(self):
    self.q.data /= torch.norm(self.q.data, dim=-1, keepdim=True)
    return self





  

if __name__ == '__main__':
  pass