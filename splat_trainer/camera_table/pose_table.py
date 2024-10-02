
from torch import nn
import torch
import torch.nn.functional as F


from splat_trainer.util.transforms import join_rt, split_rt
import roma



class RigPoseTable(nn.Module):
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

  
  def forward(self, frame_camera_indices:torch.Tensor):
    assert frame_camera_indices.dim() == 2 and frame_camera_indices.shape[1] == 2, \
      f"Expected (rig_index, camera_index) N, 2 tensor, got: {frame_camera_indices.shape}"

    rig_index, camera_index = frame_camera_indices.unbind(-1)

    camera_t_rig = self.camera_t_rig(camera_index)
    rig_t_world = self.rig_t_world(rig_index)

    return camera_t_rig @ rig_t_world





class PoseTable(nn.Module):
  def __init__(self, m:torch.Tensor):
    super().__init__()

    assert m.shape[-2:] == (4, 4), f"Expected (..., 4, 4) tensor, got: {m.shape}"


    R, t = split_rt(m)
    q = roma.rotmat_to_unitquat(R)

    self.t = nn.Parameter(t.to(torch.float32), requires_grad=False)
    self.q = nn.Parameter(q.to(torch.float32), requires_grad=False)


  def forward(self, indices):
    assert indices.dim() == 1, f"Expected 1D tensor, got: {indices.shape}"
    assert (indices < self.q.shape[0]).all(), f"Index out of bounds: {indices} >= {self.q.shape[0]}"

    q, t = F.normalize(self.q[indices], dim=-1), self.t[indices]
    pose = join_rt(roma.unitquat_to_rotmat(q), t)
    return pose
  
  def __getitem__(self, idx:int):
    return self.forward(torch.tensor([idx], device=self.t.device)).squeeze(0)
  

  def normalize(self):
    self.q.data = F.normalize(self.q.data, dim=-1)
    return self

  def __len__(self):
    return self.t.shape[0]


  

if __name__ == '__main__':
  torch.set_printoptions(precision=6, sci_mode=False)
  torch.manual_seed(0)

  for i in range(10):
    q = F.normalize(torch.randn(10, 4))
    m = roma.unitquat_to_rotmat(q)

    assert roma.rotmat_to_unitquat(m).allclose(q), f"{roma.rotmat_to_unitquat(m)} != {q}"
