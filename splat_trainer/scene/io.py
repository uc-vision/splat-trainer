
from pathlib import Path
import tempfile
import plyfile

import numpy as np
import torch
import torch.nn.functional as F

from taichi_splatting import Gaussians3D


def to_plydata(gaussians:Gaussians3D, with_sh:bool=False) -> plyfile.PlyData:
  gaussians = gaussians.to('cpu')


  dtype = [
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('opacity', 'f4'),
    ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
    ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
   ] 
  
  if with_sh is True:
    assert gaussians.feature.ndim == 3, f"Expected ndim=3 sh_feature tensor, got {gaussians.feature.shape}"
    num_sh = gaussians.feature.shape[2] *  gaussians.feature.shape[1]


    dtype += [('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')] + [
      ('f_rest_' + str(i), 'f4') for i in range(num_sh - 3)]
  else:
    assert gaussians.feature.ndim == 2, f"Expected ndim=2 feature tensor, got {gaussians.feature.shape}"

    num_features = gaussians.feature.shape[1] 
    dtype += [('f_' + str(i), 'f4') for i in range(num_features)]

  vertex = np.zeros(gaussians.batch_size[0], dtype=dtype )

  for i, name in enumerate(['x', 'y', 'z']):
    vertex[name] = gaussians.position[:, i].numpy()

  for i in range(3):
    vertex['scale_' + str(i)] = gaussians.log_scaling[:, i].numpy()

  rotation = torch.roll(F.normalize(gaussians.rotation), 1, dims=(1,))
  for i in range(4):
    vertex['rot_' + str(i)] = rotation[:, i].numpy()

  vertex['opacity'] = gaussians.alpha_logit[:, 0].numpy()

  if with_sh is True:
    sh_dc, sh_rest = gaussians.feature[:, :, 0], gaussians.feature[:, :, 1:]

    sh_dc = sh_dc.view(-1, 3)
    sh_rest = sh_rest.reshape(sh_rest.shape[0], sh_rest.shape[1] * sh_rest.shape[2])

    for i in range(3):
      vertex['f_dc_' + str(i)] = sh_dc[:, i].numpy()

    for i in range(sh_rest.shape[1]):
      vertex['f_rest_' + str(i)] = sh_rest[:, i].numpy()
  else:
    for i in range(num_features):
      vertex['f_' + str(i)] = gaussians.feature[:, i].numpy()

  el = plyfile.PlyElement.describe(vertex, 'vertex')
  return plyfile.PlyData([el])


def from_plydata(plydata:plyfile.PlyData, with_sh:bool=False) -> Gaussians3D:
  
  vertex = plydata['vertex']

  def get_keys(ks):
    values = [torch.from_numpy(vertex[k].copy()) for k in ks]
    return torch.stack(values, dim=-1)

  
  positions = torch.stack(
    [ torch.from_numpy(vertex[i].copy()) for i in ['x', 'y', 'z']], dim=-1)

  attrs = sorted(plydata['vertex'].data.dtype.names)
  log_scaling = get_keys([f'scale_{k}' for k in range(3)])
  
  if with_sh is True:
    sh_attrs = [k for k in attrs if k.startswith('f_rest_') or k.startswith('f_dc_')]
    
    n_sh = len(sh_attrs) // 3
    deg = int(np.sqrt(n_sh))

    assert deg * deg == n_sh, f"SH feature count must be square ({deg} * {deg} != {n_sh}), got {len(sh_attrs)}"

    sh_dc = get_keys([f'f_dc_{k}' for k in range(3)]).view(positions.shape[0], 3, 1)
    sh_rest = get_keys([f'f_rest_{k}' for k in range(3 * (deg * deg - 1))])
    sh_rest = sh_rest.view(positions.shape[0], 3, n_sh - 1)

    features = torch.cat([sh_dc, sh_rest], dim=2)  
  else:
    feature_attrs = [k for k in attrs if k.startswith('f_')]
    features = get_keys([f'f_{k}' for k in range(len(feature_attrs))])

  rotation = get_keys([f'rot_{k}' for k in range(4)])
  # convert from wxyz to xyzw quaternion and normalize
  rotation = torch.roll(F.normalize(rotation), -1, dims=(1,))
  
  alpha_logit = get_keys(['opacity'])

  
  return Gaussians3D(
    position = positions, 
    rotation = rotation,
    alpha_logit = alpha_logit,
    log_scaling = log_scaling,
    feature = features,

    batch_size = (positions.shape[0],)
  )

def write_gaussians(filename:Path | str, gaussians:Gaussians3D, with_sh:bool = True):
  filename = Path(filename)

  plydata = to_plydata(gaussians.apply(torch.detach), with_sh=with_sh)
  plydata.write(str(filename))



def read_gaussians(filename:Path | str, with_sh:bool = True) -> Gaussians3D:
  filename = Path(filename) 

  plydata = plyfile.PlyData.read(str(filename))
  return from_plydata(plydata, with_sh=with_sh)


  

def random_gaussians(n:int, sh_degree:int):
  points = torch.randn(n, 3)

  return Gaussians3D( 
    position = points,
    rotation = F.normalize(torch.randn(n, 4), dim=1),
    alpha_logit = torch.randn(n, 1),
    log_scaling = torch.randn(n, 3) * 4,
    feature = torch.randn(n, 3, (sh_degree + 1)**2),

    batch_size = (n,)
  )

def test_read_write():
  temp_path = Path(tempfile.mkdtemp())

  print("Testing write/read")
  for i in range(10):
    g = random_gaussians((i + 1) * 1000, 3)
    write_gaussians(temp_path / f'gaussians_{i}.ply', g)
    g2 = read_gaussians(temp_path / f'gaussians_{i}.ply')

    assert torch.allclose(g.position, g2.position)
    assert torch.allclose(g.rotation, g2.rotation)
    assert torch.allclose(g.alpha(), g2.alpha())
    assert torch.allclose(g.log_scaling, g2.log_scaling)
    assert torch.allclose(g.feature, g2.feature)





if __name__ == '__main__':
  test_read_write()


