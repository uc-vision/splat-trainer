
from dataclasses import dataclass
from typing import Optional
from tensordict import TensorDict, tensorclass
import torch

from splat_trainer.util.pointcloud import PointCloud
from splat_viewer.gaussians import Workspace

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting import Gaussians3D
    

@dataclass 
class LearningRates:
  position: float = 0.001
  log_scaling: float = 0.005
  rotation: float = 0.001
  alpha_logit: float = 0.05

  base_sh: float = 0.0025
  higher_sh: float = 0.0002
  



def scale_gradients(packed, sh_features, lr:LearningRates):
  scales = torch.tensor(3 * [lr['position']] +
                        3 * [lr['log_scaling']] +
                        4 * [lr['rotation']] +
                        [lr['alpha_logit']])

  packed.grad *= scales.unsqueeze(0)

  sh_features.grad[..., 0] *= torch.tensor(lr['base_sh'])
  sh_features.grad[..., 1:] *= torch.tensor(lr['higher_sh'])


@tensorclass
class PackedPoints:
  gaussians3d: torch.Tensor  # (N, 11)
  sh_feature: torch.Tensor  # (N, (D+1)**2)


class Scene:
  def __init__(self, points: Gaussians3D, lr:LearningRates):

    self.lr = lr
    
    packed = TensorDict(
      gaussians3d=points.packed(),
      sh_feature=points.feature
    )

    self.points = ParameterClass.create(packed, 
      learning_rates=dict(gaussians3d = 1.0, sh_feature = 1.0))


  def step(self):


    scale_gradients(self.points.gaussians3d, self.points.sh_feature, self.lr)
    self.points.step()



  @staticmethod
  def initialize(points: PointCloud):
    pass

  @staticmethod 
  def load_model(workspace_path:str, model_name:Optional[str] = None):
    workspace = Workspace.load(workspace_path)
    gaussians = workspace.load_model(model_name)

    return Scene(gaussians)

