from dataclasses import dataclass, replace
from typing import Tuple
import time

import nerfview
import torch
import viser

from taichi_splatting.perspective import CameraParams
from splat_trainer.dataset import Dataset
from splat_trainer.scene import GaussianScene


@dataclass(frozen=True)
class NerfviewConfig:
  port: int = 8080

  def create_viewer(self, dataset: Dataset, scene: GaussianScene):
      return Viewer(self, dataset, scene)

class Viewer():
  def __init__(self, config: NerfviewConfig, dataset: Dataset, scene: GaussianScene):
    self.config = config
    
    self.device = config.device
    self.dataset = dataset
    self.scene = scene
    self.tic = None
    self.server = viser.ViserServer(port=self.config.port, verbose=False)
    self.viewer = nerfview.Viewer(
      server=self.server,
      render_fn=self._viewer_render_fn,
      mode="training",)




  def update(self, step: int):
    image = self.dataset.all_cameras[0].image
    num_train_steps_per_sec = self.config.log_interval / (time.time() - self.tic)

    
    num_train_rays_per_step = image.shape[0] * image.shape[1] * image.shape[2]
    num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec

    self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
    self.viewer.update(step, num_train_rays_per_step)

  
  def _viewer_render_fn(self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
    c2w = camera_state.c2w
    K = camera_state.get_K(img_wh)
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    near, far = self.dataset.depth_range()
    camera_params = CameraParams(projection=torch.tensor([fx, fy, cx, cy], device=self.device),
                                T_camera_world=torch.from_numpy(c2w).float().to(self.device),
                                near_plane=near,
                                far_plane=far,
                                image_size=img_wh)

    config = replace(self.config.raster_config, compute_split_heuristics=True,
                    antialias=self.config.antialias,
                    blur_cov=self.config.blur_cov)
    rendering = self.scene.render(camera_params, config, 0)

    return rendering.image.detach().cpu().numpy()