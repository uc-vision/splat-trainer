from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, replace
from functools import partial
import time
from typing import Any

from beartype import beartype
import numpy as np
from termcolor import colored
import torch
import torch.nn.functional as F


import viser
import splatview
from splatview.util import with_traceback


from taichi_splatting.perspective import CameraParams

from splat_trainer.camera_table.camera_table import Camera, Cameras
from splat_trainer.trainer import Trainer, TrainerState
from splat_trainer.visibility.query_points import foreground_points

from .viewer import ViewerConfig, Viewer
from .logger import StatsLogger

# Hack to work around a bug in torch 
# RuntimeError: lazy wrapper should be called at most once
torch.inverse(torch.ones((1, 1), device="cuda:0"))


@dataclass(frozen=True)
class SplatviewConfig(ViewerConfig):
  port: int = 8080
  host: str = "0.0.0.0"

  @beartype
  def create_viewer(self, trainer: Trainer, enable_training: bool = False) -> 'SplatviewViewer':
      return SplatviewViewer(self, trainer, enable_training)


def to_splatview_camera(camera: Cameras) -> splatview.Camera:
  return splatview.Camera(
    fov = camera.projection.fov[0].item(),
    aspect=camera.projection.aspect_ratio.item(),
    world_t_camera=camera.world_t_camera.cpu().numpy(),
    image_size=camera.projection.image_size.cpu().numpy())


class SplatviewViewer(Viewer):
  def __init__(self, config: SplatviewConfig, trainer: Trainer, enable_training: bool = False):

    self.config = config  
    self.trainer = trainer
    self.zoom = 1.0
    self.far_modifier = 1.0
    self.near_modifier = 1.0
    self.server = viser.ViserServer(port=self.config.port, verbose=False)

    self.logger = StatsLogger()

    # trainer.add_logger(self.logger)
    trainer.bind(on_update=self.update)
    self.original_scene = self.trainer.scene.clone()

    self.create_ui(enable_training)

    self.cameras = self.trainer.camera_table.cameras

    up = F.normalize(self.cameras.up).mean(dim=0)
    self.server.scene.set_up_direction(up.cpu().numpy())

    self.viewer = splatview.Viewer(
      server=self.server,
      render_fn=self.render_fn,
      config = splatview.RenderConfig(initial_view=self.get_camera(0)))
    
  def get_camera(self, idx:int) -> splatview.Camera:
    return to_splatview_camera(self.trainer.camera_table.cameras[idx])


  def create_ui(self,  enable_training: bool):
    with self.server.atomic():
      self.logger_folder = Section.create(self.server, "Log", {})
      
      with self.server.gui.add_folder("Progress") as self.progress_folder:    
        self.training = self.server.gui.add_markdown(self.progress_text)
        self.progress = self.server.gui.add_progress_bar(self.trainer.progress.t * 100.)

        self.paused_checkbox = self.server.gui.add_checkbox("Paused", initial_value=self.trainer.state == TrainerState.Paused, disabled=not enable_training)
        self.paused_checkbox.on_update(self.on_pause_train)

      with self.server.gui.add_folder("Camera") as self.camera_folder:
        self.camera_slider = self.server.gui.add_slider("Camera", 0, self.trainer.camera_table.num_images - 1, step=1, initial_value=0)
        self.camera_slider.on_update(self.on_set_camera)

        self.show_depth = self.server.gui.add_checkbox("Show Depth", initial_value=False)
        self.show_depth.on_update(self.on_show_depth)

        self.depth_scale = self.server.gui.add_slider("Depth Scale", min=1, max=10, step=0.1, initial_value=3)
        self.depth_scale.on_update(self.on_set_depth_scale)

        self.zoom_slider = self.server.gui.add_slider("Zoom", min=0.1, max=10, step=0.1, initial_value=1.0)
        self.zoom_slider.on_update(self.on_set_zoom)

        self.far_slider = self.server.gui.add_slider("Far", min=0.1, max=10., step=0.01, initial_value=1.0)
        self.far_slider.on_update(self.on_set_far)

        self.near_slider = self.server.gui.add_slider("Near", min=0.1, max=1, step=0.01, initial_value=0.1)
        self.near_slider.on_update(self.on_set_near)


      with self.server.gui.add_folder("Renderer") as self.renderer_folder:
        self.antialias_checkbox = self.server.gui.add_checkbox("Antialias", initial_value=self.trainer.config.antialias)
        self.antialias_checkbox.on_update(self.on_set_antialias)

        self.specular_checkbox = self.server.gui.add_checkbox("Enable Specular", initial_value=True)
        self.specular_checkbox.on_update(self.on_set_specular)

      with self.server.gui.add_folder("Scene") as self.scene_folder:
        self.quantile_slider = self.server.gui.add_slider("Depth quantile", min=0.0, max=1.0, step=0.01, initial_value=1.0)
        self.min_overlap_slider = self.server.gui.add_slider("Min overlap", min=0.0, max=1.0, step=0.01, initial_value=0.01)

        self.crop_button = self.server.gui.add_button("Crop")
        self.crop_button.on_click(self.on_crop)
        

  @property
  def progress_text(self):
    progress = self.trainer.progress
    return f"<sub>{progress.step}/{progress.total_steps}</sub>"

  def update_training(self):
    self.training.content = self.progress_text
    self.progress.value = self.trainer.progress.t * 100.

    self.paused_checkbox.value = self.trainer.state == TrainerState.Paused
    
  @with_traceback
  def on_pause_train(self, event: viser.GuiEvent[viser.GuiCheckboxHandle]):
    self.trainer.set_paused(event.target.value)

  @property
  def current_camera_idx(self) -> int:
    return self.camera_slider.value
  
  @property
  def current_camera(self) -> Camera:
    return self.trainer.camera_table.cameras[self.current_camera_idx].item()
  
  @with_traceback
  def on_set_zoom(self, event: viser.GuiEvent[viser.GuiSliderHandle]):
    self.zoom = event.target.value
    self.viewer.client(event.client.client_id).redraw()

  @with_traceback
  def on_set_far(self, event: viser.GuiEvent[viser.GuiSliderHandle]):
    self.far_modifier = event.target.value
    self.viewer.client(event.client.client_id).redraw()

  @with_traceback
  def on_set_near(self, event: viser.GuiEvent[viser.GuiSliderHandle]):
    self.near_modifier = event.target.value
    self.viewer.client(event.client.client_id).redraw()


  @with_traceback
  def on_set_antialias(self, event: viser.GuiEvent[viser.GuiCheckboxHandle]):
    self.trainer.update_config(antialias=event.target.value)
    self.viewer.client(event.client.client_id).redraw()

  @with_traceback
  def on_set_specular(self, event: viser.GuiEvent[viser.GuiCheckboxHandle]):
    scene_config = self.trainer.config.scene

    if not hasattr(scene_config, 'enable_specular'):
      print("Warning: scene config does not have enable_specular field")
      return
    

    self.trainer =self.trainer.replace(
      scene=replace(scene_config, enable_specular=event.target.value))
    self.viewer.client(event.client.client_id).redraw()

  @with_traceback
  def on_crop(self, event: viser.GuiEvent[viser.GuiButtonHandle]):
    quantile = self.quantile_slider.value
    min_overlap = self.min_overlap_slider.value

    mask = foreground_points(self.cameras, self.original_scene.gaussians.position, quantile=quantile, min_overlap=min_overlap)
    scene = self.original_scene.clone()
    scene.split_and_prune(keep_mask=mask)

    self.trainer.scene = scene
    self.viewer.client(event.client.client_id).redraw()


  @with_traceback
  def on_set_camera(self, event: viser.GuiEvent[viser.GuiSliderHandle]):
    self.viewer.client(event.client.client_id).set_camera(self.get_camera(event.target.value))

  @with_traceback
  def on_show_depth(self, event: viser.GuiEvent[viser.GuiCheckboxHandle]):
    self.show_depth = event.target.value
    self.viewer.client(event.client.client_id).show_depth(self.show_depth)

  @with_traceback
  def on_set_depth_scale(self, event: viser.GuiEvent[viser.GuiSliderHandle]):
    near = self.current_camera.near
    self.viewer.set_depth_scale(event.target.value * near)
    


  def update(self):
    with self.server.atomic():
      self.logger_folder.update(self.server, self.logger.current)
      self.update_training()
      self.viewer.update(self.trainer.state == TrainerState.Training)

    while self.viewer.last_moved > time.time() - 0.2:
      time.sleep(0.1)




  def spin(self):
    print("Running viewer...")
    while True:
      self.viewer.update()
      time.sleep(0.1)

  def wait_for_exit(self):
    print(f"Waiting for {self.viewer.num_clients} clients to exit...")
    while self.viewer.num_clients > 0:
      self.viewer.update()
      time.sleep(0.1)

  @beartype
  @torch.no_grad()
  def render_fn(self, camera: splatview.Camera):

    camera = camera.zoomed(self.zoom)
    projection = camera.projection
    
    near, far = self.current_camera.depth_range

    img_idx = None
    world_t_camera = torch.from_numpy(camera.world_t_camera).to(device=self.trainer.device, dtype=torch.float32)
    
    # if torch.allclose(world_t_camera, current.world_t_camera, rtol=1e-3, atol=1e-3):
    #   img_idx = self.current_camera

    camera_params = CameraParams(projection=torch.tensor(projection, device=self.trainer.device),
                                 T_camera_world=torch.inverse(world_t_camera),
                                 near_plane=near * self.near_modifier, far_plane=far * self.far_modifier,
                                 image_size=camera.image_size)
    
    camera_params = camera_params.to(device=self.trainer.device, dtype=torch.float32)


    rendering = self.trainer.render(camera_params, image_idx=img_idx, render_median_depth=True)
    return rendering.image.detach().cpu().numpy(), rendering.median_depth_image.detach().cpu().numpy()
  


class Section(metaclass=ABCMeta):
  @staticmethod
  def create(server: viser.ViserServer, name:str, value: Any):
    if isinstance(value, dict):
      return FolderSection(server.gui.add_folder(name, expand_by_default=False), {})
    else:
      return MarkdownSection(server.gui.add_markdown(f"{name}: {value}"), name)
    
  @abstractmethod
  def update(self, server: viser.ViserServer, value: Any):
    pass

@dataclass
class FolderSection:
  folder: viser.GuiFolderHandle
  children: dict[str, 'Section']

  def update(self, server: viser.ViserServer, log_entries:dict[str, Any]):
    with self.folder:
      for key, value in log_entries.items():
        if key in self.children:
          self.children[key].update(server, value)
        else:
          self.children[key] = Section.create(server, key, value)

@dataclass
class MarkdownSection:
  markdown: viser.GuiMarkdownHandle
  name: str

  def update(self, server: viser.ViserServer, value: Any):
    self.markdown.content = f"{self.name}: {value}"