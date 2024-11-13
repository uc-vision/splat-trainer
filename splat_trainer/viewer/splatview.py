from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
import time
from typing import Any

import splatview
from termcolor import colored
import torch
import viser

from taichi_splatting.perspective import CameraParams

from splat_trainer.trainer import Trainer, TrainerState
from splat_trainer.viewer import ViewConfig

from .logger import ViewLogger


@dataclass(frozen=True)
class SplatviewConfig(ViewConfig):
  port: int = 8080
  host: str = "0.0.0.0"

  def create_viewer(self, trainer: Trainer):
      return SplatviewViewer(self, trainer)


class SplatviewViewer():
  def __init__(self, config: SplatviewConfig, trainer: Trainer, enable_training: bool):
    self.config = config  
    self.trainer = trainer

    self.server = viser.ViserServer(port=self.config.port, verbose=False)
    address = f'http://{self.config.host}:{self.config.port}'
    print(f"Running viewer on: {colored(address, 'light_magenta')}")

    self.logger = ViewLogger()

    trainer.add_logger(self.logger)
    trainer.on_update.bind(self.update)
    self.create_ui(enable_training)

    self.viewer = splatview.Viewer(
      server=self.server,
      render_fn=self.render_fn)
    


  def create_ui(self,  enable_training: bool):
    with self.server.atomic():
      self.logger_folder = Section.create(self.server, "Log", {})
      
      with self.server.gui.add_folder("Progress") as self.progress_folder:    
        self.training = self.server.gui.add_markdown(self.progress_text)
        self.progress = self.server.gui.add_progress_bar(self.trainer.t * 100.)


      with self.server.gui.add_folder("Training", visible=enable_training) as self.training_folder:    
        self.pause_train_button = self.server.gui.add_button("Pause")
        self.pause_train_button.on_click(partial(self.on_pause_train, True))
        self.resume_train_button = self.server.gui.add_button("Resume")
        self.resume_train_button.on_click(partial(self.on_pause_train, False))
          

  @property
  def progress_text(self):
    return f"<sub>{self.trainer.step}/{self.trainer.total_steps}</sub>"

  def update_training(self):
    self.training.content = self.progress_text
    self.progress.value = self.trainer.t * 100.

    self.resume_train_button.visible = self.trainer.state == TrainerState.Paused
    self.pause_train_button.visible = self.trainer.state == TrainerState.Training


  def on_pause_train(self, pause: bool):
    self.trainer.set_paused(pause)
    self.update_training()


  def update(self):
    with self.server.atomic():
      self.logger_folder.update(self.server, self.logger.sections)
      self.update_training()

    while self.trainer.state == TrainerState.Paused:
      self.viewer.update()
      time.sleep(0.1)



  def spin(self):
    print("Running viewer...")
    while True:
      self.viewer.update()
      time.sleep(0.1)
    
  def render_fn(self, camera: splatview.Camera):

    projection = [*camera.focal_length, *camera.principal_point]
    near, far = camera.depth_range()

    camera_params = CameraParams(projection=torch.tensor(projection, device=self.device),
                                T_camera_world=torch.from_numpy(camera.T_camera_world).to(self.device),
                                near_plane=near,
                                far_plane=far,
                                image_size=camera.image_size)

    rendering = self.trainer.render(camera_params, render_median_depth=True)
    return rendering.image.detach().cpu().numpy(), rendering.median_depth.detach().cpu().numpy()
  



class Section(metaclass=ABCMeta):
  @staticmethod
  def create(server: viser.ViserServer, name:str, value: Any):
    if isinstance(value, dict):
      return FolderSection(server.gui.add_folder(name), {})
    else:
      return MarkdownSection(server.gui.add_markdown(f"{name}: {value}"))
    
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