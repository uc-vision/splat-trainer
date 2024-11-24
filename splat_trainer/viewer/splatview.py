from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
import time
from typing import Any

from beartype import beartype
import splatview
from termcolor import colored
import torch
import viser

from taichi_splatting.perspective import CameraParams

from splat_trainer.trainer import Trainer, TrainerState

from .viewer import ViewerConfig, Viewer
from .logger import StatsLogger


@dataclass(frozen=True)
class SplatviewConfig(ViewerConfig):
  port: int = 8080
  host: str = "0.0.0.0"

  @beartype
  def create_viewer(self, trainer: Trainer) -> 'SplatviewViewer':
      return SplatviewViewer(self, trainer)


class SplatviewViewer(Viewer):
  def __init__(self, config: SplatviewConfig, trainer: Trainer, enable_training: bool = False):
    self.config = config  
    self.trainer = trainer

    self.server = viser.ViserServer(port=self.config.port, verbose=False)
    address = f'http://{self.config.host}:{self.config.port}'
    print(f"Running viewer on: {colored(address, 'light_magenta')}")

    # up_direction = 

    self.logger = StatsLogger()

    trainer.add_logger(self.logger)
    trainer.bind(on_update=self.update)
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
      self.logger_folder.update(self.server, self.logger.current)
      self.update_training()

    while self.trainer.state == TrainerState.Paused:
      self.viewer.update()
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

    projection = camera.projection
    near, far = self.trainer.depth_range

    camera_params = CameraParams(projection=torch.tensor(projection, device=self.trainer.device),
                                T_camera_world=torch.from_numpy(camera.camera_t_world).to(self.trainer.device),
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