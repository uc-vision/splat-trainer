from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import torch

from torch.utils.tensorboard import SummaryWriter

from taichi_splatting import perspective
from tqdm import tqdm

from splat_trainer.scene.gaussians import Scene, LearningRates

@dataclass 
class TrainConfig:
  model_path: str
  device: str
  load_model: Optional[str] = None
  iterations: int = 30000
  learning_rates: LearningRates = LearningRates()



class Trainer:
  def __init__(self, dataset, config):

    self.device = torch.device(config.device)
    self.model_path = Path(config.model_path)

    self.dataset = dataset
    self.config = config

    self.camera_poses = dataset.camera_poses().to(self.device)
    self.camera_projection = dataset.camera_projection().to(self.device)

    if config.load_model:
      self.scene = Scene.load_model(
        self.model_path, config.load_model, lr=config.learning_rates)
    else:
      pcd = dataset.pointcloud()
      self.scene = Scene.from_pointcloud(pcd, lr=config.learning_rates)
      
    self.scene.to(self.device)
    

  def camera_params(self, cam_idx, image):
      near, far = self.dataset.depth_range

      return perspective.CameraParams(
          T_camera_world=self.camera_poses(cam_idx.unsqueeze(0)),
          T_image_camera=self.camera_projection[cam_idx[1]],
          image_size=(image.shape[1], image.shape[0]),
          near_plane=near,
          far_plane=far
      ).to(self.device)


  def train(self):

    print("Writing to model path", self.model_path)
    self.model_path.mkdir(parents=True, exist_ok=True)

    self.logger = SummaryWriter(log_dir = str(self.model_path))

    pbar = tqdm(total=self.config.iterations, desc="Training")
    iteration = 0

    while True:
      for filename, image, cam_idx in self.dataset.train():
        image, idx = [x.to(self.device, non_blocking=True) for x in (image, cam_idx)]
        camera_params = self.camera_params(cam_idx, image)
        
        iteration += 1
        if iteration % 10 == 0:
          pbar.update(10)

        